"""注意力池化 vs 平均池化消融实验的训练入口。

与 ``char_token/train.py`` 唯一的区别在 **模型的池化方式**：数据、损失、分组采样、按句子切分
train/val、检索评测、CSV 日志全部复用 ``char_token`` 里的同一份实现，保证"池化"是唯一变量。

用法::

    uv run python char_token_pooling/train.py --pooling attn
    uv run python char_token_pooling/train.py --pooling mean
    # 一次跑完两种变体（顺序执行）：
    uv run python char_token_pooling/train.py --pooling attn mean
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 支持 `python char_token_pooling/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import CharTokenEEGDataset, char_token_collate
from char_token.sampler import SentenceGroupBatchSampler
from char_token.text_encoder import LaBSETextEncoder

# 复用 char_token 的同一份训练/评测实现，保证对照的只有池化方式
from char_token.train import (
    EPOCH_LOG_FIELDS,
    TRAIN_STEP_FIELDS,
    append_csv_row,
    build_text_lookup,
    evaluate,
    set_seed,
    split_by_sentence,
    train_one_epoch,
)
from char_token_pooling.model import POOLING_MODES, build_encoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_pooling"


def parse_args():
    parser = argparse.ArgumentParser(
        description="注意力池化 vs 平均池化消融实验（数据与其余一切同 char_token）。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试，如 sub-01")
    parser.add_argument(
        "--day", type=int, nargs="+", default=[1],
        help="day 编号（1~8），可传多个，如 --day 1 2 3 4；默认 1",
    )
    parser.add_argument(
        "--event", default="inner", choices=["inner", "reading"],
        help="取 inner（默读，默认）或 reading（朗读）",
    )
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument("--char-duration", type=float, default=0.4, help="每个字 token 的时长（秒，默认 0.4）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集句子比例（默认 0.2）")
    parser.add_argument(
        "--n-sentences", type=int, default=8,
        help="每个训练 batch 里不同句子的数量（默认 8）",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=4,
        help="每个句子在每个 batch 里抽多少条 trial 当正样本（默认 4）",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="验证集 DataLoader 的 batch 大小（默认 32）")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数（默认 30）")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率（默认 1e-3）")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW 权重衰减（默认 1e-3）")
    parser.add_argument(
        "--lr-schedule", default="cosine", choices=["none", "cosine"],
        help="学习率调度：cosine 按 epoch 余弦衰减到 0（默认），none 恒定",
    )
    parser.add_argument(
        "--select-by", default="val_loss",
        choices=["val_loss", "eeg2text_top1", "eeg2text_top5", "text2eeg_top5"],
        help="用哪个指标选 best checkpoint（默认 val_loss，越小越好；其余越大越好）",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="对比学习温度系数（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")

    # ---- 消融变量 ----
    parser.add_argument(
        "--pooling", nargs="+", default=["attn"], choices=list(POOLING_MODES),
        help="池化方式，可传多个顺序跑完，如 --pooling attn mean",
    )

    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args()


def run_variant(args, pooling, dataset, train_set, val_set, train_loader, val_loader,
                lookup, embedding_dim, device):
    """在给定池化方式下完整跑一遍训练 + 评测，返回 (best_score, metrics)。"""
    set_seed(args.seed)

    model = build_encoder(
        pooling,
        n_chans=dataset.n_chans,
        token_samples=dataset.token_samples,
        embedding_dim=embedding_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n{'=' * 88}\n[Variant] pooling={pooling}  参数量={n_params:,} "
        f"d_model={model.d_model} layers={len(model.transformer.layers)}"
        f"\n{'=' * 88}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    out_dir = Path(args.out_dir) / f"{pooling}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    train_epochs_path = out_dir / "train_epochs.csv"
    train_steps_path = out_dir / "train_steps.csv"

    lower_is_better = args.select_by == "val_loss"
    best_score = float("inf") if lower_is_better else -float("inf")
    header = (
        f"{'epoch':>5} {'lr':>8} {'train_loss':>10} {'pos/anch':>9} {'val_loss':>9} "
        f"{'e2t@1':>7} {'e2t@5':>7} {'t2e@1':>7} {'t2e@5':>7}"
    )
    print(header)
    print("-" * len(header))

    metrics = {}
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_pos = train_one_epoch(
            model, train_loader, lookup, optimizer, device, args.temperature,
            epoch=epoch, step_log_path=train_steps_path,
        )
        if scheduler is not None:
            scheduler.step()
        metrics = evaluate(model, val_loader, lookup, device, args.temperature)

        score = metrics[args.select_by]
        improved = score < best_score if lower_is_better else score > best_score
        flag = ""
        if improved:
            best_score = score
            best_metrics = dict(metrics)
            flag = " *"
            torch.save(
                {
                    "model": model.state_dict(),
                    "pooling": pooling,
                    "args": vars(args),
                    "epoch": epoch,
                    "n_chans": dataset.n_chans,
                    "token_samples": dataset.token_samples,
                    "embedding_dim": embedding_dim,
                    "metrics": metrics,
                },
                ckpt_path,
            )

        print(
            f"{epoch:>5} {epoch_lr:>8.2e} {train_loss:>10.4f} {train_pos:>9.2f} "
            f"{metrics['val_loss']:>9.4f} "
            f"{metrics['eeg2text_top1']:>7.4f} {metrics['eeg2text_top5']:>7.4f} "
            f"{metrics['text2eeg_top1']:>7.4f} {metrics['text2eeg_top5']:>7.4f}{flag}"
        )

        append_csv_row(train_epochs_path, EPOCH_LOG_FIELDS, {
            "epoch": epoch,
            "lr": f"{epoch_lr:.6f}",
            "train_loss": f"{train_loss:.6f}",
            "train_pos_per_anchor": f"{train_pos:.6f}",
            "val_loss": f"{metrics['val_loss']:.6f}",
            "n_val": metrics["n_val"],
            "n_unique_sentences": metrics["n_unique_sentences"],
            "pos_per_anchor": f"{metrics['pos_per_anchor']:.6f}",
            "n_text_candidates": metrics["n_text_candidates"],
            "eeg2text_top1": f"{metrics['eeg2text_top1']:.6f}",
            "eeg2text_top5": f"{metrics['eeg2text_top5']:.6f}",
            "text2eeg_top1": f"{metrics['text2eeg_top1']:.6f}",
            "text2eeg_top5": f"{metrics['text2eeg_top5']:.6f}",
            "is_best": int(improved),
            "temperature": args.temperature,
        })

    print(
        f"[Done] pooling={pooling} best {args.select_by}={best_score:.4f}  "
        f"best 轮指标 e2t@1={best_metrics['eeg2text_top1']:.4f} "
        f"e2t@5={best_metrics['eeg2text_top5']:.4f} "
        f"t2e@5={best_metrics['text2eeg_top5']:.4f}\n"
        f"       epochs: {train_epochs_path}"
    )
    return best_score, best_metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"[Pooling-Ablation] device={device} pooling={args.pooling} "
        f"temperature={args.temperature} batch={args.n_sentences}句×{args.n_repeats}视图 "
        f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
        f"schedule={args.lr_schedule} select={args.select_by} seed={args.seed}"
    )

    # ---------------- 数据（与 char_token 完全一致） ----------------
    dataset = CharTokenEEGDataset(
        event=args.event,
        sample=args.sample,
        subject=args.subject,
        days=args.day,
        char_duration=args.char_duration,
    )
    dataset[0]  # 触发一次读取，确定 n_chans
    train_set, val_set, info = split_by_sentence(dataset, args.val_ratio, args.seed)

    days_label = ",".join(f"{d:02d}" for d in sorted(args.day))
    print(
        f"[Data] {args.subject} day-{days_label} event={args.event} "
        f"token_samples={dataset.token_samples} n_chans={dataset.n_chans}\n"
        f"       句子 {len(dataset.meta['sentence'].unique())} 条 -> "
        f"train {info['n_unique_train']} / val {info['n_unique_val']} "
        f"(trial {len(train_set)} / {len(val_set)})，句子重叠={info['sentence_overlap']}"
    )

    # ---------------- 分组 batch sampler（与 char_token 完全一致） ----------------
    all_sentences = dataset.meta["sentence"].tolist()
    train_sentences = [all_sentences[i] for i in train_set.indices]
    max_repeats = max(Counter(train_sentences).values())

    target_batch = args.n_sentences * args.n_repeats
    n_repeats = min(args.n_repeats, max_repeats)
    n_sentences = max(1, target_batch // n_repeats)
    if n_repeats != args.n_repeats:
        print(
            f"[Sampler] 单句最多只有 {max_repeats} 条 trial，n_repeats "
            f"{args.n_repeats} -> {n_repeats}，n_sentences 相应调整为 {n_sentences} "
            f"（保持 batch={target_batch}）"
        )

    train_sampler = SentenceGroupBatchSampler(
        train_sentences, n_sentences=n_sentences, n_repeats=n_repeats, seed=args.seed
    )
    print(
        f"[Sampler] 每 batch {train_sampler.n_sentences} 句 × "
        f"{train_sampler.n_repeats} 视图 = {train_sampler.batch_size} 条；"
        f"可用句子 {len(train_sampler.groups)}/{len(set(train_sentences))}"
        + (
            f"（{len(train_sampler.excluded_sentences)} 句因 trial 不足被排除）"
            if train_sampler.excluded_sentences
            else ""
        )
        + f"；每轮 {len(train_sampler)} 个 batch"
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        collate_fn=char_token_collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=char_token_collate,
        num_workers=args.num_workers,
    )

    # ---------------- 文本 embedding（冻结，一次性算好） ----------------
    print("[Text] 加载 LaBSE 并预计算全部句子 embedding ...")
    text_encoder = LaBSETextEncoder()
    lookup, embedding_dim = build_text_lookup(
        text_encoder, dataset.meta["sentence"].tolist(), device
    )
    print(f"       已缓存 {len(lookup)} 条句子，embedding_dim={embedding_dim}")

    # ---------------- 逐个池化变体训练 ----------------
    summary = {}
    for pooling in args.pooling:
        best_score, metrics = run_variant(
            args, pooling, dataset, train_set, val_set,
            train_loader, val_loader, lookup, embedding_dim, device,
        )
        summary[pooling] = {"best": best_score, **metrics}

    # ---------------- 汇总 ----------------
    if len(summary) > 1:
        print(f"\n{'=' * 88}\n[Summary] 池化消融汇总（best by {args.select_by}）\n{'=' * 88}")
        print(f"{'pooling':>8} {'val_loss':>10} {'e2t@1':>8} {'e2t@5':>8} {'t2e@1':>8} {'t2e@5':>8}")
        for pooling, m in summary.items():
            print(
                f"{pooling:>8} {m['val_loss']:>10.4f} {m['eeg2text_top1']:>8.4f} "
                f"{m['eeg2text_top5']:>8.4f} {m['text2eeg_top1']:>8.4f} {m['text2eeg_top5']:>8.4f}"
            )
        n_cand = next(iter(summary.values()))["n_text_candidates"]
        n_val = next(iter(summary.values()))["n_val"]
        print(
            f"\n[Baseline] eeg2text 候选 {n_cand} 句，随机 top-1={1 / n_cand:.4f} "
            f"top-5={min(5, n_cand) / n_cand:.4f}；text2eeg 候选 {n_val} 条，"
            f"随机 top-5={min(5 * 4 / n_val, 1.0):.4f}"
        )


if __name__ == "__main__":
    main()