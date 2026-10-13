"""按 day 切分的跨天泛化实验训练入口。

与 ``char_token/train.py`` 唯一的区别在 **train / val 的切分方式**：
``char_token`` 用 ``split_by_sentence`` 随机按句子做 8:2 切分；本实验改用
``split_by_day`` 按天整体切分（默认 sub-01 的 day-05/06/07 做 train，day-08 做 val）。
损失、分组采样、模型、评测、CSV 日志全部复用 ``char_token`` 里的同一份实现，
从而保证"切分口径"是唯一变量。

重要：day-05 ~ day-08 是同一批 252 句的重复录制（每天 4 个 run，每 run 一整遍），
train / val 的句子集合完全重合，因此这里的 val 指标衡量的是"同一句子在未见过的
某一天录到的信号"的跨天泛化，与 ``split_by_sentence``（新句子泛化）不可直接比较。

用法::

    uv run python char_token_day_holdout/train.py
    uv run python char_token_day_holdout/train.py --epochs 20 --seed 0
    uv run python char_token_day_holdout/train.py --train-days 5 6 7 --val-days 8

    uv run python char_token_day_holdout/train.py --train-days 2 3 4 --val-days 1
    uv run python char_token_day_holdout/train.py --train-days 1 3 4 --val-days 2
    uv run python char_token_day_holdout/train.py --train-days 1 2 4 --val-days 3
    uv run python char_token_day_holdout/train.py --train-days 1 2 3 --val-days 4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 支持 `python char_token_day_holdout/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import CharTokenEEGDataset, char_token_collate
from char_token.model import CharTokenEEGEncoder
from char_token.sampler import SentenceGroupBatchSampler
from char_token.text_encoder import LaBSETextEncoder

# 复用 char_token 的同一份训练/评测实现，保证对照的只有切分方式
from char_token.train import (
    EPOCH_LOG_FIELDS,
    TRAIN_STEP_FIELDS,
    append_csv_row,
    build_text_lookup,
    dump_args_json,
    evaluate,
    set_seed,
    train_one_epoch,
)
from char_token_day_holdout.split import split_by_day

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_day_holdout"


def parse_args():
    parser = argparse.ArgumentParser(
        description="按 day 切分的跨天泛化实验：整天划到同一侧，其余复用 char_token。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试，如 sub-01（默认）")
    parser.add_argument(
        "--train-days", type=int, nargs="+", default=[5, 6, 7],
        help="划给 train 的 day 编号；默认 5 6 7",
    )
    parser.add_argument(
        "--val-days", type=int, nargs="+", default=[8],
        help="划给 val 的 day 编号；默认 8",
    )
    parser.add_argument(
        "--event", default="inner", choices=["inner", "reading"],
        help="取 inner（默读，默认）或 reading（朗读）",
    )
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument("--char-duration", type=float, default=0.4, help="每个字 token 的时长（秒，默认 0.4）")
    parser.add_argument(
        "--n-sentences", type=int, default=18,
        help="每个训练 batch 里不同句子的数量（默认 18；252 句可整除，14 batch/epoch）",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=3,
        help="每个句子在每个 batch 里抽多少条 trial 当正样本（默认 3）",
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
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args()


def run_experiment(args, out_dir, text_encoder, device):
    """完整跑一遍训练 + 评测，返回 ``(best_score, best_metrics, info)``。

    训练/评测循环与 ``char_token`` 完全一致，只有 ``split_by_day`` 这一处不同。
    """
    set_seed(args.seed)

    subject = args.subject
    print(f"\n{'=' * 88}\n[Subject] {subject}  (split by day)\n{'=' * 88}")

    # ---------------- 数据（train / val 的 day 合并成同一份数据集） ----------------
    all_days = sorted(set(int(d) for d in args.train_days) | set(int(d) for d in args.val_days))
    dataset = CharTokenEEGDataset(
        event=args.event,
        sample=args.sample,
        subject=subject,
        days=all_days,
        char_duration=args.char_duration,
    )
    dataset[0]  # 触发一次读取，确定 n_chans
    train_set, val_set, info = split_by_day(dataset, args.train_days, args.val_days)

    days_label = ",".join(f"{d:02d}" for d in all_days)
    print(
        f"[Data] {subject} day-{days_label} event={args.event} "
        f"token_samples={dataset.token_samples} n_chans={dataset.n_chans}\n"
        f"       切分 train day={info['train_days']} / val day={info['val_days']}\n"
        f"       trial train {info['n_train']} / val {info['n_val']}；"
        f"句子 train {info['n_unique_train']} / val {info['n_unique_val']}，"
        f"句子重叠={info['sentence_overlap']}"
    )
    if info["sentence_overlap"] == info["n_unique_val"]:
        print(
            "       [注意] val 的句子全部在 train 里出现过 -> 这是**跨天（跨 session）"
            "泛化**，不是新句子泛化。"
        )

    # ---------------- 分组 batch sampler ----------------
    # 一句最多能提供多少条 trial：同句跨天/跨 run 重复才有多个视图
    all_sentences = dataset.meta["sentence"].tolist()
    train_sentences = [all_sentences[i] for i in train_set.indices]
    max_repeats = max(Counter(train_sentences).values())

    # 把 n_repeats 限到实际可用值，再按用户原本想要的 batch 大小回推 n_sentences
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
    lookup, embedding_dim = build_text_lookup(
        text_encoder, dataset.meta["sentence"].tolist(), device
    )
    print(f"[Text] 已缓存 {len(lookup)} 条句子，embedding_dim={embedding_dim}")

    # ---------------- 模型 ----------------
    model = CharTokenEEGEncoder(
        n_chans=dataset.n_chans,
        token_samples=dataset.token_samples,
        embedding_dim=embedding_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {type(model).__name__} 参数量={n_params:,} "
        f"d_model={model.d_model} layers={len(model.transformer.layers)} "
        f"max_tokens={model.max_tokens}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    # ---------------- 训练 ----------------
    ckpt_path = out_dir / f"{subject}_best.pt"
    train_epochs_path = out_dir / f"{subject}_train_epochs.csv"
    train_steps_path = out_dir / f"{subject}_train_steps.csv"

    lower_is_better = args.select_by == "val_loss"
    best_score = float("inf") if lower_is_better else -float("inf")
    best_metrics = None
    metrics = {}
    header = (
        f"{'epoch':>5} {'lr':>8} {'train_loss':>10} {'pos/anch':>9} {'val_loss':>9} "
        f"{'e2t@1':>7} {'e2t@5':>7} {'t2e@1':>7} {'t2e@5':>7}"
    )
    print(f"\n{header}  (best by {args.select_by})")
    print("-" * len(header))

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

    # 两个方向的候选池不同，随机基线也要分别算
    n_val = metrics["n_val"]
    n_cand = metrics["n_text_candidates"]
    pos_per_anchor = metrics["pos_per_anchor"]
    print(
        f"\n[Done] best {args.select_by} = {best_score:.4f}\n"
        f"       eeg2text 候选 {n_cand} 句，随机基线 top-1={1 / n_cand:.4f} "
        f"top-5={min(5, n_cand) / n_cand:.4f}\n"
        f"       text2eeg 候选 {n_val} 条（每 anchor {pos_per_anchor:.2f} 个同句），"
        f"随机基线 top-1={min(pos_per_anchor / n_val, 1.0):.4f} "
        f"top-5={min(5 * pos_per_anchor / n_val, 1.0):.4f}\n"
        f"       checkpoint: {ckpt_path}\n"
        f"       epochs:     {train_epochs_path}\n"
        f"       steps:      {train_steps_path}"
    )
    return best_score, best_metrics, info


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"[Day-Holdout] device={device} subject={args.subject} "
        f"train_days={args.train_days} val_days={args.val_days} "
        f"temperature={args.temperature} batch={args.n_sentences}句×{args.n_repeats}视图 "
        f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
        f"schedule={args.lr_schedule} select={args.select_by} seed={args.seed}"
    )

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    print("[Text] 加载 LaBSE（冻结） ...")
    text_encoder = LaBSETextEncoder()

    run_experiment(args, out_dir, text_encoder, device)

    print(f"\n[Output] {out_dir}")


if __name__ == "__main__":
    main()
