"""随机噪声对照实验的训练入口。

与 ``char_token/train.py`` 唯一的区别在 **数据来源**：EEG 数值被替换成随机噪声。
损失函数、分组采样、train/val 按句子切分、检索评测、CSV 日志全部复用 ``char_token``
里的同一份实现，从而保证"信号 vs 噪声"是干净的单变量对照。

参照读法
--------
``char_token_noise`` 的 val 指标若落在随机基线附近，就说明 ``char_token`` 的高于随机
的成绩来自真实脑电；若仍显著更高，则说明模型在利用与信号无关的旁路信息（例如 token
个数 = 句子长度），需要回头检查 ``char_token`` 的设计。

用法:

    uv run python char_token_noise/train.py
    uv run python char_token_noise/train.py --day 1 2 3 4 --n-sentences 32 --n-repeats 4
    uv run python char_token_noise/train.py --noise-mode uniform --noise-std 0.5
    uv run python char_token_noise/train.py --noise-refresh never   # 固定噪声，完全可复现
    uv run python char_token_noise/train.py --day 1 2 3 4 --n-sentences 32 --n-repeats 4 --epochs 20 --seed 0 --noise-mode gaussian --noise-std 1.0 --noise-seed 0 --noise-refresh epoch
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 支持 `python char_token_noise/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import char_token_collate
from char_token.model import CharTokenEEGEncoder
from char_token.sampler import SentenceGroupBatchSampler
from char_token.text_encoder import LaBSETextEncoder

# 复用 char_token 的同一份训练/评测实现，保证对照的只有数据来源
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
from char_token_noise.dataset import NoiseTokenEEGDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_noise"


def parse_args():
    parser = argparse.ArgumentParser(
        description="随机噪声对照实验：形状与 char_token 同构，EEG 数值换成随机噪声。"
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
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0；>0 时噪声刷新不生效）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")

    # ---- 噪声相关 ----
    parser.add_argument(
        "--noise-mode", default="gaussian", choices=["gaussian", "uniform"],
        help="噪声分布：gaussian 标准正态（默认），uniform [-1,1] 均匀",
    )
    parser.add_argument("--noise-std", type=float, default=1.0, help="噪声幅度缩放（默认 1.0）")
    parser.add_argument("--noise-seed", type=int, default=0, help="噪声随机种子（默认 0）")
    parser.add_argument(
        "--noise-refresh", default="epoch", choices=["never", "epoch"],
        help="epoch 每轮换噪（默认）；never 固定噪声，完全可复现",
    )

    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"[Noise-Control] device={device} temperature={args.temperature} "
        f"batch={args.n_sentences}句×{args.n_repeats}视图 epochs={args.epochs} "
        f"lr={args.lr} wd={args.weight_decay} schedule={args.lr_schedule} "
        f"select={args.select_by} seed={args.seed}"
    )
    print(
        f"[Noise] mode={args.noise_mode} std={args.noise_std} "
        f"seed={args.noise_seed} refresh={args.noise_refresh}"
    )
    if args.num_workers > 0 and args.noise_refresh == "epoch":
        print("[Warn] num-workers>0 时每轮换噪不会生效（子进程里 dataset 副本的 epoch 不会更新）")

    # ---------------- 数据（与 char_token 同构，仅数值换成噪声） ----------------
    dataset = NoiseTokenEEGDataset(
        event=args.event,
        sample=args.sample,
        subject=args.subject,
        days=args.day,
        char_duration=args.char_duration,
        noise_mode=args.noise_mode,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
        refresh=args.noise_refresh,
    )
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

    # ---------------- 模型（与 char_token 同一结构） ----------------
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
    out_dir = Path(args.out_dir) / f"{args.subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    print(f"\n{header}  (best by {args.select_by})")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        # 噪声模式下让 dataset 感知 epoch（refresh="epoch" 时换噪）
        dataset.set_epoch(epoch)

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

    n_val = metrics["n_val"]
    n_cand = metrics["n_text_candidates"]
    pos_per_anchor = metrics["pos_per_anchor"]
    print(
        f"\n[Done] best {args.select_by} = {best_score:.4f}  （随机噪声对照组）\n"
        f"       eeg2text 候选 {n_cand} 句，随机基线 top-1={1 / n_cand:.4f} "
        f"top-5={min(5, n_cand) / n_cand:.4f}\n"
        f"       text2eeg 候选 {n_val} 条（每 anchor {pos_per_anchor:.2f} 个同句），"
        f"随机基线 top-1={min(pos_per_anchor / n_val, 1.0):.4f} "
        f"top-5={min(5 * pos_per_anchor / n_val, 1.0):.4f}\n"
        f"       checkpoint: {ckpt_path}\n"
        f"       epochs:     {train_epochs_path}\n"
        f"       steps:      {train_steps_path}"
    )
    print(
        "       [判读] 指标若落在随机基线附近 -> char_token 的成绩依赖真实脑电；"
        "若显著更高 -> 存在与信号无关的泄漏通道。"
    )


if __name__ == "__main__":
    main()