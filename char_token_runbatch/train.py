"""按 run 分 batch 的字符级 EEG–文本对比学习（day 5~8）。

实验设计
--------
day 5~8 共 4 天 × 4 run = 16 个 run；每个 run 都包含全部 252 句，且每句在
一个 run 里只出现一次。因此可以：

    一个 run = 一个 batch（252 条 trial，252 个不同句子）

这样 batch 内 run 恒定，模型无法用 "这是哪个 run/主题块" 区分候选；想降低
252 路 InfoNCE，只能利用句子级信息。这正好针对此前 "run 压制句子" 的问题。

采样
----
``RunBatchSampler`` 每个 epoch 打乱 16 个 run 的顺序，逐个作为一个 batch；
batch 内 trial 顺序默认也打乱。InfoNCE 对样本顺序等变，batch 内打乱不改变
loss，只影响日志/可视化顺序。

注意：day 5~8 里同一句在 4 个 run、4 天中的 ``trial_index`` 完全固定，
即句子在 run 内的位置不变。batch 内打乱不能消除这个数据层面的位置/上下文
混淆；如需排除，必须改变 run 内顺序或做 run 中心化对照。

损失 / 模型 / 数据
------------------
本目录自带以下文件，不依赖外部实验代码：

- ``char_token_runbatch.dataset``：``CharTokenEEGDataset`` / ``char_token_collate``
- ``char_token_runbatch.model``：``CharTokenEEGEncoder``
- ``char_token_runbatch.text_encoder``：``LaBSETextEncoder``
- ``char_token_runbatch.common``：``build_text_lookup`` / ``gather_text`` /
  ``sentence_group_ids`` / ``contrastive_loss`` 等工具

本实验先不做验证集：只记录训练 loss，保存 last / best-train checkpoint。

用法::

    uv run python char_token_runbatch/train.py
    uv run python char_token_runbatch/train.py --epochs 50
    uv run python char_token_runbatch/train.py --token-chunk 512   # 显存不足时
    uv run python char_token_runbatch/train.py --max-per-run 16 --max-steps 1  # 冒烟
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

# 支持 `python char_token_runbatch/train.py` 直接执行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_runbatch.common import (
    append_csv_row,
    build_text_lookup,
    contrastive_loss,
    dump_args_json,
    gather_text,
    sentence_group_ids,
    set_seed,
)
from char_token_runbatch.dataset import CharTokenEEGDataset, char_token_collate
from char_token_runbatch.model import CharTokenEEGEncoder
from char_token_runbatch.sampler import RunBatchSampler
from char_token_runbatch.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_runbatch"

EPOCH_FIELDS = [
    "epoch", "lr", "train_loss", "pos_per_anchor",
    "n_batches", "mean_batch_size", "temperature",
]
STEP_FIELDS = [
    "epoch", "step", "day", "run", "loss", "lr", "pos_per_anchor", "batch_size",
]


class RunMetaDataset(Dataset):
    """在 char_token 数据集外再包一层，额外返回 ``(day, run)`` 便于日志。"""

    def __init__(self, base: CharTokenEEGDataset):
        self.base = base
        self.meta = base.meta

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        tokens, sentence = self.base[index]
        row = self.meta.iloc[index]
        return tokens, sentence, int(row["day"]), int(row["run"])


def run_collate(batch):
    """collate 成 ``(tokens, mask, sentences, days, runs)``。"""
    tokens, mask, sentences = char_token_collate(
        [(tokens_i, sentence_i) for tokens_i, sentence_i, _, _ in batch]
    )
    days = torch.tensor([day for _, _, day, _ in batch], dtype=torch.long)
    runs = torch.tensor([run for _, _, _, run in batch], dtype=torch.long)
    return tokens, mask, sentences, days, runs


class ChunkedTokenEncoder(nn.Module):
    """把 token encoder 按样本分块执行，降低大 batch 的峰值显存。

    ``checkpoint`` 在 backward 时重算每块，因此显存约为一块而不是整个 batch；
    CPU 上会增加计算量，默认关闭（``--token-chunk 0``）。
    """

    def __init__(self, base: nn.Module, chunk: int):
        super().__init__()
        self.base = base
        self.chunk = int(chunk)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.chunk <= 0 or tokens.size(0) <= self.chunk:
            return self.base(tokens)

        parts = []
        for part in tokens.split(self.chunk, dim=0):
            if self.training:
                parts.append(checkpoint(self.base, part, use_reentrant=False))
            else:
                parts.append(self.base(part))
        return torch.cat(parts, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="按 (day, run) 分 batch 的字符级 EEG–文本对比学习。")
    parser.add_argument("--subject", default="sub-01", help="被试（默认 sub-01）")
    parser.add_argument(
        "--day", type=int, nargs="+", default=[5, 6, 7, 8],
        help="day 编号（默认 5 6 7 8；共 16 个 run）",
    )
    parser.add_argument("--event", default="inner", choices=["inner", "reading"])
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument("--char-duration", type=float, default=0.4, help="每个字 token 的时长（秒）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（默认 50；每轮 16 个 batch）")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率（默认 1e-3）")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW 权重衰减（默认 1e-3）")
    parser.add_argument("--temperature", type=float, default=0.2, help="对比学习温度系数（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    parser.add_argument(
        "--max-per-run", type=int, default=0,
        help="每个 run 每轮最多取多少条；0=全部（默认），仅用于冒烟/降显存",
    )
    parser.add_argument(
        "--token-chunk", type=int, default=0,
        help="token encoder 分块大小；0=不分块。显存不足时设 256/512",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="每轮最多跑多少 step；0=不限制。仅用于冒烟",
    )
    parser.add_argument(
        "--shuffle-within", action="store_true", default=True,
        help="是否打乱 batch 内 trial 顺序（默认是；不改变 loss）",
    )
    parser.add_argument(
        "--no-shuffle-within", action="store_false", dest="shuffle_within",
        help="不打乱 batch 内 trial 顺序",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"[Config] device={device} subject={args.subject} days={args.day} "
        f"event={args.event} epochs={args.epochs} lr={args.lr} "
        f"temperature={args.temperature} seed={args.seed}"
    )

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    # ---------------- 数据：一个 (day, run) 一个 group ----------------
    base_dataset = CharTokenEEGDataset(
        event=args.event,
        sample=args.sample,
        subject=args.subject,
        days=args.day,
        char_duration=args.char_duration,
    )
    base_dataset[0]  # 触发一次读取，确定 n_chans
    n_chans = base_dataset.n_chans
    token_samples = base_dataset.token_samples

    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in base_dataset.meta.iterrows():
        key = (int(row["day"]), int(row["run"]))
        groups.setdefault(key, []).append(index)

    sizes = sorted({len(indices) for indices in groups.values()})
    print(
        f"[Data] {args.subject} days={sorted(args.day)} event={args.event} "
        f"n_chans={n_chans} token_samples={token_samples}\n"
        f"       trial={len(base_dataset)}  唯一句子={base_dataset.meta.sentence.nunique()}  "
        f"run 数={len(groups)}  每个 run 的 trial 数={sizes}"
    )
    if len(groups) == 0:
        raise ValueError("没有可用的 (day, run) group")

    dataset = RunMetaDataset(base_dataset)
    sampler = RunBatchSampler(
        groups,
        seed=args.seed,
        shuffle=True,
        shuffle_within=args.shuffle_within,
        max_per_run=args.max_per_run,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=run_collate,
        num_workers=args.num_workers,
    )
    print(
        f"[Sampler] 每 epoch {len(sampler)} 个 run/batch；"
        f"batch 内打乱={args.shuffle_within}；"
        f"每 run 取 {args.max_per_run or '全部'} 条"
    )

    # ---------------- 文本 embedding（冻结，一次性算好） ----------------
    print("[Text] 加载 LaBSE（冻结） ...")
    text_encoder = LaBSETextEncoder()
    lookup, embedding_dim = build_text_lookup(
        text_encoder, base_dataset.meta["sentence"].tolist(), device
    )
    print(f"[Text] 已缓存 {len(lookup)} 条句子，embedding_dim={embedding_dim}")

    # ---------------- 模型 ----------------
    model = CharTokenEEGEncoder(
        n_chans=n_chans,
        token_samples=token_samples,
        embedding_dim=embedding_dim,
    ).to(device)
    if args.token_chunk > 0:
        model.token_encoder = ChunkedTokenEncoder(model.token_encoder, args.token_chunk)
        print(f"[Model] token encoder 分块={args.token_chunk}（checkpoint 重算）")
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {type(model).__name__} 参数量={n_params:,} "
        f"d_model={model.d_model} layers={len(model.transformer.layers)} "
        f"max_tokens={model.max_tokens}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    epochs_path = out_dir / f"{args.subject}_train_epochs.csv"
    steps_path = out_dir / f"{args.subject}_train_steps.csv"
    last_path = out_dir / f"{args.subject}_last.pt"
    best_path = out_dir / f"{args.subject}_best_train.pt"

    header = (
        f"{'epoch':>5} {'lr':>8} {'train_loss':>10} {'pos/anch':>9} "
        f"{'batches':>7} {'mean_bs':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    best_train = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss, total_n, total_pos = 0.0, 0, 0.0
        n_batches = 0

        for step, (tokens, mask, sentences, days, runs) in enumerate(loader, start=1):
            # 每个 batch 必须恰好来自一个 (day, run)
            if days.unique().numel() != 1 or runs.unique().numel() != 1:
                raise RuntimeError("一个 batch 内出现了多个 (day, run)")
            day, run = int(days[0]), int(runs[0])

            tokens = tokens.to(device)
            mask = mask.to(device)
            text = gather_text(sentences, lookup, device)
            groups_t = sentence_group_ids(sentences, device)

            eeg = model(tokens, mask)
            loss = contrastive_loss(eeg, text, args.temperature, groups_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch = len(sentences)
            pos = (groups_t[:, None] == groups_t[None, :]).sum().item()
            append_csv_row(steps_path, STEP_FIELDS, {
                "epoch": epoch,
                "step": step,
                "day": day,
                "run": run,
                "loss": f"{loss.item():.6f}",
                "lr": f"{epoch_lr:.6f}",
                "pos_per_anchor": f"{pos / batch:.4f}",
                "batch_size": batch,
            })

            total_loss += loss.item() * batch
            total_n += batch
            total_pos += pos
            n_batches += 1

            if args.max_steps and step >= args.max_steps:
                break

        scheduler.step()
        train_loss = total_loss / max(total_n, 1)
        train_pos = total_pos / max(total_n, 1)
        mean_bs = total_n / max(n_batches, 1)

        append_csv_row(epochs_path, EPOCH_FIELDS, {
            "epoch": epoch,
            "lr": f"{epoch_lr:.6f}",
            "train_loss": f"{train_loss:.6f}",
            "pos_per_anchor": f"{train_pos:.6f}",
            "n_batches": n_batches,
            "mean_batch_size": f"{mean_bs:.2f}",
            "temperature": args.temperature,
        })
        print(
            f"{epoch:>5} {epoch_lr:>8.2e} {train_loss:>10.4f} "
            f"{train_pos:>9.2f} {n_batches:>7} {mean_bs:>8.2f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "n_chans": n_chans,
            "token_samples": token_samples,
            "embedding_dim": embedding_dim,
            "train_loss": train_loss,
        }
        torch.save(ckpt, last_path)
        if train_loss < best_train:
            best_train = train_loss
            torch.save(ckpt, best_path)
            print(f"        * new best train_loss={best_train:.6f}")

    print(
        f"\n[Done] best train_loss={best_train:.6f}\n"
        f"       last : {last_path}\n"
        f"       best : {best_path}\n"
        f"       epochs: {epochs_path}\n"
        f"       steps : {steps_path}"
    )


if __name__ == "__main__":
    main()
