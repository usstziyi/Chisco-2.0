"""按 run 分 batch 的字符级 EEG–文本对比学习（day 5~8）。

实验设计
--------
day 5~8 共 4 天 × 4 run = 16 个 run；每个 run 都包含全部 252 句，且每句在
一个 run 里只出现一次。因此可以：

    一个 run = 一个 batch

验证集
------
从 252 句里按类别分层留出 63 句（默认 25%），训练只用剩下的 189 句：

- 37 个「2 句类」各留 1 句；
- 社交和聚会活动留 14/98，问候和情感状态留 12/80。

清单写到 ``<run>/<subject>_val_split.json``，可用 ``--val-split`` 复用。

训练候选池
----------
训练 query 只来自 189 句，但文本 gallery 使用全部 252 句：189 个训练句 +
63 个验证句。验证句的 EEG 不参与训练，它们的文本只作为额外的 hard
negative，使训练和验证的候选规模一致（252 路）。

验证
----
每个验证 run 的 63 条 EEG 作为 query，在全部 252 句文本里检索，报告
e2t@1 / e2t@5 / mean_rank。每 ``--val-every`` 个 epoch 验证 ``--val-runs``
个 run（默认 16 个，即全部验证 run，每个 epoch 都验满 1008 条 trial）；每
``--split-half-every`` 个 epoch 用全部 16 个验证 run 算一次同句 split-half
一致性。checkpoint 按验证 e2t@1 选择。

本目录自带以下文件，不依赖外部实验代码：

- ``char_token_runbatch.dataset``：``CharTokenEEGDataset`` / ``char_token_collate``
- ``char_token_runbatch.model``：``CharTokenEEGEncoder``
- ``char_token_runbatch.text_encoder``：``LaBSETextEncoder``
- ``char_token_runbatch.split``：``build_val_split`` / ``save_split`` / ``load_split``
- ``char_token_runbatch.common``：``build_text_lookup`` / ``gather_text`` /
  ``contrastive_loss`` 等工具

用法::

    uv run python char_token_runbatch/train.py
    uv run python char_token_runbatch/train.py --epochs 50
    uv run python char_token_runbatch/train.py --epochs 50 --val-runs 16 --val-every 1  # 每 epoch 验证全部 16 个 run
    uv run python char_token_runbatch/train.py --val-fraction 0        # 关闭验证
    uv run python char_token_runbatch/train.py --val-split path/to.json
    uv run python char_token_runbatch/train.py --max-per-run 16 --max-steps 1 --split-half-every 0  # 冒烟
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Subset

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
from char_token_runbatch.split import build_val_split, load_split, save_split
from char_token_runbatch.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_runbatch"

EPOCH_FIELDS = [
    "epoch", "lr", "train_loss", "pos_per_anchor",
    "n_batches", "mean_batch_size", "temperature",
    "val_e2t1", "val_e2t5", "val_mean_rank", "val_loss_e2t", "val_n",
]
STEP_FIELDS = [
    "epoch", "step", "day", "run", "loss", "lr", "pos_per_anchor", "batch_size",
]
VAL_FIELDS = ["epoch", "day", "run", "n", "e2t1", "e2t5", "mean_rank", "loss_e2t"]
SPLIT_FIELDS = [
    "epoch", "n_sentences", "n_splits", "same_sentence", "different_sentence", "gap",
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
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（默认 50）")
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
    parser.add_argument(
        "--val-fraction", type=float, default=0.25,
        help="留作验证的句子比例（默认 0.25；0=关闭验证）",
    )
    parser.add_argument("--val-seed", type=int, default=0, help="分层留句的随机种子（默认 0）")
    parser.add_argument(
        "--val-split", default=None,
        help="复用已有验证句清单 json；不给则按 --val-fraction/--val-seed 重新生成",
    )
    parser.add_argument(
        "--val-every", type=int, default=1,
        help="每多少个 epoch 验证一次（默认 1）",
    )
    parser.add_argument(
        "--val-runs", type=int, default=16,
        help="每次验证使用多少个 run（默认 16，即全部；按字典序轮换）",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=32,
        help="验证时 EEG 编码的 batch 大小（默认 32）",
    )
    parser.add_argument(
        "--split-half-every", type=int, default=10,
        help="每多少个 epoch 算一次 split-half 一致性（默认 10；0=关闭）",
    )
    parser.add_argument(
        "--split-half-splits", type=int, default=100,
        help="split-half 的随机分半次数（默认 100）",
    )
    args = parser.parse_args()

    if args.val_every < 1:
        parser.error("--val-every 必须 >= 1")
    if args.val_runs < 1:
        parser.error("--val-runs 必须 >= 1")
    if args.split_half_every < 0:
        parser.error("--split-half-every 必须 >= 0")
    if args.split_half_splits < 1:
        parser.error("--split-half-splits 必须 >= 1")
    return args


@torch.no_grad()
def encode_indices(model, dataset, indices, device, batch_size: int):
    """编码指定 trial 下标，返回 ``(embedding, sentences)``。"""
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=char_token_collate,
    )
    embeddings, sentences = [], []
    for tokens, mask, batch_sentences in loader:
        embeddings.append(model(tokens.to(device), mask.to(device)))
        sentences.extend(batch_sentences)
    return torch.cat(embeddings, dim=0), sentences


def retrieval_metrics(eeg, gallery, targets, temperature: float) -> dict:
    """EEG→文本检索：``targets[i]`` 是第 i 条 EEG 在 ``gallery`` 中的正确列。"""
    sim = eeg @ gallery.t()
    correct = sim.gather(1, targets[:, None]).squeeze(1)
    rank = (sim > correct[:, None]).sum(dim=1) + 1
    loss = F.cross_entropy(sim / temperature, targets)
    return {
        "n": int(eeg.size(0)),
        "e2t1": float((rank == 1).float().mean()),
        "e2t5": float((rank <= 5).float().mean()),
        "mean_rank": float(rank.float().mean()),
        "loss_e2t": float(loss),
    }


@torch.no_grad()
def evaluate_runs(model, dataset, groups, keys, text_all, col_of, device, batch_size, temperature):
    """在若干验证 run 上做 252 路检索，返回（汇总指标, 每个 run 的明细）。"""
    model.eval()
    embeddings, targets, rows = [], [], []
    for day, run in keys:
        eeg, sentences = encode_indices(model, dataset, groups[(day, run)], device, batch_size)
        target = torch.tensor([col_of[s] for s in sentences], dtype=torch.long, device=device)
        metrics = retrieval_metrics(eeg, text_all, target, temperature)
        rows.append({"day": day, "run": run, **metrics})
        embeddings.append(eeg)
        targets.append(target)

    pooled = retrieval_metrics(
        torch.cat(embeddings, dim=0), text_all, torch.cat(targets, dim=0), temperature
    )
    model.train()
    return pooled, rows


@torch.no_grad()
def split_half_stats(model, dataset, groups, device, batch_size, n_splits, seed):
    """同一句跨 16 个 run 的 split-half 一致性。

    每个 run 的 63 条验证 EEG 按句子对齐后得到 ``(R, S, D)``；对每句随机
    分成两半、各自求均值，算两半的余弦（同句）。同样大小的两半均值在不同句
    之间配对，得到噪声水平一致的异句基线。
    """
    keys = sorted(groups)
    if len(keys) < 2:
        return None

    model.eval()
    per_run, reference = [], None
    for day, run in keys:
        eeg, sentences = encode_indices(model, dataset, groups[(day, run)], device, batch_size)
        if reference is None:
            reference = sorted(sentences)
            col = {sentence: i for i, sentence in enumerate(reference)}
        elif sorted(sentences) != reference:
            raise ValueError("不同验证 run 的句子集合不一致")

        matrix = np.zeros((len(reference), eeg.size(1)), dtype=np.float32)
        counts = np.zeros(len(reference), dtype=np.int32)
        eeg = eeg.cpu().numpy()
        for i, sentence in enumerate(sentences):
            matrix[col[sentence]] = eeg[i]
            counts[col[sentence]] += 1
        if not np.all(counts == 1):
            raise ValueError("同一个验证 run 内出现重复或缺失的验证句")
        per_run.append(matrix)
    model.train()

    stacked = np.stack(per_run)  # (R, S, D)
    n_runs, n_sent, _ = stacked.shape
    half = n_runs // 2
    if half == 0:
        return None

    rng = np.random.default_rng(seed)
    same, different = [], []
    rows = np.arange(n_sent)[:, None]
    for _ in range(n_splits):
        order = np.argsort(rng.random((n_sent, n_runs)), axis=1)
        a = stacked[order[:, :half], rows].mean(axis=1)
        b = stacked[order[:, half:], rows].mean(axis=1)
        a /= np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
        b /= np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
        same.append(float((a * b).sum(axis=1).mean()))

        perm = rng.permutation(n_sent)
        mask = perm != np.arange(n_sent)
        if mask.any():
            different.append(float((a[mask] * a[perm[mask]]).sum(axis=1).mean()))

    same_mean = float(np.mean(same))
    diff_mean = float(np.mean(different)) if different else float("nan")
    return {
        "n_sentences": n_sent,
        "n_splits": n_splits,
        "same_sentence": same_mean,
        "different_sentence": diff_mean,
        "gap": same_mean - diff_mean,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"[Config] device={device} subject={args.subject} days={args.day} "
        f"event={args.event} epochs={args.epochs} lr={args.lr} "
        f"temperature={args.temperature} seed={args.seed} "
        f"val_fraction={args.val_fraction}"
    )

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    # ---------------- 数据 ----------------
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

    # ---------------- 验证集切分 ----------------
    if args.val_split:
        split = load_split(Path(args.val_split))
        missing = set(split["val_sentences"]) - set(base_dataset.meta["sentence"])
        if missing:
            raise ValueError(f"验证清单里有 {len(missing)} 句不在当前数据中")
    else:
        split = build_val_split(
            base_dataset.meta, fraction=args.val_fraction, seed=args.val_seed
        )
    split_path = save_split(out_dir / f"{args.subject}_val_split.json", split)

    val_set = set(split["val_sentences"])
    is_val = base_dataset.meta["sentence"].isin(val_set).to_numpy()

    all_groups: dict[tuple[int, int], list[int]] = {}
    train_groups: dict[tuple[int, int], list[int]] = {}
    val_groups: dict[tuple[int, int], list[int]] = {}
    for index, row in base_dataset.meta.iterrows():
        key = (int(row["day"]), int(row["run"]))
        all_groups.setdefault(key, []).append(index)
        target = val_groups if is_val[index] else train_groups
        target.setdefault(key, []).append(index)

    n_sentences = base_dataset.meta.sentence.nunique()
    print(
        f"[Data] {args.subject} days={sorted(args.day)} event={args.event} "
        f"n_chans={n_chans} token_samples={token_samples}\n"
        f"       trial={len(base_dataset)}  唯一句子={n_sentences}  "
        f"run 数={len(all_groups)}"
    )
    print(
        f"[Split] 验证句 {len(val_set)}/{n_sentences} → {split_path}\n"
        f"        训练句 {n_sentences - len(val_set)}；"
        f"每 run 训练={sorted({len(v) for v in train_groups.values()})} "
        f"验证={sorted({len(v) for v in val_groups.values()}) if val_groups else '无'}"
    )
    if val_set:
        big = {k: v for k, v in split["per_category"].items() if v > 1}
        single = sum(v == 1 for v in split["per_category"].values())
        print(f"        分层：{big}，其余 {single} 类各 1 句")

    if len(train_groups) == 0:
        raise ValueError("没有可用的训练 (day, run) group")

    run_keys = sorted(all_groups)
    if val_groups and sorted(val_groups) != run_keys:
        raise ValueError("验证 run 与数据中的 run 不一致")

    dataset = RunMetaDataset(base_dataset)
    sampler = RunBatchSampler(
        train_groups,
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
    sentences_all = sorted(lookup)
    text_all = torch.stack([lookup[sentence] for sentence in sentences_all])
    col_of = {sentence: i for i, sentence in enumerate(sentences_all)}
    print(
        f"[Text] 已缓存 {len(lookup)} 条句子，embedding_dim={embedding_dim}；"
        f"训练 gallery={text_all.size(0)} 句"
    )

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
    val_path = out_dir / f"{args.subject}_val_runs.csv"
    split_half_path = out_dir / f"{args.subject}_split_half.csv"
    last_path = out_dir / f"{args.subject}_last.pt"
    best_train_path = out_dir / f"{args.subject}_best_train.pt"
    best_val_path = out_dir / f"{args.subject}_best_val.pt"

    header = (
        f"{'epoch':>5} {'lr':>8} {'train_loss':>10} {'pos/anch':>9} "
        f"{'batches':>7} {'mean_bs':>8}"
    )
    if val_groups:
        header += f" {'val@1':>8} {'val@5':>8} {'val_rank':>9}"
    print(f"\n{header}")
    print("-" * len(header))

    val_keys = sorted(val_groups)
    val_cursor = 0
    best_train = float("inf")
    best_val_score = None

    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss, total_n, total_pos = 0.0, 0, 0.0
        n_batches = 0

        for step, (tokens, mask, sentences, days, runs) in enumerate(loader, start=1):
            # 每个 batch 必须恰好来自一个 (day, run)
            if days.unique().numel() != 1 or runs.unique().numel() != 1:
                raise RuntimeError("一个 batch 内出现了多个 (day, run)")
            if len(set(sentences)) != len(sentences):
                raise RuntimeError("一个 batch 内出现了重复句子")
            day, run = int(days[0]), int(runs[0])

            tokens = tokens.to(device)
            mask = mask.to(device)
            text = gather_text(sentences, lookup, device)
            target = torch.tensor(
                [col_of[sentence] for sentence in sentences],
                dtype=torch.long, device=device,
            )
            groups_t = sentence_group_ids(sentences, device)

            eeg = model(tokens, mask)
            loss = contrastive_loss(
                eeg, text, args.temperature, gallery=text_all, targets=target
            )

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

        # ---------------- 验证 ----------------
        val_metrics = None
        val_detail = None
        if val_keys and epoch % args.val_every == 0:
            take = min(args.val_runs, len(val_keys))
            chosen = [val_keys[(val_cursor + i) % len(val_keys)] for i in range(take)]
            val_cursor = (val_cursor + take) % len(val_keys)

            val_metrics, rows = evaluate_runs(
                model, base_dataset, val_groups, chosen,
                text_all, col_of, device, args.eval_batch_size, args.temperature,
            )
            for row in rows:
                append_csv_row(val_path, VAL_FIELDS, {
                    "epoch": epoch,
                    "day": row["day"],
                    "run": row["run"],
                    "n": row["n"],
                    "e2t1": f"{row['e2t1']:.6f}",
                    "e2t5": f"{row['e2t5']:.6f}",
                    "mean_rank": f"{row['mean_rank']:.6f}",
                    "loss_e2t": f"{row['loss_e2t']:.6f}",
                })
            val_detail = (
                f"        [Val] run={','.join(f'day{d}run{r}' for d, r in chosen)} "
                f"n={val_metrics['n']} e2t@1={val_metrics['e2t1']:.4f} "
                f"e2t@5={val_metrics['e2t5']:.4f} "
                f"mean_rank={val_metrics['mean_rank']:.1f}"
            )

        # ---------------- split-half ----------------
        split_metrics = None
        split_detail = None
        if (
            val_keys
            and args.split_half_every > 0
            and (epoch % args.split_half_every == 0 or epoch == args.epochs)
        ):
            split_metrics = split_half_stats(
                model, base_dataset, val_groups, device,
                args.eval_batch_size, args.split_half_splits, args.val_seed,
            )
            if split_metrics is not None:
                append_csv_row(split_half_path, SPLIT_FIELDS, {
                    "epoch": epoch,
                    "n_sentences": split_metrics["n_sentences"],
                    "n_splits": split_metrics["n_splits"],
                    "same_sentence": f"{split_metrics['same_sentence']:.6f}",
                    "different_sentence": f"{split_metrics['different_sentence']:.6f}",
                    "gap": f"{split_metrics['gap']:.6f}",
                })
                split_detail = (
                    f"        [Split-half] 同句={split_metrics['same_sentence']:+.4f} "
                    f"异句={split_metrics['different_sentence']:+.4f} "
                    f"差={split_metrics['gap']:+.4f}"
                )

        # ---------------- 日志 / checkpoint ----------------
        epoch_row = {
            "epoch": epoch,
            "lr": f"{epoch_lr:.6f}",
            "train_loss": f"{train_loss:.6f}",
            "pos_per_anchor": f"{train_pos:.6f}",
            "n_batches": n_batches,
            "mean_batch_size": f"{mean_bs:.2f}",
            "temperature": args.temperature,
            "val_e2t1": "" if val_metrics is None else f"{val_metrics['e2t1']:.6f}",
            "val_e2t5": "" if val_metrics is None else f"{val_metrics['e2t5']:.6f}",
            "val_mean_rank": "" if val_metrics is None else f"{val_metrics['mean_rank']:.6f}",
            "val_loss_e2t": "" if val_metrics is None else f"{val_metrics['loss_e2t']:.6f}",
            "val_n": "" if val_metrics is None else val_metrics["n"],
        }
        append_csv_row(epochs_path, EPOCH_FIELDS, epoch_row)

        line = (
            f"{epoch:>5} {epoch_lr:>8.2e} {train_loss:>10.4f} "
            f"{train_pos:>9.2f} {n_batches:>7} {mean_bs:>8.2f}"
        )
        if val_groups:
            if val_metrics is None:
                line += f" {'-':>8} {'-':>8} {'-':>9}"
            else:
                line += (
                    f" {val_metrics['e2t1']:>8.4f} {val_metrics['e2t5']:>8.4f} "
                    f"{val_metrics['mean_rank']:>9.1f}"
                )
        print(line)
        if val_detail:
            print(val_detail)
        if split_detail:
            print(split_detail)

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "n_chans": n_chans,
            "token_samples": token_samples,
            "embedding_dim": embedding_dim,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "split_half": split_metrics,
            "val_sentences": split["val_sentences"],
        }
        torch.save(ckpt, last_path)
        if train_loss < best_train:
            best_train = train_loss
            torch.save(ckpt, best_train_path)
            print(f"        * new best train_loss={best_train:.6f}")
        if val_metrics is not None:
            score = (val_metrics["e2t1"], -val_metrics["mean_rank"])
            if best_val_score is None or score > best_val_score:
                best_val_score = score
                torch.save(ckpt, best_val_path)
                print(
                    f"        * new best val e2t@1={val_metrics['e2t1']:.4f} "
                    f"mean_rank={val_metrics['mean_rank']:.1f}"
                )

    print(
        f"\n[Done] best train_loss={best_train:.6f}"
        + (
            f"  best val e2t@1={best_val_score[0]:.4f}" if best_val_score else ""
        )
        + f"\n       last      : {last_path}"
        + f"\n       best train: {best_train_path}"
        + (f"\n       best val  : {best_val_path}" if best_val_score else "")
        + f"\n       split     : {split_path}"
        + f"\n       epochs    : {epochs_path}"
        + f"\n       steps     : {steps_path}"
        + (f"\n       val runs  : {val_path}" if val_groups else "")
        + (f"\n       split-half: {split_half_path}" if val_groups else "")
    )


if __name__ == "__main__":
    main()
