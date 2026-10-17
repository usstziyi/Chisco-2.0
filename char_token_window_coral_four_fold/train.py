"""滑窗多尺度 token + CORAL 的按 day 切分单折训练入口（自包含实现）。

本目录是独立实验：数据切分、模型、按句分组采样、文本编码、InfoNCE + CORAL
损失、评测与 CSV 日志全部在本目录内实现，不依赖其他实验目录。

与 ``char_token_window_coral`` 的区别只在 **train / val 的切分方式**：
基实验用 ``split_by_sentence`` 随机按句子做 8:2 切分；本实验用 ``split_by_day``
按天整体切分（默认 day 1 2 3 做 train、day 4 做 val）。

注意：day 1~4 是同一批句子的重复录制，train / val 的句子集合完全重合，
因此这里的 val 指标衡量的是"同一句子在未见过的某一天录到的信号"的跨天泛化，
与按句子切分（新句子泛化）不可直接比较。

用法::

    uv run python char_token_window_coral_four_fold/train.py
    uv run python char_token_window_coral_four_fold/train.py --train-days 1 2 3 --val-days 4
    uv run python char_token_window_coral_four_fold/train.py --train-days 2 3 4 --val-days 1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# 支持 `python char_token_window_coral_four_fold/train.py` 直接执行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_window_coral_four_fold.dataset import CharTokenEEGDataset, char_token_collate
from char_token_window_coral_four_fold.model import CharTokenEEGEncoder
from char_token_window_coral_four_fold.sampler import SentenceGroupBatchSampler
from char_token_window_coral_four_fold.split import split_by_day
from char_token_window_coral_four_fold.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_window_coral_four_fold"

# train_epochs.csv 每轮追加一行，字段顺序即表头顺序
EPOCH_LOG_FIELDS = [
    "epoch", "lr", "train_loss", "train_infonce", "train_coral", "train_pos_per_anchor",
    "val_loss", "val_coral", "val_total", "n_val", "n_unique_sentences", "pos_per_anchor",
    "n_text_candidates",
    "eeg2text_top1", "eeg2text_top5", "text2eeg_top1", "text2eeg_top5",
    "is_best", "temperature", "coral_weight",
]

# train_steps.csv 每个训练 step 追加一行：第 0 列 epoch、第 1 列 step
TRAIN_STEP_FIELDS = [
    "epoch", "step", "loss", "infonce", "coral", "lr", "pos_per_anchor", "batch_size",
]


def append_csv_row(path: Path, fields: list, row: dict) -> None:
    """把一行追加到 CSV；文件不存在时先写表头。"""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def dump_args_json(args, out_dir: Path, filename: str = "args.json") -> Path:
    """把本次运行的所有命令行参数（key/value）以 JSON 写到时间戳输出目录。"""
    path = Path(out_dir) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_text_lookup(encoder, sentences, device, chunk: int = 64):
    """把所有唯一句子编码成 embedding，返回 ``{sentence: (D,)}`` 查表。"""
    unique = sorted(set(sentences))
    vectors = []
    with torch.no_grad():
        for start in range(0, len(unique), chunk):
            vectors.append(encoder(unique[start : start + chunk]))
    table = torch.cat(vectors, dim=0).to(device)
    embedding_dim = table.shape[1]
    return dict(zip(unique, table)), embedding_dim


def gather_text(sentences, lookup, device) -> torch.Tensor:
    """按句子取回文本 embedding，堆成 (B, D)。"""
    return torch.stack([lookup[s] for s in sentences]).to(device)


def sentence_group_ids(sentences, device) -> torch.Tensor:
    """把句子列表映射成组 id：同句的 trial 得到同一个 id。

    多天训练时同一句会在一批里出现多次，它们互为对方的正样本。
    """
    mapping: dict[str, int] = {}
    ids = []
    for sentence in sentences:
        if sentence not in mapping:
            mapping[sentence] = len(mapping)
        ids.append(mapping[sentence])
    return torch.tensor(ids, dtype=torch.long, device=device)


def _multi_positive_ce(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """多正样本交叉熵：每个 anchor 对"所有正样本列"求平均的 -log softmax。

    positive 为 (B, B) 的 0/1 掩码，第 i 行的正样本列就是 anchor i 该拉近的候选。
    每行至少有一个正样本（含 anchor 自己），所以分母不会为 0。
    """
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * positive).sum(dim=1).div(positive.sum(dim=1)).mean()


def contrastive_loss(
    eeg: torch.Tensor,
    text: torch.Tensor,
    temperature: float,
    groups: torch.Tensor | None = None,
) -> torch.Tensor:
    """多正样本对称 InfoNCE：EEG->文本 与 文本->EEG 两个方向取平均。

    正样本由 ``groups`` 决定：只要两句的 group id 相同，就互为正样本。
    """
    logits = eeg @ text.t() / temperature
    if groups is None:
        groups = torch.arange(logits.size(0), device=logits.device)
    positive = (groups[:, None] == groups[None, :]).to(logits.dtype)
    # positive 是对称的，两个方向共用同一套正样本掩码
    return 0.5 * (
        _multi_positive_ce(logits, positive) + _multi_positive_ce(logits.t(), positive)
    )


def coral_loss(eeg: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """Deep CORAL：对齐 batch 内 EEG 与文本表征的协方差矩阵（Frobenius 范数）。

        L_CA = ||Σ_EEG - Σ_Text||_F

    本实现中 ``Σ`` 取 **batch 内 B 个表征**的协方差矩阵（D×D，无偏估计，分母 B-1），
    对齐的是两模态表征的二阶统计结构，与 InfoNCE 只约束单位向量夹角互补。
    两侧输入已 L2 归一化，故协方差即"相关结构"，尺度一致、无需额外归一。

    ``B < 2`` 时协方差无定义，返回 0（该 batch 不贡献梯度）。
    """
    n = eeg.size(0)
    if n < 2:
        return eeg.new_zeros(())

    # 去均值后计算协方差：(B, D) -> (D, D)
    eeg_c = eeg - eeg.mean(dim=0, keepdim=True)
    text_c = text - text.mean(dim=0, keepdim=True)
    denom = n - 1
    cov_eeg = eeg_c.t() @ eeg_c / denom
    cov_text = text_c.t() @ text_c / denom

    return torch.linalg.matrix_norm(cov_eeg - cov_text)  # 默认即 Frobenius 范数


def train_one_epoch(
    model, loader, lookup, optimizer, device, temperature, coral_weight, epoch,
    step_log_path=None,
):
    """跑一个 epoch，返回 ``(总损失, infoNCE 项, coral 项, pos/anch)`` 四个均值。

    传了 ``step_log_path`` 就把每个 step 追加到 CSV：第 0 列 epoch、第 1 列 step。
    总损失 = infoNCE + coral_weight × coral，反传只用总损失。
    """
    model.train()
    total_loss, total_info, total_coral, total_n, total_pos = 0.0, 0.0, 0.0, 0, 0.0
    lr = optimizer.param_groups[0]["lr"]
    for step, (tokens, mask, sentences) in enumerate(loader, start=1):
        tokens = tokens.to(device) # (B,n,C,T)
        mask = mask.to(device) # (B,n)
        text = gather_text(sentences, lookup, device) # (B,D)
        groups = sentence_group_ids(sentences, device)

        eeg = model(tokens, mask)
        info = contrastive_loss(eeg, text, temperature, groups)
        coral = coral_loss(eeg, text)
        loss = info + coral_weight * coral

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = len(sentences)
        pos = (groups[:, None] == groups[None, :]).sum().item()

        if step_log_path is not None:
            append_csv_row(step_log_path, TRAIN_STEP_FIELDS, {
                "epoch": epoch,
                "step": step,
                "loss": f"{loss.item():.6f}",
                "infonce": f"{info.item():.6f}",
                "coral": f"{coral.item():.6f}",
                "lr": f"{lr:.6f}",
                "pos_per_anchor": f"{pos / batch:.4f}",
                "batch_size": batch,
            })

        total_loss += loss.item() * batch
        total_info += info.item() * batch
        total_coral += coral.item() * batch
        total_n += batch
        total_pos += pos
    denom = max(total_n, 1)
    return (
        total_loss / denom, total_info / denom, total_coral / denom, total_pos / denom,
    )


@torch.no_grad()
def evaluate(model, loader, lookup, device, temperature) -> dict:
    """在整个 val 集上算检索指标与全库对比损失（含 coral 分量）。

    两个方向的候选池不同：

    - **eeg->text**：候选池 = **去重后的唯一句子**。
    - **text->eeg**：候选池 = 全部 EEG 样本，命中 = 前 k 名里出现任意同句。

    ``val_loss`` = 全库 infoNCE；``val_coral`` = 全库 coral。
    """
    model.eval()
    eeg_parts, sentences = [], []
    for tokens, mask, batch_sentences in loader:
        eeg_parts.append(model(tokens.to(device), mask.to(device)))
        sentences.extend(batch_sentences)

    eeg = torch.cat(eeg_parts, dim=0)
    text = gather_text(sentences, lookup, device)
    groups = sentence_group_ids(sentences, device)
    n = eeg.size(0)
    positive = groups[:, None] == groups[None, :]

    info = float(contrastive_loss(eeg, text, temperature, groups))
    coral = float(coral_loss(eeg, text))

    metrics = {
        "val_loss": info,
        "val_coral": coral,
        "n_val": n,
        "n_unique_sentences": len(set(sentences)),
        "pos_per_anchor": float(positive.sum(dim=1).float().mean()),
    }

    # ---- eeg -> text：候选池去重成唯一句子，每个 anchor 恰好 1 个正样本 ----
    unique_sentences = sorted(set(sentences))
    sentence_to_col = {s: i for i, s in enumerate(unique_sentences)}
    text_unique = gather_text(unique_sentences, lookup, device)
    target = torch.tensor(
        [sentence_to_col[s] for s in sentences], device=device, dtype=torch.long
    ) # 每条 trial 的正确答案列号
    ranked = (eeg @ text_unique.t()).argsort(dim=1, descending=True)
    hit = ranked.eq(target[:, None])
    metrics["n_text_candidates"] = len(unique_sentences)
    metrics["eeg2text_top1"] = float(hit[:, :1].any(dim=1).float().mean())
    metrics["eeg2text_top5"] = float(hit[:, :5].any(dim=1).float().mean())

    # ---- text -> eeg：候选池是全部 EEG 样本，命中 = 前 k 名里出现任意同句 ----
    ranked = (text @ eeg.t()).argsort(dim=1, descending=True)
    hit = positive[torch.arange(n, device=device)[:, None], ranked]
    metrics["text2eeg_top1"] = float(hit[:, :1].any(dim=1).float().mean())
    metrics["text2eeg_top5"] = float(hit[:, :5].any(dim=1).float().mean())

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="滑窗多尺度 token + CORAL 的按 day 切分四折实验（单折入口）。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试，如 sub-01（默认）")
    parser.add_argument(
        "--train-days", type=int, nargs="+", default=[1, 2, 3],
        help="划给 train 的 day 编号；默认 1 2 3",
    )
    parser.add_argument(
        "--val-days", type=int, nargs="+", default=[4],
        help="划给 val 的 day 编号；默认 4",
    )
    parser.add_argument(
        "--event", default="inner", choices=["inner", "reading"],
        help="取 inner（默读，默认）或 reading（朗读）",
    )
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument(
        "--window-duration", type=float, default=1.0,
        help="滑动窗口长度（秒，默认 1.0）",
    )
    parser.add_argument(
        "--stride-duration", type=float, default=0.5,
        help="相邻窗口起点的间隔（秒，默认 0.5）",
    )
    parser.add_argument(
        "--kernels", type=int, nargs="+", default=[5, 15, 35, 75],
        help="多尺度时间卷积的核宽（奇数，单位采样点，默认 5 15 35 75）",
    )
    parser.add_argument(
        "--n-sentences", type=int, default=42,
        help="每个训练 batch 里不同句子的数量（默认 42）",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=3,
        help="每个句子在每个 batch 里抽多少条 trial 当正样本（默认 3）",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="验证集 DataLoader 的 batch 大小（默认 32）")
    parser.add_argument("--epochs", type=int, default=30, help="每折训练轮数（默认 30）")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率（默认 1e-3）")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW 权重衰减（默认 1e-3）")
    parser.add_argument(
        "--coral-weight", type=float, default=0.1,
        help="协方差对齐损失（CORAL）权重，0 即退化为纯 InfoNCE（默认 0.1）",
    )
    parser.add_argument(
        "--lr-schedule", default="cosine", choices=["none", "cosine"],
        help="学习率调度：cosine 按 epoch 余弦衰减到 0（默认），none 恒定",
    )
    parser.add_argument(
        "--select-by", default="val_loss",
        choices=["val_loss", "val_total", "eeg2text_top1", "eeg2text_top5", "text2eeg_top5"],
        help="用哪个指标选 best checkpoint（默认 val_loss，越小越好；其余越大越好）",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="对比学习温度系数（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args()


def run_experiment(args, out_dir, text_encoder, device):
    """完整跑一遍训练 + 评测，返回 ``(best_score, best_metrics, info)``。"""
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
        window_duration=args.window_duration,
        stride_duration=args.stride_duration,
    )
    dataset[0]  # 触发一次读取，确定 n_chans
    train_set, val_set, info = split_by_day(dataset, args.train_days, args.val_days)

    days_label = ",".join(f"{d:02d}" for d in all_days)
    n_win = dataset.meta["n_tokens"]
    print(
        f"[Data] {subject} day-{days_label} event={args.event} "
        f"window={dataset.token_samples} stride={dataset.stride_samples} "
        f"n_chans={dataset.n_chans}\n"
        f"       窗口数 {n_win.min()}~{n_win.max()}（均值 {n_win.mean():.1f}）\n"
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
    # 一句最多能提供多少条 trial：同句跨天 / 跨 run 重复才有多个视图
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
        kernels=tuple(args.kernels),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {type(model).__name__} 参数量={n_params:,} "
        f"d_model={model.d_model} layers={len(model.transformer.layers)} "
        f"max_tokens={model.max_tokens} kernels={model.token_encoder.kernels} "
        f"out_time={model.token_encoder.out_time}"
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

    lower_is_better = args.select_by in ("val_loss", "val_total")
    best_score = float("inf") if lower_is_better else -float("inf")
    best_metrics = None
    metrics = {}
    header = (
        f"{'epoch':>5} {'lr':>8} {'train_loss':>10} {'pos/anch':>9} "
        f"{'val_loss':>9} {'val_coral':>10} "
        f"{'e2t@1':>7} {'e2t@5':>7} {'t2e@1':>7} {'t2e@5':>7}"
    )
    print(f"\n{header}  (best by {args.select_by}, coral_weight={args.coral_weight})")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_info, train_coral, train_pos = train_one_epoch(
            model, train_loader, lookup, optimizer, device, args.temperature,
            args.coral_weight, epoch=epoch, step_log_path=train_steps_path,
        )
        if scheduler is not None:
            scheduler.step()
        metrics = evaluate(model, val_loader, lookup, device, args.temperature)
        metrics["val_total"] = metrics["val_loss"] + args.coral_weight * metrics["val_coral"]

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
            f"{metrics['val_loss']:>9.4f} {metrics['val_coral']:>10.4f} "
            f"{metrics['eeg2text_top1']:>7.4f} {metrics['eeg2text_top5']:>7.4f} "
            f"{metrics['text2eeg_top1']:>7.4f} {metrics['text2eeg_top5']:>7.4f}{flag}"
        )

        # 本轮指标追加到 train_epochs.csv（首轮自动写表头）
        append_csv_row(train_epochs_path, EPOCH_LOG_FIELDS, {
            "epoch": epoch,
            "lr": f"{epoch_lr:.6f}",
            "train_loss": f"{train_loss:.6f}",
            "train_infonce": f"{train_info:.6f}",
            "train_coral": f"{train_coral:.6f}",
            "train_pos_per_anchor": f"{train_pos:.6f}",
            "val_loss": f"{metrics['val_loss']:.6f}",
            "val_coral": f"{metrics['val_coral']:.6f}",
            "val_total": f"{metrics['val_total']:.6f}",
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
            "coral_weight": args.coral_weight,
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
        f"window={args.window_duration}s stride={args.stride_duration}s "
        f"kernels={args.kernels} "
        f"temperature={args.temperature} batch={args.n_sentences}句×{args.n_repeats}视图 "
        f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
        f"coral_weight={args.coral_weight} "
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
