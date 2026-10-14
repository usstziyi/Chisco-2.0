"""字符级 token 方案的对比学习训练循环。

数据
----
被试 / 天数 / 事件类型 / 采样率均可通过命令行指定，默认 sub-01 + day-01。
``--subject`` 可一次传多个被试（如 ``--subject sub-01 sub-02``），脚本会顺序跑完每个被试
做对照；所有被试共用同一个时间戳输出目录，CSV / checkpoint 以被试名前缀区分。
train / val 按 **句子** 做 8:2 切分（不是按 trial），保证同一句子的所有 trial
只会落在同一侧，不会跨集泄漏。

损失
----
多正样本对称 InfoNCE（对比学习交叉熵）：

    logits = eeg_embedding @ text_embedding.T / temperature
    loss   = 0.5 * (MP-CE(logits) + MP-CE(logits.T))

温度系数默认 0.2。正样本由"句子"决定：同一句的所有 trial 互为正样本。
一批里每句只出现一次时（单天数据），正样本矩阵退化成单位阵，损失就是
标准的单正样本 CLIP 式 InfoNCE；一批里同一句出现多次时（多天数据），
同句的 trial 互为正样本，不会被当成负样本互相推开（假负样本）。

注意句子只在**跨天**之间重复：day-01 里每句只录了一次（无重复）。
所以想拿到"同句多视图"的正样本，必须一次取多天，例如 ``--day 1 2 3 4``。

采样
----
训练用 ``SentenceGroupBatchSampler``：每个 batch 抽 ``n_sentences`` 个不同句子，
每句再抽 ``n_repeats`` 条 trial，于是同句的多视图**必然同框**，``pos/anch`` 恒等于
``n_repeats``。若改用随机 shuffle，多视图撞进同一 batch 的概率极低
（实测 day-01~04 时 pos/anch 只有 1.03），多正样本机制等于空转。
``--n-repeats 1`` 可退化成随机 shuffle 的效果，用作对照。

评测
----
两个检索方向的候选池不同：

- ``eeg2text``：候选 = **去重后的唯一句子**。同句的多个 trial 共享同一个文本向量，
  不去重的话同一句会占掉 4 个连续名次，``top-5`` 实际只等于 top-2。
- ``text2eeg``：候选 = 全部 EEG 样本（互不相同），命中 = 前 k 名里出现任意同句。

``val_loss`` 越小越好，其余指标越大越好；best checkpoint 默认按 ``val_loss`` 选
（``eeg2text_top1`` 在 900 条 val 上噪声很大，单条翻转就是 0.11%）。

文本侧用冻结的 LaBSE，所有句子的文本 embedding 在训练前一次性算好、做成查表，
训练时只做索引，不重复跑 LaBSE。

过滤
----
``--threshold`` 在训练前把"近重复句"整句剔除：先用冻结的 LaBSE 对所有唯一句子算
两两余弦相似度，相似度 >= threshold 的句子用单链接（并查集）并成组，凡落在多句组
里的句子全部去掉，只保留"单例句"（与其它任何句都不相似）。

过滤在**句子文本层面**进行，并对本次运行涉及的所有 subject / day 统一生效（keep
集在 main 里全局算一次、各被试共用），因此同句跨天的多视图正样本不会被拆散，
train / val 按句子切分依然无泄漏。

用法:

    uv run python char_token_dissim_eegnet/train.py
    uv run python char_token_dissim_eegnet/train.py --day 1 2 3 4 --threshold 0.9
    uv run python char_token_dissim_eegnet/train.py --subject sub-01 sub-02 --day 1 2 3 4 --n-sentences 32 --n-repeats 4
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
from torch.utils.data import DataLoader, Subset

# 支持 `python char_token_dissim_eegnet/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_dissim_eegnet.dataset import CharTokenEEGDataset, char_token_collate
from char_token_dissim_eegnet.model import CharTokenEEGEncoder
from char_token_dissim_eegnet.sampler import SentenceGroupBatchSampler
from char_token_dissim_eegnet.sentence_filter import keep_singleton_sentences
from char_token_dissim_eegnet.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_dissim_eegnet"

# train_epochs.csv 每轮追加一行，字段顺序即表头顺序
EPOCH_LOG_FIELDS = [
    "epoch", "lr", "train_loss", "train_pos_per_anchor",
    "val_loss", "n_val", "n_unique_sentences", "pos_per_anchor",
    "n_text_candidates",
    "eeg2text_top1", "eeg2text_top5", "text2eeg_top1", "text2eeg_top5",
    "is_best", "temperature",
]

# train_steps.csv 每个训练 step 追加一行：第 0 列 epoch、第 1 列 step
TRAIN_STEP_FIELDS = [
    "epoch", "step", "loss", "lr", "pos_per_anchor", "batch_size",
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


def split_by_sentence(dataset, val_ratio: float, seed: int):
    """按句子切分 train / val，返回 (train_subset, val_subset, 统计信息)。

    以句子为单位而非 trial，避免同一句子的不同 trial 同时出现在两侧。
    """
    sentences = dataset.meta["sentence"].tolist()
    unique = sorted(set(sentences))
    if len(unique) < 2:
        raise ValueError(f"只有 {len(unique)} 个唯一句子，无法切分")

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_ratio))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    val_sentences = set(shuffled[:n_val])

    val_idx = [i for i, s in enumerate(sentences) if s in val_sentences]
    train_idx = [i for i, s in enumerate(sentences) if s not in val_sentences]

    # 泄漏检查：按句子切分后，train / val 的句子集合必须完全不相交
    train_sentences = {sentences[i] for i in train_idx}
    info = {
        "n_unique_train": len(unique) - n_val,
        "n_unique_val": n_val,
        "sentence_overlap": len(train_sentences & val_sentences),
    }
    return Subset(dataset, train_idx), Subset(dataset, val_idx), info


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

    - 一批里每句只出现一次时（例如单天数据），``groups`` 两两不同，
      正样本矩阵退化成单位阵，损失与"对角线单正样本"的 CLIP 式写法完全一致；
    - 一批里同一句出现多次时（多天数据），同句的所有 trial 互为正样本，
      不会像单正样本写法那样把它们当成负样本互相推开。
    """
    logits = eeg @ text.t() / temperature
    if groups is None:
        groups = torch.arange(logits.size(0), device=logits.device)
    positive = (groups[:, None] == groups[None, :]).to(logits.dtype)
    # positive 是对称的，两个方向共用同一套正样本掩码
    return 0.5 * (
        _multi_positive_ce(logits, positive) + _multi_positive_ce(logits.t(), positive)
    )


def train_one_epoch(
    model, loader, lookup, optimizer, device, temperature, epoch, step_log_path=None
):
    """跑一个 epoch，返回 ``(平均 loss, 每个 anchor 的平均正样本数)``。

    传了 ``step_log_path`` 就把每个 step 追加到 CSV：第 0 列 epoch、第 1 列 step。
    """
    model.train()
    total_loss, total_n, total_pos = 0.0, 0, 0.0
    lr = optimizer.param_groups[0]["lr"]
    for step, (tokens, mask, sentences) in enumerate(loader, start=1):
        tokens = tokens.to(device) # (B,n,C,T)
        mask = mask.to(device) # (B,n)
        text = gather_text(sentences, lookup, device) # (B,D)
        groups = sentence_group_ids(sentences, device)

        eeg = model(tokens, mask)
        loss = contrastive_loss(eeg, text, temperature, groups)

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
                "lr": f"{lr:.6f}",
                "pos_per_anchor": f"{pos / batch:.4f}",
                "batch_size": batch,
            })

        total_loss += loss.item() * batch
        total_n += batch
        total_pos += pos
    return total_loss / max(total_n, 1), total_pos / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, lookup, device, temperature) -> dict:
    """在整个 val 集上算检索指标与全库对比损失。

    两个方向的候选池不同，这是有意为之：

    - **eeg->text**：候选池 = **去重后的唯一句子**。同句的多个 trial 共享同一个
      文本向量，若不去重，同一句的 4 个副本会在排名里占掉 4 个连续位置，
      ``top-5`` 实际只等于"正确答案排在前 2 个不同句子里"。
    - **text->eeg**：候选池 = 全部 EEG 样本（彼此互不相同），命中判定为
      "前 k 名里出现任意一个同句候选"。
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

    metrics = {
        "val_loss": float(contrastive_loss(eeg, text, temperature, groups)),
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
    metrics["eeg2text_top1"] = float(hit[:, :1].any(dim=1).float().mean()) # 第一名命中 → top-1
    metrics["eeg2text_top5"] = float(hit[:, :5].any(dim=1).float().mean()) # 前五名里出现过正确答案 → top-5

    # ---- text -> eeg：候选池是全部 EEG 样本，命中 = 前 k 名里出现任意同句 ----
    ranked = (text @ eeg.t()).argsort(dim=1, descending=True)
    hit = positive[torch.arange(n, device=device)[:, None], ranked]
    metrics["text2eeg_top1"] = float(hit[:, :1].any(dim=1).float().mean())
    metrics["text2eeg_top5"] = float(hit[:, :5].any(dim=1).float().mean())

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="字符级 token EEG–文本对比学习训练。")
    parser.add_argument(
        "--subject", nargs="+", default=["sub-01"],
        help="被试，可传多个一次跑完对照，如 --subject sub-01 sub-02；默认 sub-01",
    )
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
        "--n-sentences", type=int, default=32,
        help="每个训练 batch 里不同句子的数量（默认 32）",
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
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="近重复过滤阈值：相似度 >= threshold 的句子并成组，多句组成员整句剔除，"
             "只保留单例句（默认 0.95）",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args()


def compute_keep_sentences(args, text_encoder):
    """全局算一次要保留的"单例句"，所有被试 / 所有 day 共用同一份。

    用与训练相同的筛选条件（event / sample / subject / day）把本次涉及的全部
    句子取出来求并集（只读文本表、不读 FIF），按 threshold 剔除近重复后返回保序列表。
    """
    pool_ds = CharTokenEEGDataset(
        event=args.event,
        sample=args.sample,
        subject=args.subject,
        days=args.day,
        char_duration=args.char_duration,
    )
    pool = sorted(set(pool_ds.meta["sentence"].astype(str)))
    keep, stats = keep_singleton_sentences(pool, text_encoder, args.threshold)
    print(
        f"[Filter] threshold={args.threshold:g}：唯一句子 {stats['n_unique']} 条 -> "
        f"保留 {stats['n_kept']} 条（剔除近重复 {stats['n_dropped']} 条，"
        f"来自 {stats['n_multi_groups']} 个多句组）"
    )
    if len(keep) < 2:
        raise ValueError(
            f"threshold={args.threshold} 过滤后只剩 {len(keep)} 个句子，无法训练；"
            f"请调低阈值或更换数据集"
        )
    return keep


def run_subject(args, subject, out_dir, text_encoder, device, keep_sentences):
    """对单个被试完整跑一遍训练 + 评测，返回 ``(best_score, best_metrics)``。

    所有被试共用 ``main`` 里创建好的同一个时间戳目录 ``out_dir``，
    CSV / checkpoint 以被试名前缀区分（如 ``sub-01_train_epochs.csv``）。
    ``keep_sentences`` 是全局算好的"单例句"保留集，按句子文本过滤，保证跨 day 一致。
    """
    set_seed(args.seed)

    print(f"\n{'=' * 88}\n[Subject] {subject}\n{'=' * 88}")

    # ---------------- 数据 ----------------
    dataset = CharTokenEEGDataset(
        event=args.event,
        sample=args.sample,
        subject=subject,
        days=args.day,
        char_duration=args.char_duration,
        keep_sentences=keep_sentences,
    )
    dataset[0]  # 触发一次读取，确定 n_chans
    train_set, val_set, info = split_by_sentence(dataset, args.val_ratio, args.seed)

    days_label = ",".join(f"{d:02d}" for d in sorted(args.day))
    print(
        f"[Data] {subject} day-{days_label} event={args.event} "
        f"token_samples={dataset.token_samples} n_chans={dataset.n_chans}\n"
        f"       句子 {len(dataset.meta['sentence'].unique())} 条 -> "
        f"train {info['n_unique_train']} / val {info['n_unique_val']} "
        f"(trial {len(train_set)} / {len(val_set)})，句子重叠={info['sentence_overlap']}"
    )

    # ---------------- 分组 batch sampler ----------------
    # 一句最多能提供多少条 trial：单天是 1（无重复），跨天才有多个视图
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
    # 恒定学习率在 val_loss 走平后仍会继续拟合，余弦衰减到 0 有助于收敛
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    # ---------------- 训练 ----------------
    # 所有被试共用同一时间戳目录，文件以被试名前缀区分
    ckpt_path = out_dir / f"{subject}_best.pt"
    train_epochs_path = out_dir / f"{subject}_train_epochs.csv"
    train_steps_path = out_dir / f"{subject}_train_steps.csv"

    lower_is_better = args.select_by == "val_loss"
    best_score = float("inf") if lower_is_better else -float("inf")
    best_metrics = None
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

        # 本轮指标追加到 train_epochs.csv（首轮自动写表头）
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
    return best_score, best_metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Config] device={device} subjects={args.subject} temperature={args.temperature} "
          f"threshold={args.threshold} "
          f"batch={args.n_sentences}句×{args.n_repeats}视图 "
          f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
          f"schedule={args.lr_schedule} select={args.select_by} seed={args.seed}")

    # 所有被试共用同一个时间戳目录，文件以被试名前缀区分
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    # LaBSE 只加载一次，各被试复用同一个文本编码器
    print("[Text] 加载 LaBSE（冻结） ...")
    text_encoder = LaBSETextEncoder()

    # 近重复过滤：全局算一次保留集，所有被试 / 所有 day 共用
    keep_sentences = compute_keep_sentences(args, text_encoder)

    # ---------------- 逐个被试训练 ----------------
    summary = {}
    for subject in args.subject:
        best_score, best_metrics = run_subject(
            args, subject, out_dir, text_encoder, device, keep_sentences
        )
        summary[subject] = {"best": best_score, **best_metrics}

    # ---------------- 汇总 ----------------
    if len(summary) > 1:
        print(f"\n{'=' * 88}\n[Summary] 被试对照（best by {args.select_by}）\n{'=' * 88}")
        print(f"{'subject':>8} {'val_loss':>10} {'e2t@1':>8} {'e2t@5':>8} {'t2e@1':>8} {'t2e@5':>8}")
        for subject, m in summary.items():
            print(
                f"{subject:>8} {m['val_loss']:>10.4f} {m['eeg2text_top1']:>8.4f} "
                f"{m['eeg2text_top5']:>8.4f} {m['text2eeg_top1']:>8.4f} {m['text2eeg_top5']:>8.4f}"
            )
        n_cand = next(iter(summary.values()))["n_text_candidates"]
        n_val = next(iter(summary.values()))["n_val"]
        pos_per_anchor = next(iter(summary.values()))["pos_per_anchor"]
        print(
            f"\n[Baseline] eeg2text 候选 {n_cand} 句，随机 top-1={1 / n_cand:.4f} "
            f"top-5={min(5, n_cand) / n_cand:.4f}；text2eeg 候选 {n_val} 条，"
            f"随机 top-5={min(5 * pos_per_anchor / n_val, 1.0):.4f}"
        )
    print(f"\n[Output] {out_dir}")


if __name__ == "__main__":
    main()