"""字符级 token 方案的对比学习训练循环。

数据
----
被试 / 天数 / 事件类型 / 采样率均可通过命令行指定，默认 sub-01 + day-01。
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

文本侧用冻结的 LaBSE，所有句子的文本 embedding 在训练前一次性算好、做成查表，
训练时只做索引，不重复跑 LaBSE。

用法::

    uv run python char_token/train.py
    uv run python char_token/train.py --day 1 2 3 4 --n-sentences 8 --n-repeats 4
    uv run python char_token/train.py --day 1 2 3 4 --temperature 0.1 --lr 5e-4
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# 支持 `python char_token/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import CharTokenEEGDataset, char_token_collate
from char_token.model import CharTokenEEGEncoder
from char_token.sampler import SentenceGroupBatchSampler
from char_token.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token"


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
    return dict(zip(unique, table)), table.shape[1]


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


def train_one_epoch(model, loader, lookup, optimizer, device, temperature):
    """跑一个 epoch，返回 ``(平均 loss, 每个 anchor 的平均正样本数)``。"""
    model.train()
    total_loss, total_n, total_pos = 0.0, 0, 0.0
    for tokens, mask, sentences in loader:
        tokens = tokens.to(device)
        mask = mask.to(device)
        text = gather_text(sentences, lookup, device)
        groups = sentence_group_ids(sentences, device)

        eeg = model(tokens, mask)
        loss = contrastive_loss(eeg, text, temperature, groups)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = len(sentences)
        total_loss += loss.item() * batch
        total_n += batch
        total_pos += (groups[:, None] == groups[None, :]).sum().item()
    return total_loss / max(total_n, 1), total_pos / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, lookup, device, temperature) -> dict:
    """在整个 val 集上算检索指标与全库对比损失。

    命中判定是"分组感知"的：只要 top-k 里出现**任意一个同句**的候选就算命中。
    单天数据下每句只有一个 trial，分组掩码退化成单位阵，结果与"对角线命中"一致。
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
    sim = eeg @ text.t()
    row = torch.arange(n, device=device)[:, None]

    metrics = {
        "val_loss": float(contrastive_loss(eeg, text, temperature, groups)),
        "n_val": n,
        "n_unique_sentences": len(set(sentences)),
        "pos_per_anchor": float(positive.sum(dim=1).float().mean()),
    }
    for name, matrix in (("eeg2text", sim), ("text2eeg", sim.t())):
        # 按相似度降序排出候选，再用正样本掩码判断每个名次是否命中
        ranked = matrix.argsort(dim=1, descending=True)
        hit = positive[row, ranked]
        metrics[f"{name}_top1"] = float(hit[:, :1].any(dim=1).float().mean())
        metrics[f"{name}_top5"] = float(hit[:, :5].any(dim=1).float().mean())
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="字符级 token EEG–文本对比学习训练。")
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
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减（默认 1e-4）")
    parser.add_argument("--temperature", type=float, default=0.2, help="对比学习温度系数（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Config] device={device} temperature={args.temperature} "
          f"batch={args.n_sentences}句×{args.n_repeats}视图 "
          f"epochs={args.epochs} lr={args.lr} seed={args.seed}")

    # ---------------- 数据 ----------------
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
    print("[Text] 加载 LaBSE 并预计算全部句子 embedding ...")
    text_encoder = LaBSETextEncoder()
    lookup, embedding_dim = build_text_lookup(
        text_encoder, dataset.meta["sentence"].tolist(), device
    )
    print(f"       已缓存 {len(lookup)} 条句子，embedding_dim={embedding_dim}")

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

    # ---------------- 训练 ----------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    log_path = out_dir / "metrics.csv"

    best_top1 = -1.0
    rows = []
    header = (
        f"{'epoch':>5} {'train_loss':>10} {'pos/anch':>9} {'val_loss':>9} "
        f"{'e2t@1':>7} {'e2t@5':>7} {'t2e@1':>7} {'t2e@5':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pos = train_one_epoch(
            model, train_loader, lookup, optimizer, device, args.temperature
        )
        metrics = evaluate(model, val_loader, lookup, device, args.temperature)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_pos_per_anchor": train_pos,
            **metrics,
            "lr": args.lr,
            "temperature": args.temperature,
        }
        rows.append(row)

        flag = ""
        if metrics["eeg2text_top1"] > best_top1:
            best_top1 = metrics["eeg2text_top1"]
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
            f"{epoch:>5} {train_loss:>10.4f} {train_pos:>9.2f} {metrics['val_loss']:>9.4f} "
            f"{metrics['eeg2text_top1']:>7.4f} {metrics['eeg2text_top5']:>7.4f} "
            f"{metrics['text2eeg_top1']:>7.4f} {metrics['text2eeg_top5']:>7.4f}{flag}"
        )

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 多正样本下随机基线随"每个 anchor 有几个同句候选"上升
    n_val, pos_per_anchor = metrics["n_val"], metrics["pos_per_anchor"]
    chance1 = min(pos_per_anchor / n_val, 1.0)
    chance5 = min(5 * pos_per_anchor / n_val, 1.0)
    print(
        f"\n[Done] 最佳 eeg2text top-1 = {best_top1:.4f}（随机基线 {chance1:.4f}）\n"
        f"       随机 top-5 基线 = {chance5:.4f}，val 共 {n_val} 条 / "
        f"{metrics['n_unique_sentences']} 句，每个 anchor 平均 {pos_per_anchor:.2f} 个同句候选\n"
        f"       checkpoint: {ckpt_path}\n"
        f"       metrics:    {log_path}"
    )


if __name__ == "__main__":
    main()