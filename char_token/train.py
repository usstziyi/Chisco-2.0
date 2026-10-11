"""字符级 token 方案的对比学习训练循环。

数据
----
被试 / 天数 / 事件类型 / 采样率均可通过命令行指定，默认 sub-01 + day-01。
train / val 按 **句子** 做 8:2 切分（不是按 trial），保证同一句子的所有 trial
只会落在同一侧，不会跨集泄漏。

损失
----
对称 InfoNCE（对比学习交叉熵），正样本是"EEG 与它自己的文本"：

    logits = eeg_embedding @ text_embedding.T / temperature
    loss   = 0.5 * (CE(logits, labels) + CE(logits.T, labels))

温度系数默认 0.2。

注意 day-01 里每句话只录了一次（无重复），所以 batch 内不存在"同句的第二个
trial"当正样本，正样本只能是配对的那条文本，负样本是 batch 内其它句子的文本，
即 CLIP 式的 in-batch negatives。句子只重复出现在跨天之间。

文本侧用冻结的 LaBSE，所有句子的文本 embedding 在训练前一次性算好、做成查表，
训练时只做索引，不重复跑 LaBSE。

用法::

    uv run python char_token/train.py
    uv run python char_token/train.py --event reading --epochs 50
    uv run python char_token/train.py --temperature 0.1 --batch-size 64 --lr 5e-4
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# 支持 `python char_token/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import CharTokenEEGDataset, char_token_collate
from char_token.model import CharTokenEEGEncoder
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


def contrastive_loss(eeg: torch.Tensor, text: torch.Tensor, temperature: float) -> torch.Tensor:
    """对称 InfoNCE：EEG->文本 与 文本->EEG 两个方向取平均。"""
    logits = eeg @ text.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_e2t = F.cross_entropy(logits, labels)
    loss_t2e = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_e2t + loss_t2e)


def train_one_epoch(model, loader, lookup, optimizer, device, temperature) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for tokens, mask, sentences in loader:
        tokens = tokens.to(device)
        mask = mask.to(device)
        text = gather_text(sentences, lookup, device)

        eeg = model(tokens, mask)
        loss = contrastive_loss(eeg, text, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(sentences)
        total_n += len(sentences)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, lookup, device, temperature) -> dict:
    """在整个 val 集上算检索指标与全库对比损失。

    因为切分保证 val 内每句互不相同，对角线就是唯一正样本，
    所以 ``sim.argmax(1) == 自身下标`` 即可判定 top-1 命中。
    """
    model.eval()
    eeg_parts, sentences = [], []
    for tokens, mask, batch_sentences in loader:
        eeg_parts.append(model(tokens.to(device), mask.to(device)))
        sentences.extend(batch_sentences)

    eeg = torch.cat(eeg_parts, dim=0)
    text = gather_text(sentences, lookup, device)
    n = eeg.size(0)
    labels = torch.arange(n, device=device)
    sim = eeg @ text.t()

    metrics = {
        "val_loss": float(contrastive_loss(eeg, text, temperature)),
        "n_val": n,
        "n_unique_sentences": len(set(sentences)),
    }
    for name, matrix in (("eeg2text", sim), ("text2eeg", sim.t())):
        hit = matrix.argsort(dim=1, descending=True).eq(labels[:, None])
        metrics[f"{name}_top1"] = float(hit[:, :1].any(dim=1).float().mean())
        metrics[f"{name}_top5"] = float(hit[:, :5].any(dim=1).float().mean())
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="字符级 token EEG–文本对比学习训练。")
    parser.add_argument("--subject", default="sub-01", help="被试，如 sub-01")
    parser.add_argument("--day", type=int, default=1, help="day 编号（1~8），默认 1")
    parser.add_argument(
        "--event", default="inner", choices=["inner", "reading"],
        help="取 inner（默读，默认）或 reading（朗读）",
    )
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument("--char-duration", type=float, default=0.4, help="每个字 token 的时长（秒，默认 0.4）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集句子比例（默认 0.2）")
    parser.add_argument("--batch-size", type=int, default=32, help="batch 大小（默认 32）")
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
    print(f"[Config] device={device} temperature={args.temperature} batch={args.batch_size} "
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

    print(
        f"[Data] {args.subject} day-{args.day:02d} event={args.event} "
        f"token_samples={dataset.token_samples} n_chans={dataset.n_chans}\n"
        f"       句子 {len(dataset.meta['sentence'].unique())} 条 -> "
        f"train {info['n_unique_train']} / val {info['n_unique_val']} "
        f"(trial {len(train_set)} / {len(val_set)})，句子重叠={info['sentence_overlap']}"
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=char_token_collate,
        num_workers=args.num_workers,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

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
        f"{'epoch':>5} {'train_loss':>10} {'val_loss':>9} "
        f"{'e2t@1':>7} {'e2t@5':>7} {'t2e@1':>7} {'t2e@5':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, lookup, optimizer, device, args.temperature
        )
        metrics = evaluate(model, val_loader, lookup, device, args.temperature)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{k: v for k, v in metrics.items()},
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
            f"{epoch:>5} {train_loss:>10.4f} {metrics['val_loss']:>9.4f} "
            f"{metrics['eeg2text_top1']:>7.4f} {metrics['eeg2text_top5']:>7.4f} "
            f"{metrics['text2eeg_top1']:>7.4f} {metrics['text2eeg_top5']:>7.4f}{flag}"
        )

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    chance1 = 1.0 / metrics["n_val"]
    chance5 = min(5, metrics["n_val"]) / metrics["n_val"]
    print(
        f"\n[Done] 最佳 eeg2text top-1 = {best_top1:.4f}（随机基线 {chance1:.4f}）\n"
        f"       随机 top-5 基线 = {chance5:.4f}，val 共 {metrics['n_val']} 句\n"
        f"       checkpoint: {ckpt_path}\n"
        f"       metrics:    {log_path}"
    )


if __name__ == "__main__":
    main()