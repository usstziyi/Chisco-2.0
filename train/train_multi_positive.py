"""训练 Chisco-2.0 的 EEG→Text 检索模型（对比学习）。
组合的三个子包:
    train.eeg_encoder   → TSConvFixedEEGEncoder
    train.text_encoder  → LaBSETextEncoder（冻结，带文本缓存）
    train.dataloader    → n_chars=9 的 train / val DataLoader
对齐方式:
    EEG  encoder 输出 (B, D)，已 L2 归一化
    LaBSE        输出 (B, D)，已 L2 归一化（LaBSE 的 D = 768）
    两者点积即 cosine similarity。
损失（对称 Multi-Positive InfoNCE，正样本概率求和版）:
    logits = eeg_emb @ text_emb.T / temperature
    positive_mask[i, j] = (sentences[i] == sentences[j])
    每个方向: -log(sum(exp(positive_logits)) / sum(exp(all_logits)))
    最终取 EEG→Text 和 Text→EEG 两个方向的平均。
    按完整字符串精确匹配；不做 strip、标点替换或语义合并。
    无重复文本时退化为原来的对称 InfoNCE。
    整个 batch 只有一种文本（含 B=1）时 loss 为 0，没有负样本信号。
    这是正样本概率求和版本，不是对各正样本 log 概率求平均的 SupCon。
    保留随机 DataLoader；只将当前 batch 中的同文本样本标记为正样本。
指标:
    val loss 仍为各 batch 损失按 EEG 样本数加权平均，受 batch 构成影响。
    val top-1 / top-5 在整个已评估 val 集的去重文本候选上计算。
    max_batches > 0 时，仅评估实际读到的验证子集。
EEG 单位:
    数据集的 __getitem__ 已把原始伏特乘上 scale=1e6（V → µV），
    因此这里不再重复缩放，直接 .to(device) 即可。
用法:
    uv run python train/train_multi_positive.py --n-chars 9 --event inner
    uv run python train/train_multi_positive.py --n-chars 9 --event inner --subject sub-01 --epochs 50
快速冒烟（只跑少量 batch）:
    uv run python train/train_multi_positive.py --n-chars 9 --event inner --max-batches 3
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 直接 `python train/train.py` 时 sys.path[0] 是本目录（train/），
# 把项目根目录放到最前面，`import train` 才会解析到 train/ 这个包，
# 而不是被同目录下的 train.py 抢占。
sys.path.insert(0, str(PROJECT_ROOT))
from train.dataloader import build_nchars9_dataloaders
from train.eeg_encoder import TSConvFixedEEGEncoder
from train.text_encoder import LaBSETextEncoder


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
# 每轮 epoch 追加写入的日志字段
EPOCH_LOG_FIELDS = [
    "epoch", "train_loss", "train_top1",
    "val_loss", "val_top1", "val_topk", "val_k", "val_cos_pos",
    "is_best", "best_top1", "seconds",
]
# 本次运行最终 val 结果的汇总字段
FINAL_VAL_FIELDS = [
    "n_chars", "event", "sample", "subject", "run_id", "device",
    "n_chans", "n_times", "epochs", "best_epoch",
    "best_val_loss", "best_val_top1", "best_val_topk", "best_val_k", "best_val_cos_pos",
    "last_val_loss", "last_val_top1", "last_val_topk", "last_val_cos_pos",
    "checkpoint",
]


def append_csv_row(path: Path, fields: list, row: dict) -> None:
    """把一行追加到 CSV；文件不存在时先写表头。"""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv_row(path: Path, fields: list, row: dict) -> None:
    """把一行写入 CSV（覆盖已有内容）。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


# ============================================================
# Loss / 指标
# ============================================================


def group_sentences(sentences) -> tuple[list[str], list[int]]:
    """按首次出现顺序去重，并返回每条原始文本的候选索引。"""
    unique_texts: list[str] = []
    text_to_id: dict[str, int] = {}
    sentence_ids: list[int] = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise TypeError("sentences 必须是字符串序列")
        if sentence not in text_to_id:
            text_to_id[sentence] = len(unique_texts)
            unique_texts.append(sentence)
        sentence_ids.append(text_to_id[sentence])
    if not sentence_ids:
        raise ValueError("sentences 不能为空")
    return unique_texts, sentence_ids


@torch.no_grad()
def encode_text_batch(text_encoder, sentences, device) -> torch.Tensor:
    """先编码去重文本，再还原为与 EEG 一一对应的 (B, D)。

    沿用原项目 encode_unique 保持输入顺序的接口约定。
    显式还原重复行，避免把 (U, D) 的去重结果误当作 (B, D)。
    """
    unique_texts, sentence_ids = group_sentences(sentences)
    unique_emb = text_encoder.encode_unique(unique_texts).to(device)
    if unique_emb.ndim != 2 or unique_emb.size(0) != len(unique_texts):
        raise ValueError("encode_unique 必须按输入顺序为每个唯一文本返回一行 embedding")
    indices = torch.tensor(sentence_ids, dtype=torch.long, device=device)
    return unique_emb.index_select(0, indices)


def build_positive_mask(sentences, device) -> torch.Tensor:
    """同一个 batch 中，字符串相等的所有 EEG/Text 配对均为正样本。"""
    _, sentence_ids = group_sentences(sentences)
    ids = torch.tensor(sentence_ids, dtype=torch.long, device=device)
    # 跨模态配对包含对角线，不应像 EEG–EEG 损失那样排除自身。
    return ids[:, None].eq(ids[None, :])


def multi_positive_info_nce(
    eeg_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    """对称多正样本损失；输入 embedding 应已经 L2 归一化。

    logsumexp 稳定计算概率和，无需显式 exp 后再取 log。
    保留重复文本行，因此文本出现次数仍会影响候选权重。
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须是有限正数")
    if eeg_emb.ndim != 2 or text_emb.ndim != 2 or eeg_emb.shape != text_emb.shape:
        raise ValueError("eeg_emb 和 text_emb 必须具有相同的 (B, D) shape")
    batch_size = eeg_emb.size(0)
    if batch_size == 0:
        raise ValueError("batch 不能为空")
    if eeg_emb.device != text_emb.device:
        raise ValueError("两种 embedding 必须在同一 device 上")
    if positive_mask.shape != (batch_size, batch_size) or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask 必须是 (B, B) 的布尔张量")
    positive_mask = positive_mask.to(eeg_emb.device)
    if not (positive_mask.any(dim=1).all() & positive_mask.any(dim=0).all()).item():
        raise ValueError("每个 EEG 和每个 Text 必须至少有一个正样本")

    # 低精度输入提升到 float32，float64 输入保留以便数值检验。
    work_dtype = torch.float64 if torch.float64 in (eeg_emb.dtype, text_emb.dtype) else torch.float32
    with torch.autocast(device_type=eeg_emb.device.type, enabled=False):
        logits = eeg_emb.to(work_dtype) @ text_emb.to(work_dtype).t() / temperature
        positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        loss_eeg_to_text = (
            torch.logsumexp(logits, dim=1)
            - torch.logsumexp(positive_logits, dim=1)
        ).mean()
        loss_text_to_eeg = (
            torch.logsumexp(logits, dim=0)
            - torch.logsumexp(positive_logits, dim=0)
        ).mean()
    return 0.5 * (loss_eeg_to_text + loss_text_to_eeg)


def in_batch_top1(
    eeg_emb: torch.Tensor, text_emb: torch.Tensor, positive_mask: torch.Tensor,
) -> float:
    """检索到任何一条相同文本均视为正确，不要求候选行索引等于 EEG 行索引。"""
    predictions = (eeg_emb @ text_emb.t()).argmax(dim=1)
    return (
        positive_mask.gather(1, predictions[:, None])
        .float().mean().item()
    )


# ============================================================
# 训练 / 验证
# ============================================================


def train_one_epoch(
    eeg_encoder, loader, text_encoder, optimizer, device, temperature, max_batches,
) -> dict:
    eeg_encoder.train()
    total_loss = 0.0
    total_top1 = 0.0
    seen = 0
    total_steps = min(len(loader), max_batches) if max_batches else len(loader)
    for step, (eeg, sentences) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        # 数据集已把 V → µV，这里不再乘 EEG_SCALE
        eeg = eeg.to(device, non_blocking=True)
        text_emb = encode_text_batch(text_encoder, sentences, device)
        eeg_emb = eeg_encoder(eeg)
        positive_mask = build_positive_mask(sentences, eeg_emb.device)
        loss = multi_positive_info_nce(eeg_emb, text_emb, temperature, positive_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = len(sentences)
        top1 = in_batch_top1(eeg_emb.detach(), text_emb, positive_mask)
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        seen += batch_size
        print(
            f"  step {step + 1:4d}/{total_steps} | "
            f"loss {loss.item():.4f} | top1 {top1:.3f} | bs {batch_size}",
            flush=True,
        )
    if seen == 0:
        raise ValueError("训练 loader 没有产生任何样本")
    return {"loss": total_loss / seen, "top1": total_top1 / seen, "n": seen}


@torch.no_grad()
def evaluate(eeg_encoder, loader, text_encoder, device, temperature, max_batches, topk) -> dict:
    if topk < 1:
        raise ValueError("topk 必须 >= 1")
    eeg_encoder.eval()
    total_loss = 0.0
    seen = 0
    eeg_embeddings = []
    texts: list[str] = []
    for step, (eeg, sentences) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        eeg = eeg.to(device, non_blocking=True)
        text_emb = encode_text_batch(text_encoder, sentences, device)
        eeg_emb = eeg_encoder(eeg)
        positive_mask = build_positive_mask(sentences, eeg_emb.device)
        total_loss += multi_positive_info_nce(
            eeg_emb, text_emb, temperature, positive_mask,
        ).item() * len(sentences)
        seen += len(sentences)
        eeg_embeddings.append(eeg_emb.cpu())
        texts.extend(sentences)
    if seen == 0:
        raise ValueError("验证 loader 没有产生任何样本")
    eeg_all = torch.cat(eeg_embeddings, dim=0)              # (N, D)
    unique_texts, target_ids = group_sentences(texts)
    # unique_texts 已去重，直接编码即可得到 (U, D)
    text_all = text_encoder.encode_unique(unique_texts).cpu()  # (U, D)
    similarity = eeg_all @ text_all.t()                     # (N, U)
    targets = torch.tensor(target_ids, dtype=torch.long)
    k = min(topk, similarity.size(1))
    top1 = similarity.argmax(dim=1).eq(targets).float().mean().item()
    topk_acc = (
        similarity.topk(k, dim=1).indices
        .eq(targets[:, None])
        .any(dim=1)
        .float()
        .mean()
        .item()
    )
    return {
        "loss": total_loss / seen,
        "top1": top1,
        "topk": topk_acc,
        "k": k,
        "cos_pos": similarity[torch.arange(len(texts)), targets].mean().item(),
        "n": len(texts),
        "n_unique_texts": len(unique_texts),
    }


# ============================================================
# Main
# ============================================================


def probe_eeg_shape(loader) -> tuple[int, int]:
    """取一条样本的 EEG shape，得到 (n_chans, n_times)。
    loader.dataset 是 Subset，这里直接取第 0 条即可；
    同一 n_chars 下所有样本的 T 一致，无需遍历。
    """
    return tuple(loader.dataset[0][0].shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Chisco-2.0 EEG→Text 检索模型")
    parser.add_argument("--n-chars", type=int, default=9, help="句子字符数（筛选条件）")
    parser.add_argument("--event", default="inner", choices=["reading", "inner"], help="事件类型")
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（对应 tsv 的 sample 列）")
    parser.add_argument("--subject", default=None, help="被试过滤，如 sub-01；默认 None 表示全部")
    parser.add_argument("--val-size", type=int, default=100, help="验证集样本数")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--temperature", type=float, default=0.07, help="Multi-Positive InfoNCE 温度")
    parser.add_argument("--topk", type=int, default=5, help="检索 top-k 的 k")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数")
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动选择")
    parser.add_argument("--text-model", default="sentence-transformers/LaBSE", help="文本编码模型名")
    parser.add_argument("--max-batches", type=int, default=0, help="每轮最多跑多少个 batch（0 表示不限制，用于冒烟测试）")
    # TSConv 结构超参
    parser.add_argument("--k", type=int, default=40, help="TSConv 时间卷积输出通道数")
    parser.add_argument("--m1", type=int, default=25, help="时间卷积核长度")
    parser.add_argument("--m2", type=int, default=51, help="时间平均池化窗口")
    parser.add_argument("--s", type=int, default=5, help="时间平均池化 stride")
    parser.add_argument("--projection-hidden-dim", type=int, default=512, help="投影 MLP 隐藏层维度")
    parser.add_argument("--drop-prob", type=float, default=0.5, help="TSConv dropout")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature 必须是有限正数")
    if args.epochs < 1 or args.topk < 1 or args.max_batches < 0:
        parser.error("--epochs、--topk 必须 >= 1，--max-batches 必须 >= 0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # ---------------- 数据 ----------------
    sub_tag = args.subject if args.subject else "all"
    train_loader, val_loader = build_nchars9_dataloaders(
        n_chars=args.n_chars,
        event=args.event,
        sample=args.sample,
        subject=args.subject,
        val_size=args.val_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    n_chans, n_times = probe_eeg_shape(train_loader)
    # ---------------- 模型 ----------------
    text_encoder = LaBSETextEncoder(
        model_name=args.text_model,
    ).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    eeg_encoder = TSConvFixedEEGEncoder(
        n_chans=n_chans,
        n_times=n_times,
        k=args.k,
        m1=args.m1,
        m2=args.m2,
        s=args.s,
        projection_hidden_dim=args.projection_hidden_dim,
        embedding_dim=text_encoder.embedding_dim,
        drop_prob=args.drop_prob,
    ).to(device)
    optimizer = torch.optim.AdamW(
        eeg_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    n_params = sum(p.numel() for p in eeg_encoder.parameters() if p.requires_grad)
    print(
        f"n_chars={args.n_chars} / event={args.event} / sample={args.sample} / subject={sub_tag}\n"
        f"  device        : {device}\n"
        f"  EEG           : ({n_chans}, {n_times})\n"
        f"  train / val   : {len(train_loader.dataset)} / {len(val_loader.dataset)}\n"
        f"  参数 / 温度   : {n_params:,} / {args.temperature}\n"
        f"  loss          : symmetric multi-positive (sum of positive probabilities)\n"
    )
    # --------------- 输出目录 ---------------
    # 每次运行新建一个独立文件夹：outputs/nchars-<n>_event-<e>_sample-<s>_sub-<x>_<时间戳>/
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"nchars-{args.n_chars}_event-{args.event}_sample-{args.sample}"
        f"_sub-{sub_tag}_{run_id}"
    )
    run_dir = OUTPUTS_DIR /"trainlog"/run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"             # 最佳权重
    epoch_log_path = run_dir / "epoch_log.csv"  # 每个 epoch 追加一行
    final_val_path = run_dir / "final_val.csv"  # 本次运行最终 val 结果
    print(f"  输出目录      : {run_dir}\n")
    # ---------------- 训练/验证 ----------------
    best_top1 = -1.0
    best_epoch = 0
    best_val_metrics = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(
            eeg_encoder, train_loader, text_encoder, optimizer,
            device, args.temperature, args.max_batches,
        )
        val_metrics = evaluate(
            eeg_encoder, val_loader, text_encoder, device,
            args.temperature, args.max_batches, args.topk,
        )
        dt = time.perf_counter() - t0
        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} top1 {train_metrics['top1']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"top1 {val_metrics['top1']:.3f} "
            f"top{val_metrics['k']} {val_metrics['topk']:.3f} "
            f"cos+ {val_metrics['cos_pos']:.3f} | "
            f"候选文本 {val_metrics['n_unique_texts']} | "
            f"{dt:.1f}s"
        )
        is_best = val_metrics["top1"] > best_top1
        if is_best:
            best_top1 = val_metrics["top1"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            torch.save(
                {
                    "model_state": eeg_encoder.state_dict(),
                    "args": vars(args),
                    "n_chans": n_chans,
                    "n_times": n_times,
                    "epoch": epoch,
                    "val_top1": val_metrics["top1"],
                    "val_topk": val_metrics["topk"],
                    "loss_type": "symmetric_multi_positive_sum",
                    "positive_match": "exact_text",
                    "retrieval_candidates": "unique_validation_texts",
                },
                ckpt_path,
            )
        # 把本轮日志追加到本次运行的 CSV
        append_csv_row(epoch_log_path, EPOCH_LOG_FIELDS, {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_top1": f"{train_metrics['top1']:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_top1": f"{val_metrics['top1']:.6f}",
            "val_topk": f"{val_metrics['topk']:.6f}",
            "val_k": val_metrics["k"],
            "val_cos_pos": f"{val_metrics['cos_pos']:.6f}",
            "is_best": int(is_best),
            "best_top1": f"{best_top1:.6f}",
            "seconds": f"{dt:.3f}",
        })
        if is_best:
            print(f"  ↑ best top1 {best_top1:.3f}，已保存 {ckpt_path.name}")
    # 本次运行最终 val 结果（含最佳与最后一轮）
    write_csv_row(final_val_path, FINAL_VAL_FIELDS, {
        "n_chars": args.n_chars,
        "event": args.event,
        "sample": args.sample,
        "subject": sub_tag,
        "run_id": run_id,
        "device": str(device),
        "n_chans": n_chans,
        "n_times": n_times,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": f"{best_val_metrics['loss']:.6f}",
        "best_val_top1": f"{best_val_metrics['top1']:.6f}",
        "best_val_topk": f"{best_val_metrics['topk']:.6f}",
        "best_val_k": best_val_metrics["k"],
        "best_val_cos_pos": f"{best_val_metrics['cos_pos']:.6f}",
        "last_val_loss": f"{val_metrics['loss']:.6f}",
        "last_val_top1": f"{val_metrics['top1']:.6f}",
        "last_val_topk": f"{val_metrics['topk']:.6f}",
        "last_val_cos_pos": f"{val_metrics['cos_pos']:.6f}",
        "checkpoint": ckpt_path.name,
    })
    print(
        f"\n完成。最佳 val top1 = {best_top1:.3f} (epoch {best_epoch})\n"
        f"运行目录  : {run_dir}\n"
        f"检查点    : {ckpt_path}\n"
        f"逐轮日志  : {epoch_log_path.name}\n"
        f"最终结果  : {final_val_path.name}"
    )
if __name__ == "__main__":
    main()
