"""绘制 best.pt 在**预测集**上的 EEG × 文本 跨模态相似度矩阵。

和 ``plot_sim_matrix.py`` 的区别
------------------------------
``plot_sim_matrix.py`` 画的是 **EEG × EEG 自相似**（两侧都是 EEG 分支 embedding，
矩阵对称）。本脚本画的是 **EEG × 文本**：

    行 = EEG 分支 embedding   (900 条 trial，可训练的 CharTokenEEGEncoder 输出)
    列 = 文本分支 embedding   (同序 900 条 trial 各自句子的冻结 BGE-M3 输出)
    格子 (i, j) = cos(EEG_i, 文本_j)

因此矩阵**一般不对称**，且对角线 = "第 i 条 trial 与它自己那句文本"的**配对相似度**——
它正是训练里对比损失 ``logits = eeg @ text.T / temperature`` 那条对角线。

要注意：同一句在 day 1~4 各录一次，所以**文本侧每句的 4 列完全相同**（同一个句子 →
同一个文本向量）。这不是 bug，是数据本身的性质；也正因如此，图上会看到"列成组重复"。

预测集与训练时**完全一致**：用 checkpoint 里存下的 ``val_ratio`` / ``seed`` 调
``split_by_sentence``，和 ``plot_sim_matrix.py`` 取的是同一批 trial。

用法::

    uv run python char_token_bge_m3/plot_cross_modal_matrix.py
    uv run python char_token_bge_m3/plot_cross_modal_matrix.py --order raw
    uv run python char_token_bge_m3/plot_cross_modal_matrix.py --start 0 --end 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

# 支持 `python char_token_bge_m3/plot_cross_modal_matrix.py` 直接执行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_bge_m3.dataset import CharTokenEEGDataset
from char_token_bge_m3.model import CharTokenEEGEncoder
from char_token_bge_m3.text_encoder import BGEM3TextEncoder
from char_token_bge_m3.train import build_text_lookup, gather_text, split_by_sentence

# 复用绘图脚本里的工具函数（checkpoint 定位 / 编码 / 排序 / 画图 / 导出映射），避免重复实现。
# 注意：import 该模块同时会把中文字体等 rcParams 设好。
from char_token_bge_m3.plot_sim_matrix import (
    encode_indices,
    find_ckpt,
    group_order,
    plot_matrix,
    save_order_csv,
    subject_from_ckpt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="加载 best.pt，在预测集上画 EEG × 文本 跨模态相似度矩阵。"
    )
    parser.add_argument("--ckpt", default=None, help="checkpoint 路径；默认自动取最新")
    parser.add_argument("--subject", default=None, help="被试（如 sub-01）；默认从 checkpoint 推断")
    parser.add_argument("--day", type=int, nargs="+", default=None, help="day 编号；默认沿用训练参数")
    parser.add_argument("--event", default=None, help="inner / reading；默认沿用训练参数")
    parser.add_argument("--sample", type=int, default=None, help="采样率 Hz；默认沿用训练参数")
    parser.add_argument("--char-duration", type=float, default=None, help="每个字 token 时长（秒）")
    parser.add_argument("--val-ratio", type=float, default=None, help="验证集句子比例；默认沿用训练参数")
    parser.add_argument("--seed", type=int, default=None, help="切分种子；默认沿用训练参数")
    parser.add_argument("--batch-size", type=int, default=64, help="EEG 编码时的 batch 大小（默认 64）")
    parser.add_argument(
        "--order", default="sentence", choices=["sentence", "raw"],
        help="排图顺序：sentence 按句子分组（默认），raw 用预测集原始顺序",
    )
    parser.add_argument("--start", type=int, default=0, help="绘图起始下标（默认 0）")
    parser.add_argument("--end", type=int, default=None, help="绘图结束下标，不含（默认到末尾）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=None, help="图输出目录；默认存到 checkpoint 同级目录")
    return parser.parse_args()


@torch.no_grad()
def retrieval_eeg2text(eeg: torch.Tensor, text_unique: torch.Tensor, targets: torch.Tensor):
    """eeg->text 检索：候选池 = 去重后的唯一句子（口径同训练评测）。"""
    ranked = (eeg @ text_unique.t()).argsort(dim=1, descending=True)
    hit = ranked.eq(targets[:, None])
    return float(hit[:, :1].any(dim=1).float().mean()), float(hit[:, :5].any(dim=1).float().mean())


def main():
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = find_ckpt(args.ckpt, args.subject)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args") or {}

    subject = args.subject or subject_from_ckpt(ckpt_path)
    days = args.day if args.day is not None else (train_args.get("day") or [1])
    event = args.event or train_args.get("event") or "inner"
    sample = args.sample if args.sample is not None else int(train_args.get("sample", 250))
    char_duration = (
        args.char_duration if args.char_duration is not None
        else float(train_args.get("char_duration", 0.4))
    )
    val_ratio = args.val_ratio if args.val_ratio is not None else float(train_args.get("val_ratio", 0.2))
    seed = args.seed if args.seed is not None else int(train_args.get("seed", 0))

    n_chans = int(ckpt["n_chans"])
    token_samples = int(ckpt["token_samples"])
    embedding_dim = int(ckpt["embedding_dim"])

    print(
        f"[Ckpt] {ckpt_path}\n"
        f"       epoch={ckpt.get('epoch')} n_chans={n_chans} "
        f"token_samples={token_samples} embedding_dim={embedding_dim}"
    )

    # ---------------- 数据：与训练完全一致的句子级切分 ----------------
    dataset = CharTokenEEGDataset(
        event=event, sample=sample, subject=subject, days=days, char_duration=char_duration
    )
    dataset[0]  # 触发一次读取，确定 n_chans
    if dataset.n_chans != n_chans:
        raise ValueError(f"数据集通道数 {dataset.n_chans} 与 checkpoint 的 {n_chans} 不一致")

    _, val_set, info = split_by_sentence(dataset, val_ratio, seed)
    val_idx = list(val_set.indices)
    steps_label = ",".join(f"{d:02d}" for d in sorted(days))
    print(
        f"[Data] {subject} day-{steps_label} event={event}\n"
        f"       train {info['n_unique_train']} 句/{len(dataset) - len(val_set)} trial，"
        f"预测集 {info['n_unique_val']} 句/{len(val_set)} trial，句子重叠={info['sentence_overlap']}"
    )

    # ---------------- EEG 分支 ----------------
    model = CharTokenEEGEncoder(
        n_chans=n_chans, token_samples=token_samples, embedding_dim=embedding_dim
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    eeg = encode_indices(model, dataset, val_idx, device, args.batch_size)  # (N, D) 已 L2 归一化

    # ---------------- 文本分支（冻结 BGE-M3） ----------------
    print("[Text] 加载 BGE-M3（冻结） ...")
    text_encoder = BGEM3TextEncoder()
    lookup, text_dim = build_text_lookup(
        text_encoder, dataset.meta["sentence"].tolist(), device
    )
    if text_dim != embedding_dim:
        raise ValueError(f"文本 embedding 维度 {text_dim} 与 checkpoint 的 {embedding_dim} 不一致")

    sentences = [dataset.meta["sentence"].tolist()[i] for i in val_idx]
    text = gather_text(sentences, lookup, device)  # (N, D) 每 trial 取它那句的文本向量

    # ---------------- 跨模态相似度矩阵 ----------------
    sim = (eeg @ text.t()).float().cpu().numpy()  # 行=EEG trial，列=文本 trial
    n = sim.shape[0]
    n_sent = len(dict.fromkeys(sentences))
    print(
        f"[Enc] EEG={tuple(eeg.shape)} 文本={tuple(text.shape)} -> sim={sim.shape}"
        f"（{n} trial / {n_sent} 唯一句）"
    )

    # ---------------- 诊断 ----------------
    # 同一句的 4 列完全相同（同句同一个文本向量），所以"同句不同天"也是有效正样本
    sent_arr = np.array(sentences)
    same = sent_arr[:, None] == sent_arr[None, :]
    off_diag = ~np.eye(n, dtype=bool)
    aligned = float(np.diag(sim).mean())                      # 对角 = 配对（正确）
    within = float(sim[off_diag & same].mean())               # 同句不同天（文本相同）
    between = float(sim[off_diag & ~same].mean())             # 不同句
    print(
        f"\n[Sim] 配对(对角) mean={aligned:+.4f}  同句非对角 mean={within:+.4f}  "
        f"不同句 mean={between:+.4f}\n"
        f"      配对 - 不同句 = {aligned - between:+.4f}（>0 说明 EEG 更靠近它自己的句子）"
    )

    unique_sentences = sorted(set(sentences))
    sent_to_col = {s: i for i, s in enumerate(unique_sentences)}
    text_unique = gather_text(unique_sentences, lookup, device)
    targets = torch.tensor([sent_to_col[s] for s in sentences], device=device)
    top1, top5 = retrieval_eeg2text(eeg, text_unique, targets)
    print(
        f"[Retr] eeg2text top-1={top1:.4f}  top-5={top5:.4f}"
        f"（候选 {len(unique_sentences)} 句）\n"
        f"       随机基线 top-1={1 / len(unique_sentences):.4f} "
        f"top-5={min(5, len(unique_sentences)) / len(unique_sentences):.4f}"
    )

    # ---------------- 排序并截取 ----------------
    if args.order == "sentence":
        order = group_order(sentences)
        order_label, tag = "按句子分组", "crossmodal_bysentence"
    else:
        order = list(range(n))
        order_label, tag = "原始顺序", "crossmodal_raw"

    start = max(0, args.start)
    end = n if args.end is None else min(args.end, n)
    select = order[start:end]
    zoom_note = "" if (start, end) == (0, n) else f"（索引 {start}:{end}）"
    if (start, end) != (0, n):
        tag += f"_idx{start}-{end}"
    sub = sim[np.ix_(select, select)]

    # ---------------- 绘图 ----------------
    out_dir = Path(args.out_dir) if args.out_dir is not None else ckpt_path.parent
    title = (
        f"{subject} 预测集 EEG×文本 跨模态相似度矩阵（顺序={order_label}）\n"
        f"行=EEG 分支 900 trial，列=文本分支（同序 trial 的句子）  "
        f"days={steps_label}  val_ratio={val_ratio:g}  seed={seed}{zoom_note}"
    )
    path = plot_matrix(
        out_dir, sub, f"{subject}_val_{tag}.png", title,
        vmin=float(sim.min()), vmax=float(sim.max()),
    )
    print(f"\n[Plot] {len(select)}×{len(select)}（{order_label}{zoom_note}）")
    print(f"[Output] {path}")

    # 行/列顺序与 EEG 自相似图一致，映射 CSV 同样适用
    csv_path = save_order_csv(out_dir, subject, tag, dataset.meta, val_idx, select)
    print(f"[Output] {csv_path}")


if __name__ == "__main__":
    main()
