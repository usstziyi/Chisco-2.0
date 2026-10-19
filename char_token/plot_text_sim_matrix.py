"""绘制训练集 / 验证集内部的 LaBSE 文本句嵌入自相似矩阵。

做什么
------
1. 按 ``subject / days / event / sample / char_duration`` 重建与训练一致的 trial 元信息；
2. 用与训练**完全相同**的句子级切分（``val_ratio`` / ``seed``）取出训练集与验证集；
3. 用冻结的 LaBSE 把所有出现过的句子编码成 L2 归一化向量 ``(D,)``；
4. 训练集、验证集分别按 trial 取对应文本向量，两两余弦得到 ``N×N`` 矩阵；
5. 两张图共用同一色标，便于直接对比训练集 / 验证集的文本结构。

为什么看这张图
--------------
- 文本侧没有 EEG 噪声，矩阵只反映"句子语义空间"本身的结构；
- 若矩阵按主题/类别成块，说明文本目标本身高度聚类，EEG 想在同块内区分句子
  需要更精细的信息；
- train / val 两张图可以看句子级切分是否改变了文本分布。

注意
----
- day 1 里每句只出现一次；用 ``--days 1 2 3 4`` 时同一句会出现多次，它们的
  文本向量完全相同，因此会形成 1.0 的对角小块。
- 文本编码与 checkpoint / epoch 无关（LaBSE 全程冻结），所以不需要 ``--ckpt``。
- ``--days`` 就是为多天实验预留的入口，默认只用 day 1。

绘图顺序
--------
- ``sentence``（默认）：按句子分组，同句 trial 相邻（多天时最直观）；
- ``raw``：沿用该 split 的原始物理顺序（day→run→trial_index）。

图上用白色细线标出 **run 边界**（按实际绘图顺序计算），便于把亮块直接对应到 run。

用法::

    uv run python char_token/plot_text_sim_matrix.py
    uv run python char_token/plot_text_sim_matrix.py --days 1 2 3 4
    uv run python char_token/plot_text_sim_matrix.py --order raw
    uv run python char_token/plot_text_sim_matrix.py --vmax 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# 支持 `python char_token/plot_text_sim_matrix.py` 直接执行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import CharTokenEEGDataset
from char_token.text_encoder import LaBSETextEncoder
from char_token.train import build_text_lookup, split_by_sentence

# 图上标题含中文，指定可用的中文字体，避免显示成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token" / "text_simmatrix"


def parse_args():
    parser = argparse.ArgumentParser(
        description="绘制训练集 / 验证集的 LaBSE 文本句嵌入自相似矩阵。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试（如 sub-01）")
    parser.add_argument(
        "--days", type=int, nargs="+", default=[1],
        help="day 编号，可传多个（如 --days 1 2 3 4）；默认 1",
    )
    parser.add_argument(
        "--event", default="inner", choices=["inner", "reading"],
        help="inner / reading；只影响 trial 筛选口径（默认 inner）",
    )
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument(
        "--char-duration", type=float, default=0.4, help="每个字 token 时长（秒，默认 0.4）"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集句子比例（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="切分种子（默认 0）")
    parser.add_argument(
        "--order", default="sentence", choices=["sentence", "raw"],
        help="绘图顺序：sentence 按句子分组（默认），raw 用 split 原始顺序",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="色标上限；默认取两张图非对角相似度的 99.5 分位数（对角线 1.0 会饱和）",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR), help="图输出目录；默认 outputs/char_token/text_simmatrix"
    )
    return parser.parse_args()


def group_order(sentences) -> list[int]:
    """按句子分组排序：同句 trial 相邻（组内保持原顺序），组间按首次出现顺序。"""
    groups: dict[str, list[int]] = {}
    for pos, sentence in enumerate[Any](sentences):
        groups.setdefault(sentence, []).append(pos)
    return [pos for members in groups.values() for pos in members]


def run_boundaries(ordered_runs) -> list[int]:
    """返回绘图顺序下 run 发生变化的位置（分隔线画在这些下标处）。"""
    return [i for i in range(1, len(ordered_runs)) if ordered_runs[i] != ordered_runs[i - 1]]


def text_similarity(lookup, sentences) -> np.ndarray:
    """按句子查表堆出 (N, D)，返回 N×N 余弦矩阵。"""
    embeddings = torch.stack([lookup[s] for s in sentences])
    return (embeddings @ embeddings.t()).float().cpu().numpy()


def similarity_stats(name: str, sim: np.ndarray, sentences) -> None:
    """打印同句 / 句间平均相似度；单天时同句对数为 0，同句均值记为 nan。"""
    n = sim.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    sent = np.asarray(sentences)
    same_sent = sent[:, None] == sent[None, :]

    pairs_same = off_diag & same_sent
    within = float(sim[pairs_same].mean()) if pairs_same.any() else float("nan")
    between = float(sim[off_diag & ~same_sent].mean())
    print(
        f"[Sim] {name}：同句 mean={within:+.4f}（{int(pairs_same.sum() // 2)} 对）  "
        f"句间 mean={between:+.4f}（{int((off_diag & ~same_sent).sum() // 2)} 对）  "
        f"差={within - between:+.4f}"
    )


def plot_matrix(out_dir, matrix, filename, title, vmin, vmax, boundaries=()) -> Path:
    """用 seaborn 画文本自相似热图；矩阵较大时隐藏刻度标签，避免糊成一片。"""
    n = matrix.shape[0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 句子越多画布越大，否则单元挤在一起看不清
    size = 8.0 if n <= 80 else min(20.0, 8.0 + (n - 80) * 0.05)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels="auto" if n <= 250 else False,
        yticklabels="auto" if n <= 250 else False,
        square=True,
        linewidths=0,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "余弦相似度（LaBSE 句嵌入）"},
    )

    ax.set_title(title)
    ax.set_xlabel("trial")
    ax.set_ylabel("trial")

    # run 边界：白色细线；seaborn 的 y 轴是反向的，axhline 仍按数据坐标画
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=0.8, alpha=0.85)
        ax.axvline(b, color="white", linewidth=0.8, alpha=0.85)

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main():
    args = parse_args()

    # 只读 all_text.tsv 的元信息，不加载任何 FIF
    dataset = CharTokenEEGDataset(
        event=args.event,
        sample=args.sample,
        subject=args.subject,
        days=args.days,
        char_duration=args.char_duration,
    )
    train_set, val_set, info = split_by_sentence(dataset, args.val_ratio, args.seed)

    all_sentences = dataset.meta["sentence"].tolist()
    all_runs = dataset.meta["run"].tolist()
    train_sentences = [all_sentences[i] for i in train_set.indices]
    val_sentences = [all_sentences[i] for i in val_set.indices]
    train_runs = [int(all_runs[i]) for i in train_set.indices]
    val_runs = [int(all_runs[i]) for i in val_set.indices]

    days_label = ",".join(f"{d:02d}" for d in sorted(args.days))
    print(
        f"[Data] {args.subject} day-{days_label} event={args.event}\n"
        f"       train {info['n_unique_train']} 句/{len(train_sentences)} trial，"
        f"预测集 {info['n_unique_val']} 句/{len(val_sentences)} trial，"
        f"句子重叠={info['sentence_overlap']}"
    )

    # LaBSE 冻结，所有唯一句子只编码一次
    print("[Text] 加载并编码 LaBSE ...")
    encoder = LaBSETextEncoder()
    lookup, embedding_dim = build_text_lookup(
        encoder, all_sentences, torch.device("cpu")
    )
    print(f"[Text] 唯一句子={len(lookup)} embedding_dim={embedding_dim}")

    # ---------------- 两张矩阵 ----------------
    splits = [
        ("train", train_sentences, train_runs),
        ("val", val_sentences, val_runs),
    ]
    sims = {}
    for name, sentences, _ in splits:
        sims[name] = text_similarity(lookup, sentences)
        similarity_stats(name, sims[name], sentences)

    # 两张图共用色标：vmin 取两图最小值，vmax 默认取非对角 99.5 分位
    vmin = float(min(sims["train"].min(), sims["val"].min()))
    if args.vmax is not None:
        vmax = float(args.vmax)
    else:
        off_diag = []
        for sim in sims.values():
            n = sim.shape[0]
            mask = ~np.eye(n, dtype=bool)
            off_diag.append(sim[mask])
        vmax = float(np.percentile(np.concatenate(off_diag), 99.5))
        vmax = min(max(vmax, 0.1), 1.0)

    order_label = "按句子分组" if args.order == "sentence" else "原始顺序"
    order_tag = "bysentence" if args.order == "sentence" else "raw"
    days_tag = f"days{'-'.join(str(d) for d in sorted(args.days))}"

    out_dir = Path(args.out_dir)
    print(f"\n[Plot] 共用色标 vmin={vmin:+.3f} vmax={vmax:.3f}（对角线 1.0 会饱和）")

    for name, sentences, runs in splits:
        sim = sims[name]
        order = group_order(sentences) if args.order == "sentence" else list(range(len(sentences)))
        sub = sim[np.ix_(order, order)]
        ordered_runs = [runs[p] for p in order]
        bounds = run_boundaries(ordered_runs)

        split_label = "训练集" if name == "train" else "验证集"
        title = (
            f"{args.subject} {split_label} LaBSE 文本自相似矩阵（顺序={order_label}·白线=run 边界）\n"
            f"N={len(order)}  唯一句子={len(set(sentences))}  days={days_label}  "
            f"val_ratio={args.val_ratio:g}  seed={args.seed}  vmax={vmax:.2f}"
        )
        path = plot_matrix(
            out_dir, sub,
            f"{args.subject}_text_simmatrix_{name}_{days_tag}_{order_tag}.png",
            title, vmin, vmax, boundaries=bounds,
        )
        print(f"[Run] {split_label} 边界(plot index)={bounds}")
        print(f"[Plot] {split_label} {len(order)}×{len(order)}")
        print(f"[Output] {path}")


if __name__ == "__main__":
    main()
