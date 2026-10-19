"""绘制 ``char_token_runbatch`` 的 **sentence × sentence** 文本自相似矩阵。

用冻结的 LaBSE 编码 day 5~8 的全部唯一句（252 句），两两余弦：

    格子 (i, j) = cos(sentence_i, sentence_j)

这是 EEG 侧对比学习的**目标空间**：EEG 学得再好，也只能逼近这个矩阵的结构。
若文本矩阵按类别成块，说明类别内句子的语义距离本来就小，EEG 在块内区分句子
天然更难。

不需要 checkpoint 的权重（LaBSE 全程冻结）；``--ckpt`` 只用来沿用训练时的
subject / day / event / sample 配置。

用法::

    uv run python char_token_runbatch/plot_text_matrix.py
    uv run python char_token_runbatch/plot_text_matrix.py --order category
    uv run python char_token_runbatch/plot_text_matrix.py --day 5 6 7 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_runbatch.plot_common import (
    add_common_args,
    build_dataset,
    days_tag,
    load_meta,
    load_text_embeddings,
    order_positions,
    plot_heatmap,
    resolve_config,
    unique_sentence_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 sentence × sentence 文本自相似矩阵。")
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config(args)
    dataset = build_dataset(cfg, touch=False)  # 只读 all_text.tsv，不碰 FIF
    meta = load_meta(dataset)

    unique = unique_sentence_rows(meta)
    positions = order_positions(meta, unique.index.to_numpy(), args.order)
    sentences = meta["sentence"].to_numpy()[positions]
    categories = meta["category_name"].fillna("?").to_numpy()[positions]

    lookup = load_text_embeddings(dataset)
    text = np.stack([lookup[s] for s in sentences])
    sim = text @ text.T

    n = len(sentences)
    off = ~np.eye(n, dtype=bool)
    same_cat = categories[:, None] == categories[None, :]

    def mean_of(mask):
        return float(sim[mask].mean()) if mask.any() else float("nan")

    print(f"[Data] {cfg['subject']} 唯一句={n} order={args.order}")
    print(
        f"[Sim] 同类 mean={mean_of(off & same_cat):+.4f}  "
        f"异类 mean={mean_of(off & ~same_cat):+.4f}  "
        f"差={mean_of(off & same_cat) - mean_of(off & ~same_cat):+.4f}"
    )

    # 对角线恒为 1，用非对角 99.5 分位作色标上限，避免整张图被压暗
    vmax = float(np.percentile(sim[off], 99.5)) if off.any() else 1.0
    title = (
        f"{cfg['subject']} LaBSE 文本自相似矩阵（唯一句，order={args.order}）\n"
        f"N={n}  非对角 99.5 分位={vmax:.3f}（对角线 1.0 会饱和）"
    )
    path = plot_heatmap(
        sim,
        cfg["out_dir"] / (
            f"{cfg['subject']}_text_text_day{days_tag(cfg['days'])}_{args.order}.png"
        ),
        title,
        xlabel="sentence",
        ylabel="sentence",
        cbar_label="余弦相似度（LaBSE）",
        vmin=float(sim[off].min()) if off.any() else 0.0,
        vmax=vmax,
    )
    print(f"[Output] {path}")


if __name__ == "__main__":
    main()
