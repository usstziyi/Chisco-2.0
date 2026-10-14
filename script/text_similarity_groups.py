"""把某个被试某一天的文本用 LaBSE 编码，再按"高相似度"把句子聚成组。

流程：

    1. 从 ``datasets/preprocessed/all_text.tsv`` 取出 --subject / --day / --sample
       命中的 trial，并按首次出现顺序去重得到句子清单；
       （每一天录的试次完全相同，所以取一天即可，day 是单个整数）
    2. 用冻结的 LaBSE（来自 ``char_token.text_encoder``）编码成单位向量 (N, D)；
    3. 两两余弦相似度 >= --threshold 的句子并进同一组（单链接 / 并查集）；
    4. 统计一共分成多少组、各组大小，打印非单例组明细，并存一份 CSV；
    5. 画两张相似度矩阵图："去除前"（全部句子）和"去除后"（仅保留单例句，
       即不含在任何多句组里的句子），两图共用同一色标便于对比。

单链接的含义：只要 A~B、B~C 各自超过阈值，A 和 C 就落进同一组，
即使 A~C 本身没到阈值。阈值取高（默认 0.95，接近"同义/近重复"）时几乎不会链式串组。

用法::

    uv run python script/text_similarity_groups.py --subject sub-01 --day 1 --threshold 0.9
    uv run python script/text_similarity_groups.py --start 0 --end 100            # 只画前 100 句
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# 图上标题含中文，指定可用的中文字体，避免显示成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 支持 `python script/text_similarity_groups.py` 直接执行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_TABLE = PROJECT_ROOT / "datasets" / "preprocessed" / "all_text.tsv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "text_similarity_groups"


class _UnionFind:
    """并查集：把超过阈值的句子对合并成连通分量。"""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            # 路径压缩
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_sentences(text_table: Path, subject, day: int, sample: int):
    """按 subject / day / sample 过滤 all_text.tsv，去重后返回句子清单。"""
    table = pd.read_csv(text_table, sep="\t")
    mask = table["sample"] == sample
    if subject is not None:
        mask &= table["subject"].isin(subject)
    mask &= table["day"] == day
    subset = table[mask]
    if subset.empty:
        raise ValueError(
            f"all_text.tsv 中没有 subject={subject}、day={day}、sample={sample} 的记录"
        )

    # 去重但保留首次出现顺序（dict.fromkeys 保序）
    sentences = list(dict.fromkeys(subset["sentence"].astype(str).tolist()))
    return subset, sentences


@torch.no_grad()
def encode_sentences(encoder, sentences, chunk: int = 64) -> torch.Tensor:
    """分批编码，返回 L2 归一化的 (N, D) 张量。"""
    parts = [encoder(sentences[start : start + chunk]) for start in range(0, len(sentences), chunk)]
    return torch.cat(parts, dim=0)


def group_by_threshold(sim: np.ndarray, threshold: float):
    """单链接聚类：cos >= threshold 的句子对连通成一个组。

    返回 ``(labels, groups)``：
        labels: (N,) 每个句子所属组号（按首次出现顺序编号）；
        groups: list[list[int]]，每组里的句子下标。
    """
    n = sim.shape[0]
    uf = _UnionFind(n)
    ii, jj = np.triu_indices(n, k=1)
    keep = sim[ii, jj] >= threshold
    for i, j in zip(ii[keep], jj[keep]):
        uf.union(int(i), int(j))

    root_to_label: dict[int, int] = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        root = uf.find(i)
        labels[i] = root_to_label.setdefault(root, len(root_to_label))

    groups = [[] for _ in range(len(root_to_label))]
    for i, label in enumerate(labels):
        groups[label].append(i)
    return labels, groups


def top_similar_pairs(sim: np.ndarray, k: int):
    """返回相似度最高的 k 对（排除对角线），用于人工核对阈值。"""
    work = sim.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(k, work.size)
    flat = np.argpartition(work.ravel(), -k)[-k:]
    flat = flat[np.argsort(work.ravel()[flat])[::-1]]
    n = sim.shape[0]
    return [(int(idx // n), int(idx % n), float(work.ravel()[idx])) for idx in flat]


def print_report(sentences, sim, labels, groups, threshold, top_k):
    n = len(sentences)
    sizes = sorted((len(g) for g in groups), reverse=True)
    n_multi = sum(1 for s in sizes if s >= 2)
    n_merged = sum(s for s in sizes if s >= 2) - n_multi  # 被并入他组的句子数

    off = sim[~np.eye(n, dtype=bool)]
    print(f"\n句子数（去重后）= {n}")
    print(f"阈值 = {threshold}")
    print(
        f"非对角相似度: max={off.max():.4f}  mean={off.mean():.4f}  "
        f">= 阈值 的对数 = {int((off >= threshold).sum() // 2)}"
    )
    print(f"\n分成 {len(groups)} 组（单例 {len(groups) - n_multi} 组，"
          f"包含多句的组 {n_multi} 个，被合并的句子 {n_merged} 句）")
    print(f"最大组大小 = {sizes[0] if sizes else 0}")

    # 组大小分布
    from collections import Counter
    dist = Counter(sizes)
    print("组大小分布:", ", ".join(f"{size}句×{cnt}" for size, cnt in sorted(dist.items())))

    # 最相似的若干对
    print(f"\n最相似的 {top_k} 对（供核对阈值）：")
    for i, j, s in top_similar_pairs(sim, top_k):
        print(f"  {s:.4f}  {sentences[i]}  |  {sentences[j]}")

    # 非单例组明细
    multi = [(g, [sentences[i] for i in g]) for g in groups if len(g) >= 2]
    if multi:
        print(f"\n包含多句的组（{len(multi)} 组）：")
        for gi, (group, items) in enumerate(multi, start=1):
            print(f"  [组 {gi}] {len(group)} 句")
            for text in items:
                print(f"      - {text}")
    else:
        print("\n没有相似度超过阈值的句子对（全是单例组）。")


def save_sentences(out_dir, subset, sentences, day):
    """把去重后的句子按原始顺序写成 CSV（附序号，便于对应相似度矩阵的行/列）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = "_".join(sorted(subset["subject"].unique()))
    path = out_dir / f"{subjects}_day-{day:02d}_sentences.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "sentence"])
        for i, text in enumerate(sentences):
            writer.writerow([i, text])
    return path


def save_groups(out_dir: Path, subset, sentences, labels, groups, threshold, day):
    """把每组的句子写成一个 CSV（一行一句，按组大小降序）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = "_".join(sorted(subset["subject"].unique()))
    path = out_dir / f"{subjects}_day-{day:02d}_thr{threshold:g}_groups.csv"

    order = sorted(range(len(groups)), key=lambda gid: (-len(groups[gid]), gid))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "group_size", "sentence"])
        for rank, gid in enumerate(order, start=1):
            members = groups[gid]
            for i in members:
                writer.writerow([rank, len(members), sentences[i]])
    return path


def plot_similarity_matrix(
    out_dir,
    subset,
    sentences,
    sim,
    threshold,
    day,
    select,
    label: str,
    tag: str = "",
    stats: str = "",
):
    """把选中句子的两两余弦相似度矩阵画成热图并存 PNG。

    select：要画的句子下标（按给定顺序，通常是全部句子或仅保留的单例句）；
    label：标题里的视图名（如"去除前"/"去除后"）；tag：文件名后缀，区分不同视图；
    stats：标题第二行的补充统计。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    select = list(select)
    if not select:
        print("[Skip] 没有可绘制的句子，跳过该相似度矩阵图")
        return None

    ordered = sim[np.ix_(select, select)]

    n = len(select)
    # 句子越多画布越大，否则单元挤在一起看不清
    size = 8.0 if n <= 80 else min(20.0, 8.0 + (n - 80) * 0.05)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))

    # 用全量 sim 的最小值作 vmin，保证"去除前/后"两图色标一致，便于对比
    im = ax.imshow(ordered, cmap="viridis", vmin=float(sim.min()), vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="余弦相似度")

    subjects = "_".join(sorted(subset["subject"].unique()))
    ax.set_title(
        f"{subjects} day-{day:02d} 句子相似度矩阵（{label}）\n"
        f"N={n}  阈值={threshold:g}{stats}"
    )
    ax.set_xlabel("句子")
    ax.set_ylabel("句子")

    fig.tight_layout()
    path = out_dir / f"{subjects}_day-{day:02d}_thr{threshold:g}_simmatrix{tag}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="LaBSE 编码 + 高相似度句子分组统计。")
    parser.add_argument("--subject", nargs="+", default=["sub-01"], help="被试，默认 sub-01")
    parser.add_argument("--day", type=int, default=1, help="day 编号（1~8），默认 1")
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument("--threshold", type=float, default=0.95, help="余弦相似度分组阈值（默认 0.95）")
    parser.add_argument("--top-k", type=int, default=10, help="打印最相似的多少对（默认 10）")
    parser.add_argument("--start", type=int, default=0, help="绘图起始句子下标（默认 0）")
    parser.add_argument("--end", type=int, default=None, help="绘图结束句子下标，不含（默认到末尾）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="分组 CSV 输出目录")
    return parser.parse_args()


def main():
    args = parse_args()

    subset, sentences = load_sentences(TEXT_TABLE, args.subject, args.day, args.sample)
    print(
        f"[Data] {sorted(subset['subject'].unique())} day-{args.day:02d} "
        f"sample={args.sample}：trial {len(subset)} 条 -> 去重句子 {len(sentences)} 句"
    )

    sent_path = save_sentences(Path(args.out_dir), subset, sentences, args.day)
    print(f"[Saved] {sent_path}")

    print("[Text] 加载 LaBSE（冻结） ...")
    encoder = LaBSETextEncoder()
    embeddings = encode_sentences(encoder, sentences)
    sim = (embeddings @ embeddings.t()).cpu().numpy()

    labels, groups = group_by_threshold(sim, args.threshold)
    print_report(sentences, sim, labels, groups, args.threshold, args.top_k)

    path = save_groups(
        Path(args.out_dir), subset, sentences, labels, groups, args.threshold, args.day
    )
    print(f"\n[Saved] {path}")

    n_total = len(sentences)
    n_multi = sum(1 for g in groups if len(g) >= 2)
    # "去除后" = 只保留不含在任何多句组里的句子（单例句），保序遍历
    kept = sorted(i for g in groups if len(g) == 1 for i in g)

    # 去除前：全部句子（--start/--end 可截取局部放大）
    start = max(0, args.start)
    end = n_total if args.end is None else min(args.end, n_total)
    zoom_note = "" if (start, end) == (0, n_total) else f"（索引 {start}:{end}）"
    print(f"\n[Plot] 去除前：{end - start} 句；去除后：{len(kept)} 句（剔除 {n_total - len(kept)} 句）")

    before_path = plot_similarity_matrix(
        Path(args.out_dir),
        subset,
        sentences,
        sim,
        args.threshold,
        args.day,
        list(range(start, end)),
        label=f"去除前{zoom_note}",
        tag="_before" + ("" if (start, end) == (0, n_total) else f"_idx{start}-{end}"),
        stats=f"  组数={len(groups)}  多句组={n_multi}",
    )
    if before_path is not None:
        print(f"[Saved] {before_path}")

    # 去除后：仅保留单例句（与原图共用色标，便于对比）
    after_path = plot_similarity_matrix(
        Path(args.out_dir),
        subset,
        sentences,
        sim,
        args.threshold,
        args.day,
        kept,
        label="去除后·仅保留单例句",
        tag="_after",
        stats=f"  剔除 {n_total - len(kept)} 句",
    )
    if after_path is not None:
        print(f"[Saved] {after_path}")


if __name__ == "__main__":
    main()
