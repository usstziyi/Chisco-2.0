"""按余弦相似度阈值过滤句子：只保留"单例句"。

给一批句子用冻结的 LaBSE 编码，两两算余弦相似度；把相似度 >= threshold 的句子
用单链接（并查集）并成组；凡落在多句组里的句子都视为"近重复"而整句剔除，
只保留与其它任何句都不相似的句子。

多天 / 重复 trial 场景的用法：调用方传入的是**唯一句子文本**，因此同一句在所有
day 会被一致地保留或剔除，不会拆散同句跨天的多视图正样本。

用法::

    from char_token_dissim.text_encoder import LaBSETextEncoder
    from char_token_dissim.sentence_filter import keep_singleton_sentences

    keep, stats = keep_singleton_sentences(sentences, LaBSETextEncoder(), threshold=0.9)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# 支持 `python char_token_dissim/sentence_filter.py` 直接执行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def group_by_threshold(sim: np.ndarray, threshold: float):
    """单链接聚类：cos >= threshold 的句子对连通成一个组。

    返回 ``(labels, groups)``：

    - labels: (N,) 每个句子所属组号（按首次出现顺序编号）；
    - groups: list[list[int]]，每组里的句子下标。
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


@torch.no_grad()
def keep_singleton_sentences(sentences, encoder, threshold: float, chunk: int = 64):
    """返回 ``(keep, stats)``。

    - keep: 保序的"单例句"列表——去重后仍没和任何其它句相似度 >= threshold 的句子；
    - stats: dict，含 ``n_unique`` / ``n_kept`` / ``n_dropped`` / ``n_multi_groups``。

    ``sentences`` 应传入**唯一句子文本**（重复文本会去重，保序）。
    """
    unique = list(dict.fromkeys(str(s) for s in sentences))
    if not unique:
        raise ValueError("sentences 为空，无法过滤")

    embeddings = torch.cat(
        [encoder(unique[start : start + chunk]) for start in range(0, len(unique), chunk)],
        dim=0,
    )
    sim = (embeddings @ embeddings.t()).cpu().numpy()
    _, groups = group_by_threshold(sim, threshold)

    # 只保留单例组里的句子：任何多句组的成员都整句剔除
    keep_set = {unique[g[0]] for g in groups if len(g) == 1}
    keep = [s for s in unique if s in keep_set]

    stats = {
        "n_unique": len(unique),
        "n_kept": len(keep),
        "n_dropped": len(unique) - len(keep),
        "n_multi_groups": sum(1 for g in groups if len(g) >= 2),
    }
    return keep, stats
