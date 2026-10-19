"""按类别分层的验证句切分。

day 5~8 的 252 句覆盖 39 个类别，其中 37 类各只有 2 句，两个大类
（社交和聚会活动 98 句、问候和情感状态 80 句）占其余 178 句。随机留句会让
多数小类在验证集中缺席，所以这里：

1. 每个类别保底留 1 句；
2. 剩余名额按各类「在保留至少 1 句训练样本后还能留多少」的比例，用最大
   余数法分配。

默认比例 0.25 时得到 63 句：37 个小类各 1 句，社交 14 句，问候 12 句。

用法::

    from char_token_runbatch.split import build_val_split, save_split, load_split

    split = build_val_split(meta, fraction=0.25, seed=0)
    save_split(path, split)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_val_split(meta: pd.DataFrame, fraction: float = 0.25, seed: int = 0) -> dict:
    """从 ``meta`` 的唯一句里分层抽取验证句。

    Parameters
    ----------
    meta:
        至少包含 ``sentence`` 与 ``category_name`` 两列，允许同一句重复多行。
    fraction:
        验证句占比，``0`` 表示不切分。
    seed:
        类内抽句的随机种子；相同数据与种子得到同一份清单。

    Returns
    -------
    dict:
        ``fraction`` / ``seed`` / ``n_total`` / ``n_val`` / ``val_sentences`` /
        ``per_category``。
    """
    if not 0 <= fraction < 1:
        raise ValueError(f"fraction 必须在 [0, 1) 内，收到 {fraction}")

    unique = meta.drop_duplicates("sentence")[["sentence", "category_name"]].copy()
    unique["category_name"] = unique["category_name"].fillna("?")
    grouped = {
        str(category): sorted(group["sentence"].tolist())
        for category, group in unique.groupby("category_name", sort=True)
    }

    total = len(unique)
    target_n = int(round(total * fraction))
    if target_n == 0:
        return {
            "fraction": float(fraction),
            "seed": int(seed),
            "n_total": total,
            "n_val": 0,
            "val_sentences": [],
            "per_category": {},
        }

    categories = list(grouped)
    counts = {category: len(grouped[category]) for category in categories}

    # 1. 每个类别保底留 1 句（目标少于类别数时按类别大小择优）
    if target_n >= len(categories):
        quota = {category: 1 for category in categories}
        remaining = target_n - len(categories)
    else:
        order = sorted(categories, key=lambda category: (-counts[category], category))
        quota = {category: int(category in order[:target_n]) for category in categories}
        remaining = 0

    # 2. 剩余名额按「留完后至少还剩 1 句训练」的容量比例分配
    capacity = {
        category: max(counts[category] - quota[category] - 1, 0)
        for category in categories
    }
    if remaining > 0:
        total_capacity = sum(capacity.values())
        if total_capacity < remaining:
            raise ValueError(
                f"fraction={fraction} 需要留出 {target_n}/{total} 句，"
                f"但按『每类至少保留 1 句训练』只能再分配 {total_capacity} 句"
            )

        raw = {
            category: remaining * capacity[category] / total_capacity
            for category in categories
        }
        extra = {category: int(np.floor(raw[category])) for category in categories}
        left = remaining - sum(extra.values())
        # 最大余数法；余数相同按类别名排序，保证可复现
        order = sorted(
            (category for category in categories if capacity[category] > 0),
            key=lambda category: (-(raw[category] - extra[category]), category),
        )
        for category in order[:left]:
            extra[category] += 1
        for category in categories:
            quota[category] += extra[category]

    # 3. 每个类别内随机抽 quota 句
    rng = np.random.default_rng(seed)
    val_sentences = []
    for category in categories:
        sentences = grouped[category]
        n = quota[category]
        if n == 0:
            continue
        picked = rng.permutation(len(sentences))[:n]
        val_sentences.extend(sentences[i] for i in picked)

    val_sentences = sorted(val_sentences)
    return {
        "fraction": float(fraction),
        "seed": int(seed),
        "n_total": total,
        "n_val": len(val_sentences),
        "val_sentences": val_sentences,
        "per_category": {category: int(quota[category]) for category in categories},
    }


def save_split(path: Path, split: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_split(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"验证集清单不存在: {path}")
    split = json.loads(path.read_text(encoding="utf-8"))
    if "val_sentences" not in split:
        raise ValueError(f"{path} 缺少 val_sentences 字段")
    return split
