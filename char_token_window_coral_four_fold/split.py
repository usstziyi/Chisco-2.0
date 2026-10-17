"""按 day 切分 train / val（四折留一天）。"""

from __future__ import annotations

from torch.utils.data import Subset


def split_by_day(dataset, train_days, val_days):
    """按 day 编号切分，返回 ``(train_subset, val_subset, info)``。

    与基实验的 ``split_by_sentence``（随机按句子 8:2）不同：
    这里把整个 day 划到同一侧，用于 **跨天（跨 session）泛化** 的四折实验。

    Parameters
    ----------
    dataset:
        ``CharTokenEEGDataset``（或其子类），需带 ``meta["day"]`` 与 ``meta["sentence"]``。
    train_days / val_days:
        划给 train / val 的 day 编号（int 可迭代）。两者必须不相交。

    info 里额外报告 ``sentence_overlap``：day 1~4 是同一批句子的重复录制，
    按 day 切分时 train / val 会共享句子，这个数字需要显式暴露，
    避免误读成"新句子泛化"。
    """
    days = dataset.meta["day"].astype(int).tolist()
    train_days = sorted({int(d) for d in train_days})
    val_days = sorted({int(d) for d in val_days})

    seen = set(train_days) & set(val_days)
    if seen:
        raise ValueError(f"train / val 的 day 不能重叠：{sorted(seen)}")

    train_idx = [i for i, d in enumerate(days) if d in train_days]
    val_idx = [i for i, d in enumerate(days) if d in val_days]
    if not train_idx:
        raise ValueError(f"没有 trial 落在 train_days={train_days} 上")
    if not val_idx:
        raise ValueError(f"没有 trial 落在 val_days={val_days} 上")

    sentences = dataset.meta["sentence"].tolist()
    train_sentences = {sentences[i] for i in train_idx}
    val_sentences = {sentences[i] for i in val_idx}

    info = {
        "train_days": train_days,
        "val_days": val_days,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_unique_train": len(train_sentences),
        "n_unique_val": len(val_sentences),
        "sentence_overlap": len(train_sentences & val_sentences),
    }
    return Subset(dataset, train_idx), Subset(dataset, val_idx), info
