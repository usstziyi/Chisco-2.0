"""按 run 切分 train / val。

与 ``char_token.train.split_by_sentence``（随机按句子 8:2 切分）不同：这里把整个 run
划到同一侧，用于**跨 run 泛化**实验——例如 sub-01 day-05 的前 3 个 run 做 train、
最后一个 run 做 val。

返回 ``torch.utils.data.Subset``，与 ``split_by_sentence`` 保持同构，便于直接替换。
"""

from __future__ import annotations

from torch.utils.data import Subset


def split_by_run(dataset, train_runs, val_runs):
    """按 run 编号切分，返回 ``(train_subset, val_subset, info)``。

    Parameters
    ----------
    dataset:
        ``CharTokenEEGDataset``（或其子类），需带 ``meta["run"]`` 与 ``meta["sentence"]``。
    train_runs / val_runs:
        划给 train / val 的 run 编号（int 可迭代）。两者必须不相交。

    info 里额外报告 ``sentence_overlap``：按 run 切分时 train / val 可能共享句子
    （day-05 就是同一批句子的重复录制），这个数字需要显式暴露，避免误读成"新句子泛化"。
    """
    runs = dataset.meta["run"].astype(int).tolist()
    train_runs = sorted({int(r) for r in train_runs})
    val_runs = sorted({int(r) for r in val_runs})

    seen = set(train_runs) & set(val_runs)
    if seen:
        raise ValueError(f"train / val 的 run 不能重叠：{sorted(seen)}")

    train_idx = [i for i, r in enumerate(runs) if r in train_runs]
    val_idx = [i for i, r in enumerate(runs) if r in val_runs]
    if not train_idx:
        raise ValueError(f"没有 trial 落在 train_runs={train_runs} 上")
    if not val_idx:
        raise ValueError(f"没有 trial 落在 val_runs={val_runs} 上")

    sentences = dataset.meta["sentence"].tolist()
    train_sentences = {sentences[i] for i in train_idx}
    val_sentences = {sentences[i] for i in val_idx}

    info = {
        "train_runs": train_runs,
        "val_runs": val_runs,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_unique_train": len(train_sentences),
        "n_unique_val": len(val_sentences),
        "sentence_overlap": len(train_sentences & val_sentences),
    }
    return Subset(dataset, train_idx), Subset(dataset, val_idx), info
