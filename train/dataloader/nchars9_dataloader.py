"""``dataset_nchars9`` 的配套 DataLoader：随机切出 val 与 train。

基于 ``train/dataloader/dataset_nchars9.py`` 的 ``ChiscoEEGTextDataset``：

- 用固定种子的随机置换切分索引：前 ``val_size`` 条做验证集，其余全部做训练集；
- 同一 ``ChiscoEEGTextDataset`` 实例被两个 ``Subset`` 共享，FIF 缓存也会复用；
- 返回 ``(train_loader, val_loader)``，样本为 ``(eeg, sentence)``。

用法::

    from train.dataloader.nchars9_dataloader import build_nchars9_dataloaders

    train_loader, val_loader = build_nchars9_dataloaders(batch_size=32, seed=0)
    for eeg, sentences in train_loader:   # eeg: (B, C, T)；sentences: list[str]
        ...
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from torch.utils.data import DataLoader, Subset

# 同时支持 `python -m train.dataloader.nchars9_dataloader` 与
# 直接 `python train/dataloader/nchars9_dataloader.py`（后者 sys.path 不含项目根目录）
try:
    from train.dataloader.dataset_nchars9 import ChiscoEEGTextDataset
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from train.dataloader.dataset_nchars9 import ChiscoEEGTextDataset


def build_nchars9_dataloaders(
    n_chars: int = 9,
    event: str = "inner",
    sample: int = 250,
    subject=None,
    val_size: int = 100,
    batch_size: int = 32,
    seed: int = 0,
    num_workers: int = 0,
    scale: float = 1e6,
):
    """构建 ``n_chars=9`` 的 train / val DataLoader。

    Parameters
    ----------
    n_chars / event / sample / subject:
        透传给 ``ChiscoEEGTextDataset`` 的四个筛选维度。
    val_size:
        验证集样本数，默认 100；其余全部作为训练集。
    batch_size:
        train / val 共用的 batch 大小。
    seed:
        随机切分种子，保证每次切分结果一致。
    num_workers:
        透传给 ``DataLoader`` 的 worker 数，默认 0。
    scale:
        EEG 幅度缩放系数，透传给 ``ChiscoEEGTextDataset``。

    Returns
    -------
    (train_loader, val_loader):
        train 已 shuffle；val 不打乱。二者共享同一个底层 dataset 与 FIF 缓存。
    """
    dataset = ChiscoEEGTextDataset(
        n_chars=n_chars,
        event=event,
        sample=sample,
        subject=subject,
        scale=scale,
    )

    total = len(dataset)
    if total <= val_size:
        raise ValueError(
            f"总样本 {total} 不足以切出 val_size={val_size} 条验证集"
        )

    # 固定种子随机置换索引，再切出 val / train
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_nchars9_dataloaders(batch_size=32, event="inner", seed=0)
    ds = train_loader.dataset.dataset  # Subset -> ChiscoEEGTextDataset

    print(
        f"event={ds.event} sample={ds.sample} n_chars={ds.n_chars} "
        f"subjects={len(ds.subjects)}\n"
        f"总样本 {len(ds)} -> train {len(train_loader.dataset)} "
        f"({len(train_loader)} batch) / val {len(val_loader.dataset)} "
        f"({len(val_loader)} batch)"
    )

    eeg, sentences = next(iter(train_loader))
    print(
        f"train batch: eeg={tuple(eeg.shape)} dtype={eeg.dtype} "
        f"sentences={len(sentences)} 条，示例 {sentences[0]!r}"
    )
    eeg, sentences = next(iter(val_loader))
    print(f"val   batch: eeg={tuple(eeg.shape)} sentences={len(sentences)} 条")
