"""数据加载。

对外导出当前训练流程使用的接口:
    ChiscoEEGTextDataset        按 sample/subject/event/n_chars 过滤的 EEG–文本数据集
    build_nchars9_dataloaders   n_chars=9 的 train / val DataLoader
    SentenceRepeatEEGDataset    面向「句子重复」对比学习的数据集
    SentenceRepeatBatchSampler  同句多 trial 成组的 batch 采样器

注: ``dataloader.py`` 里另有一个更早版本的 ``ChiscoEEGTextDataset``（采样率参数名为
``sample_rate``），与 ``dataset_nchars9.py`` 同名，为避免冲突未在此重导出；
需要时用 ``from train.dataloader.dataloader import ChiscoEEGTextDataset``。
"""

from .dataset_nchars9 import ChiscoEEGTextDataset
from .nchars9_dataloader import build_nchars9_dataloaders
from .sampler import SentenceRepeatBatchSampler, SentenceRepeatEEGDataset

__all__ = [
    "ChiscoEEGTextDataset",
    "build_nchars9_dataloaders",
    "SentenceRepeatEEGDataset",
    "SentenceRepeatBatchSampler",
]
