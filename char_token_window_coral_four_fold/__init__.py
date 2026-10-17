"""滑窗多尺度 token + CORAL 的四折跨天泛化实验（自包含目录）。

本目录是独立实验：数据切分、模型、按句分组采样、文本编码、InfoNCE + CORAL
损失、评测与 CSV 日志全部在本目录内实现，不引用其他实验目录。

在指定被试的 day 1~4 上做 leave-one-day-out 四折：

    fold 1: train day 2 3 4 -> val day 1
    fold 2: train day 1 3 4 -> val day 2
    fold 3: train day 1 2 4 -> val day 3
    fold 4: train day 1 2 3 -> val day 4

注意
----
day 1~4（para1）是**同一批句子的重复录制**，每天 4 个 run、每个 run 完整录一遍。
因此 train（3 天）与 val（1 天）的**句子集合完全重合**，val 指标衡量的是
"同一句子在未见过的某一天录到的信号"能否被正确检索，即 **跨天 / 跨 session 泛化**，
而不是新句子泛化；不要与按句子切分的结果直接比较。

    split.py       split_by_day（按 day 切分 train / val）
    dataset.py     CharTokenEEGDataset / char_token_collate
    model.py       CharTokenEncoder / CharTokenEEGEncoder
    sampler.py     SentenceGroupBatchSampler
    text_encoder.py LaBSETextEncoder
    train.py       单折训练入口（--train-days / --val-days）
    four_fold.py   leave-one-day-out 四折 + 自动汇总 mean±std

用法::

    uv run python char_token_window_coral_four_fold/train.py --train-days 1 2 3 --val-days 4
    uv run python char_token_window_coral_four_fold/four_fold.py
    uv run python char_token_window_coral_four_fold/four_fold.py --subject sub-01 --days 1 2 3 4
"""

from char_token_window_coral_four_fold.dataset import CharTokenEEGDataset, char_token_collate
from char_token_window_coral_four_fold.model import CharTokenEEGEncoder, CharTokenEncoder
from char_token_window_coral_four_fold.split import split_by_day
from char_token_window_coral_four_fold.text_encoder import LaBSETextEncoder

__all__ = [
    "CharTokenEEGDataset",
    "char_token_collate",
    "CharTokenEncoder",
    "CharTokenEEGEncoder",
    "LaBSETextEncoder",
    "split_by_day",
]
