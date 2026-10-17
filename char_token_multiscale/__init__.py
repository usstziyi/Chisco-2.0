"""多尺度时间卷积的字符级 token EEG 编码方案（独立消融目录）。

与 ``char_token/`` 的唯一差别在 token encoder 的时间前端：

    单尺度：Conv(1→32, 1×9) + AvgPool(1×13, s=2)
    多尺度：Conv(1→11, 1×3) / (1→11, 1×7) / (1→10, 1×17) 并行 + AvgPool(1×5, s=3)

其余（Dataset、按句分组采样、InfoNCE、检索评测、LaBSE 文本侧）与主实验一致，
便于把指标差异归因到时间前端。

    dataset.py       CharTokenEEGDataset / char_token_collate
    model.py         CharTokenEncoder / CharTokenEEGEncoder
    text_encoder.py  LaBSETextEncoder

运行::

    uv run python char_token_multiscale/train.py --day 1 2 3 4
"""

from char_token_multiscale.dataset import CharTokenEEGDataset, char_token_collate
from char_token_multiscale.model import CharTokenEEGEncoder, CharTokenEncoder
from char_token_multiscale.text_encoder import LaBSETextEncoder

__all__ = [
    "CharTokenEEGDataset",
    "char_token_collate",
    "CharTokenEncoder",
    "CharTokenEEGEncoder",
    "LaBSETextEncoder",
]
