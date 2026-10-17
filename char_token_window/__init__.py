"""滑窗 + 多尺度时间卷积的 EEG–文本编码方案（独立实验目录）。

与 ``char_token/`` 的差别有两处：

    切分：1.6~6.0 s 的整句 EEG → 1 s 窗、0.5 s 步长的等长 token（250 点）
    前端：5/15/35/75 四支并行时间卷积，覆盖 β/α/θ/δ

其余（按句分组采样、InfoNCE、检索评测、LaBSE 文本侧）与主实验一致。
窗口比单个字长 2.5 倍，低频第一次可以在单个 token 内被分开。

    dataset.py       CharTokenEEGDataset / char_token_collate
    model.py         CharTokenEncoder / CharTokenEEGEncoder
    text_encoder.py  LaBSETextEncoder

运行::

    uv run python char_token_window/train.py --day 1 2 3 4
"""

from char_token_window.dataset import CharTokenEEGDataset, char_token_collate
from char_token_window.model import CharTokenEEGEncoder, CharTokenEncoder
from char_token_window.text_encoder import LaBSETextEncoder

__all__ = [
    "CharTokenEEGDataset",
    "char_token_collate",
    "CharTokenEncoder",
    "CharTokenEEGEncoder",
    "LaBSETextEncoder",
]
