"""滑窗 + 多尺度时间卷积 + 协方差对齐（CORAL）的 EEG–文本编码方案。

本目录 = ``char_token_window`` 的完整拷贝，**只新增协方差对齐损失**，其余
（数据切分、模型、按句分组采样、InfoNCE、检索评测、LaBSE 文本侧）逐行保持一致，
以便与 ``char_token_window`` 做单变量对照。

    切分：1.6~6.0 s 的整句 EEG → 1 s 窗、0.5 s 步长的等长 token（250 点）
    前端：5/15/35/75 四支并行时间卷积，覆盖 β/α/θ/δ
    损失：多正样本对称 InfoNCE + coral_weight × Deep CORAL

    dataset.py       CharTokenEEGDataset / char_token_collate
    model.py         CharTokenEncoder / CharTokenEEGEncoder
    text_encoder.py  LaBSETextEncoder

运行::

    uv run python char_token_window_coral/train.py --day 1 2 3 4
"""

from char_token_window_coral.dataset import CharTokenEEGDataset, char_token_collate
from char_token_window_coral.model import CharTokenEEGEncoder, CharTokenEncoder
from char_token_window_coral.text_encoder import LaBSETextEncoder

__all__ = [
    "CharTokenEEGDataset",
    "char_token_collate",
    "CharTokenEncoder",
    "CharTokenEEGEncoder",
    "LaBSETextEncoder",
]
