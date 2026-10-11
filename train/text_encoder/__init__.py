"""文本 encoder 实现。

对外导出:
    LaBSETextEncoder    LaBSE 句子编码器（冻结，输出已 L2 归一化，带文本缓存）
"""

from .text_encoder_labse import LaBSETextEncoder

__all__ = ["LaBSETextEncoder"]
