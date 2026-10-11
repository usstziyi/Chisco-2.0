"""EEG encoder 实现。

对外导出:
    TSConvFixedEEGEncoder       固定长度 EEG 的 TSConv encoder（当前训练使用）
    TSConvAdaptiveEEGEncoder    支持 batch 内变长时间的 TSConv encoder
    TSConvPointwiseEEGEncoder   固定长度 + 1×1 pointwise conv 的 TSConv encoder
"""

from .eeg_encoder_tsconv_fixed import TSConvFixedEEGEncoder
from .eeg_encoder_tsconv_pointwise import TSConvPointwiseEEGEncoder
from .TSConvAdaptiveEEGEncoder import TSConvAdaptiveEEGEncoder

__all__ = [
    "TSConvFixedEEGEncoder",
    "TSConvPointwiseEEGEncoder",
    "TSConvAdaptiveEEGEncoder",
]
