"""字符级 token 的 EEG 编码方案（BGE-M3 文本侧，独立目录）。

把不定长 EEG 段按固定字数窗口切成等长 token，变长性从"每个 token 的采样点数"
转移到"token 个数"，再用共享权重的 CNN tokenizer + Transformer 编码，
最后用 attention pooling 汇总成定长 embedding，与 BGE-M3（1024 维）文本空间对齐。

    dataset.py       CharTokenEEGDataset / char_token_collate
    model.py         CharTokenEncoder / CharTokenEEGEncoder
    text_encoder.py  BGEM3TextEncoder

本目录不依赖 ``char_token/`` 下的代码，两套实验可以独立运行。
"""

from char_token_bge_m3_runinv.dataset import CharTokenEEGDataset, char_token_collate
from char_token_bge_m3_runinv.model import CharTokenEEGEncoder, CharTokenEncoder
from char_token_bge_m3_runinv.text_encoder import BGEM3TextEncoder

__all__ = [
    "CharTokenEEGDataset",
    "char_token_collate",
    "CharTokenEncoder",
    "CharTokenEEGEncoder",
    "BGEM3TextEncoder",
]
