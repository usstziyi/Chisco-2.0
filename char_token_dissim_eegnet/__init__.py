"""字符级 token 的 EEG 编码方案（独立目录）：EEGNet tokenizer 变体。

把不定长 EEG 段按固定字数窗口切成等长 token，变长性从"每个 token 的采样点数"
转移到"token 个数"，再用共享权重的 CNN tokenizer + Transformer 编码，
最后用 attention pooling 汇总成定长 embedding，与 LaBSE 文本空间对齐。

与 ``char_token_dissim`` 的唯一区别是 token encoder：那边是 NICE 风格的 TSConv
（时间卷积 + 平均池化 + 稠密跨电极空间卷积），这里是 EEGNet 风格的可分离卷积。
其余（数据切分、Transformer、注意力池化、对比损失、近重复句过滤）完全一致。

    dataset.py       CharTokenEEGDataset / char_token_collate
    model.py         EEGNetTokenEncoder / CharTokenEEGEncoder
    text_encoder.py  LaBSETextEncoder

本目录不依赖 ``train/``、``script/`` 下的既有代码。
"""

from char_token_dissim_eegnet.dataset import CharTokenEEGDataset, char_token_collate
from char_token_dissim_eegnet.model import CharTokenEEGEncoder, EEGNetTokenEncoder
from char_token_dissim_eegnet.text_encoder import LaBSETextEncoder

__all__ = [
    "CharTokenEEGDataset",
    "char_token_collate",
    "EEGNetTokenEncoder",
    "CharTokenEEGEncoder",
    "LaBSETextEncoder",
]