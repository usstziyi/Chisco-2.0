"""注意力池化 vs 平均池化消融实验（独立目录）。

与 ``char_token`` 共享数据、损失、采样、评测与 token 编码器，唯一变量是**整句的池化方式**：

    attn: 可学习 query + MultiheadAttention 对有效 token 加权汇总（主实验同款）
    mean: 对有效 token 做 masked mean

目的：检验主实验"attention pooling 优于 mean-pool（避免计数泄漏）"的论断。

    model.py   MeanPoolEEGEncoder / build_encoder
    train.py   消融训练入口（--pooling attn mean 可一次跑完两种变体）
"""

from char_token_pooling.model import MeanPoolEEGEncoder, POOLING_MODES, build_encoder

__all__ = ["MeanPoolEEGEncoder", "build_encoder", "POOLING_MODES"]