"""随机噪声对照实验（独立目录）。

与 ``char_token`` 共享受损、采样、评测与模型实现，唯一变量是 **EEG 数值来源**：
把 ``(n_tokens, C, token_samples)`` 的真实脑电替换成独立同分布的随机噪声，
形状、句子标签、数据划分全部保持不变。

    dataset.py   NoiseTokenEEGDataset（噪声版数据集）
    train.py     噪声对照训练入口

判读：噪声组 val 指标若回落到随机基线附近，说明 ``char_token`` 的成绩来自真实脑电；
若仍显著更高，则说明模型利用了与信号无关的旁路信息（如 token 个数=句子长度）。
"""

from char_token_noise.dataset import NoiseTokenEEGDataset

__all__ = ["NoiseTokenEEGDataset"]