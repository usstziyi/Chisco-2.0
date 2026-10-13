"""跨被试预测实验（独立目录）。

用某个被试训练出的 best checkpoint（``{source}_best.pt``）去预测另一个被试
（``target``）的 EEG 数据，直接做零样本跨被试检索评测。模型与数据分属不同被试，
评测口径（损失、检索指标、随机基线）完全复用 ``char_token``，因此可与同被试的
val 指标直接对比。

    predict.py   跨被试预测入口（source -> target，可双向）
"""
