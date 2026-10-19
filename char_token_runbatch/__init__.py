"""按 ``(day, run)`` 分 batch 的字符级 EEG–文本对比学习实验。

与原先的字符级 token 实验的区别只在采样：

- 原实验：一个 batch 抽 ``n_sentences`` 个不同句子，每句 ``n_repeats`` 条 trial；
- 本实验：一个 ``(day, run)`` 的全部 trial 直接作为一个 batch。

day 5~8 的每个 run 都包含全部 252 句，run 不再携带句子身份。一个 run 一个 batch
时，batch 内 run 恒定，模型无法用 "这是哪个 run/主题块" 来区分候选，只能学句子级
信息。数据、模型、损失均在本目录内自带，不依赖外部实验代码。

    dataset.py       CharTokenEEGDataset / char_token_collate
    model.py         CharTokenEEGEncoder / CharTokenEncoder
    text_encoder.py  LaBSETextEncoder
    common.py        文本查表 / 句子分组 / InfoNCE 等工具
    sampler.py       RunBatchSampler
    train.py         训练入口（无验证集）
"""
