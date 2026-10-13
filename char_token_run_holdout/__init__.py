"""按 run 切分的跨 run 泛化实验（独立目录）。

与 ``char_token`` 共享受损、采样、评测、模型与数据集实现，唯一变量是 **train / val
的切分方式**：``char_token`` 的 ``split_by_sentence`` 随机按句子做 8:2 切分，本实验
改成 **按 run 切分**——把 sub-01 day-05 的 4 个 run 里的前 3 个整体划给 train，
最后一个 run 整体划给 val。

注意
----
day-05（= ses-01 para2）的 4 个 run 是**同一批 252 句的重复录制**：每个 run 都完整
录一遍这 252 句，且顺序一致。因此 train run（1~3）与 val run（4）的**句子集合完全
重合**，val 测的是"同一句子的新一次录制"能否被正确检索——即**跨 run / 跨录制次
的泛化**，而不是"新句子泛化"。这与 ``split_by_sentence`` 刻意保证的句子不相交切分
是两种不同的评测口径，读结果时务必区分。

    split.py      split_by_run（按 run 切分 train / val）
    train.py      训练入口

用法::

    uv run python char_token_run_holdout/train.py
    uv run python char_token_run_holdout/train.py --epochs 20 --seed 0
    uv run python char_token_run_holdout/train.py --train-runs 1 2 3 --val-runs 4
"""

from char_token_run_holdout.split import split_by_run

__all__ = ["split_by_run"]
