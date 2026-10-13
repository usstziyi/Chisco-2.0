"""按 day 切分的跨天泛化实验（独立目录）。

与 ``char_token`` 共享受损、采样、评测、模型与数据集实现，唯一变量是 **train / val
的切分方式**：``char_token`` 的 ``split_by_sentence`` 随机按句子做 8:2 切分，本实验
改成 **按 day 切分**——把 sub-01 的 day-05/06/07 整体划给 train，day-08 整体划给 val。

注意
----
day-05 ~ day-08（均为 para2）是**同一批 252 句的重复录制**：每天 4 个 run，每个 run
完整录一遍这 252 句。因此 train（day-05/06/07）与 val（day-08）的**句子集合完全
重合**，val 测的是"同一句子在**没见过的某一天**录到的信号"能否被正确检索——即
**跨天 / 跨 session 泛化**，而不是新句子泛化。这与 ``split_by_sentence`` 刻意保证的
句子不相交切分是两种不同的评测口径，读结果时务必区分。

    split.py      split_by_day（按 day 切分 train / val）
    train.py      训练入口

用法::

    uv run python char_token_day_holdout/train.py
    uv run python char_token_day_holdout/train.py --epochs 20 --seed 0
    uv run python char_token_day_holdout/train.py --train-days 5 6 7 --val-days 8
"""

from char_token_day_holdout.split import split_by_day

__all__ = ["split_by_day"]
