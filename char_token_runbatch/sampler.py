"""按 ``(day, run)`` 分 batch 的采样器。

每个 batch = 一个 run 的全部 trial。day 5~8 里每个 run 都包含同样的 252 句，
所以 batch 内 run 恒定、句子互不相同；run 不再能作为区分句子的线索。

``shuffle_within`` 只影响 batch 内 trial 的顺序。InfoNCE 对样本顺序是等变的，
打乱不会改变 loss；真正的随机性来自每个 epoch 重新打乱 run 的顺序。
"""

from __future__ import annotations

import random

from torch.utils.data import Sampler


class RunBatchSampler(Sampler[list[int]]):
    """让一个 ``(day, run)`` 的所有样本构成一个 batch。

    Parameters
    ----------
    groups:
        ``{(day, run): [dataset_index, ...]}``。每个 key 是一个 run。
    seed:
        随机种子；给定后按 epoch 递增，每个 epoch 的 run 顺序可复现。
    shuffle:
        是否在每个 epoch 打乱 run 的顺序，默认 ``True``。
    shuffle_within:
        是否打乱 batch 内 trial 的顺序，默认 ``True``。
        InfoNCE 对顺序等变，这个选项不改变 loss，只影响日志/可视化顺序。
    max_per_run:
        每个 run 每个 epoch 最多取多少条；``0`` 表示全部（默认）。
        仅用于 CPU 冒烟或显存不足时缩小 batch，不改变实验逻辑。
    """

    def __init__(
        self,
        groups: dict[tuple[int, int], list[int]],
        seed: int | None = None,
        shuffle: bool = True,
        shuffle_within: bool = True,
        max_per_run: int = 0,
    ):
        if not groups:
            raise ValueError("groups 不能为空")
        if max_per_run < 0:
            raise ValueError(f"max_per_run 必须 >= 0，收到 {max_per_run}")

        self.groups = {tuple(key): list(indices) for key, indices in groups.items()}
        self.run_keys = sorted(self.groups)
        self.group_sizes = {key: len(indices) for key, indices in self.groups.items()}
        self.seed = seed
        self.shuffle = bool(shuffle)
        self.shuffle_within = bool(shuffle_within)
        self.max_per_run = int(max_per_run)
        self.epoch = 0

    def __len__(self) -> int:
        """一个 epoch 的 batch 数 = run 数（默认 16）。"""
        return len(self.run_keys)

    def __iter__(self):
        rng = random.Random(None if self.seed is None else self.seed + self.epoch)
        self.epoch += 1

        keys = list(self.run_keys)
        if self.shuffle:
            rng.shuffle(keys)

        for key in keys:
            indices = list(self.groups[key])
            if self.max_per_run and len(indices) > self.max_per_run:
                indices = rng.sample(indices, self.max_per_run)
            if self.shuffle_within:
                rng.shuffle(indices)
            yield indices
