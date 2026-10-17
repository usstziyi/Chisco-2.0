"""按句子分组的 batch sampler：一个 batch = n_sentences 句 × 每句 n_repeats 条 trial。

为对比学习服务：同一个句子的 ``n_repeats`` 条 trial 互为正样本（同句多视图），
不同句子之间构成 in-batch 负样本。

动机：句子只在**跨天**之间重复。随机 shuffle 下同一句的多个视图几乎不可能落进
同一个 batch——实测 day-01~04 的 4496 条 trial 里，每个 anchor 的平均正样本数
只有 1.03，也就是说"多视图"这套正样本信号基本被浪费了。分组采样把这件事从
"碰运气"变成确定性保证：``pos_per_anchor`` 直接等于 ``n_repeats``。

用法::

    from torch.utils.data import DataLoader

    from char_token_multiscale.sampler import SentenceGroupBatchSampler

    sampler = SentenceGroupBatchSampler(
        train_sentences, n_sentences=8, n_repeats=4, seed=0
    )
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=char_token_collate)

    for tokens, mask, sentences in loader:   # 每个 batch 恰好 8 句 × 4 视图 = 32 条
        ...
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

from torch.utils.data import Sampler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SentenceGroupBatchSampler(Sampler[list[int]]):
    """每个 batch 抽 ``n_sentences`` 个句子，每个句子再抽 ``n_repeats`` 条 trial。

    Parameters
    ----------
    sentences:
        与数据集索引一一对应的句子列表（``sentences[i]`` 是第 i 个样本的句子）。
        用 train subset 时传对应的子列表即可。
    n_sentences:
        每个 batch 中不同句子的数量。
    n_repeats:
        每个句子在同一 batch 内被抽到的 trial 数量。
    seed:
        随机种子；给定后按 epoch 递增，每个 epoch 可复现。
    shuffle:
        是否在每个 epoch 打乱句子顺序及试次选取，默认 ``True``。
    drop_last:
        句子池最后凑不满 ``n_sentences`` 的一组是否丢弃，默认 ``True``，
        以保证每个 batch 大小恒为 ``n_sentences * n_repeats``。

    Attributes
    ----------
    groups:
        ``{句子: [样本索引, ...]}``，只保留 trial 数 >= ``n_repeats`` 的句子。
    excluded_sentences:
        ``{句子: trial 数}``，trial 数不足 ``n_repeats`` 而被排除的句子。
    batch_size:
        ``n_sentences * n_repeats``。

    Notes
    -----
    ``n_repeats=1`` 时退化成"每 batch 抽 n_sentences 个不同句子、各取 1 条"，
    与随机 shuffle 的效果基本一致，可作为对照实验的配置。
    """

    def __init__(
        self,
        sentences,
        n_sentences: int = 8,
        n_repeats: int = 4,
        seed: int | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        if n_sentences <= 0 or n_repeats <= 0:
            raise ValueError(
                f"n_sentences 与 n_repeats 必须为正整数，收到 {n_sentences=}, {n_repeats=}"
            )

        groups: dict[str, list[int]] = defaultdict(list)
        # 遍历每个样本，按句子分组收集对应的索引
        for index, sentence in enumerate(sentences):
            groups[str(sentence)].append(index)

        # 只有 trial 数 >= n_repeats 的句子才能提供完整的重复组
        self.groups = {
            sentence: indices for sentence, indices in groups.items()
            if len(indices) >= n_repeats
        }
        self.excluded_sentences = {
            sentence: len(indices) for sentence, indices in groups.items()
            if len(indices) < n_repeats
        }
        if len(self.groups) < n_sentences:
            raise ValueError(
                f"trial 数 >= n_repeats={n_repeats} 的句子只有 {len(self.groups)} 个，"
                f"不足以凑出 n_sentences={n_sentences} 个不同句子的 batch"
            )

        self.n_sentences = int(n_sentences)
        self.n_repeats = int(n_repeats)
        self.batch_size = self.n_sentences * self.n_repeats
        self.seed = seed
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """显式设置 epoch（影响打乱顺序）。"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """整个 sampler 的调度单位是 group（一个唯一句子） ，不是单条 trial。"""
        """`__len__` 统计的是 一个 epoch 会产出多少个 batch ，也就是`DataLoader` 一轮迭代的次数。"""
        n_groups = len(self.groups)
        if self.drop_last:
            return n_groups // self.n_sentences
        return -(-n_groups // self.n_sentences)

    def __iter__(self):
        rng = random.Random(None if self.seed is None else self.seed + self.epoch)
        self.epoch += 1

        # 把self.groups 的 键取出来、拷成一个新列表，得到这一轮要排队的"句子池"。
        pool = list(self.groups)
        if self.shuffle:
            rng.shuffle(pool)

        for start in range(0, len(pool), self.n_sentences):
            chunk = pool[start : start + self.n_sentences]
            if len(chunk) < self.n_sentences and self.drop_last:
                break  # 最后不足一组，丢弃

            batch: list[int] = []
            for sentence in chunk:
                indices = self.groups[sentence]
                if self.shuffle:
                    batch.extend(rng.sample(indices, self.n_repeats))
                else:
                    batch.extend(indices[: self.n_repeats])
            yield batch


if __name__ == "__main__":
    from collections import Counter

    from char_token_multiscale.dataset import CharTokenEEGDataset


    all_dataset = CharTokenEEGDataset(
        event="inner", sample=250, subject="sub-01", days=[5,6]
    )
    print(f"dataset N={len(all_dataset)}")
    all_sentences = all_dataset.meta["sentence"].tolist()

    sampler = SentenceGroupBatchSampler(
        all_sentences, 
        n_sentences=8, 
        n_repeats=4, 
        seed=0
    )

    for i,(sentence,indices) in enumerate(sampler.groups.items()):
        print(i,sentence,indices)   
        if i >= 5:
            break


    for batch in sampler:
        for index in batch:
            print(all_sentences[index])
        exit(0)
