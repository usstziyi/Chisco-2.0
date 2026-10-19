"""按句子分组的 batch sampler：一个 batch = n_sentences 句 × 每句 n_repeats 条 trial。

为对比学习服务：同一个句子的 ``n_repeats`` 条 trial 互为正样本（同句多视图），
不同句子之间构成 in-batch 负样本。

动机：句子只在**跨天**之间重复。随机 shuffle 下同一句的多个视图几乎不可能落进
同一个 batch——实测 day-01~04 的 4496 条 trial 里，每个 anchor 的平均正样本数
只有 1.03，也就是说"多视图"这套正样本信号基本被浪费了。分组采样把这件事从
"碰运气"变成确定性保证：``pos_per_anchor`` 直接等于 ``n_repeats``。

用法::

    from torch.utils.data import DataLoader

    from char_token_debias.sampler import SentenceGroupBatchSampler

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


class LengthBucketBatchSampler(Sampler[list[int]]):
    """按**句长分桶**的句子分组采样器：一个 batch 内的句子句长相同。

    动机
    ----
    run / 类别被去偏后，模型会退而使用"句长"这个跨天完全一致的线索：
    同一句 4 天的字数相同，而且 token 个数恰好等于句长（inner 是 n_chars+1）。
    于是只要把同长度句子放在同一个 batch，**句长在 batch 内就是常数**，
    对 softmax 排序没有任何区分力，梯度不会再奖励"数 token"。

    行为
    ----
    - 句子先按 ``n_chars`` 分桶；每桶内部再按 ``n_sentences`` 切块；
    - 桶太小时（``< min_bucket`` 句）并入相邻桶，避免出现过小的 batch；
    - 每个 batch = 桶内若干句 × ``n_repeats`` 视图，batch 大小可能略小于
      ``n_sentences * n_repeats``（最后一组或小桶），这在对比损失里是允许的。

    注意
    ----
    这不是"句长不变"的训练目标，而只是让句长在 batch 内失去判别力。
    真正的句长控制还要配合 ``--length-residual linear``（推理时去掉句长方向）。
    """

    def __init__(
        self,
        sentences,
        lengths,
        n_sentences: int = 32,
        n_repeats: int = 4,
        seed: int | None = None,
        shuffle: bool = True,
        min_bucket: int = 4,
    ):
        if len(sentences) != len(lengths):
            raise ValueError("sentences 与 lengths 长度必须一致")
        if n_sentences <= 0 or n_repeats <= 0:
            raise ValueError("n_sentences 与 n_repeats 必须为正")

        groups: dict[str, list[int]] = defaultdict(list)
        for index, sentence in enumerate(sentences):
            groups[str(sentence)].append(index)

        self.groups = {
            s: idx for s, idx in groups.items() if len(idx) >= n_repeats
        }
        self.excluded_sentences = {
            s: len(idx) for s, idx in groups.items() if len(idx) < n_repeats
        }
        if len(self.groups) < 2:
            raise ValueError("可用句子不足 2 个，无法构造对比 batch")

        # 句子 -> 句长（同一句的所有 trial 句长相同）
        length_of = {}
        for s, idx in self.groups.items():
            length_of[s] = int(lengths[idx[0]])

        raw_buckets: dict[int, list[str]] = defaultdict(list)
        for s, L in length_of.items():
            raw_buckets[L].append(s)

        # 小桶并入相邻桶（先向前，再向后）
        keys = sorted(raw_buckets)
        buckets: dict[int, list[str]] = {}
        for i, L in enumerate(keys):
            pool = raw_buckets[L]
            if len(pool) < min_bucket and buckets:
                prev = sorted(buckets)[-1]
                buckets[prev].extend(pool)
            else:
                buckets[L] = list(pool)
        if len(buckets) == 1 and len(next(iter(buckets.values()))) < 2:
            raise ValueError("句长分桶后可用句子不足 2 个")

        self.buckets = {L: pool for L, pool in buckets.items() if len(pool) >= 2}
        self.length_of = length_of
        self.n_sentences = int(n_sentences)
        self.n_repeats = int(n_repeats)
        self.seed = seed
        self.shuffle = bool(shuffle)
        self.epoch = 0

    @staticmethod
    def _chunks(pool: list[str], k: int) -> list[list[str]]:
        """把桶切成大小 k 的块；最后一块只有 1 句时并入前一块。"""
        if len(pool) <= k:
            return [pool]
        out = [pool[i : i + k] for i in range(0, len(pool), k)]
        if len(out) > 1 and len(out[-1]) < 2:
            out[-2].extend(out[-1])
            out.pop()
        return out

    def __len__(self) -> int:
        return sum(len(self._chunks(pool, self.n_sentences)) for pool in self.buckets.values())

    def __iter__(self):
        rng = random.Random(None if self.seed is None else self.seed + self.epoch)
        self.epoch += 1

        keys = list(self.buckets)
        if self.shuffle:
            rng.shuffle(keys)

        for L in keys:
            pool = list(self.buckets[L])
            if self.shuffle:
                rng.shuffle(pool)
            for chunk in self._chunks(pool, self.n_sentences):
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

    from char_token_debias.dataset import CharTokenEEGDataset


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
