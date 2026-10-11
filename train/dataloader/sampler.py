"""面向"句子重复"对比学习的自包含 Dataset + BatchSampler。

本文件不依赖 ``train.dataloader.dataloader``，数据集读取与 FIF 切片都写在内部：

- ``SentenceRepeatEEGDataset``：从 ``datasets/preprocessed/all_text.tsv`` 取
  ``n_chars``/``sample`` 匹配、且在指定 ``days`` 内的 trial，按 ``event``
  （reading / inner）从预处理 FIF 切出对应 EEG 段，返回 ``(eeg, sentence)``。
- ``SentenceRepeatBatchSampler``：一个 batch 由 ``n_sentences`` 个不同句子组成，
  每个句子贡献 ``n_repeats`` 条 trial。同句子的多条录制互为正样本，
  不同句子构成 in-batch 负样本，可直接用于对比 / 语义对齐训练。

EEG 段长度只由 ``n_chars`` 决定，因此同一 ``n_chars`` 下所有样本的 ``T`` 一致，
batch 可直接 stack，无需 padding。

用法::

    from torch.utils.data import DataLoader

    from train.dataloader.sampler import (
        SentenceRepeatEEGDataset,
        SentenceRepeatBatchSampler,
    )

    ds = SentenceRepeatEEGDataset(
        n_chars=10, event="reading", sample_rate=250, days=(5, 6, 7, 8)
    )
    sampler = SentenceRepeatBatchSampler(ds, n_sentences=16, n_repeats=4)
    loader = DataLoader(ds, batch_sampler=sampler)

    for eeg, sentences in loader:   # eeg: (64, C, T)；sentences: list[str]，长度 64
        ...
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "preprocessed"
FIF_FOLDER = PREPROCESSED_FOLDER / "fif"
TEXT_TABLE = PREPROCESSED_FOLDER / "all_text.tsv"

# event 类型 -> (起始采样点列, 持续采样点列, onset 列)
EVENT_COLUMNS = {
    "reading": ("reading_sample", "reading_dur_samples", "reading_onset"),
    "inner": ("inner_sample", "inner_dur_samples", "inner_onset"),
}


class SentenceRepeatEEGDataset(Dataset):
    """按句子长度、天数、事件类型筛选的 EEG–文本数据集。

    Parameters
    ----------
    n_chars:
        只取句子非标点字符数等于该值的 trial。
    event:
        取哪类事件：``"reading"`` 或 ``"inner"``。
    sample_rate:
        采样率（Hz），同时决定数据所在的 ``<sample_rate>hz`` 目录与切片换算。
    days:
        只保留这些 day 的 trial；``None`` 表示不限天数。例如 ``(5, 6, 7, 8)``。
    scale:
        EEG 幅度缩放系数；默认 ``1e6``，把原始伏特转成微伏。
    fif_folder:
        FIF 根目录，默认 ``datasets/preprocessed/fif``。
    text_table:
        汇总表路径，默认 ``datasets/preprocessed/all_text.tsv``。

    Attributes
    ----------
    meta:
        与样本一一对应的元数据 DataFrame，含 ``sentence`` 列供采样器使用。
    days:
        实际生效的天数过滤（tuple 或 None）。
    eeg_shape:
        单个样本的形状 ``(n_chans, n_times)``（读取首条样本后确定）。
    """

    def __init__(
        self,
        n_chars: int,
        event: str = "reading",
        sample_rate: int = 250,
        days=None,
        scale: float = 1e6,
        fif_folder: Path = FIF_FOLDER,
        text_table: Path = TEXT_TABLE,
    ):
        if event not in EVENT_COLUMNS:
            raise ValueError(
                f"event 必须是 {sorted(EVENT_COLUMNS)} 之一，收到 {event!r}"
            )

        table = pd.read_csv(text_table, sep="\t")
        mask = (table["n_chars"] == int(n_chars)) & (
            table["sample"] == int(sample_rate)
        )
        self.days = None if days is None else tuple(sorted(int(d) for d in days))
        if self.days is not None:
            mask &= table["day"].isin(self.days)

        subset = table[mask].reset_index(drop=True)
        if subset.empty:
            raise ValueError(
                f"all_text.tsv 中没有 n_chars={n_chars}、sample={sample_rate}、"
                f"days={self.days} 的 trial"
            )

        self.n_chars = int(n_chars)
        self.event = event
        self.sample_rate = int(sample_rate)
        self.scale = float(scale)
        self.fif_folder = Path(fif_folder)
        self.sample_col, self.dur_col, self.onset_col = EVENT_COLUMNS[event]

        # 同一 n_chars 下段长必须一致，否则 T 不固定
        durations = subset[self.dur_col].unique()
        if len(durations) != 1:
            raise ValueError(
                f"n_chars={n_chars} 对应多种段长 {durations.tolist()}，T 不固定"
            )

        self._subset = subset
        self._n_times = int(durations[0])
        self._raw_cache: dict[Path, tuple] = {}
        self.eeg_shape: tuple[int, int] | None = None

        self.meta = subset[
            [
                "sample",
                "subject",
                "day",
                "run",
                "trial_index",
                "sentence",
                "n_chars",
            ]
        ].copy()
        self.meta["event"] = event
        self.meta["start_sample"] = subset[self.sample_col].astype(int)
        self.meta["n_times"] = self._n_times

    def __len__(self) -> int:
        return len(self._subset)

    def _fif_path(self, row) -> Path:
        subject = str(row["subject"])
        return (
            self.fif_folder
            / f"{self.sample_rate}hz"
            / subject
            / f"{subject}_day-{int(row['day']):02d}_run-{int(row['run']):02d}_raw.fif"
        )

    def _get_raw(self, path: Path):
        """惰性读取并缓存 FIF（preload=False，只按需读取需要的采样点）。"""
        if path not in self._raw_cache:
            if not path.exists():
                raise FileNotFoundError(f"FIF 不存在: {path}")
            raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
            # raw.first_samp 不一定为 0，切片索引需要减去它
            first = int(raw.first_samp)
            picks = [
                i
                for i, ch_type in enumerate(raw.get_channel_types())
                if ch_type == "eeg"
            ]
            self._raw_cache[path] = (raw, first, picks)
        return self._raw_cache[path]

    def __getitem__(self, index: int):
        row = self._subset.iloc[index]
        path = self._fif_path(row)
        raw, first_samp, picks = self._get_raw(path)

        sfreq = float(raw.info["sfreq"])
        if abs(sfreq - self.sample_rate) > 1e-3:
            raise ValueError(
                f"FIF 采样率 {sfreq} 与指定 sample_rate={self.sample_rate} 不一致: {path}"
            )

        start = int(row[self.sample_col]) - first_samp
        stop = start + self._n_times
        if start < 0 or stop > raw.n_times:
            raise ValueError(
                f"切片越界: [{start}, {stop}) 超出 FIF 长度 {raw.n_times}（{path.name}）"
            )

        data = raw.get_data(picks=picks, start=start, stop=stop)
        eeg = torch.from_numpy((data * self.scale).astype(np.float32))

        if self.eeg_shape is None:
            self.eeg_shape = tuple(eeg.shape)

        return eeg, str(row["sentence"])


class SentenceRepeatBatchSampler(Sampler[list[int]]):
    """每个 batch 抽 ``n_sentences`` 个句子，每个句子再抽 ``n_repeats`` 条 trial。

    Parameters
    ----------
    dataset:
        需带 ``meta['sentence']`` 列的数据集（如 ``SentenceRepeatEEGDataset``）。
    n_sentences:
        每个 batch 中不同句子的数量。
    n_repeats:
        每个句子在同一 batch 内被抽到的 trial 数量。
    sentences:
        可选，显式指定每个样本的句子标签；不给则从 ``dataset.meta['sentence']`` 读。
    drop_last:
        句子池最后凑不满 ``n_sentences`` 的一组是否丢弃，默认 ``True``，
        以保证每个 batch 大小恒为 ``n_sentences * n_repeats``。
    shuffle:
        是否在每个 epoch 打乱句子顺序及试次选取，默认 ``True``。
    seed:
        随机种子；给定后每个 epoch 可复现。

    Notes
    -----
    只有 trial 数 >= ``n_repeats`` 的句子才能凑满一组正样本，少于 ``n_repeats``
    的句子会被排除，记录在 ``excluded_sentences``（句子 -> trial 数）。
    """

    def __init__(
        self,
        dataset,
        n_sentences: int,
        n_repeats: int,
        sentences=None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        if n_sentences <= 0 or n_repeats <= 0:
            raise ValueError(
                f"n_sentences 与 n_repeats 必须为正整数，收到 {n_sentences=}, {n_repeats=}"
            )

        labels = self._extract_sentences(dataset, sentences)

        groups: dict[str, list[int]] = defaultdict(list)
        for index, sentence in enumerate(labels):
            groups[str(sentence)].append(index)

        # 只有 trial 数 >= n_repeats 的句子才能提供完整的重复组
        self.groups = {
            sentence: indices
            for sentence, indices in groups.items()
            if len(indices) >= n_repeats
        }
        self.excluded_sentences = {
            sentence: len(indices)
            for sentence, indices in groups.items()
            if len(indices) < n_repeats
        }
        if len(self.groups) < n_sentences:
            raise ValueError(
                f"trial 数 >= n_repeats={n_repeats} 的句子只有 {len(self.groups)} 个，"
                f"不足以凑出 n_sentences={n_sentences} 个不同句子的 batch"
            )

        self.dataset = dataset
        self.n_sentences = int(n_sentences)
        self.n_repeats = int(n_repeats)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.epoch = 0

    @staticmethod
    def _extract_sentences(dataset, sentences) -> list:
        if sentences is not None:
            return list(sentences)
        meta = getattr(dataset, "meta", None)
        if meta is None or "sentence" not in meta:
            raise ValueError(
                "无法从 dataset 读取句子标签：请显式传入 sentences=，"
                "或让 dataset 提供带 'sentence' 列的 meta"
            )
        return list(meta["sentence"])

    def set_epoch(self, epoch: int) -> None:
        """显式设置 epoch（影响打乱顺序）。"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        n_groups = len(self.groups)
        if self.drop_last:
            return n_groups // self.n_sentences
        return (n_groups + self.n_sentences - 1) // self.n_sentences

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch if self.seed is not None else None)
        self.epoch += 1

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

    from torch.utils.data import DataLoader

    ds = SentenceRepeatEEGDataset(
        n_chars=10, event="reading", sample_rate=250, days=(5, 6, 7, 8)
    )





    sampler = SentenceRepeatBatchSampler(ds, n_sentences=16, n_repeats=4, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler)

    print(
        f"dataset N={len(ds)}（days={ds.days}），可用句子 {len(sampler.groups)} 个，"
        f"被排除（trial < 4）{len(sampler.excluded_sentences)} 个，"
        f"每轮 {len(sampler)} 个 batch"
    )

    for step, (eeg, sentences) in enumerate(loader):
        counts = Counter(sentences)
        print(
            f"batch {step}: eeg={tuple(eeg.shape)} dtype={eeg.dtype} "
            f"不同句子 {len(counts)} 个，每句重复 {sorted(set(counts.values()))} 次"
        )

    # 复现第 0 个 batch 的样本索引，逐条打印 sentence 及其来源 day / run
    sampler.set_epoch(0)
    first_batch = next(iter(sampler))
    meta = ds.meta.iloc[first_batch]

    print(f"\n=== batch 0 明细（{len(first_batch)} 条）===")
    for order, (_, row) in enumerate(meta.iterrows(), start=1):
        print(
            f"{order:2d}. day-{int(row['day']):02d} run-{int(row['run']):02d} "
            f"trial-{int(row['trial_index']):03d}  {row['sentence']}"
        )
