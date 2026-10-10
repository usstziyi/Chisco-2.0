"""Chisco-2.0 EEG–文本配对数据集。

从 ``datasets/preprocessed/all_text.tsv`` 取 ``n_chars`` 匹配的 trial，
按 ``event`` 指定的类型（reading / inner）从预处理 FIF 中切出对应 EEG 段，
与句子文本配对，返回 ``(eeg, sentence)``。

EEG 段长度只由 ``n_chars`` 决定，因此同一 ``n_chars`` 下所有样本的 ``T`` 一致，
可以直接喂固定长度的 encoder，无需 padding。

用法::

    from train.dataloader.dataloader import ChiscoEEGTextDataset

    ds = ChiscoEEGTextDataset(n_chars=10, event="reading", sample_rate=250)
    eeg, sentence = ds[0]   # eeg: (C, T) float32；sentence: str
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "preprocessed"
FIF_FOLDER = PREPROCESSED_FOLDER / "fif"
TEXT_TABLE = PREPROCESSED_FOLDER / "all_text.tsv"

# event 类型 -> (起始采样点列, 持续采样点列, onset 列)
EVENT_COLUMNS = {
    "reading": ("reading_sample", "reading_dur_samples", "reading_onset"),
    "inner": ("inner_sample", "inner_dur_samples", "inner_onset"),
}


class ChiscoEEGTextDataset(Dataset):
    """按句子长度筛选的 EEG–文本数据集。

    Parameters
    ----------
    n_chars:
        只取句子非标点字符数等于该值的 trial。
    event:
        取哪类事件：``"reading"`` 或 ``"inner"``。
    sample_rate:
        采样率（Hz），同时决定数据所在的 ``<sample_rate>hz`` 目录与切片换算。
    scale:
        EEG 幅度缩放系数；默认 ``1e6``，把原始伏特转成微伏。
    fif_folder:
        FIF 根目录，默认 ``datasets/preprocessed/fif``。
    text_table:
        汇总表路径，默认 ``datasets/preprocessed/all_text.tsv``。

    Attributes
    ----------
    meta:
        与样本一一对应的元数据 DataFrame。
    eeg_shape:
        单个样本的形状 ``(n_chans, n_times)``（读取首条样本后确定）。
    """

    def __init__(
        self,
        n_chars: int,
        event: str = "reading",
        sample_rate: int = 250,
        scale: float = 1e6,
        fif_folder: Path = FIF_FOLDER,
        text_table: Path = TEXT_TABLE,
    ):
        if event not in EVENT_COLUMNS:
            raise ValueError(
                f"event 必须是 {sorted(EVENT_COLUMNS)} 之一，收到 {event!r}"
            )

        table = pd.read_csv(text_table, sep="\t")
        subset = table[
            (table["n_chars"] == int(n_chars))
            & (table["sample"] == int(sample_rate))
        ].reset_index(drop=True)
        if subset.empty:
            raise ValueError(
                f"all_text.tsv 中没有 n_chars == {n_chars} 且 "
                f"sample == {sample_rate} 的 trial"
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


if __name__ == "__main__":
    for event in ("reading", "inner"):
        ds = ChiscoEEGTextDataset(n_chars=10, event=event, sample_rate=250)
        eeg, sentence = ds[0]
        print(
            f"event={event:7s} sample_rate={ds.sample_rate} N={len(ds):4d} "
            f"eeg={tuple(eeg.shape)} dtype={eeg.dtype} sentence={sentence!r}"
        )
        print(len(ds))
