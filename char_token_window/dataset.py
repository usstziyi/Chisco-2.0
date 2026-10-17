"""滑窗切分的 EEG–文本数据集（字级 token 的句级多尺度变体）。

核心思路：不再按“一个字 0.4 s”切分，而是用固定长度的滑动窗口把整句 EEG 切成
等长 token，让变长性从“每段采样点数”转移到“窗口个数”：

    (C, T)  --window=1.0 s (250 点), stride=0.5 s (125 点)-->  (n_windows, C, 250)

Chisco-2.0 的段长：

    reading 段  T = n_chars       × 100     （1.6 ~ 5.6 s）
    inner 段    T = (n_chars + 1) × 100     （2.0 ~ 6.0 s）

默认 1 s 窗 / 0.5 s 步长下，inner 段切出 3~11 个窗口，reading 段 2~10 个。
窗口边界与字边界**不对齐**，模型无法依赖“窗首 = 字首”这种位置捷径。

窗口数：

    n = floor((T - token_samples) / stride_samples) + 1

末尾不足一个完整窗口的部分会被丢弃。切分之后：
    - 每个窗口形状 (C, token_samples) 恒定 -> 复用同一套 token encoder；
    - 每句窗口数不等 -> 交给 collate 做窗口级 padding + mask。

用法::

    from char_token_window.dataset import CharTokenEEGDataset, char_token_collate

    ds = CharTokenEEGDataset(event="inner", sample=250)
    tokens, sentence = ds[0]        # tokens: (n_windows, C, 250) float32

    loader = DataLoader(ds, batch_size=8, collate_fn=char_token_collate)
    tokens, mask, sentences = next(iter(loader))
    # tokens: (B, n_max, C, token_samples)；mask: (B, n_max) bool，True 表示有效窗口
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "preprocessed"
FIF_FOLDER = PREPROCESSED_FOLDER / "fif"
TEXT_TABLE = PREPROCESSED_FOLDER / "all_text.tsv"

# event 类型 -> (起始采样点列, 持续采样点列)
EVENT_COLUMNS = {
    "reading": ("reading_sample", "reading_dur_samples"),
    "inner": ("inner_sample", "inner_dur_samples"),
}

# 滑动窗口长度与步长（秒）；250 Hz 下对应 250 / 125 个采样点
WINDOW_DURATION = 1.0
STRIDE_DURATION = 0.5


def _normalize_subject(subject):
    """None 不过滤；单个 str 或可迭代统一成 tuple。"""
    if subject is None:
        return None
    if isinstance(subject, str):
        return (subject,)
    if isinstance(subject, Iterable):
        return tuple(str(s) for s in subject)
    return (str(subject),)


def _normalize_days(days):
    """None 不过滤；单个 int 或可迭代统一成升序 tuple。"""
    if days is None:
        return None
    if isinstance(days, (int, np.integer)):
        return (int(days),)
    return tuple(sorted(int(d) for d in days))


class CharTokenEEGDataset(Dataset):
    """把每条 trial 切成等长字 token 的 EEG–文本数据集。

    Parameters
    ----------
    event:
        取哪类事件：``"reading"`` 或 ``"inner"``。
    sample:
        采样率（Hz），决定数据所在的 ``<sample>hz`` 目录。
    subject:
        只保留这些被试：``None`` 不限；``"sub-01"`` 或 ``["sub-01", "sub-02"]``。
    days:
        只保留这些 day：``None`` 不限；单个 ``1`` 或 ``[1, 2]``。
        注意 day 与 session 的对应见 ``script/preprocessing.py``：
        day 1~4 = ses-01~04 的 para1，day 5~8 = ses-01~04 的 para2。
    window_duration:
        滑动窗口长度（秒），默认 1.0。乘以采样率得到 ``token_samples``。
    stride_duration:
        相邻窗口起点的间隔（秒），默认 0.5。小于窗口长度时窗口重叠。
    scale:
        EEG 幅度缩放系数；默认 ``1e6``，把伏特转成微伏。
    fif_folder / text_table:
        数据位置，一般不用改。

    Attributes
    ----------
    token_samples:
        每个窗口的采样点数（默认 250 Hz × 1.0 s = 250）。
    stride_samples:
        相邻窗口起点的采样点间隔（默认 250 Hz × 0.5 s = 125）。
    n_chans:
        传感器（EEG）通道数，读取首条样本后确定。
    meta:
        与样本一一对应的元数据，含 ``n_tokens`` 列。
    """

    def __init__(
        self,
        event: str = "inner",
        sample: int = 250,
        subject=None,
        days=None,
        window_duration: float = WINDOW_DURATION,
        stride_duration: float = STRIDE_DURATION,
        scale: float = 1e6,
        fif_folder: Path = FIF_FOLDER,
        text_table: Path = TEXT_TABLE,
    ):
        if event not in EVENT_COLUMNS:
            raise ValueError(
                f"event 必须是 {sorted(EVENT_COLUMNS)} 之一，收到 {event!r}"
            )

        self.event = event
        self.sample = int(sample)
        self.scale = float(scale)
        self.fif_folder = Path(fif_folder)
        self.sample_col, self.dur_col = EVENT_COLUMNS[event]

        self.token_samples = int(round(float(window_duration) * self.sample))
        self.stride_samples = int(round(float(stride_duration) * self.sample))
        if self.token_samples <= 0 or self.stride_samples <= 0:
            raise ValueError(
                f"window_duration={window_duration}, stride_duration={stride_duration} "
                f"在 {self.sample} Hz 下算不出有效窗口长度"
            )

        self.subject = _normalize_subject(subject)
        self.days = _normalize_days(days)

        # 行顺序沿用 all_text.tsv 的物理顺序：该文件由 script/summarize_text.py 按
        # sample → subject → day → run → trial_index 升序写出；这里只做布尔筛选与
        # reset_index，不改变相对次序，因此相邻 trial 通常落在同一个 run/FIF 上。
        table = pd.read_csv(text_table, sep="\t")
        mask = table["sample"] == self.sample
        if self.subject is not None:
            mask &= table["subject"].isin(self.subject)
        if self.days is not None:
            mask &= table["day"].isin(self.days)
        # 只保留至少能产出一个完整窗口的 trial；滑窗不要求段长被窗口整除
        mask &= table[self.dur_col] >= self.token_samples

        subset = table[mask].reset_index(drop=True)
        if subset.empty:
            raise ValueError(
                f"all_text.tsv 中没有 event={event}、sample={sample}、"
                f"subject={subject}、days={days} 且段长 >= {self.token_samples} 的 trial"
            )

        self._subset = subset
        self._raw_cache: dict[Path, tuple] = {}
        self.n_chans: int | None = None

        self.meta = subset[
            ["sample", "subject", "day", "run", "trial_index", "sentence", "n_chars"]
        ].copy()
        self.meta["event"] = event
        self.meta["start_sample"] = subset[self.sample_col].astype(int)
        self.meta["n_tokens"] = (
            (subset[self.dur_col] - self.token_samples) // self.stride_samples + 1
        ).astype(int)
        self.meta["token_samples"] = self.token_samples
        self.meta["stride_samples"] = self.stride_samples

    def __len__(self) -> int:
        return len(self._subset)

    def _fif_path(self, row) -> Path:
        subject = str(row["subject"])
        return (
            self.fif_folder
            / f"{self.sample}hz"
            / subject
            / f"{subject}_day-{int(row['day']):02d}_run-{int(row['run']):02d}_raw.fif"
        )

    def _get_raw(self, path: Path):
        """惰性读取并缓存 FIF（preload=False，只按需读取需要的那一段）。"""
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
        if abs(sfreq - self.sample) > 1e-3:
            raise ValueError(
                f"FIF 采样率 {sfreq} 与指定 sample={self.sample} 不一致: {path}"
            )

        start = int(row[self.sample_col]) - first_samp
        stop = start + int(row[self.dur_col])
        if start < 0 or stop > raw.n_times:
            raise ValueError(
                f"切片越界: [{start}, {stop}) 超出 FIF 长度 {raw.n_times}（{path.name}）"
            )

        # 一次性读出整段 (C, T)，再按 stride 展开成 (n_tokens, C, token_samples)。
        # unfold 的窗口起点间隔为 stride_samples，末尾不足一个窗口的部分自动丢弃。
        data = raw.get_data(picks=picks, start=start, stop=stop)
        tokens = torch.from_numpy((data * self.scale).astype(np.float32))
        tokens = tokens.unfold(-1, self.token_samples, self.stride_samples)
        tokens = tokens.permute(1, 0, 2).contiguous()  # (n_tokens, C, token_samples)

        if self.n_chans is None:
            self.n_chans = len(picks)

        return tokens, str(row["sentence"])


def char_token_collate(batch):
    """token 级 collate：把 batch 内不同 token 个数的样本 pad 到同一个 n_max。

    因为每个 token 的形状 (C, token_samples) 已经统一，这里只需要在 **token 数**
    这一维上 padding，代价极小（最多十几个位置）。

    Parameters
    ----------
    batch:
        ``[(tokens_i, sentence_i), ...]``，``tokens_i`` 形状 ``(n_i, C, token_samples)``。

    Returns
    -------
    (tokens, mask, sentences):
        tokens:    ``(B, n_max, C, token_samples)``，pad 位置填 0；
        mask:      ``(B, n_max)`` bool，``True`` 表示该位置是有效 token；
        sentences: ``list[str]``，长度 B。
    """
    tokens_list, sentences = zip(*batch)
    n_max = max(int(t.shape[0]) for t in tokens_list)
    n_chans, token_samples = tokens_list[0].shape[1], tokens_list[0].shape[2]

    tokens = tokens_list[0].new_zeros((len(tokens_list), n_max, n_chans, token_samples))
    mask = torch.zeros(len(tokens_list), n_max, dtype=torch.bool)

    for i, token_tensor in enumerate(tokens_list):
        n = int(token_tensor.shape[0])
        tokens[i, :n] = token_tensor
        mask[i, :n] = True

    return tokens, mask, list(sentences)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = CharTokenEEGDataset(event="inner", days=1, sample=250)
    loader = DataLoader(
        ds, batch_size=10, collate_fn=char_token_collate, shuffle=False,
        drop_last=False,
    )
    print(f"ds N={len(ds)}")
    print(f"loader N={len(loader)}")
    # b 是 collate 返回的 (tokens, mask, sentences) 三元组，len(b) 恒为 3；
    # 样本数看 tokens 的第 0 维
    batch_size_list = [b[0].shape[0] for b in loader]
    print(f"最后一个batch的长度是 {batch_size_list[-1]}")


    tokens, mask, sentences = next(iter(loader))
    print(f"tokens.shape={tuple(tokens.shape)}")
    print(f"mask.shape={tuple(mask.shape)}")
    print(f"sentences N={len(sentences)}")

    for i,s in enumerate(sentences):
        print(f"sentences[{i}]={s}")
