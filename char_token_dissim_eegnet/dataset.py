"""字符级 token 的 EEG–文本数据集。

核心思路：把不定长 EEG 段按固定字数窗口切成**等长小 token**，让"变长"从
每个 token 的采样点数转移到 token 的个数：

    (C, T)  --按 token_samples 非重叠切分-->  (n_tokens, C, token_samples)

Chisco-2.0 的段长天然适合这样切：

    reading 段  T = n_chars       × 100
    inner 段    T = (n_chars + 1) × 100      （250 Hz × 0.4 s）

两条都能被 100 整除，所以 token 边界正好落在字的边界上，每个 token 就是一个字。

切分之后：
    - 同一个 token 的形状 (C, token_samples) 恒定 -> 可复用同一套 token encoder；
    - 每条 trial 的 token 个数 n ∈ [4, 15] 不等 -> 交给 collate 做 token 级 padding + mask。

用法::

    from char_token_dissim_eegnet.dataset import CharTokenEEGDataset, char_token_collate

    ds = CharTokenEEGDataset(event="reading", sample=250)
    tokens, sentence = ds[0]        # tokens: (n_tokens, C, token_samples) float32

    loader = DataLoader(ds, batch_size=8, collate_fn=char_token_collate)
    tokens, mask, sentences = next(iter(loader))
    # tokens: (B, n_max, C, token_samples)；mask: (B, n_max) bool，True 表示有效 token
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

# 每个字的时长（秒）；250 Hz 下对应 100 个采样点
CHAR_DURATION = 0.4


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
    char_duration:
        每个字 token 的时长（秒），默认 0.4。乘以采样率得到 ``token_samples``。
        传 0.2 就得到"半字 token"，token 个数翻倍。
    scale:
        EEG 幅度缩放系数；默认 ``1e6``，把伏特转成微伏。
    fif_folder / text_table:
        数据位置，一般不用改。
    keep_sentences:
        只保留这些句子文本（``None`` 表示不过滤）。用于"按相似度阈值剔除近重复句"
        的实验：传入保留句集合，其余 trial 在构造阶段就被过滤掉。

    Attributes
    ----------
    token_samples:
        每个 token 的采样点数（默认 250 Hz × 0.4 s = 100）。
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
        char_duration: float = CHAR_DURATION,
        scale: float = 1e6,
        fif_folder: Path = FIF_FOLDER,
        text_table: Path = TEXT_TABLE,
        keep_sentences=None,
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

        self.token_samples = int(round(float(char_duration) * self.sample))
        if self.token_samples <= 0:
            raise ValueError(
                f"char_duration={char_duration} 在 {self.sample} Hz 下算不出有效 token 长度"
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
        # 只保留能被 token_samples 整除的 trial，保证不产生残缺 token
        mask &= (table[self.dur_col] % self.token_samples) == 0

        # 可选：只保留指定句子（近重复过滤后剩下的"单例句"），按句子文本过滤，
        # 使同一句在所有 day 保持一致地保留 / 剔除
        if keep_sentences is not None:
            mask &= table["sentence"].astype(str).isin(set(map(str, keep_sentences)))

        subset = table[mask].reset_index(drop=True)
        if subset.empty:
            raise ValueError(
                f"all_text.tsv 中没有 event={event}、sample={sample}、"
                f"subject={subject}、days={days} 且段长能被 {self.token_samples} 整除的 trial"
            )

        self._subset = subset
        self._raw_cache: dict[Path, tuple] = {}
        self.n_chans: int | None = None

        self.meta = subset[
            ["sample", "subject", "day", "run", "trial_index", "sentence", "n_chars"]
        ].copy()
        self.meta["event"] = event
        self.meta["start_sample"] = subset[self.sample_col].astype(int)
        self.meta["n_tokens"] = (subset[self.dur_col] // self.token_samples).astype(int)
        self.meta["token_samples"] = self.token_samples

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

        n_tokens = int(row[self.dur_col]) // self.token_samples
        start = int(row[self.sample_col]) - first_samp
        stop = start + n_tokens * self.token_samples
        if start < 0 or stop > raw.n_times:
            raise ValueError(
                f"切片越界: [{start}, {stop}) 超出 FIF 长度 {raw.n_times}（{path.name}）"
            )

        # 一次性读出整段 (C, T)，再 reshape 成 (n_tokens, C, token_samples)
        data = raw.get_data(picks=picks, start=start, stop=stop)
        tokens = data.reshape(len(picks), n_tokens, self.token_samples)
        tokens = np.transpose(tokens, (1, 0, 2))  # (n_tokens, C, token_samples)

        if self.n_chans is None:
            self.n_chans = len(picks)

        return torch.from_numpy((tokens * self.scale).astype(np.float32)), str(
            row["sentence"]
        )


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
