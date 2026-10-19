"""字符级 token 的 EEG–文本数据集（多天 + 可选欧氏对齐）。

与 ``char_token/dataset.py`` 一致的部分
--------------------------------------
把不定长 EEG 段按固定字数窗口切成等长小 token：

    (C, T)  --按 token_samples 非重叠切分-->  (n_tokens, C, token_samples)

Chisco-2.0 的段长天然适合这样切：

    reading 段  T = n_chars       × 100
    inner 段    T = (n_chars + 1) × 100      （250 Hz × 0.4 s）

新增
----
1. ``__getitem__`` 额外返回该 trial 的 **记录块 (day, run)**，训练脚本据此做
   run 级去偏和 within-run 检索；
2. ``align="ea"`` 时对每个 ``(day, run)`` 块做 **欧氏对齐（Euclidean Alignment）**：

   每个块的连续 EEG 先算通道均值和协方差

       μ_g = E[x],   C_g = E[(x-μ_g)(x-μ_g)^T]

   再做白化 + 去均值

       x' = C_g^{-1/2} (x - μ_g)

   run 之间的差异主要来自电极阻抗、参考、接触状态，表现为**逐通道的偏移和
   空间协方差**；EA 恰好把这两项一次性抹掉，而保留单 trial 内的时域波形。
   协方差用收缩估计 ``(1-s)·C + s·tr(C)/C·I`` 避免病态。

用法::

    ds = CharTokenEEGDataset(event="inner", days=[1, 2, 3, 4], align="ea")
    tokens, sentence, group = ds[0]      # group = (day, run)
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


def group_key(day: int, run: int) -> str:
    """把记录块编码成稳定字符串，用于 CSV / checkpoint 里的索引表。"""
    return f"d{int(day):02d}r{int(run):02d}"


class CharTokenEEGDataset(Dataset):
    """把每条 trial 切成等长字 token 的 EEG–文本数据集。

    Parameters
    ----------
    event / sample / subject / days / char_duration / scale / fif_folder / text_table:
        与 ``char_token.dataset.CharTokenEEGDataset`` 完全一致。
    align:
        ``"none"`` 不做对齐；``"ea"`` 对每个 ``(day, run)`` 块做欧氏对齐。
    ea_shrinkage:
        EA 协方差的收缩系数（0 表示不收缩）。122 通道、几十万采样点，协方差估计
        本身不病态，但 bad 通道插值后仍有近共线方向，留一点收缩更稳。
    align_cache:
        对齐矩阵缓存路径（``.npz``）。给了就复用/写入，避免每次重新扫一遍 FIF。
    """

    def __init__(
        self,
        event: str = "inner",
        sample: int = 250,
        subject=None,
        days=None,
        char_duration: float = CHAR_DURATION,
        scale: float = 1e6,
        align: str = "none",
        ea_shrinkage: float = 0.05,
        align_cache: Path | str | None = None,
        fif_folder: Path = FIF_FOLDER,
        text_table: Path = TEXT_TABLE,
    ):
        if event not in EVENT_COLUMNS:
            raise ValueError(
                f"event 必须是 {sorted(EVENT_COLUMNS)} 之一，收到 {event!r}"
            )
        if align not in ("none", "ea"):
            raise ValueError(f"align 必须是 'none' 或 'ea'，收到 {align!r}")
        if not 0 <= ea_shrinkage < 1:
            raise ValueError(f"ea_shrinkage 必须满足 0 <= s < 1，收到 {ea_shrinkage}")

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

        subset = table[mask].reset_index(drop=True)
        if subset.empty:
            raise ValueError(
                f"all_text.tsv 中没有 event={event}、sample={sample}、"
                f"subject={subject}、days={days} 且段长能被 {self.token_samples} 整除的 trial"
            )

        self._subset = subset
        self._raw_cache: dict[Path, tuple] = {}
        self.n_chans: int | None = None

        meta_cols = [
            "sample", "subject", "day", "run", "trial_index", "sentence", "n_chars",
        ]
        # 类别是 all_text.tsv 后来加上的列；有就带进 meta，评测时用它做类别控制
        meta_cols += [
            c for c in ("category_name", "category_number") if c in subset.columns
        ]
        self.meta = subset[meta_cols].copy()
        self.meta["event"] = event
        self.meta["start_sample"] = subset[self.sample_col].astype(int)
        self.meta["n_tokens"] = (subset[self.dur_col] // self.token_samples).astype(int)
        self.meta["token_samples"] = self.token_samples
        # 记录块：run 的偏移是"天 × 段"级别的，跨天不复用
        self.meta["group"] = [
            group_key(d, r) for d, r in zip(self.meta["day"], self.meta["run"])
        ]

        self.align = align
        self.ea_shrinkage = float(ea_shrinkage)
        self.align_cache = Path(align_cache) if align_cache is not None else None
        # {(day, run): (W, mu)}，W 是 C^{-1/2}，mu 是通道均值
        self._align_mats: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        if self.align == "ea":
            self._build_alignment()

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

    # ------------------------------------------------------------------
    # 欧氏对齐
    # ------------------------------------------------------------------
    def _align_groups(self) -> list[tuple[int, int]]:
        """所有出现过的 (day, run) 块，按 (day, run) 升序。"""
        pairs = self._subset[["day", "run"]].drop_duplicates()
        return [(int(d), int(r)) for d, r in zip(pairs["day"], pairs["run"])]

    def _build_alignment(self) -> None:
        """为每个 (day, run) 块估计均值与白化矩阵（单次顺序扫描，按块缓存）。"""
        groups = self._align_groups()

        if self.align_cache is not None and self.align_cache.exists():
            cached = np.load(self.align_cache)
            expected = {group_key(d, r) for d, r in groups}
            if all(f"{key}_W" in cached for key in expected):
                for d, r in groups:
                    key = group_key(d, r)
                    self._align_mats[(d, r)] = (
                        cached[f"{key}_W"], cached[f"{key}_mu"]
                    )
                print(f"[EA] 复用对齐缓存 {self.align_cache}（{len(groups)} 个块）")
                return

        rows = self._subset
        chunk = 100_000  # 每次读 400 s，避免整段 FIF 一次性进内存
        for day, run in groups:
            sel = rows[(rows["day"] == day) & (rows["run"] == run)]
            path = self._fif_path(sel.iloc[0])
            raw, _, picks = self._get_raw(path)
            n_chans = len(picks)
            n_total = int(raw.n_times)
            sum_x = np.zeros(n_chans, dtype=np.float64)
            second = np.zeros((n_chans, n_chans), dtype=np.float64)
            count = 0
            for start in range(0, n_total, chunk):
                stop = min(start + chunk, n_total)
                x = raw.get_data(picks=picks, start=start, stop=stop).astype(np.float64)
                sum_x += x.sum(axis=1)
                second += x @ x.T
                count += x.shape[1]
            mu = sum_x / count
            cov = second / count - np.outer(mu, mu)
            # 收缩：往各向同性靠，抑制插值通道带来的近共线方向
            s = self.ea_shrinkage
            if s > 0:
                cov = (1.0 - s) * cov + s * (np.trace(cov) / n_chans) * np.eye(n_chans)
            eigval, eigvec = np.linalg.eigh(cov)
            eigval = np.clip(eigval, 1e-12, None)
            whitener = (eigvec * (1.0 / np.sqrt(eigval))) @ eigvec.T
            self._align_mats[(day, run)] = (
                whitener.astype(np.float32),
                mu.astype(np.float32),
            )
            print(
                f"[EA] {group_key(day, run)}: {n_chans} ch, {count} 采样点, "
                f"trace={np.trace(cov):.3e}"
            )

        if self.align_cache is not None:
            self.align_cache.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            for d, r in groups:
                key = group_key(d, r)
                whitener, mu = self._align_mats[(d, r)]
                payload[f"{key}_W"] = whitener
                payload[f"{key}_mu"] = mu
            np.savez_compressed(self.align_cache, **payload)
            print(f"[EA] 对齐矩阵已缓存到 {self.align_cache}")

    # ------------------------------------------------------------------
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

        if self.align == "ea":
            day, run = int(row["day"]), int(row["run"])
            whitener, mu = self._align_mats[(day, run)]
            # (C, T)：先减块均值，再白化到各通道单位方差、去相关
            data = whitener @ (data.astype(np.float32) - mu[:, None])

        tokens = data.reshape(len(picks), n_tokens, self.token_samples)
        tokens = np.transpose(tokens, (1, 0, 2))  # (n_tokens, C, token_samples)

        if self.n_chans is None:
            self.n_chans = len(picks)

        group = (int(row["day"]), int(row["run"]))
        return (
            torch.from_numpy((tokens * self.scale).astype(np.float32)),
            str(row["sentence"]),
            group,
        )


def char_token_collate(batch):
    """token 级 collate：把 batch 内不同 token 个数的样本 pad 到同一个 n_max。

    ``batch`` 是 ``[(tokens_i, sentence_i, group_i), ...]``，返回

        tokens:  ``(B, n_max, C, token_samples)``，pad 位置填 0；
        mask:    ``(B, n_max)`` bool，``True`` 表示该位置是有效 token；
        sentences / groups: ``list``，长度 B（groups 的元素是 ``(day, run)``）。
    """
    tokens_list, sentences, groups = zip(*batch)
    n_max = max(int(t.shape[0]) for t in tokens_list)
    n_chans, token_samples = tokens_list[0].shape[1], tokens_list[0].shape[2]

    tokens = tokens_list[0].new_zeros((len(tokens_list), n_max, n_chans, token_samples))
    mask = torch.zeros((len(tokens_list), n_max), dtype=torch.bool)

    for i, token_tensor in enumerate(tokens_list):
        n = int(token_tensor.shape[0])
        tokens[i, :n] = token_tensor
        mask[i, :n] = True

    return tokens, mask, list(sentences), list(groups)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = CharTokenEEGDataset(event="inner", subject="sub-01", days=[1, 2], align="ea")
    loader = DataLoader(ds, batch_size=8, collate_fn=char_token_collate, shuffle=False)
    tokens, mask, sentences, groups = next(iter(loader))
    print(f"N={len(ds)} tokens={tuple(tokens.shape)} mask={tuple(mask.shape)}")
    print(f"groups={groups}")
