"""随机噪声对照实验的数据集。

目的
----
``char_token`` 的对比学习能在 EEG→文本 检索上做出高于随机的成绩。本实验要回答：
这个成绩来自真实的脑电信号，还是来自"架构 + 句子长度 + 数据划分"这些与信号无关的
旁路信息？

做法
----
除 EEG 数值外，一切照搬 ``char_token``：

    - 同一条 trial 的 token 形状 ``(n_tokens, C, token_samples)`` 完全保留，
      ``n_tokens`` 仍等于该句的非标点字数；
    - 句子标签、train/val 划分、分组采样、损失函数、评测口径全部沿用；
    - 只把 ``(n_tokens, C, token_samples)`` 的值替换成独立同分布的随机噪声。

噪声按 **样本索引** 独立取种，因此同一个句子的不同 trial（跨天多视图）拿到的是互不
相关的噪声，多正样本之间不存在任何可学习的共同信号。

判读
----
- val 指标回落到随机基线附近 -> ``char_token`` 的成绩确实依赖真实脑电；
- val 指标仍显著高于随机 -> 存在与信号无关的泄漏通道（例如靠 token 个数反推句子
  长度），需要回头检查 ``char_token`` 的设计。

噪声刷新
--------
``refresh="epoch"``（默认）每个 epoch 重新采样，连"记住某一轮训练噪声"这条路也走
不通；``refresh="never"`` 固定为 ``(noise_seed, index)`` 决定的确定性噪声，便于完全
复现。两种模式都按样本索引独立取种，绝不让同一句的多视图共享噪声。

用法::

    from char_token_noise.dataset import NoiseTokenEEGDataset
    from char_token.dataset import char_token_collate

    ds = NoiseTokenEEGDataset(event="inner", days=[1, 2], noise_std=1.0)
    tokens, sentence = ds[0]       # tokens: (n_tokens, C, token_samples) float32
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# 支持 `python char_token_noise/dataset.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import CharTokenEEGDataset, char_token_collate  # noqa: F401

NOISE_MODES = ("gaussian", "uniform")
REFRESH_MODES = ("never", "epoch")


class NoiseTokenEEGDataset(CharTokenEEGDataset):
    """与 ``CharTokenEEGDataset`` 同构，但把 EEG 数值替换成随机噪声。

    只读取元数据（``all_text.tsv``）来确定每条的 token 形状，**不读 FIF 数据**，
    因此比真实数据集快得多。

    Parameters
    ----------
    noise_mode:
        ``"gaussian"``（默认，标准正态）或 ``"uniform"``（[-1, 1] 均匀分布）。
    noise_std:
        噪声幅度缩放；默认 1.0。token encoder 用 GroupNorm 逐样本归一化，
        对整体幅度不敏感，该参数主要用于对照不同信噪比。
    noise_seed:
        噪声随机种子。
    refresh:
        ``"epoch"`` 每轮换一批噪声（默认）；``"never"`` 固定不放。
    **kwargs:
        透传给 ``CharTokenEEGDataset``（event / sample / subject / days /
        char_duration / scale 等）。其中 ``scale`` 在这里不生效（噪声不经过 FIF）。

    Attributes
    ----------
    n_chans:
        通道数，构造时即从首条 trial 的 FIF 表头读定。
    epoch:
        当前 epoch，影响 ``refresh="epoch"`` 时的噪声取值。
    """

    def __init__(
        self,
        *,
        noise_mode: str = "gaussian",
        noise_std: float = 1.0,
        noise_seed: int = 0,
        refresh: str = "epoch",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if noise_mode not in NOISE_MODES:
            raise ValueError(f"noise_mode 必须是 {NOISE_MODES} 之一，收到 {noise_mode!r}")
        if noise_std <= 0:
            raise ValueError(f"noise_std 必须为正，收到 {noise_std}")
        if refresh not in REFRESH_MODES:
            raise ValueError(f"refresh 必须是 {REFRESH_MODES} 之一，收到 {refresh!r}")

        self.noise_mode = noise_mode
        self.noise_std = float(noise_std)
        self.noise_seed = int(noise_seed)
        self.refresh = refresh
        self.epoch = 0

        # 形状里的 C 唯一依赖 FIF，构造时读一次表头即可（preload=False，不解码数据）
        self.n_chans = self._resolve_n_chans()

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch；``refresh="epoch"`` 时噪声随之更新。"""
        self.epoch = int(epoch)

    def _resolve_n_chans(self) -> int:
        """从首条 trial 的 FIF 表头读出 EEG 通道数。"""
        row = self._subset.iloc[0]
        _, _, picks = self._get_raw(self._fif_path(row))
        return len(picks)

    def _make_noise(self, shape: tuple[int, int, int], index: int) -> torch.Tensor:
        """按 (noise_seed, [epoch,] index) 生成独立噪声。

        用索引（而非全局 RNG 顺序）取种，保证：同一句子跨天多视图互不相关、
        且结果与 DataLoader 的遍历顺序 / worker 数无关，可复现。
        """
        seed = self.noise_seed * 1_000_003 + index
        if self.refresh == "epoch":
            seed += self.epoch * 7_919

        generator = torch.Generator().manual_seed(seed)
        if self.noise_mode == "gaussian":
            noise = torch.randn(*shape, generator=generator)
        else:
            noise = torch.rand(*shape, generator=generator).mul_(2).sub_(1)
        return noise.mul_(self.noise_std).to(torch.float32)

    def __getitem__(self, index: int):
        row = self._subset.iloc[index]
        # 与真实数据集完全一致的 token 数：只由段长与 token_samples 决定
        n_tokens = int(row[self.dur_col]) // self.token_samples
        tokens = self._make_noise((n_tokens, self.n_chans, self.token_samples), index)
        return tokens, str(row["sentence"])


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = NoiseTokenEEGDataset(event="inner", days=[1, 2], noise_std=1.0)
    print(f"N={len(ds)} n_chans={ds.n_chans} token_samples={ds.token_samples}")
    print("n_tokens 分布:", ds.meta["n_tokens"].value_counts().sort_index().to_dict())

    tokens, sentence = ds[0]
    print(f"ds[0]: tokens={tuple(tokens.shape)} std={tokens.std():.4f} sentence={sentence!r}")

    # 跨天多视图必须拿到互不相关的噪声：同句两条 trial 的噪声应几乎不相关
    sentence_rows = ds.meta.groupby("sentence")
    multi = next((s for s, g in sentence_rows if len(g) >= 2), None)
    if multi is not None:
        idx = ds.meta.index[ds.meta["sentence"] == multi].tolist()
        a, b = ds[idx[0]][0], ds[idx[1]][0]
        n = min(a.shape[0], b.shape[0])
        corr = torch.corrcoef(torch.stack([a[:n].flatten(), b[:n].flatten()]))[0, 1]
        print(f"同句两视图噪声相关系数={corr:.4f}（应接近 0）")

    loader = DataLoader(ds, batch_size=8, collate_fn=char_token_collate, shuffle=False)
    batch_tokens, batch_mask, batch_sentences = next(iter(loader))
    print(f"batch: tokens={tuple(batch_tokens.shape)} mask={tuple(batch_mask.shape)} "
          f"sentences={len(batch_sentences)}")