"""多尺度时间卷积的字符级 token EEG 编码器。

与 ``char_token/model.py`` 的唯一差别在 ``CharTokenEncoder`` 的时间前端：

    单尺度（char_token/）                 多尺度（本目录）
    Conv(1→32, 1×9)                      Conv(1→11, 1×3)  ┐
    AvgPool(1×13, s=2)                   Conv(1→11, 1×7)  ├→ concat → (N, 32, C, T)
                                         Conv(1→10, 1×17) ┘
                                         AvgPool(1×5, s=3)

两个改动是配套的，缺一不可：

1. **多尺度**：250 Hz 下 9 抽头卷积的第一零点约 27.8 Hz，等效于一个低通，beta/gamma
   段几乎进不来。并行 3/7/17 三个核对应约 12/28/68 ms 的波形片段，让模型同时看到
   快振荡和慢漂移，由后面 32×1 的空间卷积与线性层自行融合。
2. **池化放缓**：(1×13, s=2) 的第一零点约 19 Hz，会把短核刚带进来的高频再平均掉；
   改成 (1×5, s=3) 后输出时间长度由 40 变为 32，高频得以保留到 Transformer 之前。

其余部分——GroupNorm、跨电极空间卷积、位置编码、Transformer、attention pooling、
投影与 L2 归一化——与主实验逐行一致，指标差异才可以归因到时间前端。

用法::

    from char_token_multiscale.model import CharTokenEEGEncoder

    model = CharTokenEEGEncoder(n_chans=122, token_samples=100, embedding_dim=768)
    eeg = model(tokens, mask)      # tokens: (B, n, 122, 100)；mask: (B, n) bool
    # eeg: (B, 768)，已 L2 归一化
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# 默认三支核宽（采样点）：约 12 / 28 / 68 ms
DEFAULT_KERNELS = (3, 7, 17)


def _split_channels(k: int, n_branches: int) -> list[int]:
    """把 k 个特征图尽量平均分给各分支，余数从前往后依次 +1。"""
    base, remainder = divmod(k, n_branches)
    if base == 0:
        raise ValueError(
            f"k={k} 不足以分给 {n_branches} 个分支，每个分支至少需要 1 个通道"
        )
    return [base + (1 if i < remainder else 0) for i in range(n_branches)]


class MultiScaleTemporalConv(nn.Module):
    """并行多支时间卷积，same padding 保证各支输出时间长度一致。

    padding 使用 ``replicate`` 而非默认的补零：token 只是连续 EEG 的一段切片，
    补零会在两端制造人为阶跃，且核越宽污染越重（17 抽头时两端各 8 个采样点）。
    复制边界值不改变任何维度，但对生理信号更合理。

    输入 ``(N, 1, C, T)`` -> 输出 ``(N, k, C, T)``，通道维由各分支拼接而成。

    Parameters
    ----------
    k:
        输出特征图总数，按 ``kernels`` 个数平均分配。
    kernels:
        各分支的核宽（采样点），必须为正奇数且不超过 ``token_samples``。
    token_samples:
        单个 token 的时间长度，用于校验核宽。
    """

    def __init__(self, k: int, kernels=DEFAULT_KERNELS, token_samples: int = 100):
        super().__init__()

        kernels = tuple(int(m) for m in kernels)
        if not kernels:
            raise ValueError("kernels 不能为空")
        if any(m <= 0 or m % 2 == 0 for m in kernels):
            raise ValueError(f"kernels 必须全为正奇数，收到 {kernels}")
        if len(set(kernels)) != len(kernels):
            raise ValueError(f"kernels 不能重复，收到 {kernels}")
        if max(kernels) > token_samples:
            raise ValueError(
                f"最大核宽 {max(kernels)} 超过 token_samples={token_samples}"
            )

        branch_channels = _split_channels(k, len(kernels))
        self.kernels = kernels
        self.branch_channels = tuple(branch_channels)
        self.branches = nn.ModuleList(
            # replicate：token 是连续 EEG 的切片，补零相当于在两端叠加一个归零阶跃，
            # 受影响的位置数随 m//2 增长（3/7/17 对应两端各 1/3/8 个采样点）。
            # 复制边界值消除了这一数值阶跃，但仍假设边界外局部恒定，是近似而非真实信号。
            nn.Conv2d(
                1, out_ch, kernel_size=(1, m), padding=(0, m // 2),
                padding_mode="replicate",
            )
            for out_ch, m in zip(branch_channels, kernels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 1, C, T)`` -> ``(N, k, C, T)``。"""
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class CharTokenEncoder(nn.Module):
    """把一个字 token ``(C, token_samples)`` 编码成 ``d_model`` 维向量。

    流程与主实验一致，只有时间前端换成多尺度：

        (N, C, 100)
            ↓ 三支并行时间卷积 + same padding
        (N, k, C, 100)
            ↓ AvgPool(1×m2, s)
        (N, k, C, T2)
            ↓ GroupNorm + ELU
            ↓ Conv(k→k, n_chans×1)  跨全部电极的空间卷积
        (N, k, 1, T2)
            ↓ GroupNorm + ELU + Dropout
            ↓ Linear(k·T2 → d_model)

    Parameters
    ----------
    k:
        时间卷积的特征图总数，默认 32（与主实验一致，便于对照）。
    kernels:
        多尺度核宽，默认 ``(3, 7, 17)``。传 ``(9,)`` 可退回单支结构。
    m2 / s:
        时间平均池化的核宽与步长，默认 ``(5, 3)``；主实验为 ``(13, 2)``。
    """

    def __init__(
        self,
        n_chans: int = 122,
        token_samples: int = 100,
        d_model: int = 128,
        k: int = 32,
        kernels=DEFAULT_KERNELS,
        m2: int = 5,
        s: int = 3,
        drop_prob: float = 0.3,
    ):
        super().__init__()

        if n_chans <= 0 or token_samples <= 0 or d_model <= 0 or k <= 0:
            raise ValueError("n_chans / token_samples / d_model / k 必须为正")
        if m2 <= 0 or s <= 0:
            raise ValueError("m2 / s 必须为正")
        if m2 > token_samples:
            raise ValueError(
                f"m2={m2} 超过 token_samples={token_samples}，无法池化"
            )
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob 必须满足 0 <= drop_prob < 1")

        t2 = (token_samples - m2) // s + 1
        if t2 <= 0:
            raise ValueError(
                f"token_samples={token_samples}, m2={m2}, s={s} 算不出有效输出长度"
            )

        self.n_chans = n_chans
        self.token_samples = token_samples
        self.d_model = d_model
        self.m2 = int(m2)
        self.s = int(s)
        self.out_time = t2
        self.flatten_dim = k * t2

        self.temporal = MultiScaleTemporalConv(k, kernels, token_samples)
        self.kernels = self.temporal.kernels
        self.branch_channels = self.temporal.branch_channels
        self.pool = nn.AvgPool2d(kernel_size=(1, m2), stride=(1, s))
        # 与主实验一致：时间池化之后再归一化
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=k)
        self.act = nn.ELU()
        # 一次跨越全部电极的空间卷积
        self.spatial = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(n_chans, 1))
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=k)
        self.dropout = nn.Dropout(drop_prob)
        self.proj = nn.Linear(self.flatten_dim, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: ``(N, C, token_samples)`` -> ``(N, d_model)``。"""
        if tokens.ndim != 3:
            raise ValueError(
                f"token 必须是 (N, C, T) 三维，收到 {tuple(tokens.shape)}"
            )
        if tokens.shape[1] != self.n_chans:
            raise ValueError(
                f"期望 {self.n_chans} 个 EEG 通道，收到 {tokens.shape[1]}"
            )
        if tokens.shape[2] != self.token_samples:
            raise ValueError(
                f"期望每个 token {self.token_samples} 个采样点，收到 {tokens.shape[2]}"
            )

        x = self.temporal(tokens.unsqueeze(1))  # (N, 1, C, T) -> (N, k, C, T)
        x = self.pool(x)                        # (N, k, C, T2)
        x = self.act(self.norm1(x))
        x = self.spatial(x)                     # (N, k, 1, T2)
        x = self.act(self.norm2(x))
        x = self.dropout(x)
        return self.proj(x.flatten(start_dim=1))  # (N, d_model)


class CharTokenEEGEncoder(nn.Module):
    """字符级 token 的整句 EEG 编码器，输出与文本共享空间的定长 embedding。

    与 ``char_token.model.CharTokenEEGEncoder`` 完全相同，只是透传给
    ``CharTokenEncoder`` 的 ``token_kwargs`` 默认值不同（多尺度核宽 + 更小的池化）。

    Parameters
    ----------
    n_chans / token_samples:
        必须与 ``CharTokenEEGDataset`` 的输出一致（默认 122 / 100）。
    d_model:
        token 空间与 Transformer 的宽度。
    n_heads / n_layers / dim_feedforward / dropout:
        Transformer 配置，token 数只有 4~15、数据量也小，保持 2 层、4 头。
    max_tokens:
        位置编码支持的最大 token 数，默认 32。
    embedding_dim:
        最终 embedding 维度；与 LaBSE 的 768 对齐。
    token_kwargs:
        透传给 ``CharTokenEncoder``（k / kernels / m2 / s / drop_prob）。
    """

    def __init__(
        self,
        n_chans: int = 122,
        token_samples: int = 100,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_tokens: int = 32,
        embedding_dim: int = 768,
        **token_kwargs,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须为正")

        self.d_model = d_model
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim

        self.token_encoder = CharTokenEncoder(
            n_chans=n_chans,
            token_samples=token_samples,
            d_model=d_model,
            **token_kwargs,
        )

        # token 顺序就是字序，位置信息必须显式给出
        self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True 时 PyTorch 无法启用 nested tensor 加速，显式关掉以免告警
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # attention pooling：一个可学习 query 对所有有效 token 做加权汇总
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, embedding_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``tokens (B, n, C, T)`` + ``mask (B, n)`` -> ``(B, embedding_dim)``（L2 归一化）。"""
        if tokens.ndim != 4:
            raise ValueError(
                f"tokens 必须是 (B, n, C, T) 四维，收到 {tuple(tokens.shape)}"
            )
        if mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"mask 形状 {tuple(mask.shape)} 与 tokens 前两维 {tuple(tokens.shape[:2])} 不一致"
            )

        batch_size, n_tokens = tokens.shape[:2]
        if n_tokens > self.max_tokens:
            raise ValueError(
                f"token 数 {n_tokens} 超过 max_tokens={self.max_tokens}，"
                "请调大 max_tokens 或换更粗的切分粒度"
            )

        # (B, n, C, T) -> (B·n, C, T)，所有 token 共用同一套权重
        flat = tokens.reshape(batch_size * n_tokens, *tokens.shape[2:])
        hidden = self.token_encoder(flat).reshape(batch_size, n_tokens, self.d_model)
        hidden = hidden + self.position[:, :n_tokens]

        # True 表示要忽略的位置，所以取反
        key_padding_mask = ~mask
        hidden = self.transformer(hidden, src_key_padding_mask=key_padding_mask)

        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(
            query, hidden, hidden, key_padding_mask=key_padding_mask
        )  # (B, 1, d_model)
        pooled = pooled.squeeze(1)  # (B, d_model)

        return F.normalize(self.out_proj(pooled), p=2, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = CharTokenEEGEncoder(n_chans=122, token_samples=100, embedding_dim=768)
    encoder = model.token_encoder
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total:,}（可训练 {trainable:,}）")
    print(f"kernels={encoder.kernels} 各分支通道={encoder.branch_channels}")
    print(
        f"token encoder 输出时间长度={encoder.out_time} "
        f"flatten={encoder.flatten_dim} pool=({encoder.m2}, s={encoder.s})"
    )

    batch_size, n_tokens = 6, 11
    tokens = torch.randn(batch_size, n_tokens, 122, 100)
    mask = torch.ones(batch_size, n_tokens, dtype=torch.bool)
    for i, keep in enumerate([4, 6, 8, 9, 10, 11]):
        mask[i, keep:] = False
        tokens[i, keep:] = 0.0

    model.eval()
    with torch.no_grad():
        eeg = model(tokens, mask)

    print(f"\ntokens={tuple(tokens.shape)} -> eeg={tuple(eeg.shape)}")
    print("每行有效 token 数:", mask.sum(dim=1).tolist())
    print("L2 范数:", eeg.norm(dim=-1).tolist())

    # pad 位置灌噪声，输出应保持不变
    noisy = tokens.clone()
    for i, keep in enumerate([4, 6, 8, 9, 10, 11]):
        noisy[i, keep:] = torch.randn_like(noisy[i, keep:]) * 100.0
    with torch.no_grad():
        eeg_noisy = model(noisy, mask)
    print("padding 位置灌噪声后的最大输出偏差:", (eeg - eeg_noisy).abs().max().item())
