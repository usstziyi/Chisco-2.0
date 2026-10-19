"""按记录块 ``(day, run)`` 的 embedding 中心化去偏。

为什么需要它
------------
单天训练时，一个 batch 里约 8 条同 run、24 条跨 run。对比损失只要把 EEG 编成
"哪个 run"就能一次排掉 24 个负样本，loss 从 ln32 降到 ln8；而 run 方向在 batch
内**梯度相干**（同 run 的样本都往同一方向推），句级方向却互相抵消。于是 run 方向
被快速放大，句级信息被当成噪声压掉。

这里的做法是：维护每个 ``(day, run)`` 块的平均 embedding ``μ_g``（EMA），前向时

    e' = normalize(e - μ_g)

同一块内的样本减去同一个偏移，run 的公共分量被抵消，排名只能靠块内**残差**。
``μ_g`` 只在训练 batch 上更新，验证时使用训练估计值，不读取验证集统计量。

注意这不会退化成"全部输出相同"：如果模型真把所有 trial 编成同一个向量，
``e - μ_g ≈ 0``，归一化后 logits 全为 0，loss = ln(B)，比保持区分度更差。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GroupDebiaser(nn.Module):
    """维护 ``(day, run)`` 的 EMA 均值，并在前向时减掉。

    Parameters
    ----------
    group_vocab:
        ``{(day, run): index}``，索引须连续（0..G-1）。
    dim:
        embedding 维度。
    momentum:
        EMA 动量。样本数少于 ``1/(1-momentum)`` 时自动退化为累计平均，
        避免前几个 batch 的估计过噪。
    """

    def __init__(
        self,
        group_vocab: dict[tuple[int, int], int],
        dim: int,
        momentum: float = 0.9,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim 必须为正，收到 {dim}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum 必须满足 0 <= m < 1，收到 {momentum}")

        self.group_vocab = {tuple(k): int(v) for k, v in group_vocab.items()}
        self.dim = int(dim)
        self.momentum = float(momentum)

        self.register_buffer("means", torch.zeros(len(self.group_vocab), dim))
        self.register_buffer("seen", torch.zeros(len(self.group_vocab), dtype=torch.long))

    @property
    def n_groups(self) -> int:
        return len(self.group_vocab)

    def index(self, groups) -> torch.Tensor:
        """把 ``(day, run)`` 列表映射成索引张量。"""
        return torch.tensor(
            [self.group_vocab[tuple(g)] for g in groups], dtype=torch.long
        )

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, group_index: torch.Tensor) -> None:
        """用当前 batch 的块均值更新 EMA（不参与梯度）。"""
        embeddings = embeddings.detach()
        for group in torch.unique(group_index):
            mask = group_index == group
            batch_mean = embeddings[mask].mean(dim=0)
            g = int(group)
            n = int(self.seen[g])
            # 早期用累计平均（n/(n+1)），后期切到固定动量
            mom = min(self.momentum, n / (n + 1)) if n > 0 else 0.0
            if n == 0:
                self.means[g] = batch_mean
            else:
                self.means[g] = mom * self.means[g] + (1.0 - mom) * batch_mean
            self.seen[g] = n + 1

    def forward(self, embeddings: torch.Tensor, group_index: torch.Tensor) -> torch.Tensor:
        """``(B, D)`` -> ``(B, D)``：减去所属块的均值后重新 L2 归一化。"""
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings 必须是 (B, D)，收到 {tuple(embeddings.shape)}")
        if group_index.ndim != 1 or group_index.size(0) != embeddings.size(0):
            raise ValueError("group_index 必须是长度 B 的一维张量")

        centered = embeddings - self.means[group_index.to(embeddings.device)]
        return F.normalize(centered, p=2, dim=-1, eps=1e-6)

    def residual_norm(self, embeddings: torch.Tensor, group_index: torch.Tensor) -> float:
        """诊断：块内残差的平均范数；越接近 0 说明压得越塌。"""
        with torch.no_grad():
            centered = embeddings - self.means[group_index.to(embeddings.device)]
            return float(centered.norm(dim=-1).mean())
