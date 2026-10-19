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


class LengthResidualizer(nn.Module):
    """在线估计并去掉 embedding 中"线性可预测句长"的方向。

    为什么需要它
    ------------
    run 与类别被去偏后，模型会退到下一个跨天完全一致的线索：**句长**。
    本数据里 token 个数恰好等于字数（inner 为 ``n_chars + 1``），attention pooling
    可以数 token；实测训练到后期，残差第一主成分与 ``n_chars`` 的相关可达 0.75。

    做法
    ----
    用流式最小二乘估计"句长 -> embedding"的斜率方向 ``w``：

        w ∝ Cov(e, n),   n = 句长

    前向时去掉 embedding 在该方向上的投影：

        e' = normalize(e - (e·ŵ) ŵ)

    这不是把长度信息"藏起来"，而是让对比损失无法再用长度做判别。运行统计量用
    EMA + 偏差校正，因此能跟上训练过程中表示的变化。

    Parameters
    ----------
    dim:
        embedding 维度。
    momentum:
        矩估计的 EMA 动量；0.99 约等于最近 100 个 batch。
    eps:
        方差下限，避免句长在 batch 内为常数（配合长度分桶采样时）导致除零。
    """

    def __init__(self, dim: int, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim 必须为正，收到 {dim}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum 必须满足 0 <= m < 1，收到 {momentum}")

        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)

        self.register_buffer("direction", torch.zeros(dim))
        self.register_buffer("ema_e", torch.zeros(dim))
        self.register_buffer("ema_n", torch.zeros(()))
        self.register_buffer("ema_nn", torch.zeros(()))
        self.register_buffer("ema_ne", torch.zeros(dim))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> None:
        """用当前 batch 更新"句长方向"的流式估计（不参与梯度）。"""
        embeddings = embeddings.detach()
        lengths = lengths.detach().to(embeddings.dtype).flatten()
        m = self.momentum
        t = int(self.updates) + 1
        bias = 1.0 - m ** t

        self.ema_e.mul_(m).add_(embeddings.mean(dim=0) * (1 - m))
        self.ema_n.mul_(m).add_(lengths.mean() * (1 - m))
        self.ema_nn.mul_(m).add_((lengths * lengths).mean() * (1 - m))
        self.ema_ne.mul_(m).add_(
            (lengths[:, None] * embeddings).mean(dim=0) * (1 - m)
        )

        mean_e = self.ema_e / bias
        mean_n = self.ema_n / bias
        mean_nn = self.ema_nn / bias
        mean_ne = self.ema_ne / bias

        cov = mean_ne - mean_n * mean_e
        var = mean_nn - mean_n * mean_n
        direction = cov / (var + self.eps)
        self.direction.copy_(torch.nn.functional.normalize(direction, p=2, dim=0, eps=self.eps))
        self.updates += 1

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """去掉沿句长方向的投影，再重新 L2 归一化。"""
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings 必须是 (B, D)，收到 {tuple(embeddings.shape)}")
        direction = self.direction.to(embeddings.device)
        projection = embeddings @ direction
        debiased = embeddings - projection.unsqueeze(1) * direction
        return torch.nn.functional.normalize(debiased, p=2, dim=-1, eps=1e-6)
