"""字符级 token 的 EEG 编码器：EEGNet 风格 CNN tokenizer + Transformer + attention pooling。

整体结构：

    tokens  (B, n, C, token_samples)        # n 个等长字 token，n 可变
        ↓  reshape 成 (B·n, C, token_samples)
    EEGNetTokenEncoder（共享权重，逐 token 独立编码）
        ↓
    (B, n, d_model)
        ↓  + 可学习位置编码（token 顺序就是字序，必须有）
    TransformerEncoder（n_layers 层）
        ↓  按 mask 屏蔽 padding 的 token
    (B, n, d_model)
        ↓  attention pooling（可学习 query 对有效 token 做加权汇总）
    (B, d_model)
        ↓  Linear
    (B, embedding_dim)
        ↓  L2 归一化

本目录是 ``char_token_dissim`` 的**架构消融**：数据切分、Transformer、attention
pooling、对比损失、近重复句过滤全部原样保留，只把 token encoder 从 NICE 风格的
TSConv 换成 EEGNet 风格的可分离卷积。

    TSConv（基线）  时间卷积 -> 平均池化 -> 稠密跨电极空间卷积
    EEGNet（本实验）时间卷积 -> 深度可分离空间卷积 -> 平均池化
                    -> 深度可分离时间卷积 -> 逐点卷积 -> 平均池化

三个关键选择（前两条与基线一致）：

1. **token encoder 权重共享** —— 它相当于 ViT 的 patch embedding，把所有字映射到
   同一个 d_model 空间。若每个字符位置各用一套权重，参数会爆炸，也无法泛化到
   没见过的句子长度。

2. **attention pooling 而不是 mean-pool** —— token 个数恰好等于字符数，mean-pool
   会让模型靠"平均了多少个 token"反推句子长度；可学习 query 的加权汇总表达力更强，
   也避免了这种计数泄漏。

3. **空间卷积从稠密改成 depthwise（EEGNet 的核心改动）** —— 基线的跨电极空间卷积
   是一个 ``(C, 1)`` 的稠密卷积，单这一步在 122 电极下就有约 12.5 万权重。EEGNet
   用一个 **逐时间滤波器独立** 的 depthwise 空间卷积（groups = 时间滤波器个数）替掉
   它：参数量降一个数量级，但保留"每个时间滤波器各自学一套空间权重"的表达形式。
   这正是本次消融要回答的问题——少得多的空间参数够不够用。

用法::

    from char_token_dissim_eegnet.model import CharTokenEEGEncoder

    model = CharTokenEEGEncoder(n_chans=122, token_samples=100, embedding_dim=768)
    eeg = model(tokens, mask)      # tokens: (B, n, 122, 100)；mask: (B, n) bool
    # eeg: (B, 768)，已 L2 归一化
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EEGNetTokenEncoder(nn.Module):
    """把一个字 token ``(C, token_samples)`` 编码成 ``d_model`` 维向量（EEGNet 风格）。

    两个 block 把 ``(N, 1, C, T)`` 一路压到 ``(N, F2, 1, T4)``：

        Block1  时间卷积   (1 -> F1, 核 (1, K))
                深度可分离空间卷积 (F1 -> F1·D, groups=F1, 核 (C, 1))
                平均池化   (1, pool1)
        Block2  深度可分离时间卷积 (F1·D -> F1·D, groups=F1·D, 核 (1, K2))
                逐点卷积   (F1·D -> F2, 核 (1, 1))
                平均池化   (1, pool2)

    与 NICE 风格 TSConv 的两处差别：

    - **空间卷积是 depthwise 的**：TSConv 用一个 ``(C, 1)`` 稠密卷积让所有时间滤波
      器共享一套跨电极权重，参数量是 ``F1·F2·C``；这里每个时间滤波器独立学一套，
      参数量是 ``F1·D·C``，在 F2 = F1·D 时直接降了 F1 倍。
    - **时间方向用可分离卷积**：Block2 先沿时间做 depthwise，再用 1×1 逐点卷积混合
      特征，比一个大核稠密卷积省得多。

    归一化用 ``InstanceNorm2d(affine=True)`` 而不是 EEGNet 原文的 ``BatchNorm2d``，
    这是必须的：collate 会往 batch 里塞 pad 出来的全零 token，拍平后它们会作为独立
    "样本"进入 (B·n) 维，把零样本算进 batch 统计量就会污染真实 token 的归一化。
    ``InstanceNorm2d`` 只做**逐样本、逐通道**归一化（等价于 ``GroupNorm(num_groups=C)``），
    是 BatchNorm 在"拿不到 batch 统计量"时最直接的替代，且完全不受 padding 影响：
    全零 token 的方差为 0，归一化后仍是 0，不会产生 NaN。

    卷积一律 ``bias=False``：归一化自带仿射偏置，偏置冗余；同时让全零的 pad token
    在初始状态下逐层保持严格为零，便于 ``__main__`` 里的 padding 不变性自检。

    Parameters
    ----------
    n_chans / token_samples:
        必须与 ``CharTokenEEGDataset`` 的输出一致（默认 122 / 100）。
    d_model:
        输出维度，与 Transformer 的宽度一致。
    F1:
        Block1 的时间滤波器个数（EEGNet 的 F1）。
    D:
        深度乘子（EEGNet 的 D），空间卷积把滤波器数扩到 ``F1·D``。
    F2:
        Block2 逐点卷积的输出滤波器数；``None`` 表示沿用 EEGNet 的 ``F1·D``。
    K / K2:
        Block1 / Block2 的时间卷积核长度，**必须是正奇数**，这样 ``padding=K//2``
        才是真正的 "same" 卷积，token 长度不会漂移。
    pool1 / pool2:
        两个 block 之后的非重叠平均池化宽度。
    drop_prob:
        两个 block 末尾的 dropout 概率。
    """

    def __init__(
        self,
        n_chans: int = 122,
        token_samples: int = 100,
        d_model: int = 128,
        F1: int = 16,
        D: int = 2,
        F2: int | None = None,
        K: int = 25,
        K2: int = 15,
        pool1: int = 2,
        pool2: int = 5,
        drop_prob: float = 0.3,
    ):
        super().__init__()

        if n_chans <= 0 or token_samples <= 0 or d_model <= 0:
            raise ValueError("n_chans / token_samples / d_model 必须为正")
        if F1 <= 0 or D <= 0:
            raise ValueError("F1 / D 必须为正")
        if F2 is None:
            F2 = F1 * D  # EEGNet 原文的默认取法
        if F2 <= 0:
            raise ValueError("F2 必须为正")
        if K <= 0 or K2 <= 0 or K % 2 == 0 or K2 % 2 == 0:
            raise ValueError(
                f"K / K2 必须是正奇数，收到 K={K}, K2={K2}；偶数核配 padding=K//2 "
                "会让时间长度每过一层多 1"
            )
        if pool1 <= 0 or pool2 <= 0:
            raise ValueError("pool1 / pool2 必须为正")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob 必须满足 0 <= drop_prob < 1")

        # 奇数核 + padding=K//2 => 时间长度不变，逐层就是 token_samples -> //pool1 -> //pool2
        t2 = token_samples // pool1
        t4 = t2 // pool2
        if t4 < 1:
            raise ValueError(
                f"token_samples={token_samples} 经过 pool1={pool1}, pool2={pool2} 后"
                f"只剩 {t4} 个时间点，池化过于激进"
            )

        self.n_chans = n_chans
        self.token_samples = token_samples
        self.d_model = d_model
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.out_time = t4
        self.flatten_dim = F2 * t4

        self.blocks = nn.Sequential(
            # ---- Block1：时间卷积 + 深度可分离空间卷积 ----
            # 时间卷积：(N, 1, C, T) -> (N, F1, C, T)
            nn.Conv2d(1, F1, kernel_size=(1, K), padding=(0, K // 2), bias=False),
            nn.InstanceNorm2d(F1, affine=True),
            # 空间卷积：每个时间滤波器独立跨全部电极卷一次，
            # (N, F1, C, T) -> (N, F1·D, 1, T)
            nn.Conv2d(
                F1, F1 * D, kernel_size=(n_chans, 1), groups=F1, bias=False
            ),
            nn.InstanceNorm2d(F1 * D, affine=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(drop_prob),
            # ---- Block2：深度可分离时间卷积 + 逐点卷积 ----
            # (N, F1·D, 1, T2) -> (N, F1·D, 1, T2)
            nn.Conv2d(
                F1 * D, F1 * D, kernel_size=(1, K2), padding=(0, K2 // 2),
                groups=F1 * D, bias=False,
            ),
            # 1×1 逐点卷积把 depthwise 的时间特征混合回 F2 个通道，
            # (N, F1·D, 1, T2) -> (N, F2, 1, T2)
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.InstanceNorm2d(F2, affine=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(drop_prob),
        )
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

        x = self.blocks(tokens.unsqueeze(1))  # (N, 1, C, T) -> (N, F2, 1, T4)
        return self.proj(x.flatten(start_dim=1))  # (N, d_model)


class CharTokenEEGEncoder(nn.Module):
    """字符级 token 的整句 EEG 编码器，输出与文本共享空间的定长 embedding。

    Parameters
    ----------
    n_chans / token_samples:
        必须与 ``CharTokenEEGDataset`` 的输出一致（默认 122 / 100）。
    d_model:
        token 空间与 Transformer 的宽度。
    n_heads / n_layers / dim_feedforward / dropout:
        Transformer 配置。token 数只有 4~15、数据量也小，默认给得很浅
        （2 层、4 头），别往上堆。
    max_tokens:
        位置编码支持的最大 token 数；``n_chars`` 最大 14（inner 段 +1 = 15），
        默认留到 32 以便换用更细的切分粒度。
    embedding_dim:
        最终 embedding 维度；与文本侧对齐，LaBSE 是 768。
    token_kwargs:
        透传给 ``EEGNetTokenEncoder``（F1 / D / F2 / K / K2 / pool1 / pool2 /
        drop_prob）。默认配置下 token encoder 约 3.1 万参数，而基线的 TSConv
        约 29 万，做对比时留意这个容量差。

    Attributes
    ----------
    token_encoder:
        逐 token 的 EEGNet，权重在所有字之间共享。
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

        self.token_encoder = EEGNetTokenEncoder(
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

        # attention pooling：叫注意力池化，一个可学习 query 对所有有效 token 做加权汇总。
        # 它的唯一目的就是把变长序列池化成一个定长向量。
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, embedding_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 只有这两个是手工建的 torch.zeros Parameter，必须自己打散；
        # 其余模块（含 out_proj 的 nn.Linear）交给 PyTorch 默认初始化即可。
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens:
            ``(B, n, C, token_samples)``；pad 位置填 0。
        mask:
            ``(B, n)`` bool，``True`` 表示有效 token。

        Returns
        -------
        torch.Tensor:
            ``(B, embedding_dim)``，已做 L2 归一化；与文本 embedding 点积即余弦相似度。
        """
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
        ) # (B, 1, d_model)
        pooled = pooled.squeeze(1) # (B, d_model)

        return F.normalize(self.out_proj(pooled), p=2, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = CharTokenEEGEncoder(n_chans=122, token_samples=100, embedding_dim=768)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    token_params = sum(p.numel() for p in model.token_encoder.parameters())
    print(f"参数量: {total:,}（可训练 {trainable:,}）")
    print(f"其中 token encoder: {token_params:,}")
    print(
        f"token encoder 时间长度={model.token_encoder.out_time} "
        f"flatten={model.token_encoder.flatten_dim}"
    )

    batch_size, n_tokens = 6, 11
    tokens = torch.randn(batch_size, n_tokens, 122, 100)
    mask = torch.ones(batch_size, n_tokens, dtype=torch.bool)
    # 模拟 batch 内长度不齐：不同样本保留不同数目的 token
    for i, keep in enumerate([4, 6, 8, 9, 10, 11]):
        mask[i, keep:] = False
        tokens[i, keep:] = 0.0

    model.eval()
    with torch.no_grad():
        eeg = model(tokens, mask)

    print(f"\ntokens={tuple(tokens.shape)} -> eeg={tuple(eeg.shape)}")
    print("每行有效 token 数:", mask.sum(dim=1).tolist())
    print("L2 范数:", eeg.norm(dim=-1).tolist())

    # token 之外的 padding 数量不应影响结果：给 pad 位置灌噪声，输出应保持不变
    noisy = tokens.clone()
    for i, keep in enumerate([4, 6, 8, 9, 10, 11]):
        noisy[i, keep:] = torch.randn_like(noisy[i, keep:]) * 100.0
    with torch.no_grad():
        eeg_noisy = model(noisy, mask)
    deviation = (eeg - eeg_noisy).abs().max().item()
    print(f"padding 位置灌噪声后的最大输出偏差: {deviation:.3e}")
    assert deviation < 1e-6, "padding 不变性被破坏：pad token 影响了有效 token"
    print("padding 不变性自检通过")
