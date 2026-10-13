"""注意力池化 vs 平均池化的消融模型。

消融要干净，关键是**只改池化方式**，其余（token 编码器、位置编码、Transformer、投影、
文本侧、损失、采样、评测）全部不变。为此：

- ``pooling="attn"``：**直接返回** ``char_token.model.CharTokenEEGEncoder``，与主实验所用
  的类**字面上是同一个**，不存在实现漂移的风险；
- ``pooling="mean"``：``MeanPoolEEGEncoder`` 继承同一个父类，**只覆写 ``forward`` 里的池化
  那几行**，并在 ``__init__`` 中删除 ``pool_query`` / ``pool_attn`` 两个专用参数，使参数量
  如实反映"没有池化参数"。

平均池化采用 **masked mean**（只在有效 token 上求平均），这是与 attention pooling 对齐的标准
基线；若把 pad 位置也平均进去，反而会额外引入一个"有效 token 占比"通道，混淆消融结论。

用法::

    from char_token_pooling.model import build_encoder

    model = build_encoder("mean", n_chans=122, token_samples=100, embedding_dim=768)
    model = build_encoder("attn", n_chans=122, token_samples=100, embedding_dim=768)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

# 支持 `python char_token_pooling/model.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.model import CharTokenEEGEncoder  # noqa: F401

POOLING_MODES = ("attn", "mean")


class MeanPoolEEGEncoder(CharTokenEEGEncoder):
    """把整句的 attention pooling 换成 **masked mean pooling**。

    与父类唯一的差别在 ``forward`` 的最后一步：

        attn: 可学习 query + MultiheadAttention 对有效 token 加权汇总
        mean: 对有效 token 直接取平均（pad 位置用 ``masked_fill`` 置 0，不参与分母）

    这类模型正是"计数泄漏"论断所要检验的对照：mean pooling 的输出统计量会随 token 个数
    变化，模型可能借此反推句子长度。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 删除注意力池化专用参数，使参数量如实反映 mean-pool 变体
        del self.pool_query
        del self.pool_attn

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """与父类 ``forward`` 前半段逐行一致，只在池化处不同。"""
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

        flat = tokens.reshape(batch_size * n_tokens, *tokens.shape[2:])
        hidden = self.token_encoder(flat).reshape(batch_size, n_tokens, self.d_model)
        hidden = hidden + self.position[:, :n_tokens]

        key_padding_mask = ~mask
        hidden = self.transformer(hidden, src_key_padding_mask=key_padding_mask)

        # ---- 唯一区别：masked mean 取代 attention pooling ----
        # masked_fill 把 pad 位置（可能受 masked attention 影响取到异常值）显式置 0，
        # 保证既不进入分子也不进入分母
        keep = mask.unsqueeze(-1).to(hidden.dtype)  # (B, n, 1)
        hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        pooled = hidden.sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)  # (B, d)

        return F.normalize(self.out_proj(pooled), p=2, dim=-1)


def build_encoder(pooling: str = "attn", **kwargs) -> nn.Module:
    """按池化方式构造编码器。

    Parameters
    ----------
    pooling:
        ``"attn"`` -> ``CharTokenEEGEncoder``（主实验同款）；
        ``"mean"`` -> ``MeanPoolEEGEncoder``（masked mean）。
    **kwargs:
        透传给编码器（n_chans / token_samples / d_model / embedding_dim 等）。
    """
    if pooling == "attn":
        return CharTokenEEGEncoder(**kwargs)
    if pooling == "mean":
        return MeanPoolEEGEncoder(**kwargs)
    raise ValueError(f"pooling 必须是 {POOLING_MODES} 之一，收到 {pooling!r}")


if __name__ == "__main__":
    torch.manual_seed(0)

    for mode in POOLING_MODES:
        model = build_encoder(mode, n_chans=122, token_samples=100, embedding_dim=768)
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{mode}] 参数量={n_params:,}（可训练 {n_train:,}）")

    # 两种池化都应满足：pad 位置的数值不影响输出
    batch_size, n_max = 6, 11
    mask = torch.ones(batch_size, n_max, dtype=torch.bool)
    for i, keep in enumerate([5, 6, 8, 9, 10, 11]):
        mask[i, keep:] = False

    tokens = torch.randn(batch_size, n_max, 122, 100)
    tokens = tokens * mask.unsqueeze(-1).unsqueeze(-1)  # pad 位置置 0

    for mode in POOLING_MODES:
        model = build_encoder(mode, n_chans=122, token_samples=100, embedding_dim=768)
        model.eval()
        with torch.no_grad():
            out = model(tokens, mask)

        noisy = tokens.clone()
        for i, keep in enumerate([5, 6, 8, 9, 10, 11]):
            noisy[i, keep:] = torch.randn_like(noisy[i, keep:]) * 100.0
        with torch.no_grad():
            out_noisy = model(noisy, mask)

        print(
            f"\n[{mode}] tokens={tuple(tokens.shape)} -> out={tuple(out.shape)}"
            f"  L2={out.norm(dim=-1).mean():.4f}"
            f"  pad 灌噪声后最大偏差={(out - out_noisy).abs().max().item():.3e}"
        )