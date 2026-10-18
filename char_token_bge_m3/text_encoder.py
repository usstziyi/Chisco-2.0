"""BGE-M3 文本编码器：句子 -> 定长、L2 归一化的 embedding。

与 EEG 侧配合的方式：两边输出都做 L2 归一化后，点积即余弦相似度。

    sentence (list[str], 长度 B)
        ↓
    BGE-M3（SentenceTransformer，冻结）
        ↓
    (B, 1024)
        ↓
    L2 归一化
        ↓
    (B, 1024)

要点：

- 参数全程冻结（``requires_grad = False``），只做 forward，不参与梯度更新；
- embedding 维度从模型自身读取（BGE-M3 为 1024），EEG 侧通过
  ``embedding_dim`` 参数对齐，不在代码里写死；
- 这里显式做 L2 归一化，保留"输出一定是单位向量"的契约；
- ``sentence_transformers`` 为延迟导入：只 import 本模块不会加载模型。

注意：BGE-M3 权重约 2 GB，首次运行会从 HuggingFace 镜像下载。

用法::

    from char_token_bge_m3.text_encoder import BGEM3TextEncoder

    encoder = BGEM3TextEncoder()
    text = encoder(["附近还有其他的酒店吗"])   # (1, 1024)
"""

from __future__ import annotations

import os

import torch
from torch import nn
import torch.nn.functional as F


class BGEM3TextEncoder(nn.Module):
    """BGE-M3 句子编码器（冻结）。

    Parameters
    ----------
    model_name:
        SentenceTransformer 模型名或本地路径，默认 ``"BAAI/bge-m3"``。

    Input
    -----
    texts:
        ``list[str]``，也允许单个 ``str``（按 B = 1 处理）。

    Output
    ------
    torch.Tensor:
        ``(B, embedding_dim)``，已 L2 归一化；BGE-M3 对应 ``(B, 1024)``。
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        super().__init__()

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name 必须是非空字符串")

        # 必须在导入 sentence_transformers 之前设置，否则 huggingface_hub 已经
        # 读走了默认 endpoint。用 setdefault 不覆盖用户已有配置。
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        actual_dim = self.model.get_embedding_dimension()
        if actual_dim is None:
            raise ValueError(f"无法确定 {model_name!r} 的 embedding 维度")
        self._embedding_dim = int(actual_dim)

        # 文本侧在训练中不更新
        # 1.固定 Dropout / BatchNorm 行为
        self.model.eval()
        # 2.冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        """模型实际的输出维度（BGE-M3 为 1024）。"""
        return self._embedding_dim

    def forward(self, texts: str | list[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            raise ValueError("texts 不能为空")
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("texts 的每个元素都必须是 str")

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        embeddings = embeddings.float()
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"文本 embedding 形状异常: {tuple(embeddings.shape)}，"
                f"期望 (B, {self.embedding_dim})"
            )

        return F.normalize(embeddings, p=2, dim=-1)


if __name__ == "__main__":
    encoder = BGEM3TextEncoder()
    print(f"model={encoder.model_name} embedding_dim={encoder.embedding_dim}")
    print(f"frozen={not any(p.requires_grad for p in encoder.parameters())}")

    sentences = ["附近还有其他的酒店吗", "今天天气很好"]
    embeddings = encoder(sentences)
    print(f"\nembedding={tuple(embeddings.shape)} dtype={embeddings.dtype}")
    print("L2 范数:", embeddings.norm(dim=-1).tolist())
    print("相似度矩阵:")
    print(embeddings @ embeddings.T)
