"""从训练脚本中抽出的通用工具，供本目录独立使用。

包含：CSV/JSON 记录、随机种子、LaBSE 文本查表、按句子取文本、句子分组 id，
以及多正样本对称 InfoNCE。本目录不 import 任何外部实验代码。
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def append_csv_row(path: Path, fields: list, row: dict) -> None:
    """把一行追加到 CSV；文件不存在时先写表头。"""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def dump_args_json(args, out_dir: Path, filename: str = "args.json") -> Path:
    """把本次运行的所有命令行参数（key/value）以 JSON 写到时间戳输出目录。"""
    path = Path(out_dir) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_text_lookup(encoder, sentences, device, chunk: int = 64):
    """把所有唯一句子编码成 embedding，返回 ``{sentence: (D,)}`` 查表。"""
    unique = sorted(set(sentences))
    vectors = []
    with torch.no_grad():
        for start in range(0, len(unique), chunk):
            vectors.append(encoder(unique[start : start + chunk]))
    table = torch.cat(vectors, dim=0).to(device)
    embedding_dim = table.shape[1]
    return dict(zip(unique, table)), embedding_dim


def gather_text(sentences, lookup, device) -> torch.Tensor:
    """按句子取回文本 embedding，堆成 (B, D)。"""
    return torch.stack([lookup[s] for s in sentences]).to(device)


def sentence_group_ids(sentences, device) -> torch.Tensor:
    """把句子列表映射成组 id：同句的 trial 得到同一个 id。"""
    mapping: dict[str, int] = {}
    ids = []
    for sentence in sentences:
        if sentence not in mapping:
            mapping[sentence] = len(mapping)
        ids.append(mapping[sentence])
    return torch.tensor(ids, dtype=torch.long, device=device)


def _multi_positive_ce(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """多正样本交叉熵：每个 anchor 对所有正样本列求平均的 -log softmax。"""
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * positive).sum(dim=1).div(positive.sum(dim=1)).mean()


def contrastive_loss(
    eeg: torch.Tensor,
    text: torch.Tensor,
    temperature: float,
    groups: torch.Tensor | None = None,
    gallery: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """多正样本对称 InfoNCE：EEG->文本 与 文本->EEG 两个方向取平均。

    默认 ``gallery=None``：EEG->文本方向的候选就是本 batch 的 ``text``，
    与旧行为一致。

    给定 ``gallery`` 与 ``targets`` 时，EEG->文本方向改用更大的候选池
    （例如全部 252 句），``targets[i]`` 是第 i 条 EEG 在 ``gallery`` 中的列号；
    文本->EEG 方向仍只在本 batch 内计算。这样训练与验证的候选规模可以一致，
    同时验证句的文本只作为负样本、不携带任何 EEG。
    """
    text_e2t = text if gallery is None else gallery
    logits_e2t = eeg @ text_e2t.t() / temperature

    if targets is None:
        if groups is None:
            groups = torch.arange(logits_e2t.size(0), device=logits_e2t.device)
        positive_e2t = (groups[:, None] == groups[None, :]).to(logits_e2t.dtype)
    else:
        if targets.ndim != 1 or targets.numel() != logits_e2t.size(0):
            raise ValueError(
                f"targets 形状 {tuple(targets.shape)} 与 EEG 行数 {logits_e2t.size(0)} 不匹配"
            )
        positive_e2t = torch.zeros_like(logits_e2t)
        positive_e2t.scatter_(1, targets[:, None], 1.0)

    logits_t2e = text @ eeg.t() / temperature
    if groups is None:
        groups = torch.arange(logits_t2e.size(0), device=logits_t2e.device)
    positive_t2e = (groups[:, None] == groups[None, :]).to(logits_t2e.dtype)

    return 0.5 * (
        _multi_positive_ce(logits_e2t, positive_e2t)
        + _multi_positive_ce(logits_t2e, positive_t2e)
    )
