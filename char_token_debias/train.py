"""多视图 + run 去偏的字符级 EEG–文本对比学习。

为什么另起一个实验
------------------
``char_token`` 的单天训练（``--day 1``）里，每句只有一条 trial，``pos/anch = 1``，
"同句多视图"机制等于没有启动。此时 InfoNCE 只需把 EEG 编成"哪个 run"就能把
32 路候选砍到 8 路（loss 从 ln32 掉到 ln8），实验和可视化都证实：embedding
≈ run 均值 + 无句子信息的残差。

本脚本按四步走：

1. **多视图**：``--day 1 2 3 4 --n-repeats 4``。同一句在 4 天各有一条 trial，
   互为正样本；模型才有"跨视图一致 = 保留、视图特异 = 丢弃"的判据。
2. **run 去偏**：``--debias``
   - ``none``：只开多视图（基线，用来隔离去偏的贡献）；
   - ``ea``：输入空间欧氏对齐，按 ``(day, run)`` 去掉逐通道偏移与协方差差异；
   - ``center``：embedding 空间按 ``(day, run)`` 减去 EMA 均值；
   - ``ea+center``：两者叠加。
3. **句长控制**（run/类别被去掉后，模型会退而使用句长）：
   - ``--length-negatives``：按句长分桶采样，一个 batch 内句长相同，
     句长在 softmax 里失去判别力；
   - ``--length-residual linear``：在线估计并去掉 embedding 中线性可预测句长的方向。
4. **分层指标**：``e2t_run@k``（同块）、``e2t_cat@k``（同块∩同类别）、
   ``e2t_match@k``（同块∩同类别∩同句长）。随机基线分别约 1/56、1/20、1/4。
   默认按 ``cos_gap_catlen``（同句跨天 − 同块同类同长异句）选模型，因为
   不带句长控制的 ``cos_gap`` 会被"类别 + 句长"混淆。

其余训练细节（LaBSE 冻结查表、按句子切分、多正样本 InfoNCE、余弦退火、
句子分组采样）与 ``char_token/train.py`` 保持一致。

用法::

    # 推荐主实验：多视图 + 输入对齐 + 输出中心化
    uv run python char_token_debias/train.py --day 1 2 3 4 --debias ea+center

    # 消融：只多视图 / 只 EA / 只中心化
    uv run python char_token_debias/train.py --day 1 2 3 4 --debias none
    uv run python char_token_debias/train.py --day 1 2 3 4 --debias ea
    uv run python char_token_debias/train.py --day 1 2 3 4 --debias center

    # 句长控制：同长负样本 / 线性残差化 / 两者叠加
    uv run python char_token_debias/train.py --day 1 2 3 4 --debias center --length-negatives
    uv run python char_token_debias/train.py --day 1 2 3 4 --debias center --length-residual linear
    uv run python char_token_debias/train.py --day 1 2 3 4 --debias center --length-negatives --length-residual linear

    # 快速冒烟
    uv run python char_token_debias/train.py --day 1 2 --n-sentences 8 --n-repeats 2 \
        --epochs 1 --d-model 64 --n-layers 1 --length-negatives --length-residual linear
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# 支持 `python char_token_debias/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_debias.dataset import CharTokenEEGDataset, char_token_collate, group_key
from char_token_debias.debias import GroupDebiaser, LengthResidualizer
from char_token_debias.model import CharTokenEEGEncoder
from char_token_debias.sampler import LengthBucketBatchSampler, SentenceGroupBatchSampler
from char_token_debias.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_debias"

EPOCH_LOG_FIELDS = [
    "epoch", "lr", "train_loss", "train_pos_per_anchor",
    "val_loss", "val_loss_run",
    "e2t@1", "e2t@5", "e2t_run@1", "e2t_run@5",
    "e2t_cat@1", "e2t_cat@5", "e2t_match@1", "e2t_match@5",
    "t2e@1", "t2e@5", "t2e_xday@1", "t2e_xday@5",
    "cos_sent_xday", "cos_same_block_diff_sent", "cos_diff_block", "cos_gap",
    "cos_gap_cat", "cos_gap_catlen",
    "n_val", "n_unique_sentences", "pos_per_anchor", "n_text_candidates",
    "n_block_candidates", "n_cat_candidates", "n_match_candidates",
    "is_best", "temperature", "debias", "length_negatives", "length_residual",
]

TRAIN_STEP_FIELDS = [
    "epoch", "step", "loss", "lr", "pos_per_anchor", "batch_size", "resid_norm",
]


# ----------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------
def append_csv_row(path: Path, fields: list, row: dict) -> None:
    """把一行追加到 CSV；文件不存在时先写表头。"""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def dump_args_json(args, out_dir: Path, filename: str = "args.json") -> Path:
    """把本次运行的所有命令行参数以 JSON 写到时间戳输出目录。"""
    path = Path(out_dir) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_group_vocab(meta) -> dict[tuple[int, int], int]:
    """``(day, run)`` -> 连续索引；训练与推理共用同一套编号。"""
    pairs = sorted({(int(d), int(r)) for d, r in zip(meta["day"], meta["run"])})
    return {pair: i for i, pair in enumerate(pairs)}


def split_by_sentence(dataset, val_ratio: float, seed: int):
    """按句子切分 train / val，返回 ``(train_subset, val_subset, info)``。

    以句子为单位而非 trial，避免同一句的不同视图同时出现在两侧。
    """
    sentences = dataset.meta["sentence"].tolist()
    unique = sorted(set(sentences))
    if len(unique) < 2:
        raise ValueError(f"只有 {len(unique)} 个唯一句子，无法切分")

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_ratio))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    val_sentences = set(shuffled[:n_val])

    val_idx = [i for i, s in enumerate(sentences) if s in val_sentences]
    train_idx = [i for i, s in enumerate(sentences) if s not in val_sentences]

    train_sentences = {sentences[i] for i in train_idx}
    info = {
        "n_unique_train": len(unique) - n_val,
        "n_unique_val": n_val,
        "sentence_overlap": len(train_sentences & val_sentences),
    }
    return Subset(dataset, train_idx), Subset(dataset, val_idx), info


def build_text_lookup(
    encoder, sentences, device, sentence_run: dict[str, int] | None = None, chunk: int = 64
):
    """把所有唯一句子编码成 embedding，返回 ``{sentence: (D,)}`` 查表。

    ``sentence_run`` 非空时按 **run 均值** 对文本做中心化：EEG 侧去掉 ``(day, run)``
    偏移、文本侧去掉 ``run`` 偏移后，两侧才处在同一"只剩句子残差"的空间里。不这样
    做的话，EEG 无法（也不该）预测被自己减掉的 run 公共分量，全局检索会被压低。
    """
    unique = sorted(set(sentences))
    vectors = []
    with torch.no_grad():
        for start in range(0, len(unique), chunk):
            vectors.append(encoder(unique[start : start + chunk]))
    table = torch.cat(vectors, dim=0).to(device)

    if sentence_run is not None:
        runs = torch.tensor([sentence_run[s] for s in unique], device=device)
        for run in torch.unique(runs):
            mask = runs == run
            table[mask] = table[mask] - table[mask].mean(dim=0, keepdim=True)
        table = torch.nn.functional.normalize(table, p=2, dim=-1)

    return dict(zip(unique, table)), int(table.shape[1])


def gather_text(sentences, lookup, device) -> torch.Tensor:
    """按句子取回文本 embedding，堆成 (B, D)。"""
    return torch.stack([lookup[s] for s in sentences]).to(device)


def sentence_group_ids(sentences, device) -> torch.Tensor:
    """同句 trial 得到同一个 id（多视图互为正样本）。"""
    mapping: dict[str, int] = {}
    ids = []
    for sentence in sentences:
        if sentence not in mapping:
            mapping[sentence] = len(mapping)
        ids.append(mapping[sentence])
    return torch.tensor(ids, dtype=torch.long, device=device)


def _multi_positive_ce(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """多正样本交叉熵：每个 anchor 对"所有正样本列"求平均的 -log softmax。"""
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * positive).sum(dim=1).div(positive.sum(dim=1)).mean()


def contrastive_loss(
    eeg: torch.Tensor,
    text: torch.Tensor,
    temperature: float,
    groups: torch.Tensor | None = None,
) -> torch.Tensor:
    """多正样本对称 InfoNCE（EEG->文本、文本->EEG 两方向取平均）。"""
    logits = eeg @ text.t() / temperature
    if groups is None:
        groups = torch.arange(logits.size(0), device=logits.device)
    positive = (groups[:, None] == groups[None, :]).to(logits.dtype)
    return 0.5 * (
        _multi_positive_ce(logits, positive) + _multi_positive_ce(logits.t(), positive)
    )


# ----------------------------------------------------------------------
# 训练 / 评测
# ----------------------------------------------------------------------
def train_one_epoch(
    model, debiaser, length_resid, loader, lookup, optimizer, device, temperature,
    epoch, step_log_path=None,
):
    """跑一个 epoch，返回 ``(平均 loss, 每个 anchor 的平均正样本数)``。"""
    model.train()
    total_loss, total_n, total_pos, total_resid = 0.0, 0, 0.0, 0.0

    for step, (tokens, mask, sentences, groups) in enumerate(loader, start=1):
        lr = optimizer.param_groups[0]["lr"]
        tokens = tokens.to(device)
        mask = mask.to(device)
        text = gather_text(sentences, lookup, device)
        groups_sent = sentence_group_ids(sentences, device)

        raw = model(tokens, mask)
        if debiaser is not None:
            group_index = debiaser.index(groups).to(device)
            eeg = debiaser(raw, group_index)
            resid = debiaser.residual_norm(raw, group_index)
        else:
            eeg, resid = raw, 0.0

        # 句长控制：token 数 = 句长（inner 为 n_chars+1，单调等价），用它在 batch 内
        # 估计并去掉"线性可预测句长"的方向。注意用的是去块均值后的表示。
        if length_resid is not None:
            lengths = mask.sum(dim=1).float()
            length_resid.update(eeg.detach(), lengths)
            eeg = length_resid(eeg)

        loss = contrastive_loss(eeg, text, temperature, groups_sent)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 用更新前的块均值算完 loss 后再更新 EMA，保证同一 batch 内一致
        if debiaser is not None:
            debiaser.update(raw, group_index)

        batch = len(sentences)
        pos = (groups_sent[:, None] == groups_sent[None, :]).sum().item()

        if step_log_path is not None:
            append_csv_row(step_log_path, TRAIN_STEP_FIELDS, {
                "epoch": epoch,
                "step": step,
                "loss": f"{loss.item():.6f}",
                "lr": f"{lr:.6f}",
                "pos_per_anchor": f"{pos / batch:.4f}",
                "batch_size": batch,
                "resid_norm": f"{resid:.4f}",
            })

        total_loss += loss.item() * batch
        total_n += batch
        total_pos += pos
        total_resid += resid * batch

    return (
        total_loss / max(total_n, 1),
        total_pos / max(total_n, 1),
        total_resid / max(total_n, 1),
    )


@torch.no_grad()
def evaluate(
    model, debiaser, length_resid, loader, lookup, device, temperature, val_meta
) -> dict:
    """在预测集上算全局 / within-run / within-category / 同长 检索与诊断余弦。

    ``val_meta`` 必须与 DataLoader 的顺序一一对应（``shuffle=False``），用于取
    ``category_name`` 与 ``n_chars``。类别缺失时相关指标返回 nan。
    """
    model.eval()

    parts, sentences, groups = [], [], []
    for tokens, mask, batch_sentences, batch_groups in loader:
        raw = model(tokens.to(device), mask.to(device))
        if debiaser is not None:
            raw = debiaser(raw, debiaser.index(batch_groups).to(device))
        if length_resid is not None:
            raw = length_resid(raw)
        parts.append(raw)
        sentences.extend(batch_sentences)
        groups.extend(batch_groups)

    eeg = torch.cat(parts, dim=0)
    n = eeg.size(0)
    if len(val_meta) != n:
        raise ValueError(f"val_meta 行数 {len(val_meta)} 与评测样本数 {n} 不一致")

    uniq = sorted(set(sentences))
    sent_to_col = {s: i for i, s in enumerate(uniq)}
    target = torch.tensor([sent_to_col[s] for s in sentences], device=device)
    text_all = gather_text(uniq, lookup, device)
    text = text_all[target]
    sent_groups = sentence_group_ids(sentences, device)

    metrics = {
        "val_loss": float(contrastive_loss(eeg, text, temperature, sent_groups)),
        "n_val": n,
        "n_unique_sentences": len(uniq),
        "pos_per_anchor": float(
            (sent_groups[:, None] == sent_groups[None, :]).sum(1).float().mean()
        ),
        "n_text_candidates": len(uniq),
    }

    def topk(logits: torch.Tensor, tgt: torch.Tensor, k: int) -> float:
        hit = logits.argsort(dim=1, descending=True).eq(tgt[:, None])
        return float(hit[:, :k].any(dim=1).float().mean())

    # ---- 全局 eeg -> text（候选 = 去重后的唯一句子） ----
    metrics["e2t@1"] = topk(eeg @ text_all.t(), target, 1)
    metrics["e2t@5"] = topk(eeg @ text_all.t(), target, 5)

    # ---- 数组：块 / 类别 / 句长 ----
    sent_arr = np.array(sentences, dtype=object)
    day_arr = np.array([g[0] for g in groups])
    run_arr = np.array([g[1] for g in groups])
    block_arr = np.array([f"{d}-{r}" for d, r in groups], dtype=object)
    len_arr = val_meta["n_chars"].to_numpy()
    cat_arr = (
        val_meta["category_name"].astype(str).to_numpy()
        if "category_name" in val_meta.columns else None
    )

    # ---- within-(day, run) eeg -> text ----
    hit1_sum = hit5_sum = loss_sum = 0.0
    weight = 0
    cand_sizes = []
    for block in sorted(set(block_arr)):
        idx = np.where(block_arr == block)[0]
        if len(idx) == 0:
            continue
        idx_t = torch.as_tensor(idx, device=device)
        cand = sorted({sentences[i] for i in idx})
        cand_col = {s: i for i, s in enumerate(cand)}
        tgt = torch.tensor([cand_col[sentences[i]] for i in idx], device=device)
        logits = eeg[idx_t] @ gather_text(cand, lookup, device).t()

        hit1_sum += topk(logits, tgt, 1) * len(idx)
        hit5_sum += topk(logits, tgt, 5) * len(idx)
        logp = logits / temperature
        logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
        loss_sum += float(-logp[torch.arange(len(idx), device=device), tgt].mean()) * len(idx)
        weight += len(idx)
        cand_sizes.append(len(cand))

    metrics["e2t_run@1"] = hit1_sum / max(weight, 1)
    metrics["e2t_run@5"] = hit5_sum / max(weight, 1)
    metrics["val_loss_run"] = loss_sum / max(weight, 1)
    metrics["n_block_candidates"] = float(np.mean(cand_sizes)) if cand_sizes else 0.0

    # ---- within-(block, category)：排除 run + 类别 ----
    cat_h1 = cat_h5 = cat_w = 0.0
    cat_sizes = []
    if cat_arr is not None:
        for block in sorted(set(block_arr)):
            m = np.where(block_arr == block)[0]
            for c in sorted(set(cat_arr[m])):
                idx = m[cat_arr[m] == c]
                cand = sorted({sentences[i] for i in idx})
                if len(cand) < 5:
                    continue
                col = {s: i for i, s in enumerate(cand)}
                tgt = torch.tensor([col[sentences[i]] for i in idx], device=device)
                logits = eeg[torch.as_tensor(idx, device=device)] @ gather_text(
                    cand, lookup, device
                ).t()
                cat_h1 += topk(logits, tgt, 1) * len(idx)
                cat_h5 += topk(logits, tgt, 5) * len(idx)
                cat_w += len(idx)
                cat_sizes.append(len(cand))
    metrics["e2t_cat@1"] = cat_h1 / cat_w if cat_w else float("nan")
    metrics["e2t_cat@5"] = cat_h5 / cat_w if cat_w else float("nan")
    metrics["n_cat_candidates"] = float(np.mean(cat_sizes)) if cat_sizes else float("nan")

    # ---- within-(block, category, length)：再排除句长 ----
    match_h1 = match_h5 = match_w = 0.0
    match_sizes = []
    if cat_arr is not None:
        for block in sorted(set(block_arr)):
            m = np.where(block_arr == block)[0]
            for c in sorted(set(cat_arr[m])):
                base = m[cat_arr[m] == c]
                for L in sorted(set(len_arr[base])):
                    idx = base[len_arr[base] == L]
                    cand = sorted({sentences[i] for i in idx})
                    if len(cand) < 2:
                        continue
                    col = {s: i for i, s in enumerate(cand)}
                    tgt = torch.tensor([col[sentences[i]] for i in idx], device=device)
                    logits = eeg[torch.as_tensor(idx, device=device)] @ gather_text(
                        cand, lookup, device
                    ).t()
                    match_h1 += topk(logits, tgt, 1) * len(idx)
                    match_h5 += topk(logits, tgt, 5) * len(idx)
                    match_w += len(idx)
                    match_sizes.append(len(cand))
    metrics["e2t_match@1"] = match_h1 / match_w if match_w else float("nan")
    metrics["e2t_match@5"] = match_h5 / match_w if match_w else float("nan")
    metrics["n_match_candidates"] = (
        float(np.mean(match_sizes)) if match_sizes else float("nan")
    )

    # ---- 全局 text -> eeg（命中 = 前 k 名出现任意同句） ----
    uniq_arr = np.array(uniq, dtype=object)
    logits_te = text_all @ eeg.t()
    ranked_te = logits_te.argsort(dim=1, descending=True)
    row_ids = torch.arange(len(uniq), device=device)[:, None]

    pos_any = torch.from_numpy(sent_arr[None, :] == uniq_arr[:, None]).to(device)
    hit_te = pos_any[row_ids, ranked_te]
    metrics["t2e@1"] = float(hit_te[:, :1].any(dim=1).float().mean())
    metrics["t2e@5"] = float(hit_te[:, :5].any(dim=1).float().mean())

    # ---- 跨天 text -> eeg（只把"同句且不同天"算作命中） ----
    x_hit1 = x_hit5 = 0.0
    n_query = 0
    for day in sorted(set(day_arr)):
        pos = torch.from_numpy(
            (sent_arr[None, :] == uniq_arr[:, None]) & (day_arr[None, :] != day)
        ).to(device)
        hit = pos[row_ids, ranked_te]
        x_hit1 += float(hit[:, :1].any(1).float().mean())
        x_hit5 += float(hit[:, :5].any(1).float().mean())
        n_query += 1
    metrics["t2e_xday@1"] = x_hit1 / max(n_query, 1)
    metrics["t2e_xday@5"] = x_hit5 / max(n_query, 1)

    # ---- 诊断余弦 ----
    sim = (eeg @ eeg.t()).float().cpu().numpy()
    day_eq = day_arr[:, None] == day_arr[None, :]
    sent_eq = sent_arr[:, None] == sent_arr[None, :]
    block_eq = block_arr[:, None] == block_arr[None, :]
    upper = np.triu(np.ones_like(sim, dtype=bool), 1)
    same_sent_xday = (sent_eq & ~day_eq)[upper]

    def mean_of(mask) -> float:
        values = sim[upper][mask]
        return float(values.mean()) if values.size else float("nan")

    metrics["cos_sent_xday"] = mean_of(same_sent_xday)
    metrics["cos_same_block_diff_sent"] = mean_of(block_eq[upper] & ~sent_eq[upper])
    metrics["cos_diff_block"] = mean_of(~block_eq[upper])
    metrics["cos_gap"] = (
        metrics["cos_sent_xday"] - metrics["cos_same_block_diff_sent"]
    )

    if cat_arr is not None:
        cat_eq = cat_arr[:, None] == cat_arr[None, :]
        len_eq = len_arr[:, None] == len_arr[None, :]
        metrics["cos_gap_cat"] = metrics["cos_sent_xday"] - mean_of(
            block_eq[upper] & ~sent_eq[upper] & cat_eq[upper]
        )
        metrics["cos_gap_catlen"] = metrics["cos_sent_xday"] - mean_of(
            block_eq[upper] & ~sent_eq[upper] & cat_eq[upper] & len_eq[upper]
        )
    else:
        metrics["cos_gap_cat"] = float("nan")
        metrics["cos_gap_catlen"] = float("nan")

    return metrics


# ----------------------------------------------------------------------
# 单个被试
# ----------------------------------------------------------------------
def run_subject(args, subject, out_dir, text_encoder, device):
    """对单个被试跑完整的训练 + 评测，返回 ``(best_score, best_metrics)``。"""
    set_seed(args.seed)

    print(f"\n{'=' * 92}\n[Subject] {subject}  debias={args.debias}\n{'=' * 92}")

    use_ea = "ea" in args.debias
    use_center = "center" in args.debias

    # ---------------- 数据 ----------------
    dataset = CharTokenEEGDataset(
        event=args.event,
        sample=args.sample,
        subject=subject,
        days=args.day,
        char_duration=args.char_duration,
        align="ea" if use_ea else "none",
        ea_shrinkage=args.ea_shrinkage,
        align_cache=(out_dir / f"{subject}_ea_align.npz") if use_ea else None,
    )
    dataset[0]  # 触发一次读取，确定 n_chans
    train_set, val_set, info = split_by_sentence(dataset, args.val_ratio, args.seed)

    days_label = ",".join(f"{d:02d}" for d in sorted(args.day))
    print(
        f"[Data] {subject} day-{days_label} event={args.event} align={dataset.align} "
        f"token_samples={dataset.token_samples} n_chans={dataset.n_chans}\n"
        f"       句子 {len(dataset.meta['sentence'].unique())} 条 -> "
        f"train {info['n_unique_train']} / val {info['n_unique_val']} "
        f"(trial {len(train_set)} / {len(val_set)})，句子重叠={info['sentence_overlap']}"
    )

    # ---------------- 分组 batch sampler ----------------
    all_sentences = dataset.meta["sentence"].tolist()
    train_sentences = [all_sentences[i] for i in train_set.indices]
    max_repeats = max(Counter(train_sentences).values())

    target_batch = args.n_sentences * args.n_repeats
    n_repeats = min(args.n_repeats, max_repeats)
    n_sentences = max(1, target_batch // n_repeats)
    if n_repeats != args.n_repeats:
        print(
            f"[Sampler] 单句最多只有 {max_repeats} 条 trial，n_repeats "
            f"{args.n_repeats} -> {n_repeats}，n_sentences 相应调整为 {n_sentences}"
        )

    if args.length_negatives:
        train_lengths = [int(dataset.meta["n_chars"].iloc[i]) for i in train_set.indices]
        train_sampler = LengthBucketBatchSampler(
            train_sentences, train_lengths,
            n_sentences=n_sentences, n_repeats=n_repeats, seed=args.seed,
        )
        sizes = sorted((len(p) for p in train_sampler.buckets.values()), reverse=True)
        print(
            f"[Sampler] 句长分桶：{len(train_sampler.buckets)} 桶，句子数分布={sizes}；"
            f"每 batch ≤ {n_sentences} 句 × {n_repeats} 视图；每轮 {len(train_sampler)} 个 batch"
        )
    else:
        train_sampler = SentenceGroupBatchSampler(
            train_sentences, n_sentences=n_sentences, n_repeats=n_repeats, seed=args.seed
        )
        print(
            f"[Sampler] 每 batch {train_sampler.n_sentences} 句 × "
            f"{train_sampler.n_repeats} 视图 = {train_sampler.batch_size} 条；每轮 "
            f"{len(train_sampler)} 个 batch"
        )

    train_loader = DataLoader(
        train_set, batch_sampler=train_sampler,
        collate_fn=char_token_collate, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=char_token_collate, num_workers=args.num_workers,
    )

    # ---------------- 文本 embedding（冻结，一次性算好） ----------------
    # 本数据里同一句每天都在同一个 run，所以 sentence -> run 是唯一映射；
    # 若将来数据不是这样，这里会直接报错而不是悄悄取错分组。
    sentence_run: dict[str, int] = {}
    for sentence, run in zip(dataset.meta["sentence"], dataset.meta["run"]):
        sentence = str(sentence)
        run = int(run)
        if sentence_run.setdefault(sentence, run) != run:
            raise ValueError(f"句子 {sentence!r} 出现在多个 run 中，无法做 run 级文本去偏")

    lookup, embedding_dim = build_text_lookup(
        text_encoder,
        dataset.meta["sentence"].tolist(),
        device,
        sentence_run=sentence_run if use_center else None,
    )
    print(
        f"[Text] 已缓存 {len(lookup)} 条句子，embedding_dim={embedding_dim}，"
        f"中心化={'是（按 run 均值）' if use_center else '否'}"
    )

    # ---------------- 模型 ----------------
    model = CharTokenEEGEncoder(
        n_chans=dataset.n_chans,
        token_samples=dataset.token_samples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        embedding_dim=embedding_dim,
    ).to(device)

    debiaser = None
    if use_center:
        group_vocab = build_group_vocab(dataset.meta)
        debiaser = GroupDebiaser(
            group_vocab, embedding_dim, momentum=args.debias_momentum
        ).to(device)
        print(
            f"[Debias] embedding 中心化：{len(group_vocab)} 个 (day,run) 块，"
            f"momentum={args.debias_momentum}"
        )

    length_resid = None
    if args.length_residual == "linear":
        length_resid = LengthResidualizer(
            embedding_dim, momentum=args.length_momentum
        ).to(device)
        print(f"[Length] 线性句长残差化，momentum={args.length_momentum}")

    # 与 val_loader 的 shuffle=False 顺序一一对应
    val_meta = dataset.meta.iloc[val_set.indices].reset_index(drop=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {type(model).__name__} 参数量={n_params:,} "
        f"d_model={model.d_model} layers={len(model.transformer.layers)}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine" else None
    )

    # ---------------- 训练 ----------------
    ckpt_path = out_dir / f"{subject}_best.pt"
    train_epochs_path = out_dir / f"{subject}_train_epochs.csv"
    train_steps_path = out_dir / f"{subject}_train_steps.csv"

    lower_is_better = args.select_by.startswith("val_loss")
    best_score = float("inf") if lower_is_better else -float("inf")
    best_metrics = None

    header = (
        f"{'epoch':>5} {'train':>8} {'val_run':>8} {'e2t_run@1':>9} "
        f"{'e2t_cat@1':>9} {'e2t_match@1':>11} {'gap_cl':>7}"
    )
    print(f"\n{header}  (best by {args.select_by})")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_pos, train_resid = train_one_epoch(
            model, debiaser, length_resid, train_loader, lookup, optimizer, device,
            args.temperature, epoch=epoch, step_log_path=train_steps_path,
        )
        if scheduler is not None:
            scheduler.step()

        metrics = evaluate(
            model, debiaser, length_resid, val_loader, lookup, device,
            args.temperature, val_meta,
        )
        metrics["resid_norm"] = train_resid

        score = metrics[args.select_by]
        improved = score < best_score if lower_is_better else score > best_score
        flag = ""
        if improved:
            best_score = score
            best_metrics = dict(metrics)
            flag = " *"
            torch.save(
                {
                    "model": model.state_dict(),
                    "debiaser": debiaser.state_dict() if debiaser is not None else None,
                    "group_vocab": (
                        {group_key(d, r): i for (d, r), i in build_group_vocab(dataset.meta).items()}
                        if debiaser is not None else None
                    ),
                    "length_residualizer": (
                        length_resid.state_dict() if length_resid is not None else None
                    ),
                    "args": vars(args),
                    "epoch": epoch,
                    "n_chans": dataset.n_chans,
                    "token_samples": dataset.token_samples,
                    "embedding_dim": embedding_dim,
                    "metrics": metrics,
                },
                ckpt_path,
            )

        print(
            f"{epoch:>5} {train_loss:>8.4f} {metrics['val_loss_run']:>8.4f} "
            f"{metrics['e2t_run@1']:>9.4f} {metrics['e2t_cat@1']:>9.4f} "
            f"{metrics['e2t_match@1']:>11.4f} {metrics['cos_gap_catlen']:>+7.4f}{flag}"
        )

        append_csv_row(train_epochs_path, EPOCH_LOG_FIELDS, {
            "epoch": epoch,
            "lr": f"{epoch_lr:.6f}",
            "train_loss": f"{train_loss:.6f}",
            "train_pos_per_anchor": f"{train_pos:.6f}",
            "val_loss": f"{metrics['val_loss']:.6f}",
            "val_loss_run": f"{metrics['val_loss_run']:.6f}",
            "e2t@1": f"{metrics['e2t@1']:.6f}",
            "e2t@5": f"{metrics['e2t@5']:.6f}",
            "e2t_run@1": f"{metrics['e2t_run@1']:.6f}",
            "e2t_run@5": f"{metrics['e2t_run@5']:.6f}",
            "e2t_cat@1": f"{metrics['e2t_cat@1']:.6f}",
            "e2t_cat@5": f"{metrics['e2t_cat@5']:.6f}",
            "e2t_match@1": f"{metrics['e2t_match@1']:.6f}",
            "e2t_match@5": f"{metrics['e2t_match@5']:.6f}",
            "t2e@1": f"{metrics['t2e@1']:.6f}",
            "t2e@5": f"{metrics['t2e@5']:.6f}",
            "t2e_xday@1": f"{metrics['t2e_xday@1']:.6f}",
            "t2e_xday@5": f"{metrics['t2e_xday@5']:.6f}",
            "cos_sent_xday": f"{metrics['cos_sent_xday']:+.6f}",
            "cos_same_block_diff_sent": f"{metrics['cos_same_block_diff_sent']:+.6f}",
            "cos_diff_block": f"{metrics['cos_diff_block']:+.6f}",
            "cos_gap": f"{metrics['cos_gap']:+.6f}",
            "cos_gap_cat": f"{metrics['cos_gap_cat']:+.6f}",
            "cos_gap_catlen": f"{metrics['cos_gap_catlen']:+.6f}",
            "n_val": metrics["n_val"],
            "n_unique_sentences": metrics["n_unique_sentences"],
            "pos_per_anchor": f"{metrics['pos_per_anchor']:.4f}",
            "n_text_candidates": metrics["n_text_candidates"],
            "n_block_candidates": f"{metrics['n_block_candidates']:.1f}",
            "n_cat_candidates": f"{metrics['n_cat_candidates']:.1f}",
            "n_match_candidates": f"{metrics['n_match_candidates']:.1f}",
            "is_best": int(improved),
            "temperature": args.temperature,
            "debias": args.debias,
            "length_negatives": int(args.length_negatives),
            "length_residual": args.length_residual,
        })

    print(
        f"\n[Done] best {args.select_by} = {best_score:.4f}\n"
        f"       同句跨天 cos={best_metrics['cos_sent_xday']:+.4f}  "
        f"控类别 A-B={best_metrics['cos_gap_cat']:+.4f}  "
        f"控类别+句长 A-B={best_metrics['cos_gap_catlen']:+.4f}\n"
        f"       块内 e2t@1/5={best_metrics['e2t_run@1']:.4f}/"
        f"{best_metrics['e2t_run@5']:.4f}（候选≈{best_metrics['n_block_candidates']:.0f}）\n"
        f"       同类 e2t@1/5={best_metrics['e2t_cat@1']:.4f}/"
        f"{best_metrics['e2t_cat@5']:.4f}（候选≈{best_metrics['n_cat_candidates']:.0f}）\n"
        f"       同类同长 e2t@1={best_metrics['e2t_match@1']:.4f}"
        f"（候选≈{best_metrics['n_match_candidates']:.0f}，随机≈1/候选；"
        f"@5 在候选不足 5 时会饱和，见 CSV）\n"
        f"       checkpoint: {ckpt_path}\n"
        f"       epochs:     {train_epochs_path}"
    )
    return best_score, best_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="多视图 + run 去偏的 EEG–文本对比学习。"
    )
    parser.add_argument("--subject", nargs="+", default=["sub-01"], help="被试，可多个")
    parser.add_argument(
        "--day", type=int, nargs="+", default=[1, 2, 3, 4],
        help="day 编号；多天才有同句多视图（默认 1 2 3 4）",
    )
    parser.add_argument("--event", default="inner", choices=["inner", "reading"])
    parser.add_argument("--sample", type=int, default=250)
    parser.add_argument("--char-duration", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--n-sentences", type=int, default=32, help="每 batch 不同句子数")
    parser.add_argument("--n-repeats", type=int, default=4, help="每句每 batch 的视图数")
    parser.add_argument("--batch-size", type=int, default=64, help="验证集 batch 大小")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"])
    parser.add_argument(
        "--select-by", default="cos_gap_catlen",
        choices=[
            "cos_gap_catlen", "cos_gap_cat", "cos_gap",
            "val_loss", "val_loss_run",
            "e2t@1", "e2t_run@1", "e2t_run@5", "e2t_cat@1", "e2t_match@1", "t2e@5",
        ],
        help=(
            "选 best checkpoint 的指标（默认 cos_gap_catlen = 同句跨天余弦 - 同块同类同长异句余弦）。"
            "cos_gap 会被类别/句长混淆；控类别与句长后的 A-B 才是'句子身份'的干净度量。"
        ),
    )
    parser.add_argument(
        "--length-negatives", action="store_true",
        help="按句长分桶采样：一个 batch 内句长相同，句长在 softmax 中失去判别力",
    )
    parser.add_argument(
        "--length-residual", default="none", choices=["none", "linear"],
        help="linear 表示在线估计并去掉 embedding 中线性可预测句长的方向",
    )
    parser.add_argument("--length-momentum", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--debias", default="ea+center", choices=["none", "ea", "center", "ea+center"],
        help="run 去偏方式（默认 ea+center）",
    )
    parser.add_argument("--debias-momentum", type=float, default=0.9)
    parser.add_argument("--ea-shrinkage", type=float, default=0.05)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"[Config] device={device} subjects={args.subject} days={args.day} "
        f"debias={args.debias} length_neg={args.length_negatives} "
        f"length_resid={args.length_residual} temperature={args.temperature} "
        f"batch={args.n_sentences}句×{args.n_repeats}视图 epochs={args.epochs} "
        f"lr={args.lr} select={args.select_by} seed={args.seed}"
    )

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    print("[Text] 加载 LaBSE（冻结） ...")
    text_encoder = LaBSETextEncoder()

    summary = {}
    for subject in args.subject:
        best_score, best_metrics = run_subject(
            args, subject, out_dir, text_encoder, device
        )
        summary[subject] = {"best": best_score, **best_metrics}

    if len(summary) > 1:
        print(f"\n{'=' * 92}\n[Summary] 被试对照（best by {args.select_by}）\n{'=' * 92}")
        print(
            f"{'subject':>8} {'e2t_run@1':>10} {'e2t_cat@1':>10} "
            f"{'e2t_match@1':>12} {'gap_catlen':>11}"
        )
        for subject, m in summary.items():
            print(
                f"{subject:>8} {m['e2t_run@1']:>10.4f} {m['e2t_cat@1']:>10.4f} "
                f"{m['e2t_match@1']:>12.4f} {m['cos_gap_catlen']:>+11.3f}"
            )
    print(f"\n[Output] {out_dir}")


if __name__ == "__main__":
    main()
