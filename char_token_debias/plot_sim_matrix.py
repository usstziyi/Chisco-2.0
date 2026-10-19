"""画 ``char_token_debias`` checkpoint 在预测集上的 EEG 句嵌入自相似矩阵。

与 ``char_token/plot_sim_matrix.py`` 的区别
-------------------------------------------
1. 默认按 **4 天** 重建预测集，因此同一句会贡献 4 条 trial；``--order sentence``
   会把它们排在一起，矩阵对角线附近应出现 **4×4 的亮点**（同句跨天一致性）；
2. 若 checkpoint 是 ``--debias center`` / ``ea+center`` 训练出来的，部署时默认
   沿用 checkpoint 里保存的块均值 ``μ_g``（``--apply-debias auto``）；
3. 额外输出"每个 run 一张"的图：该 run 跨 4 天的全部 trial，按句子分组。
   这是排除 run 块混淆后最直接的检验——同句 4 个点是否互相靠近。

诊断口径
--------
- **同句跨天**：``cos(e_i, e_j)``，i/j 同句不同天；越高越好；
- **同块异句**：同一 ``(day, run)``、不同句；这是"只知道 run"时唯一能达到的水平；
- **跨块**：不同 ``(day, run)``；理想情况下应接近 0（去偏后）。

用法::

    uv run python char_token_debias/plot_sim_matrix.py
    uv run python char_token_debias/plot_sim_matrix.py --ckpt outputs/char_token_debias/xxx/sub-01_best.pt
    uv run python char_token_debias/plot_sim_matrix.py --order raw --center-by day-run
    uv run python char_token_debias/plot_sim_matrix.py --start 0 --end 240   # 放大前 60 句
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_debias.dataset import CharTokenEEGDataset, char_token_collate, group_key
from char_token_debias.model import CharTokenEEGEncoder
from char_token_debias.train import split_by_sentence

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "outputs" / "char_token_debias"


def parse_args():
    parser = argparse.ArgumentParser(description="在预测集上画 EEG 句嵌入自相似矩阵。")
    parser.add_argument("--ckpt", default=None, help="checkpoint 路径；默认取最新")
    parser.add_argument("--subject", default=None, help="被试；默认从 checkpoint 推断")
    parser.add_argument("--day", type=int, nargs="+", default=None, help="默认沿用训练参数")
    parser.add_argument("--order", default="sentence", choices=["sentence", "raw"])
    parser.add_argument(
        "--center-by", default="none", choices=["none", "day-run", "run"],
        help="事后中心化：day-run 去掉当天当段偏移；run 去掉内容块公共方向",
    )
    parser.add_argument(
        "--apply-debias", default="auto", choices=["auto", "on", "off"],
        help="是否使用 checkpoint 里的块均值 μ_g（auto=有就用）",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None, help="默认存到 checkpoint 同级目录")
    return parser.parse_args()


def find_ckpt(ckpt: str | None, subject: str | None) -> Path:
    if ckpt is not None:
        path = Path(ckpt)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        return path

    pattern = f"*/{subject}_best.pt" if subject else "*/*_best.pt"
    candidates = sorted(
        DEFAULT_CKPT_ROOT.glob(pattern), key=lambda p: p.parent.name, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {DEFAULT_CKPT_ROOT} 下未找到 {pattern}，请用 --ckpt 显式指定"
        )
    return candidates[0]


def subject_from_ckpt(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_best")] if stem.endswith("_best") else stem


@torch.no_grad()
def encode(model, loader, device, debiaser_state, group_vocab) -> tuple[np.ndarray, list, list]:
    """编码预测集，返回 ``(embeddings, sentences, groups)``。"""
    model.eval()
    parts, sentences, groups = [], [], []
    for tokens, mask, batch_sentences, batch_groups in loader:
        raw = model(tokens.to(device), mask.to(device))
        parts.append(raw)
        sentences.extend(batch_sentences)
        groups.extend(batch_groups)
    emb = torch.cat(parts, dim=0)

    if debiaser_state is not None:
        means = debiaser_state["means"].to(emb.device)
        idx = torch.tensor(
            [group_vocab[group_key(d, r)] for d, r in groups], device=emb.device
        )
        emb = torch.nn.functional.normalize(emb - means[idx], p=2, dim=-1)

    return emb.float().cpu().numpy(), sentences, groups


def group_order(sentences) -> list[int]:
    """按句子分组：同句 trial 相邻，组内保持原顺序（day 升序）。"""
    groups: dict[str, list[int]] = {}
    for pos, sentence in enumerate[Any](sentences):
        groups.setdefault(sentence, []).append(pos)
    return [pos for members in groups.values() for pos in members]


def center_embeddings(embeddings: np.ndarray, groups) -> np.ndarray:
    """按 ``groups`` 分组减质心，再重新 L2 归一化。"""
    groups = np.asarray(groups, dtype=object)
    centered = embeddings.copy()
    for group in np.unique(groups):
        mask = groups == group
        centered[mask] = centered[mask] - centered[mask].mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12
    return centered / norm


def similarity_stats(sim, sentences, groups, title: str) -> None:
    """打印同句跨天 / 同块异句 / 跨块的余弦对照。"""
    n = sim.shape[0]
    sent = np.array(sentences, dtype=object)
    day = np.array([g[0] for g in groups])
    block = np.array([f"{d}-{r}" for d, r in groups])

    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    sent_eq = sent[:, None] == sent[None, :]
    day_eq = day[:, None] == day[None, :]
    block_eq = block[:, None] == block[None, :]

    def mean_of(mask) -> float:
        values = sim[upper & mask]
        return float(values.mean()) if values.size else float("nan")

    same_xday = mean_of(sent_eq & ~day_eq)
    same_block_diff = mean_of(block_eq & ~sent_eq)
    cross_block = mean_of(~block_eq)

    print(
        f"[Sim] {title}\n"
        f"      同句跨天   mean={same_xday:+.4f}\n"
        f"      同块异句   mean={same_block_diff:+.4f}\n"
        f"      跨块       mean={cross_block:+.4f}\n"
        f"      落差（同句 - 同块异句）={same_xday - same_block_diff:+.4f}"
    )


def plot_matrix(
    out_dir, matrix, filename, title, vmin, vmax=1.0,
    xlabel="trial", ylabel="trial", cbar_label="余弦相似度（EEG 句嵌入）",
) -> Path | None:
    n = matrix.shape[0]
    if n == 0:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size = 8.0 if n <= 80 else min(18.0, 8.0 + (n - 80) * 0.04)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))

    sns.heatmap(
        matrix, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
        square=True, linewidths=0, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_order_csv(out_dir, subject, tag, sentences, groups, select) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{subject}_val_simmatrix_{tag}_order.csv"

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["plot_index", "group_id", "slot_in_group", "day", "run", "sentence"])
        group_id, slot, prev = -1, 0, None
        for plot_index, pos in enumerate(select):
            sentence = str(sentences[pos])
            if sentence != prev:
                group_id += 1
                slot = 0
                prev = sentence
            writer.writerow(
                [plot_index, group_id, slot, groups[pos][0], groups[pos][1], sentence]
            )
            slot += 1
    return path


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = find_ckpt(args.ckpt, args.subject)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args") or {}

    subject = args.subject or subject_from_ckpt(ckpt_path)
    days = args.day if args.day is not None else (train_args.get("day") or [1, 2, 3, 4])
    event = train_args.get("event") or "inner"
    sample = int(train_args.get("sample", 250))
    char_duration = float(train_args.get("char_duration", 0.4))
    val_ratio = float(train_args.get("val_ratio", 0.2))
    seed = int(train_args.get("seed", 0))
    debias_mode = train_args.get("debias", "none")
    use_ea = "ea" in debias_mode

    n_chans = int(ckpt["n_chans"])
    token_samples = int(ckpt["token_samples"])
    embedding_dim = int(ckpt["embedding_dim"])

    print(
        f"[Ckpt] {ckpt_path}\n"
        f"       epoch={ckpt.get('epoch')} debias={debias_mode} "
        f"n_chans={n_chans} embedding_dim={embedding_dim}"
    )

    dataset = CharTokenEEGDataset(
        event=event, sample=sample, subject=subject, days=days,
        char_duration=char_duration, align="ea" if use_ea else "none",
        align_cache=(ckpt_path.parent / f"{subject}_ea_align.npz") if use_ea else None,
    )
    dataset[0]
    _, val_set, info = split_by_sentence(dataset, val_ratio, seed)

    days_label = ",".join(f"{d:02d}" for d in sorted(days))
    print(
        f"[Data] {subject} day-{days_label} align={'ea' if use_ea else 'none'} "
        f"预测集 {info['n_unique_val']} 句 / {len(val_set)} trial"
    )

    model = CharTokenEEGEncoder(
        n_chans=n_chans, token_samples=token_samples,
        d_model=int(train_args.get("d_model", 128)),
        n_heads=int(train_args.get("n_heads", 4)),
        n_layers=int(train_args.get("n_layers", 2)),
        embedding_dim=embedding_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    debiaser_state = ckpt.get("debiaser")
    group_vocab = ckpt.get("group_vocab")
    apply_debias = (
        args.apply_debias == "on"
        or (args.apply_debias == "auto" and debiaser_state is not None and group_vocab)
    )
    if args.apply_debias == "on" and not (debiaser_state and group_vocab):
        raise ValueError("该 checkpoint 没有保存块均值，无法 --apply-debias on")

    loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=char_token_collate, num_workers=0,
    )
    with torch.no_grad():
        emb, sentences, groups = encode(
            model, loader, device,
            debiaser_state if apply_debias else None, group_vocab,
        )
    n = len(sentences)
    print(
        f"[Enc] trial={n}，唯一句子={len(set(sentences))}，"
        f"块={len(set(groups))}，沿用 checkpoint 去偏={apply_debias}"
    )

    # ---------------- 事后中心化（可叠加在 checkpoint 去偏之上） ----------------
    tag = "bysentence" if args.order == "sentence" else "raw"
    if apply_debias:
        tag += "_ckptdebias"
    if args.center_by != "none":
        if args.center_by == "day-run":
            center_groups = [f"{d}-{r}" for d, r in groups]
            center_label = "按(day,run)减质心"
        else:
            center_groups = [r for _, r in groups]
            center_label = "按run号跨天减质心"
        emb = center_embeddings(emb, center_groups)
        tag += "_center" + args.center_by.replace("-", "")
        print(f"[Center] {center_label}")

    sim = emb @ emb.T
    similarity_stats(sim, sentences, groups, "原始顺序")

    # ---------------- 排序并截取 ----------------
    if args.order == "sentence":
        order = group_order(sentences)
        order_label = "按句子分组"
    else:
        order = list(range(n))
        order_label = "原始顺序"

    start = max(0, args.start)
    end = n if args.end is None else min(args.end, n)
    select = order[start:end]
    zoom_note = "" if (start, end) == (0, n) else f"（索引 {start}:{end}）"
    if (start, end) != (0, n):
        tag += f"_idx{start}-{end}"
    sub = sim[np.ix_(select, select)]

    out_dir = Path(args.out_dir) if args.out_dir is not None else ckpt_path.parent
    vmin = float(sim.min())

    title = (
        f"{subject} 预测集 EEG 句嵌入自相似矩阵（{order_label}）\n"
        f"N={len(select)}  days={days_label}  debias={debias_mode}"
        f"  val_ratio={val_ratio:g}  seed={seed}{zoom_note}"
    )
    path = plot_matrix(out_dir, sub, f"{subject}_val_simmatrix_{tag}.png", title, vmin)
    print(f"\n[Plot] {len(select)}×{len(select)}（{order_label}{zoom_note}）")
    print(f"[Output] {path}")

    # ---------------- 每个 run 单独一张（跨 4 天） ----------------
    run_ids = sorted({run for _, run in groups})
    for run_id in run_ids:
        sel_run = [p for p in order if groups[p][1] == run_id]
        n_sent = len({sentences[p] for p in sel_run})
        n_day = len({groups[p][0] for p in sel_run})
        title_run = (
            f"{subject} 预测集 EEG 自相似矩阵（{order_label}·run-{run_id}）\n"
            f"N={len(sel_run)}  句数={n_sent}  天数={n_day}  debias={debias_mode}"
        )
        path_run = plot_matrix(
            out_dir, sim[np.ix_(sel_run, sel_run)],
            f"{subject}_val_simmatrix_{tag}_run{run_id}.png", title_run, vmin,
        )
        print(f"[Plot] run-{run_id}: {len(sel_run)}×{len(sel_run)}（{n_sent} 句）")
        print(f"[Output] {path_run}")

    csv_path = save_order_csv(out_dir, subject, tag, sentences, groups, select)
    print(f"[Output] {csv_path}")

    counts = Counter(sentences)
    print(f"[Check] 每句 trial 数分布={dict(sorted(Counter(counts.values()).items()))}")


if __name__ == "__main__":
    main()
