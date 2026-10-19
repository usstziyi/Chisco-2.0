"""用 ``order=sentence`` 的 16×16 同句块，拆开「重复一致性」与「时段结构」。

思路
----
全部 16 个 run 拼起来共 4032 条 trial，每句恰好 16 条（4 天 × 4 run）。
按 ``order=sentence`` 重排后，EEG×EEG 矩阵的对角线上是 252 个 16×16 小块：

- **块内亮、块外暗** -> 同句的 16 次重复彼此相像，即重复 trial 一致性；
- **块外仍有大片亮/暗区域** -> run 内慢漂移（时段结构）仍在，未被同句块吸收。

本脚本同时给出两类指标，避免只看图下结论：

1. 重复一致性
   - 同句块内均值 vs 最近邻句块均值（块对比）；
   - 16×16 块同质性 ICC = 块间方差 /（块间 + 块内方差）；
   - kNN 同句命中率（最近邻是否同句），随机基线 ``15/4031``。
2. 时段结构（只看异句对，避免同句块的高相似度稀释）
   - run 内 |Δtrial| 的 lag 曲线；
   - 最近异句邻居落在同 run 的比例（随机基线约 ``251/4031``）；
   - 按 (同 run × lag 桶 / 同 day 异 run / 异 day) 分组的均值、RMS 与方差解释 R²。
3. 单 run 252×252 矩阵（去对角）的 SVD 第一分量能量占比，以及它与 trial_index 的相关，
   直接量化原图里那种整块亮/暗的平滑分量。

产出（默认写到 checkpoint 同级 ``analysis/``）::

    <subject>_blocks_sentence.png   4032×4032，order=sentence 全图
    <subject>_blocks_zoom.png       前若干句的 16×16 块纹理
    <subject>_blocks_panels.png     lag 曲线 / 分组均值 / 第一奇异向量
    <subject>_blocks_metrics.json   全部数值

用法::

    uv run python char_token_runbatch/check_sentence_blocks.py
    uv run python char_token_runbatch/check_sentence_blocks.py --run-day 5 --run-id 1
    uv run python char_token_runbatch/check_sentence_blocks.py --zoom-sentences 12

注意：本脚本固定用 ``order=sentence``，``--order`` 参数会被忽略。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_runbatch.plot_common import (
    add_common_args,
    build_dataset,
    encode_eeg,
    load_meta,
    load_model,
    load_text_embeddings,
    order_positions,
    plot_heatmap,
    resolve_config,
)


# ---------------------------------------------------------------- 参数


def parse_args():
    parser = argparse.ArgumentParser(
        description="用 order=sentence 的 16×16 同句块拆分重复一致性与时段结构。"
    )
    add_common_args(parser)
    parser.set_defaults(order="sentence")
    parser.add_argument("--run-day", type=int, default=5, help="单 run 分量分析取哪个 day（默认 5）")
    parser.add_argument("--run-id", type=int, default=1, help="单 run 分量分析取哪个 run（默认 1）")
    parser.add_argument("--zoom-sentences", type=int, default=8, help="放大图展示的句子数（默认 8）")
    parser.add_argument("--lag-max", type=int, default=60, help="lag 曲线最大 lag（默认 60）")
    return parser.parse_args()


# ---------------------------------------------------------------- 指标


def _cells_summary(sim, masks, grand_mean: float) -> tuple[list[dict], float]:
    """逐组算 n / mean / rms，并给出组均值对总方差的解释比例 R²。"""
    rows = []
    ss_between = 0.0
    for name, mask in masks:
        values = sim[mask]
        n = int(values.size)
        if n == 0:
            rows.append({"cell": name, "n": 0, "mean": float("nan"), "rms": float("nan")})
            continue
        mean = float(values.mean())
        rms = float(np.sqrt(np.square(values, dtype=np.float64).mean()))
        rows.append({"cell": name, "n": n, "mean": mean, "rms": rms})
        ss_between += n * (mean - grand_mean) ** 2
        del values
    return rows, ss_between


def structure_metrics(
    sim: np.ndarray,
    sent_ids: np.ndarray,
    run_ids: np.ndarray,
    day_ids: np.ndarray,
    trial_index: np.ndarray,
    lag_max: int,
) -> dict:
    n = sim.shape[0]
    n_sent = int(sent_ids.max()) + 1
    views = np.bincount(sent_ids)
    if not np.all(views == views[0]):
        raise ValueError(f"各句 trial 数不一致: {views.min()}~{views.max()}")
    v = int(views[0])

    off = ~np.eye(n, dtype=bool)
    same_sent = sent_ids[:, None] == sent_ids[None, :]

    within = sim[off & same_sent]
    other = sim[off & ~same_sent]
    repeat = {
        "views_per_sentence": v,
        "n_within_pairs": int(within.size),
        "within_mean": float(within.mean()),
        "within_std": float(within.std()),
        "outside_mean": float(other.mean()),
        "contrast": float(within.mean() - other.mean()),
    }

    # 同句块内 std / 块均值 -> ICC
    block_sums = np.zeros((n_sent, n_sent), dtype=np.float64)
    block_views = sim.reshape(n_sent, v, n_sent, v)
    block_sums = block_views.sum(axis=(1, 3), dtype=np.float64)
    within_pair_count = v * (v - 1)
    block = block_sums / within_pair_count  # 对角块 = 块内异 trial 对的均值

    within_std = []
    for g in range(n_sent):
        sub = block_views[g, :, g, :]
        m = ~np.eye(v, dtype=bool)
        within_std.append(float(sub[m].std()))
    block_diag = np.diag(block).copy()
    block_off = block.copy()
    np.fill_diagonal(block_off, -np.inf)
    nearest_other = block_off.max(axis=1)
    nearest_sent = block_off.argmax(axis=1)

    var_between = float(np.var(block_diag))
    var_within = float(np.mean(np.square(within_std)))
    repeat.update({
        "block_mean_median": float(np.median(block_diag)),
        "block_mean_min": float(block_diag.min()),
        "block_mean_max": float(block_diag.max()),
        "nearest_block_mean": float(nearest_other.mean()),
        "block_margin": float(block_diag.mean() - nearest_other.mean()),
        "within_block_std_mean": float(np.mean(within_std)),
        "between_block_var": var_between,
        "icc": float(var_between / (var_between + var_within)) if var_between + var_within > 0 else float("nan"),
    })

    # kNN 同句命中
    tmp = sim.copy()
    np.fill_diagonal(tmp, -np.inf)
    top1 = tmp.argmax(axis=1)
    part = np.argpartition(-tmp, 5, axis=1)[:, :6]
    part_vals = np.take_along_axis(tmp, part, axis=1)
    top5 = np.take_along_axis(part, np.argsort(-part_vals, axis=1), axis=1)[:, :5]
    repeat.update({
        "knn_top1_same_sentence": float((sent_ids[top1] == sent_ids).mean()),
        "knn_top5_same_sentence": float((sent_ids[top5] == sent_ids[:, None]).any(axis=1).mean()),
        "knn_top1_chance": float((v - 1) / (n - 1)),
    })

    # 最近「异句」邻居：看它落在哪个 run / 多近
    tmp[same_sent] = -np.inf
    nn = tmp.argmax(axis=1)
    del tmp
    nn_same_run = run_ids[nn] == run_ids
    nn_lag = np.abs(trial_index[nn] - trial_index)
    segment_neighbor = {
        "same_run_rate": float(nn_same_run.mean()),
        "same_run_chance": float((n / 16 - 1) / (n - 1)),
        "same_day_rate": float((day_ids[nn] == day_ids).mean()),
        "mean_lag_same_run": float(nn_lag[nn_same_run].mean()),
        "median_lag_same_run": float(np.median(nn_lag[nn_same_run])),
    }

    # 异句对的 lag 曲线（run 内）
    same_run = run_ids[:, None] == run_ids[None, :]
    same_day = day_ids[:, None] == day_ids[None, :]
    diff = ~same_sent
    lag_abs = np.abs(trial_index[:, None] - trial_index[None, :]).astype(np.int16)

    lag_sums = np.zeros(int(lag_abs.max()) + 1, dtype=np.float64)
    lag_cnts = np.zeros(int(lag_abs.max()) + 1, dtype=np.float64)
    for r in np.unique(run_ids):
        idx = np.where(run_ids == r)[0]
        idx = idx[np.argsort(trial_index[idx], kind="stable")]
        sub = sim[np.ix_(idx, idx)]
        ti = trial_index[idx]
        sent = sent_ids[idx]
        d = np.abs(ti[:, None] - ti[None, :]).astype(int)
        ds = (sent[:, None] != sent[None, :]).astype(np.float64)
        iu = np.triu_indices(len(idx), 1)
        dd = d[iu]
        k = int(dd.max())
        lag_sums[: k + 1] += np.bincount(dd, weights=sub[iu] * ds[iu], minlength=k + 1)
        lag_cnts[: k + 1] += np.bincount(dd, weights=ds[iu], minlength=k + 1)

    lags = np.arange(1, min(lag_max, len(lag_sums) - 1) + 1)
    lag_curve = [
        {"lag": int(l), "mean": float(lag_sums[l] / lag_cnts[l]), "n": int(lag_cnts[l])}
        for l in lags if lag_cnts[l] > 0
    ]

    def bucket_mean(lo, hi):
        sel = (lags >= lo) & (lags <= hi) & (lag_cnts[lags] > 0)
        if not sel.any():
            return float("nan")
        return float(lag_sums[lags[sel]].sum() / lag_cnts[lags[sel]].sum())

    lag_buckets = {
        "lag=1": bucket_mean(1, 1),
        "lag=2-5": bucket_mean(2, 5),
        "lag=6-20": bucket_mean(6, 20),
        "lag=21-60": bucket_mean(21, 60),
    }

    masks = [
        ("同 run, lag≤5", same_run & diff & (lag_abs <= 5)),
        ("同 run, lag 6-20", same_run & diff & (lag_abs >= 6) & (lag_abs <= 20)),
        ("同 run, lag 21-60", same_run & diff & (lag_abs >= 21) & (lag_abs <= 60)),
        ("同 run, lag>60", same_run & diff & (lag_abs > 60)),
        ("同 day 异 run", ~same_run & same_day & diff),
        ("异 day", ~same_run & ~same_day & diff),
    ]
    grand_mean = float(other.mean())
    cells, ss_between = _cells_summary(sim, masks, grand_mean)
    ss_total = float(np.square(other, dtype=np.float64).sum() - other.size * grand_mean ** 2)
    cell_rows = []
    for row in cells:
        row["share_pairs"] = float(row["n"] / other.size)
        cell_rows.append(row)

    segment = {
        "nearest_other_sentence_neighbor": segment_neighbor,
        "lag_curve": lag_curve,
        "lag_buckets": lag_buckets,
        "cells": cell_rows,
        "grand_mean": grand_mean,
        "r2_between_cells": float(ss_between / ss_total) if ss_total > 0 else float("nan"),
        "note": "R² 按全部异句对加权，绝大多数是远距离对，因此 R² 偏小属正常；"
                "看 cells 的 mean/rms 与 lag 曲线更直观。",
    }

    return {
        "repeat_consistency": repeat,
        "segment_structure": segment,
        "block_sentences": nearest_sent.tolist(),
    }


def rank1_component(matrix: np.ndarray, trial_index: np.ndarray) -> dict:
    """去对角后 SVD：第一分量的能量占比，以及它与 trial_index 的关系。"""
    m = matrix.copy()
    np.fill_diagonal(m, 0.0)
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    u1 = u[:, 0]
    if u1[0] < 0:  # 固定符号，便于看时间趋势
        u1 = -u1
    energy = s ** 2
    q = max(1, len(u1) // 4)
    return {
        "top1_energy_share": float(energy[0] / energy.sum()),
        "top5_energy_share": float(energy[:5].sum() / energy.sum()),
        "u1_corr_trial_index": float(np.corrcoef(u1, trial_index)[0, 1]),
        "u1_first_quarter_mean": float(u1[:q].mean()),
        "u1_last_quarter_mean": float(u1[-q:].mean()),
        "u1": u1,
    }


# ---------------------------------------------------------------- 绘图


def plot_zoom(matrix: np.ndarray, views: int, n_sent: int, path: Path, title: str) -> Path:
    size = min(n_sent, matrix.shape[0] // views)
    sub = matrix[: size * views, : size * views]
    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    im = ax.imshow(sub, cmap="viridis", interpolation="nearest", vmin=-0.5, vmax=1.0)
    for k in range(1, size):
        ax.axhline(k * views - 0.5, color="white", lw=0.6, alpha=0.7)
        ax.axvline(k * views - 0.5, color="white", lw=0.6, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(f"trial（每 {views} 条同句相邻）")
    ax.set_ylabel(f"trial（每 {views} 条同句相邻）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="余弦相似度（EEG × EEG）")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_panels(
    path: Path,
    lag_curve: list[dict],
    cells: list[dict],
    u1_cross: np.ndarray,
    u1_eeg: np.ndarray,
    run_label: str,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))

    ax = axes[0]
    lags = [row["lag"] for row in lag_curve]
    vals = [row["mean"] for row in lag_curve]
    ax.plot(lags, vals, color="#2a6fdb", lw=1.8)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("|Δtrial|（同 run 内，异句对）")
    ax.set_ylabel("平均余弦")
    ax.set_title("run 内 lag 曲线（异句对）")

    ax = axes[1]
    names = [row["cell"] for row in cells]
    means = [row["mean"] for row in cells]
    ax.barh(range(len(names)), means, color="#2a6fdb")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("平均余弦（异句对）")
    ax.set_title("异句对分组的平均余弦（标注 n）")
    for i, row in enumerate(cells):
        ax.text(row["mean"], i, f"  n={row['n']:,}", va="center", fontsize=7, color="gray")
    ax.invert_yaxis()

    ax = axes[2]
    ax.plot(np.arange(len(u1_cross)), u1_cross, color="#2a6fdb", lw=1.4, label="EEG×sentence")
    ax.plot(np.arange(len(u1_eeg)), u1_eeg, color="#e07a5f", lw=1.4, alpha=0.85, label="EEG×EEG")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("trial_index")
    ax.set_ylabel("第一奇异向量 u1")
    ax.set_title(f"{run_label} 去对角矩阵的第一分量")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 主流程


def main():
    args = parse_args()
    cfg = resolve_config(args)
    dataset = build_dataset(cfg, touch=True)
    meta = load_meta(dataset)
    model = load_model(cfg["ckpt"], dataset, cfg["device"])

    positions = order_positions(meta, np.arange(len(meta)), "sentence")
    eeg = encode_eeg(model, dataset, positions, cfg["device"], args.batch_size)

    sim = eeg @ eeg.T
    sent = meta["sentence"].to_numpy()[positions]
    sent_ids = pd.factorize(sent)[0]
    run_key = (meta["day"].astype(str) + "_" + meta["run"].astype(str)).to_numpy()
    run_ids = pd.factorize(run_key)[0][positions]
    day_ids = pd.factorize(meta["day"])[0][positions]
    trial_index = meta["trial_index"].to_numpy()[positions]

    n_sent = int(sent_ids.max()) + 1
    views = int(round(len(positions) / n_sent))
    print(
        f"[Data] {cfg['subject']} trial={len(positions)} 唯一句={n_sent} "
        f"每句={views} 条  order=sentence"
    )

    metrics = structure_metrics(sim, sent_ids, run_ids, day_ids, trial_index, args.lag_max)
    repeat = metrics["repeat_consistency"]
    seg = metrics["segment_structure"]

    # 单 run：跨模态与 EEG×EEG 的第一分量
    lookup = load_text_embeddings(dataset)
    sel = (meta["day"].astype(int).to_numpy()[positions] == args.run_day) & (
        meta["run"].astype(int).to_numpy()[positions] == args.run_id
    )
    run_idx = np.where(sel)[0]
    if len(run_idx) == 0:
        raise ValueError(f"(day, run)=({args.run_day}, {args.run_id}) 不在数据里")
    run_idx = run_idx[np.argsort(trial_index[run_idx], kind="stable")]
    ti_run = trial_index[run_idx]
    text_run = np.stack([lookup[s].cpu().numpy() for s in sent[run_idx]])
    cross = eeg[run_idx] @ text_run.T
    eeg_self = sim[np.ix_(run_idx, run_idx)]

    r1_cross = rank1_component(cross, ti_run)
    r1_eeg = rank1_component(eeg_self, ti_run)

    # ---------------- 出图 ----------------
    out_dir = cfg["out_dir"]

    fig_full = plot_heatmap(
        sim,
        out_dir / f"{cfg['subject']}_blocks_sentence.png",
        (
            f"{cfg['subject']} EEG×EEG 自相似矩阵（全部 {len(positions)} trial，order=sentence）\n"
            f"每块 {views}×{views} = 同一句的 {views} 条 trial；块亮=重复一致"
        ),
        xlabel="trial（同句相邻）",
        ylabel="trial（同句相邻）",
        cbar_label="余弦相似度（EEG × EEG）",
        vmin=float(np.percentile(sim, 1)),
        vmax=1.0,
    )
    fig_zoom = plot_zoom(
        sim,
        views,
        min(args.zoom_sentences, n_sent),
        out_dir / f"{cfg['subject']}_blocks_zoom.png",
        f"{cfg['subject']} 前 {min(args.zoom_sentences, n_sent)} 句：{views}×{views} 块内部纹理",
    )
    fig_panels = plot_panels(
        out_dir / f"{cfg['subject']}_blocks_panels.png",
        seg["lag_curve"],
        seg["cells"],
        r1_cross["u1"],
        r1_eeg["u1"],
        f"day{args.run_day} run{args.run_id}",
    )

    # ---------------- 汇总 ----------------
    summary = {
        "ckpt": str(cfg["ckpt_path"]),
        "n_trial": int(len(positions)),
        "n_sentence": n_sent,
        "views_per_sentence": views,
        "repeat_consistency": repeat,
        "segment_structure": seg,
        "run_component": {
            "day": int(args.run_day),
            "run": int(args.run_id),
            "cross_modal": {k: v for k, v in r1_cross.items() if k != "u1"},
            "eeg_eeg": {k: v for k, v in r1_eeg.items() if k != "u1"},
        },
    }
    out_json = out_dir / f"{cfg['subject']}_blocks_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 打印 ----------------
    print("\n" + "=" * 78)
    print("一、16×16 同句块：重复 trial 一致性")
    print("=" * 78)
    print(
        f"  同句块内      mean={repeat['within_mean']:+.4f}  std={repeat['within_std']:.4f}"
        f"（{repeat['n_within_pairs']} 对）\n"
        f"  块外全部异句  mean={repeat['outside_mean']:+.4f}\n"
        f"  块对比        {repeat['contrast']:+.4f}"
    )
    print(
        f"  最近邻句块均值={repeat['nearest_block_mean']:+.4f}  "
        f"块边距={repeat['block_margin']:+.4f}\n"
        f"  块内 std 均值={repeat['within_block_std_mean']:.4f}  "
        f"块间方差={repeat['between_block_var']:.4f}  ICC={repeat['icc']:.4f}"
    )
    print(
        f"  kNN 同句命中：top1={repeat['knn_top1_same_sentence']:.4f}  "
        f"top5={repeat['knn_top5_same_sentence']:.4f}  "
        f"（随机 {repeat['knn_top1_chance']:.4f}）"
    )
    print(
        f"  块内均值范围 {repeat['block_mean_min']:+.3f} ~ {repeat['block_mean_max']:+.3f}"
        f"（中位 {repeat['block_mean_median']:+.3f}）"
    )

    print("\n" + "=" * 78)
    print("二、异句对：时段结构")
    print("=" * 78)
    neigh = seg["nearest_other_sentence_neighbor"]
    print(
        f"  最近异句邻居：同 run={neigh['same_run_rate']:.4f}"
        f"（随机 {neigh['same_run_chance']:.4f}）  "
        f"同 day={neigh['same_day_rate']:.4f}\n"
        f"  同 run 时的 |Δtrial|：mean={neigh['mean_lag_same_run']:.1f}  "
        f"median={neigh['median_lag_same_run']:.1f}"
    )
    print("  run 内 lag 曲线（异句对）：")
    for name, value in seg["lag_buckets"].items():
        print(f"    {name:<10}{value:+.4f}")
    print(f"  {'分组':<18}{'mean':>9}{'rms':>9}{'n':>12}{'占比':>9}")
    for row in seg["cells"]:
        print(
            f"  {row['cell']:<18}{row['mean']:>+9.4f}{row['rms']:>9.4f}"
            f"{row['n']:>12,d}{row['share_pairs']:>8.1%}"
        )
    print(f"  组均值对方差的解释 R² = {seg['r2_between_cells']:.4f}")
    print(f"  注：{seg['note']}")

    print("\n" + "=" * 78)
    print(f"三、单 run 矩阵的平滑分量（day{args.run_day} run{args.run_id}）")
    print("=" * 78)
    for name, r in (("EEG×sentence", r1_cross), ("EEG×EEG", r1_eeg)):
        print(
            f"  {name:<12} 去对角后第一分量能量占比={r['top1_energy_share']:.4f}  "
            f"前 5 分量={r['top5_energy_share']:.4f}\n"
            f"  {'':<12} u1 与 trial_index 相关={r['u1_corr_trial_index']:+.4f}  "
            f"首/尾四分之一均值={r['u1_first_quarter_mean']:+.3f} / {r['u1_last_quarter_mean']:+.3f}"
        )

    print("\n" + "=" * 78)
    print("四、读法")
    print("=" * 78)
    print(
        "  1. 块内远高于块外全部异句、kNN 同句命中远高于随机 -> 重复一致性真实存在；\n"
        "  2. 最近邻句块与同句块差距若只有 1 个块内 std 量级 -> 块并不孤立，\n"
        "     块亮度主要来自句子间公共/语义分量，而非同句特异信息；\n"
        "  3. 最近异句邻居大量落在同 run、lag 曲线近正远负 -> 时段结构仍在块外；\n"
        "  4. 第一分量占比高、u1 随 trial_index 平滑变化 -> 单 run 图的整块亮暗来自慢漂移。"
    )

    print(f"\n[Output] {fig_full}")
    print(f"[Output] {fig_zoom}")
    print(f"[Output] {fig_panels}")
    print(f"[Output] {out_json}")


if __name__ == "__main__":
    main()
