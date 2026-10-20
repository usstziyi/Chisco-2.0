"""在 63 句留出验证集上绘制模型的相似度热力图。

验证集来自 ``train.py --val-fraction 0.25`` 的按类别分层留出（seed=0）：
63 句未参与训练，每句在 day 5~8 的 16 个 run 里各出现一次，共 1008 条 trial。

本脚本用 checkpoint 编码这 1008 条 trial，并画两类 63×63 矩阵：

1. **EEG × sentence（LaBSE）**：每句的 16 条 trial 先求均值，再看与 63 个候选句
   的余弦。对角线 = 正确配对；若模型能泛化到没见过的句子，对角线应明显亮于背景。
2. **EEG × EEG（句均值）**：看没见过的句子在 EEG 空间里是否自成一簇
   （同句跨 run 稳定），以及与语义/类别结构的关系。

同时输出一条 1008×63 的 trial 级热力图（每 16 行同句），并给出：
63 路 / 252 路检索指标、同句 vs 异句余弦、与 LaBSE 的 RSA、逐句 CSV。

用法::

    uv run python char_token_runbatch/plot_val_heatmap.py
    uv run python char_token_runbatch/plot_val_heatmap.py --ckpt outputs/char_token_runbatch/20261019_230758/sub-01_best_train.pt
    uv run python char_token_runbatch/plot_val_heatmap.py --order sentence
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
from scipy.stats import spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_runbatch.plot_common import (
    add_common_args,
    build_dataset,
    encode_eeg,
    load_meta,
    load_model,
    load_text_embeddings,
    normalize,
    resolve_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "char_token_runbatch"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="63 句留出验证集上的相似度热力图。")
    add_common_args(parser)
    parser.add_argument(
        "--val-split", default=None,
        help="验证句清单 JSON；默认取 checkpoint 同级的 <subject>_val_split.json",
    )
    # 复用 add_common_args 的 --order；本脚本只区分 category / sentence
    parser.set_defaults(order="category")
    return parser.parse_args()


def find_latest_ckpt(subject: str = "sub-01") -> Path:
    candidates = sorted(
        OUT_ROOT.glob(f"*/{subject}_best_train.pt"),
        key=lambda p: p.parent.name,
    )
    if not candidates:
        raise FileNotFoundError(f"{OUT_ROOT} 下没有 */{subject}_best_train.pt")
    return candidates[-1]


def load_val_sentences(args, cfg) -> list[str]:
    ckpt_sentences = list(cfg["ckpt"].get("val_sentences") or [])
    split_path = (
        Path(args.val_split) if args.val_split
        else cfg["ckpt_path"].parent / f"{cfg['subject']}_val_split.json"
    )
    if split_path.exists():
        sentences = json.loads(split_path.read_text(encoding="utf-8"))["val_sentences"]
        if ckpt_sentences and sorted(sentences) != sorted(ckpt_sentences):
            print(f"[Warn] {split_path.name} 与 checkpoint 的 val_sentences 不一致，以文件为准")
        return sorted(sentences)
    if ckpt_sentences:
        return sorted(ckpt_sentences)
    raise ValueError(
        "找不到验证句清单；请用 --val-split 指定，或换一个带 val_sentences 的 checkpoint"
    )


def retrieval(sim: np.ndarray, target: np.ndarray) -> dict:
    """按行做 top-k 检索；target[i] 是第 i 行的正确列号。"""
    n = len(target)
    correct = sim[np.arange(n), target]
    rank = (sim > correct[:, None]).sum(axis=1) + 1
    masked = sim.copy()
    masked[np.arange(n), target] = -np.inf
    best_wrong = masked.max(axis=1)
    return {
        "n": int(n),
        "top1": float((rank == 1).mean()),
        "top5": float((rank <= 5).mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(np.median(rank)),
        "cos_correct": float(correct.mean()),
        "cos_best_wrong": float(best_wrong.mean()),
        "margin": float((correct - best_wrong).mean()),
        "chance_top1": float(1.0 / sim.shape[1]),
    }


def plot_square(
    ax,
    matrix: np.ndarray,
    title: str,
    cbar_label: str,
    boundaries: list[int],
    vmin: float,
    vmax: float,
):
    im = ax.imshow(
        matrix, cmap="viridis", interpolation="nearest",
        vmin=vmin, vmax=vmax, aspect="equal",
    )
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=0.4, alpha=0.45)
        ax.axvline(b - 0.5, color="white", lw=0.4, alpha=0.45)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("sentence（63 句留出集）", fontsize=8)
    ax.set_ylabel("sentence（63 句留出集）", fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def main():
    args = parse_args()
    if args.ckpt is None:
        args.ckpt = str(find_latest_ckpt(args.subject or "sub-01"))
    cfg = resolve_config(args)
    dataset = build_dataset(cfg, touch=True)
    meta = load_meta(dataset)
    model = load_model(cfg["ckpt"], dataset, cfg["device"])

    val_sentences = load_val_sentences(args, cfg)
    val_set = set(val_sentences)
    mask = meta["sentence"].isin(val_set).to_numpy()
    positions = np.where(mask)[0]
    if len(positions) == 0:
        raise ValueError("数据里没有任何验证句 trial")

    val_meta = meta.iloc[positions].copy()
    counts = val_meta.groupby("sentence").size()
    print(
        f"[Ckpt] {cfg['ckpt_path']}  epoch={cfg['ckpt'].get('epoch')}  "
        f"train_loss={float(cfg['ckpt'].get('train_loss', float('nan'))):.4f}"
    )
    print(
        f"[Split] 验证句={len(val_sentences)}  trial={len(positions)}  "
        f"每句 {counts.min()}~{counts.max()} 次  days={cfg['days']} event={cfg['event']}"
    )

    # ---------------- 排列 / 编码 ----------------
    uniq = (
        val_meta.drop_duplicates("sentence")[["sentence", "category_name", "n_chars"]]
        .copy()
    )
    uniq["category_name"] = uniq["category_name"].fillna("?")
    if args.order == "category":
        uniq = uniq.sort_values(["category_name", "sentence"]).reset_index(drop=True)
    else:
        uniq = uniq.sort_values(["sentence"]).reset_index(drop=True)
    sentences = uniq["sentence"].tolist()
    sent_to_col = {s: i for i, s in enumerate(sentences)}

    eeg = encode_eeg(model, dataset, positions, cfg["device"], args.batch_size)
    trial_sent = val_meta["sentence"].to_numpy()
    target_trial = np.array([sent_to_col[s] for s in trial_sent])

    # 每句 16 次取均值，再做 L2 归一化
    eeg_sent = np.stack([
        normalize(eeg[trial_sent == s].mean(axis=0, keepdims=True))[0]
        for s in sentences
    ])

    lookup = load_text_embeddings(dataset)
    text_sent = normalize(np.stack([lookup[s].cpu().numpy() for s in sentences]))

    sim_cross = eeg_sent @ text_sent.T  # (63, 63)
    sim_eeg = eeg_sent @ eeg_sent.T
    sim_trial = eeg @ text_sent.T  # (1008, 63)

    # ---------------- 指标 ----------------
    m63 = retrieval(sim_cross, np.arange(len(sentences)))
    m63_trial = retrieval(sim_trial, target_trial)

    # 252 路（与训练时的验证口径一致）
    all_sentences = sorted(lookup)
    text_all = normalize(np.stack([lookup[s].cpu().numpy() for s in all_sentences]))
    col_all = {s: i for i, s in enumerate(all_sentences)}
    target_all = np.array([col_all[s] for s in trial_sent])
    sim_all = eeg @ text_all.T
    m252_trial = retrieval(sim_all, target_all)

    # 同句 / 异句一致性（trial 级）
    sim_self = eeg @ eeg.T
    n = len(eeg)
    off = ~np.eye(n, dtype=bool)
    same = trial_sent[:, None] == trial_sent[None, :]
    within = sim_self[off & same]
    other = sim_self[off & ~same]
    consistency = {
        "same_sentence_mean": float(within.mean()),
        "same_sentence_std": float(within.std()),
        "different_sentence_mean": float(other.mean()),
        "gap": float(within.mean() - other.mean()),
    }

    # EEG 句子几何 vs LaBSE 句子几何
    iu = np.triu_indices(len(sentences), 1)
    rsa = float(spearmanr(sim_eeg[iu], (text_sent @ text_sent.T)[iu]).statistic)

    # 预测是否塌缩到少数句子（左图的竖条带）
    pred = sim_trial.argmax(axis=1)
    share = np.bincount(pred, minlength=len(sentences)) / len(pred)
    hub_order = np.argsort(-share)[:3]
    hubs = [
        {"sentence": sentences[i], "share": float(share[i])}
        for i in hub_order if share[i] > 0
    ]
    positive = share[share > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    hub_stats = {
        "top3": hubs,
        "max_share": float(share.max()),
        "chance_share": float(1.0 / len(sentences)),
        "entropy_ratio": float(entropy / np.log(len(sentences))),
    }

    # 逐句明细
    ranks63 = np.empty(n, dtype=int)
    ranks63[:] = (sim_trial > sim_trial[np.arange(n), target_trial][:, None]).sum(axis=1) + 1
    ranks252 = np.empty(n, dtype=int)
    ranks252[:] = (sim_all > sim_all[np.arange(n), target_all][:, None]).sum(axis=1) + 1

    rows = []
    for i, s in enumerate(sentences):
        sel = trial_sent == s
        rows.append({
            "sentence": s,
            "category": uniq.loc[i, "category_name"],
            "n_chars": int(uniq.loc[i, "n_chars"]),
            "n_views": int(sel.sum()),
            "top1_63": float((ranks63[sel] == 1).mean()),
            "median_rank_63": float(np.median(ranks63[sel])),
            "median_rank_252": float(np.median(ranks252[sel])),
            "cos_correct_sentmean": float(sim_cross[i, i]),
            "best_wrong_sentmean": float(np.delete(sim_cross[i], i).max()),
            "margin_sentmean": float(sim_cross[i, i] - np.delete(sim_cross[i], i).max()),
        })
    per_sentence = pd.DataFrame(rows).sort_values("median_rank_63")

    # ---------------- 出图 ----------------
    out_dir = cfg["out_dir"]
    tag = f"{cfg['subject']}_val63"

    cat = uniq["category_name"].to_numpy()
    boundaries = [i for i in range(1, len(cat)) if cat[i] != cat[i - 1]]

    scale_cross = float(np.percentile(np.abs(sim_cross), 99))
    scale_cross = max(scale_cross, 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    im0 = plot_square(
        axes[0], sim_cross,
        f"EEG(句均值) × LaBSE（63 路）\n"
        f"top1={m63['top1']:.3f}（随机 {m63['chance_top1']:.3f}）  "
        f"margin={m63['margin']:+.3f}",
        "余弦相似度", boundaries, -scale_cross, scale_cross,
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, label="余弦（EEG × sentence）")

    im1 = plot_square(
        axes[1], np.where(np.eye(len(sentences), dtype=bool), np.nan, sim_eeg),
        f"EEG × EEG（句均值，对角置空）\n"
        f"异句 mean={sim_eeg[~np.eye(len(sentences), dtype=bool)].mean():+.3f}  "
        f"RSA(LaBSE)={rsa:+.3f}",
        "余弦相似度", boundaries,
        float(np.percentile(sim_eeg[~np.eye(len(sentences), dtype=bool)], 1)),
        float(np.percentile(sim_eeg[~np.eye(len(sentences), dtype=bool)], 99)),
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03, label="余弦（EEG × EEG）")

    fig.suptitle(
        f"{cfg['subject']} 留出 63 句（未参与训练，每句 16 次）  "
        f"ckpt epoch={cfg['ckpt'].get('epoch')}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_heat = out_dir / f"{tag}_heatmap.png"
    out_heat.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_heat, dpi=180)
    plt.close(fig)

    # trial 级：1008×63，每 16 行同一句
    fig, ax = plt.subplots(figsize=(8.8, 10.0))
    im = ax.imshow(
        sim_trial, cmap="viridis", aspect="auto", interpolation="nearest",
        vmin=-scale_cross, vmax=scale_cross,
    )
    for k in range(1, len(sentences)):
        ax.axhline(k * 16 - 0.5, color="white", lw=0.4, alpha=0.25)
    ax.set_title(
        f"trial 级 EEG × LaBSE（{len(positions)} trial × 63 句）\n"
        f"每 16 行同句；top1={m63_trial['top1']:.3f}（随机 {m63_trial['chance_top1']:.3f}）",
        fontsize=9,
    )
    ax.set_xlabel("sentence（63 句留出集）")
    ax.set_ylabel("trial（每 16 条同句相邻）")
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="余弦（EEG × sentence）")
    fig.tight_layout()
    out_trial = out_dir / f"{tag}_trials.png"
    fig.savefig(out_trial, dpi=180)
    plt.close(fig)

    # ---------------- 汇总 ----------------
    summary = {
        "ckpt": str(cfg["ckpt_path"]),
        "ckpt_epoch": cfg["ckpt"].get("epoch"),
        "n_val_sentences": len(sentences),
        "n_val_trials": int(len(positions)),
        "views_per_sentence": [int(counts.min()), int(counts.max())],
        "retrieval_63_sentence_mean": m63,
        "retrieval_63_trial": m63_trial,
        "retrieval_252_trial": m252_trial,
        "repeat_consistency": consistency,
        "rsa_with_labse": rsa,
        "prediction_hubs": hub_stats,
        "ckpt_stored_val_metrics": cfg["ckpt"].get("val_metrics"),
        "ckpt_stored_train_metrics": cfg["ckpt"].get("train_metrics"),
    }
    out_json = out_dir / f"{tag}_metrics.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_csv = out_dir / f"{tag}_per_sentence.csv"
    per_sentence.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ---------------- 打印 ----------------
    print("\n" + "=" * 74)
    print("一、63 句留出集检索（候选 = 63 句）")
    print("=" * 74)
    print(f"  句均值 EEG→文本：top1={m63['top1']:.4f}  top5={m63['top5']:.4f}  "
          f"mean_rank={m63['mean_rank']:.1f}  （随机 {m63['chance_top1']:.4f}）")
    print(f"  cos(正确)={m63['cos_correct']:+.4f}  cos(最佳错误)={m63['cos_best_wrong']:+.4f}  "
          f"margin={m63['margin']:+.4f}")
    print(f"  trial 级（1008 条）：top1={m63_trial['top1']:.4f}  top5={m63_trial['top5']:.4f}  "
          f"mean_rank={m63_trial['mean_rank']:.1f}")
    print(f"  252 路 trial 级（63 留出 + 189 训练）：top1={m252_trial['top1']:.4f}  "
          f"top5={m252_trial['top5']:.4f}  mean_rank={m252_trial['mean_rank']:.1f}")
    if cfg["ckpt"].get("val_metrics"):
        vm = cfg["ckpt"]["val_metrics"]
        print(f"  checkpoint 记录（val63 口径）：top1={vm['e2t1']:.4f}  top5={vm['e2t5']:.4f}  "
              f"mean_rank={vm['mean_rank']:.1f}  n={vm['n']}")
    if cfg["ckpt"].get("train_metrics"):
        tm = cfg["ckpt"]["train_metrics"]
        print(f"  checkpoint 记录（train189 口径）：top1={tm['e2t1']:.4f}  top5={tm['e2t5']:.4f}  "
              f"mean_rank={tm['mean_rank']:.1f}  n={tm['n']}")

    print("\n" + "=" * 74)
    print("二、没见过的句子在 EEG 空间里稳不稳")
    print("=" * 74)
    print(f"  同句跨 run 余弦={consistency['same_sentence_mean']:+.4f}  "
          f"异句={consistency['different_sentence_mean']:+.4f}  "
          f"差={consistency['gap']:+.4f}")
    print(f"  EEG 句间相似度 vs LaBSE 句间相似度：Spearman ρ={rsa:+.4f}")
    print(
        f"  预测塌缩：最常被选中的句子占 {hub_stats['max_share']:.1%}"
        f"（随机 {hub_stats['chance_share']:.1%}），预测熵比 {hub_stats['entropy_ratio']:.3f}"
    )
    for hub in hub_stats["top3"]:
        print(f"    {hub['share']:6.2%}  {hub['sentence']}")
    print("\n  读法：若 63 路 top1 ≈ 0.016（随机）而对角线 EEG×EEG 仍亮，")
    print("        说明模型对没见过的句子有稳定表征，但该表征没有落到文本空间里。")

    print(f"\n[Output] {out_heat}")
    print(f"[Output] {out_trial}")
    print(f"[Output] {out_json}")
    print(f"[Output] {out_csv}")


if __name__ == "__main__":
    main()
