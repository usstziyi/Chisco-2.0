"""对 ``char_token_bge_m3_runinv`` 的 checkpoint 做 raw vs run 中心化 对比分析。

为什么需要这个脚本
------------------
本分支训练时把"run 常量"在损失里扣掉了（见 ``train.py`` 的 Run 不变性一节），
所以评测也必须给两套口径：

- ``raw``：模型原始输出空间。可能仍残留 run 常量（损失对它免疫，既不奖励也不惩罚）；
- ``c_``：按 run 减均值再归一化 —— 与训练口径一致，**主指标**。

判据：如果新模型在 ``raw`` 与 ``c_`` 两套空间下的检索指标**几乎相同**，说明 run 常量
已被训练真正消除（模型不再把 run 编码进表示里）；如果 ``c_`` 明显好于 ``raw``，说明
run 常量还在，只是被评测时扣掉了。

脚本同时会把**基线**（``char_token_bge_m3``）的 checkpoint 用**同一套代码**算一遍做对照。

产出
----
- 控制台：两模型 × 两空间 的完整指标对比表；
- ``{subject}_runinv_eeg_ee_{tag}.png``：EEG×EEG 自相似矩阵（raw vs run 中心化，共色标）；
- ``{subject}_runinv_cross_modal_{tag}.png``：EEG×文本 跨模态矩阵（raw vs run 中心化）；
- ``{subject}_runinv_analysis.json``：全部指标。

用法::

    .\\.venv\\Scripts\\python.exe char_token_bge_m3_runinv/analyze.py
    .\\.venv\\Scripts\\python.exe char_token_bge_m3_runinv/analyze.py --ckpt outputs/char_token_bge_m3_runinv/<ts>/sub-01_best.pt
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
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_bge_m3_runinv.dataset import CharTokenEEGDataset, char_token_collate
from char_token_bge_m3_runinv.model import CharTokenEEGEncoder
from char_token_bge_m3_runinv.text_encoder import BGEM3TextEncoder
from char_token_bge_m3_runinv.train import (
    build_sentence_run_lookup,
    build_text_lookup,
    gather_text,
    remove_run_mean,
    retrieval_metrics,
    run_ids,
    sentence_group_ids,
    split_by_sentence,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNINV_ROOT = PROJECT_ROOT / "outputs" / "char_token_bge_m3_runinv"
BASELINE_ROOT = PROJECT_ROOT / "outputs" / "char_token_bge_m3"


def find_latest(root: Path, subject: str) -> Path | None:
    """在 ``outputs/<exp>/*/`` 下取最新的 ``{subject}_best.pt``（目录名是时间戳）。"""
    candidates = sorted(root.glob(f"*/{subject}_best.pt"), key=lambda p: p.parent.name, reverse=True)
    return candidates[0] if candidates else None


@torch.no_grad()
def encode_val(ckpt_path: Path, dataset, val_idx, device, batch_size=64):
    """加载 checkpoint，对预测集全部 trial 编码，返回 ``(E, ckpt, meta)``。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CharTokenEEGEncoder(
        n_chans=int(ckpt["n_chans"]),
        token_samples=int(ckpt["token_samples"]),
        embedding_dim=int(ckpt["embedding_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
        collate_fn=char_token_collate,
    )
    parts = [model(tokens.to(device), mask.to(device)) for tokens, mask, _ in loader]
    return torch.cat(parts, dim=0), ckpt


def space_metrics(eeg, text, sentences, lookup, positive, runs, groups, device) -> dict:
    """在给定表示空间里算相似度统计 + 检索指标。"""
    n = eeg.size(0)
    off = ~torch.eye(n, dtype=torch.bool, device=device)
    same_run = runs[:, None] == runs[None, :]
    same_sent = groups[:, None] == groups[None, :]
    ee = eeg @ eeg.t()      # EEG × EEG
    cm = eeg @ text.t()     # EEG × 文本

    out = {
        "eeg_ee_same_run_cos": float(ee[off & same_run].mean()),
        "eeg_ee_cross_run_cos": float(ee[off & ~same_run].mean()),
        "eeg_ee_same_sent_cos": float(ee[off & same_sent].mean()),
        "eeg_ee_same_run_other_sent_cos": float(ee[off & same_run & ~same_sent].mean()),
        "cm_paired_cos": float(cm.diag().mean()),
        "cm_same_run_other_sent_cos": float(cm[off & same_run & ~same_sent].mean()),
        "cm_cross_run_cos": float(cm[off & ~same_run].mean()),
    }
    out.update(retrieval_metrics(eeg, text, sentences, lookup, positive, device))
    return out


def plot_pair(out_dir, A, B, titles, filename, cbar_label, xlabel, ylabel):
    """raw / centered 两联图，共用同一色标便于直接对比。"""
    lo = float(min(A.min(), B.min()))
    hi = float(max(A.max(), B.max()))
    fig, axes = plt.subplots(1, 2, figsize=(26, 13))
    for ax, M, title in zip(axes, [A, B], titles):
        im = ax.imshow(M, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=cbar_label)
    fig.tight_layout()
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run_block_matrix(sim, runs, order):
    """按 (行 run, 列 run) 求块均值，返回 4×4（或按实际 run 数）矩阵。"""
    ro = runs[order]
    values = sorted({int(v) for v in runs.tolist()})
    M = np.zeros((len(values), len(values)))
    for a, ra in enumerate(values):
        rows = order[ro == ra]
        for b, rb in enumerate(values):
            cols = order[ro == rb]
            M[a, b] = float(sim[np.ix_(rows, cols)].mean())
    return values, M


def parse_args():
    parser = argparse.ArgumentParser(description="runinv 分支的 raw vs run 中心化 对比分析。")
    parser.add_argument("--ckpt", default=None, help="新分支 checkpoint；默认取 outputs/char_token_bge_m3_runinv 下最新")
    parser.add_argument("--baseline-ckpt", default=None, help="基线 checkpoint；默认取 outputs/char_token_bge_m3 下最新")
    parser.add_argument("--subject", default="sub-01", help="被试（默认 sub-01）")
    parser.add_argument("--batch-size", type=int, default=64, help="编码时的 batch 大小（默认 64）")
    parser.add_argument("--order", default="sentence", choices=["sentence", "raw"], help="绘图顺序（默认按句子分组）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=None, help="输出目录；默认存到新 checkpoint 同级目录")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest(RUNINV_ROOT, args.subject)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"找不到新分支 checkpoint，请用 --ckpt 指定（{RUNINV_ROOT}）")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args") or {}

    baseline_path = Path(args.baseline_ckpt) if args.baseline_ckpt else find_latest(BASELINE_ROOT, args.subject)
    if baseline_path is not None and not baseline_path.exists():
        baseline_path = None

    subject = args.subject
    days = train_args.get("day") or [1, 2, 3, 4]
    event = train_args.get("event") or "inner"
    sample = int(train_args.get("sample", 250))
    char_duration = float(train_args.get("char_duration", 0.4))
    val_ratio = float(train_args.get("val_ratio", 0.2))
    seed = int(train_args.get("seed", 0))

    print(f"[Ckpt] runinv  : {ckpt_path} (epoch={ckpt.get('epoch')})")
    print(f"[Ckpt] baseline: {baseline_path}")

    # ---------------- 数据（两模型共用同一份预测集） ----------------
    dataset = CharTokenEEGDataset(
        event=event, sample=sample, subject=subject, days=days, char_duration=char_duration
    )
    dataset[0]
    _, val_set, info = split_by_sentence(dataset, val_ratio, seed)
    val_idx = list(val_set.indices)
    steps_label = ",".join(f"{d:02d}" for d in sorted(days))
    print(
        f"[Data] {subject} day-{steps_label} event={event}  预测集 "
        f"{info['n_unique_val']} 句 / {len(val_set)} trial，句子重叠={info['sentence_overlap']}"
    )

    meta = dataset.meta
    sentences = [meta["sentence"].tolist()[i] for i in val_idx]
    run_lookup = build_sentence_run_lookup(dataset)

    print("[Text] 加载 BGE-M3（冻结） ...")
    text_encoder = BGEM3TextEncoder()
    lookup, text_dim = build_text_lookup(text_encoder, meta["sentence"].tolist(), device)

    # ---------------- 编码两模型 ----------------
    E_new, ckpt_new = encode_val(ckpt_path, dataset, val_idx, device, args.batch_size)
    E_base = None
    if baseline_path is not None:
        E_base, _ = encode_val(baseline_path, dataset, val_idx, device, args.batch_size)
        if E_base.shape[1] != E_new.shape[1]:
            E_base = None

    text = gather_text(sentences, lookup, device)
    groups = sentence_group_ids(sentences, device)
    runs = run_ids(sentences, run_lookup, device)
    positive = groups[:, None] == groups[None, :]

    def both_spaces(E, tag):
        raw = space_metrics(E, text, sentences, lookup, positive, runs, groups, device)
        cen = space_metrics(
            remove_run_mean(E, runs), text, sentences, lookup, positive, runs, groups, device
        )
        return {tag + "_raw": raw, tag + "_centered": cen}

    results = {}
    results.update(both_spaces(E_new, "runinv"))
    if E_base is not None:
        results.update(both_spaces(E_base, "baseline"))

    # ---------------- 打印对比表 ----------------
    keys = [
        ("eeg_ee_same_run_cos", "EEG×EEG 同run"),
        ("eeg_ee_cross_run_cos", "EEG×EEG 跨run"),
        ("eeg_ee_same_sent_cos", "EEG×EEG 同句"),
        ("eeg_ee_same_run_other_sent_cos", "EEG×EEG 同run异句"),
        ("cm_paired_cos", "EEG×文本 配对"),
        ("cm_same_run_other_sent_cos", "EEG×文本 同run异句"),
        ("cm_cross_run_cos", "EEG×文本 跨run"),
        ("eeg2text_top1", "e2t@1"),
        ("eeg2text_top5", "e2t@5"),
        ("text2eeg_top5", "t2e@5"),
    ]
    cols = [k for k in results]
    width = 22
    print("\n" + "=" * (width + 11 * len(cols)))
    print("指标".ljust(width) + "".join(c.rjust(11) for c in cols))
    print("-" * (width + 11 * len(cols)))
    for key, label in keys:
        print(label.ljust(width) + "".join(f"{results[c][key]:>+11.4f}" for c in cols))

    n_cand = results["runinv_raw"]["n_text_candidates"]
    print(
        f"\n随机基线：e2t@1={1 / n_cand:.4f}  e2t@5={min(5, n_cand) / n_cand:.4f}"
        f"（候选 {n_cand} 句）"
    )
    same_n = results["runinv_raw"]["eeg_ee_same_sent_cos"]
    print(
        "判据：runinv 的 raw 与 runinv_centered 若几乎相同 -> run 常量已被训练消除；\n"
        "      否则说明 run 常量仍在，只是评测时被扣掉。\n"
        f"      本 runinv：raw 同句 EEG×EEG={same_n:+.4f}；"
        f"raw 同run-跨run 差="
        f"{results['runinv_raw']['eeg_ee_same_run_cos'] - results['runinv_raw']['eeg_ee_cross_run_cos']:+.4f}"
    )

    # ---------------- run 块结构 ----------------
    order = list(range(len(sentences)))
    if args.order == "sentence":
        grp = {}
        for pos, s in enumerate(sentences):
            grp.setdefault(s, []).append(pos)
        order = np.array([p for v in grp.values() for p in v])
    else:
        order = np.arange(len(sentences))
    tag = "bysentence" if args.order == "sentence" else "raw"

    sim_new_raw = (E_new @ E_new.t()).cpu().numpy()
    E_new_c = remove_run_mean(E_new, runs)
    sim_new_cen = (E_new_c @ E_new_c.t()).cpu().numpy()
    values, block_raw = run_block_matrix(sim_new_raw, runs.cpu().numpy(), order)
    _, block_cen = run_block_matrix(sim_new_cen, runs.cpu().numpy(), order)
    print(f"\n{subject} runinv EEG×EEG 块均值（行=EEG run, 列=EEG run）")
    print("raw     " + "".join(f"run{v:>3} " for v in values))
    for i, v in enumerate(values):
        print(f"  run{v}  " + "".join(f"{block_raw[i, j]:>+7.3f}" for j in range(len(values))))
    print("centered" + "".join(f"run{v:>3} " for v in values))
    for i, v in enumerate(values):
        print(f"  run{v}  " + "".join(f"{block_cen[i, j]:>+7.3f}" for j in range(len(values))))

    # ---------------- 出图 ----------------
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_new_raw_o = sim_new_raw[np.ix_(order, order)]
    sim_new_cen_o = sim_new_cen[np.ix_(order, order)]
    p1 = plot_pair(
        out_dir, sim_new_raw_o, sim_new_cen_o,
        [f"runinv EEG×EEG 原始空间（{subject} 预测集，{tag}）",
         "runinv EEG×EEG run 中心化空间"],
        f"{subject}_runinv_eeg_ee_{tag}.png", "余弦相似度", "trial（列）", "trial（行）",
    )
    cm_raw = (E_new @ text.t()).cpu().numpy()[np.ix_(order, order)]
    cm_cen = (E_new_c @ text.t()).cpu().numpy()[np.ix_(order, order)]
    p2 = plot_pair(
        out_dir, cm_raw, cm_cen,
        [f"runinv EEG×文本 原始空间（{subject} 预测集，{tag}）",
         "runinv EEG×文本 run 中心化空间"],
        f"{subject}_runinv_cross_modal_{tag}.png", "cos(EEG, 文本)", "文本 trial（列）", "EEG trial（行）",
    )
    print(f"\n[Output] {p1}\n[Output] {p2}")

    json_path = out_dir / f"{subject}_runinv_analysis.json"
    json_path.write_text(
        json.dumps(
            {
                "ckpt": str(ckpt_path),
                "ckpt_epoch": ckpt.get("epoch"),
                "baseline_ckpt": str(baseline_path) if baseline_path else None,
                "data": {
                    "n_val": len(val_idx),
                    "n_unique_val": info["n_unique_val"],
                    "days": sorted(days),
                    "val_ratio": val_ratio,
                    "seed": seed,
                },
                "results": results,
                "run_block_raw": block_raw.tolist(),
                "run_block_centered": block_cen.tolist(),
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Output] {json_path}")


if __name__ == "__main__":
    main()
