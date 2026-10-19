"""对 ``char_token_runbatch`` 的训练结果做诊断分析。

训练时一个 batch = 一个 (day, run)，batch 内 run 恒定，所以模型无法靠
"这是哪个 run" 排序。这个脚本从 checkpoint 出发，检查训练完之后：

1. **检索是否真的学到句子**：e2t@1/@5、t2e@1/@5（候选 = 252 个唯一句 / 4032 条 trial）。
2. **run 是否还在表示里**：run 线性探针 + run 中心化前后的检索指标。
   判据：
   - 中心化前后几乎不变 -> run 常量还在或不在都不影响排序，设计生效；
   - 中心化后明显更好   -> run 分量在稀释句子信号；
   - 中心化后明显更差   -> run 分量携带了句子相关信息（位置/上下文）。
3. **下一个捷径是什么**：句长（ridge 探针 + 分层检索 + OOF 残差化）、
   类别（探针 + 错误同类别率 + 类别中心化）、run 内位置（trial_index 探针）。
4. **对照**：随机初始化模型跑同一套指标，区分"训练学到的"与"输入本身就有"。
5. **与文本结构的一致性**：EEG 句子相似度矩阵 vs LaBSE 文本相似度矩阵的 Spearman。

产出（默认写到 checkpoint 同级目录 ``analysis/``）::

    summary.json          全部指标
    per_sentence.csv      每句的 rank / margin / 视图一致性
    loss_by_run.csv       最后 10 个 epoch 每个 (day, run) 的平均 loss
    fig_diagnostics.png    loss、相似度、长度分层、中心化对比
    fig_simmatrix.png      252×252 句子均值相似度矩阵（按 trial_index 排序）

用法::

    uv run python char_token_runbatch/analyze.py
    uv run python char_token_runbatch/analyze.py --ckpt outputs/char_token_runbatch/<ts>/sub-01_best_val.pt
    uv run python char_token_runbatch/analyze.py --baseline none
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
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_runbatch.common import build_text_lookup
from char_token_runbatch.dataset import CharTokenEEGDataset, char_token_collate
from char_token_runbatch.model import CharTokenEEGEncoder
from char_token_runbatch.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "char_token_runbatch"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 加载 / 编码


def find_latest(subject: str) -> Path:
    for name in (f"{subject}_best_val.pt", f"{subject}_best_train.pt"):
        candidates = sorted(
            OUT_ROOT.glob(f"*/{name}"),
            key=lambda p: p.parent.name,
        )
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(
        f"{OUT_ROOT} 下没有 */{subject}_best_val.pt 或 */{subject}_best_train.pt"
    )


def load_encoder(ckpt_path: Path, device) -> tuple[CharTokenEEGEncoder, dict]:
    """载入 checkpoint；兼容训练时被 ``ChunkedTokenEncoder`` 包过一层的情况。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    if any(k.startswith("token_encoder.base.") for k in state):
        state = {
            k.replace("token_encoder.base.", "token_encoder.", 1): v
            for k, v in state.items()
        }

    model = CharTokenEEGEncoder(
        n_chans=int(ckpt["n_chans"]),
        token_samples=int(ckpt["token_samples"]),
        embedding_dim=int(ckpt["embedding_dim"]),
    )
    model.load_state_dict(state)
    model.eval().to(device)
    return model, ckpt


def random_encoder_like(ckpt: dict, seed: int = 1) -> CharTokenEEGEncoder:
    """同架构、随机初始化的对照模型（不训练）。"""
    torch.manual_seed(seed)
    model = CharTokenEEGEncoder(
        n_chans=int(ckpt["n_chans"]),
        token_samples=int(ckpt["token_samples"]),
        embedding_dim=int(ckpt["embedding_dim"]),
    )
    model.eval()
    return model


@torch.no_grad()
def encode_all(model, dataset, device, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=char_token_collate
    )
    parts = []
    for tokens, mask, _ in loader:
        parts.append(model(tokens.to(device), mask.to(device)))
    return torch.cat(parts, dim=0).cpu().numpy()


# ---------------------------------------------------------------- 指标


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def e2t_metrics(eeg: np.ndarray, text: np.ndarray, target: np.ndarray) -> dict:
    """EEG -> 唯一句检索；target[i] 是第 i 条 trial 的正确句列号。"""
    sim = eeg @ text.T
    n = len(eeg)
    correct = sim[np.arange(n), target]
    rank = (sim > correct[:, None]).sum(axis=1) + 1

    masked = sim.copy()
    masked[np.arange(n), target] = -np.inf
    best_wrong = masked.max(axis=1)
    return {
        "top1": float((rank == 1).mean()),
        "top5": float((rank <= 5).mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(np.median(rank)),
        "cos_correct": float(correct.mean()),
        "cos_best_wrong": float(best_wrong.mean()),
        "margin": float((correct - best_wrong).mean()),
    }


def t2e_metrics(eeg: np.ndarray, text: np.ndarray, target: np.ndarray, n_sent: int) -> dict:
    """文本 -> trial 检索；一个句子有多个正样本 trial，命中任一即算对。"""
    sim = text @ eeg.T  # (S, N)
    positive = np.zeros_like(sim, dtype=bool)
    positive[target, np.arange(len(target))] = True

    max_pos = np.where(positive, sim, -np.inf).max(axis=1)
    neg = ~positive
    rank = ((sim > max_pos[:, None]) & neg).sum(axis=1) + 1
    return {
        "top1": float((rank == 1).mean()),
        "top5": float((rank <= 5).mean()),
        "mean_rank": float(rank.mean()),
    }


def similarity_stats(eeg: np.ndarray, groups: np.ndarray, runs: np.ndarray, days: np.ndarray) -> dict:
    sim = eeg @ eeg.T
    n = len(eeg)
    off = ~np.eye(n, dtype=bool)
    same_s = groups[:, None] == groups[None, :]
    same_r = runs[:, None] == runs[None, :]
    same_d = days[:, None] == days[None, :]

    def mean(mask):
        return float(sim[mask].mean()) if mask.any() else float("nan")

    return {
        "same_sentence": mean(off & same_s),
        "same_sentence_cross_run": mean(off & same_s & ~same_r),
        "same_sentence_cross_day": mean(off & same_s & ~same_d),
        "same_run_other_sentence": mean(off & same_r & ~same_s),
        "diff_run_other_sentence": mean(off & ~same_r & ~same_s),
        "other_sentence": mean(off & ~same_s),
    }


def center_by(eeg: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """按组减均值再 L2 归一化。"""
    out = eeg.copy()
    for key in np.unique(keys):
        rows = keys == key
        out[rows] -= out[rows].mean(axis=0, keepdims=True)
    return normalize(out)


def run_matrices(eeg: np.ndarray, groups: np.ndarray, run_ids: np.ndarray):
    """把每个 run 压成 ``(n_sentence, D)``，行按唯一句排序。

    返回 ``(mats, uniq_ids)``；所有 run 覆盖同一组句子，所以行可以直接对齐。
    """
    uniq_ids = np.unique(groups)
    mats = []
    for r in np.unique(run_ids):
        rows = run_ids == r
        mats.append(np.stack([
            eeg[rows & (groups == g)].mean(axis=0) for g in uniq_ids
        ]))
    return np.stack(mats), uniq_ids


def cross_run_retrieval(mats: np.ndarray) -> dict:
    """跨 run 的 EEG→EEG 1-近邻检索：同句为正、其余 251 句为负。

    这个指标比 e2t 更接近"跨记录段稳定性"：如果模型只是把 trial 映射到
    固定文本向量，跨 run 的同一句也应对得上；如果它对每个 run 单独记忆，
    跨 run 检索会明显下降。
    """
    n_runs, n_sent, _ = mats.shape
    top1, top5, ranks = [], [], []
    for a in range(n_runs):
        for b in range(n_runs):
            if a == b:
                continue
            sim = mats[a] @ mats[b].T
            rank = (sim > np.diag(sim)[:, None]).sum(axis=1) + 1
            top1.append(float((rank == 1).mean()))
            top5.append(float((rank <= 5).mean()))
            ranks.append(float(rank.mean()))
    return {
        "n_pairs": len(top1),
        "top1": float(np.mean(top1)),
        "top5": float(np.mean(top5)),
        "mean_rank": float(np.mean(ranks)),
        "random_top1": 1.0 / n_sent,
    }


def category_diagnostics(
    eeg: np.ndarray,
    text: np.ndarray,
    sent_mean: np.ndarray,
    text_mean: np.ndarray,
    target: np.ndarray,
    cat_ids: np.ndarray,
    cat_of_sent: np.ndarray,
) -> dict:
    """区分"只学到类别"还是"类别内还有语义"。

    三个口径：
    1. 类别内检索：候选只保留同类句子，看还能不能排对；同类别文本检索作上限；
    2. 类别内 RSA：只在同类句子对里算 EEG/文本相似度的相关；
    3. 类别中心化后的 RSA：减掉每个类别的均值，看残余结构是否仍与文本一致。
    """
    big = [c for c in np.unique(cat_of_sent) if (cat_of_sent == c).sum() >= 10]
    within = {}
    for cat in big:
        cols = np.where(cat_of_sent == cat)[0]
        col_pos = {g: i for i, g in enumerate(cols)}
        rows = np.where(cat_ids == cat)[0]
        local = np.array([col_pos[g] for g in target[rows]])

        sim = eeg[rows] @ text[cols].T
        correct = sim[np.arange(len(rows)), local]
        rank = (sim > correct[:, None]).sum(axis=1) + 1

        sim_t = text[cols] @ text[cols].T
        diag_t = np.diag(sim_t).copy()
        np.fill_diagonal(sim_t, -np.inf)  # 排除自身，否则 top1 恒为 1
        rank_t = (sim_t > diag_t[:, None]).sum(axis=1) + 1

        within[str(cat)] = {
            "n_sentence": int(len(cols)),
            "n_trial": int(len(rows)),
            "e2t_top1": float((rank == 1).mean()),
            "e2t_top5": float((rank <= 5).mean()),
            "e2t_mean_rank": float(rank.mean()),
            "text_top1": float((rank_t == 1).mean()),
            "text_top5": float((rank_t <= 5).mean()),
            "random_top1": 1.0 / len(cols),
        }

    # 类别内 RSA（只取样本量 >= 10 的大类，避免 2 句小类的单对噪声）
    pe, pt = [], []
    for cat in big:
        idx = np.where(cat_of_sent == cat)[0]
        tri = np.triu_indices(len(idx), 1)
        pe.append((sent_mean[idx] @ sent_mean[idx].T)[tri])
        pt.append((text_mean[idx] @ text_mean[idx].T)[tri])
    rsa_within = float(spearmanr(np.concatenate(pe), np.concatenate(pt)).statistic)

    # 类别中心化后的 RSA
    def centered_rsa(groups_for_centering):
        e_c, t_c = sent_mean.copy(), text_mean.copy()
        for cat in groups_for_centering:
            idx = np.where(cat_of_sent == cat)[0]
            e_c[idx] -= e_c[idx].mean(axis=0, keepdims=True)
            t_c[idx] -= t_c[idx].mean(axis=0, keepdims=True)
        e_c, t_c = normalize(e_c), normalize(t_c)
        tri = np.triu_indices(len(e_c), 1)
        return float(spearmanr((e_c @ e_c.T)[tri], (t_c @ t_c.T)[tri]).statistic)

    rsa_centered_big = centered_rsa(big)
    rsa_centered_all = centered_rsa(list(np.unique(cat_of_sent)))

    # top-5 命中里同类别的比例：EEG vs 文本 vs 随机先验
    sim = eeg @ text.T
    top5 = np.argsort(-sim, axis=1)[:, :5]
    eeg_same_cat = float((cat_of_sent[top5] == cat_ids[:, None]).mean())

    sim_t = text @ text.T
    np.fill_diagonal(sim_t, -np.inf)
    top5_t = np.argsort(-sim_t, axis=1)[:, :5]
    text_same_cat = float((cat_of_sent[top5_t] == cat_of_sent[:, None]).mean())

    freq = np.bincount(cat_ids) / len(cat_ids)
    chance_same_cat = float((freq ** 2).sum())

    return {
        "within_category": within,
        "rsa_within_big_categories": rsa_within,
        "rsa_category_centered_big": rsa_centered_big,
        "rsa_category_centered_all": rsa_centered_all,
        "top5_same_category": {
            "eeg": eeg_same_cat,
            "text": text_same_cat,
            "chance": chance_same_cat,
        },
    }


def cv_logreg_accuracy(x: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """5 折分层逻辑回归准确率（标准化放在 pipeline 内，避免泄漏）。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in skf.split(x, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0),
        )
        clf.fit(x[train_idx], y[train_idx])
        scores.append(float(clf.score(x[test_idx], y[test_idx])))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


def cv_ridge_r2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """5 折 ridge 回归 R²，用于连续量（句长 / run 内位置）。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in kf.split(x):
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(x[train_idx], y[train_idx])
        pred[test_idx] = reg.predict(x[test_idx])
    pred = np.asarray(pred)
    mae = float(np.abs(y - pred).mean())
    return {"r2": float(r2_score(y, pred)), "mae": mae, "spearman": float(spearmanr(y, pred).statistic)}


def oof_residual(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """5 折 out-of-fold 回归残差（保留给后续分析用）。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in kf.split(x):
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(x[train_idx], y[train_idx])
        pred[test_idx] = reg.predict(x[test_idx])
    return y - pred


# ---------------------------------------------------------------- 绘图


def plot_diagnostics(
    out_path: Path,
    epochs: pd.DataFrame | None,
    sim_trained: dict,
    sim_random: dict | None,
    by_length: list[dict],
    spaces: dict,
    random_top1: float,
    probe: dict,
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    if epochs is not None and not epochs.empty:
        ax.plot(epochs["epoch"], epochs["train_loss"], color="#2a6fdb", lw=2)
        ax.axhline(np.log(252), color="gray", ls="--", lw=1, label="ln 252 = 5.53（随机）")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.set_title("训练 loss（e^loss ≈ 有效候选数）")
        ax.legend(frameon=False)
        top = ax.secondary_yaxis("right", functions=(np.exp, np.log))
        top.set_ylabel("e^loss")
    else:
        ax.axis("off")

    ax = axes[0, 1]
    keys = [
        ("same_sentence_cross_run", "同句\n跨 run"),
        ("same_sentence_cross_day", "同句\n跨 day"),
        ("same_run_other_sentence", "同 run\n异句"),
        ("diff_run_other_sentence", "异 run\n异句"),
    ]
    x = np.arange(len(keys))
    width = 0.36
    vals_t = [sim_trained[k] for k, _ in keys]
    ax.bar(x - width / 2, vals_t, width, label="训练后", color="#2a6fdb")
    if sim_random is not None:
        vals_r = [sim_random[k] for k, _ in keys]
        ax.bar(x + width / 2, vals_r, width, label="随机初始化", color="#c9c9c9")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in keys], fontsize=9)
    ax.set_ylabel("EEG×EEG 余弦")
    ax.set_title("同句 / 同 run 的相似度（随机模型作对照）")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    xs = [row["n_chars"] for row in by_length]
    ax.bar([str(v) for v in xs], [row["top1"] for row in by_length], color="#2a6fdb")
    ax.axhline(random_top1, color="red", ls="--", lw=1, label=f"随机 {random_top1:.4f}")
    ax.set_xlabel("句长（字数）")
    ax.set_ylabel("e2t@1")
    ax.set_title("按句长的 top-1 命中")
    ax.legend(frameon=False, fontsize=9)
    for i, row in enumerate(by_length):
        ax.text(i, row["top1"], f"{row['n_trial']}", ha="center", va="bottom", fontsize=7, color="gray")

    ax = axes[1, 1]
    names = list(spaces)
    vals = [spaces[n]["top1"] for n in names]
    ax.bar(names, vals, color=["#2a6fdb", "#6aa5f0", "#8fbcff", "#b9a3e3", "#e3a3a3"])
    ax.axhline(random_top1, color="red", ls="--", lw=1)
    ax.set_ylabel("e2t@1")
    ax.set_title(
        "不同中心化空间的检索（run 探针 %.3f / day 探针 %.3f）"
        % (probe["run_accuracy"]["mean"], probe["day_accuracy"]["mean"])
    )
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_simmatrix(out_path: Path, sent_mean: np.ndarray):
    """按 trial_index 排序的句子均值相似度矩阵；对角线即同一句。"""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(sent_mean @ sent_mean.T, cmap="viridis", interpolation="nearest")
    ax.set_title("252 句平均 EEG 相似度（按 run 内位置/trial_index 排序）")
    ax.set_xlabel("trial_index")
    ax.set_ylabel("trial_index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="余弦")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程


def parse_args():
    parser = argparse.ArgumentParser(description="run-batch 实验的诊断分析。")
    parser.add_argument("--ckpt", default=None, help="默认取 outputs/char_token_runbatch 下最新的 best_train.pt")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--day", type=int, nargs="+", default=None, help="默认沿用 checkpoint 里的训练 day")
    parser.add_argument("--batch-size", type=int, default=252)
    parser.add_argument("--device", default=None)
    parser.add_argument("--baseline", default="random", choices=["random", "none"])
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest(args.subject)
    model, ckpt = load_encoder(ckpt_path, device)
    train_args = ckpt.get("args", {}) or {}
    days = args.day or train_args.get("day") or [5, 6, 7, 8]
    event = train_args.get("event", "inner")
    sample = int(train_args.get("sample", 250))
    char_duration = float(train_args.get("char_duration", 0.4))

    print(f"[Ckpt] {ckpt_path} (epoch={ckpt.get('epoch')}, train_loss={float(ckpt.get('train_loss', float('nan'))):.4f})")
    val_metrics = ckpt.get("val_metrics")
    if val_metrics:
        print(
            f"[Val ] e2t@1={val_metrics['e2t1']:.4f} "
            f"e2t@5={val_metrics['e2t5']:.4f} "
            f"mean_rank={val_metrics['mean_rank']:.1f} "
            f"n={val_metrics['n']}（留出 {len(ckpt.get('val_sentences', []))} 句）"
        )
    print(f"[Cfg ] days={days} event={event} device={device}")

    # ---------------- 数据 ----------------
    dataset = CharTokenEEGDataset(
        event=event, sample=sample, subject=args.subject,
        days=days, char_duration=char_duration,
    )
    dataset[0]  # 触发一次读取，确定 n_chans
    meta = dataset.meta.copy()

    sentences = meta["sentence"].tolist()
    unique_sentences = sorted(set(sentences))
    sent_to_col = {s: i for i, s in enumerate(unique_sentences)}
    target = np.array([sent_to_col[s] for s in sentences], dtype=np.int64)
    groups = target.copy()
    n_sent = len(unique_sentences)

    run_key = (meta["day"].astype(str) + "_" + meta["run"].astype(str)).to_numpy()
    run_ids = pd.factorize(run_key)[0]
    day_ids = pd.factorize(meta["day"])[0]
    cat_ids = pd.factorize(meta["category_name"].fillna("?"))[0]
    n_chars = meta["n_chars"].to_numpy(dtype=np.float64)
    trial_index = meta["trial_index"].to_numpy(dtype=np.float64)

    print(
        f"[Data] trial={len(meta)}  唯一句={n_sent}  run={len(np.unique(run_ids))}  "
        f"day={len(np.unique(day_ids))}  类别={len(np.unique(cat_ids))}"
    )

    # ---------------- 文本 ----------------
    print("[Text] 加载 LaBSE（冻结） ...")
    text_encoder = LaBSETextEncoder()
    lookup, _ = build_text_lookup(text_encoder, sentences, device)
    text = torch.stack([lookup[s] for s in unique_sentences]).cpu().numpy()

    # ---------------- 编码 ----------------
    print("[Enc ] 训练后模型 ...")
    eeg = encode_all(model, dataset, device, args.batch_size)
    base_model = None
    eeg_base = None
    if args.baseline == "random":
        print("[Enc ] 随机初始化对照 ...")
        base_model = random_encoder_like(ckpt).to(device)
        eeg_base = encode_all(base_model, dataset, device, args.batch_size)

    random_top1 = 1.0 / n_sent
    random_top5 = min(5, n_sent) / n_sent
    print(f"[Base] 随机 e2t@1={random_top1:.4f}  e2t@5={random_top5:.4f}（{n_sent} 候选）")

    # ---------------- 表示空间 ----------------
    spaces = {
        "raw": eeg,
        "run-center": center_by(eeg, run_ids),
        "day-center": center_by(eeg, day_ids),
    }

    # 长度残差化：在中心化空间拟合 n_chars ~ embedding，把该线性方向投影掉。
    # 这是描述性分析（全数据拟合），用于回答"去掉长度后检索还剩多少"。
    centered = eeg - eeg.mean(axis=0, keepdims=True)
    y_len = n_chars - n_chars.mean()
    w_len = np.linalg.lstsq(centered, y_len, rcond=None)[0]
    projection = (centered @ w_len) / max(float(w_len @ w_len), 1e-12)
    spaces["len-resid"] = normalize(centered - projection[:, None] * w_len[None, :])
    spaces["cat-center"] = center_by(eeg, cat_ids)

    e2t = {name: e2t_metrics(vec, text, target) for name, vec in spaces.items()}
    t2e = t2e_metrics(eeg, text, target, n_sent)
    sim_trained = similarity_stats(eeg, groups, run_ids, day_ids)
    sim_random = (
        similarity_stats(eeg_base, groups, run_ids, day_ids)
        if eeg_base is not None else None
    )
    for stats in (sim_trained, sim_random):
        if stats is not None:
            stats["same_minus_other"] = stats["same_sentence"] - stats["other_sentence"]

    # 跨 run 的 EEG→EEG 检索：比"同句余弦"更能说明跨记录段的稳定性
    run_mats, _ = run_matrices(eeg, groups, run_ids)
    cross_run = cross_run_retrieval(run_mats)
    cross_run_random = None
    if eeg_base is not None:
        base_mats, _ = run_matrices(eeg_base, groups, run_ids)
        cross_run_random = cross_run_retrieval(base_mats)

    # ---------------- 探针 ----------------
    print("[Probe] run / day / category / 长度 / 位置 ...")
    probe = {
        "run_accuracy": cv_logreg_accuracy(eeg, run_ids),
        "day_accuracy": cv_logreg_accuracy(eeg, day_ids),
        "category_accuracy": cv_logreg_accuracy(eeg, cat_ids),
        "length_r2": cv_ridge_r2(eeg, n_chars),
        "trial_index_r2": cv_ridge_r2(eeg, trial_index),
    }
    if eeg_base is not None:
        probe["random_run_accuracy"] = cv_logreg_accuracy(eeg_base, run_ids)
        probe["random_day_accuracy"] = cv_logreg_accuracy(eeg_base, day_ids)

    # ---------------- 按句长分层 ----------------
    ranks = np.empty(len(eeg), dtype=np.int64)
    sim_all = eeg @ text.T
    ranks[:] = (sim_all > sim_all[np.arange(len(eeg)), target][:, None]).sum(axis=1) + 1
    by_length = []
    for value in sorted(set(n_chars.astype(int))):
        rows = n_chars.astype(int) == value
        by_length.append({
            "n_chars": int(value),
            "n_trial": int(rows.sum()),
            "top1": float((ranks[rows] == 1).mean()),
            "top5": float((ranks[rows] <= 5).mean()),
            "mean_rank": float(ranks[rows].mean()),
        })

    # ---------------- 错误结构 ----------------
    pred = sim_all.argmax(axis=1)
    wrong = pred != target
    same_cat_wrong = float(
        (cat_ids[pred[wrong]] == cat_ids[target[wrong]]).mean()
    ) if wrong.any() else float("nan")
    cat_freq = np.bincount(cat_ids, minlength=cat_ids.max() + 1) / len(cat_ids)
    chance_same_cat = float((cat_freq ** 2).sum())
    length_gap = float(np.abs(n_chars[pred] - n_chars[target]).mean())
    uniq_n = np.array([n_chars[groups == g][0] for g in range(n_sent)])
    chance_length_gap = float(
        np.abs(uniq_n[:, None] - uniq_n[None, :]).mean()
    )

    # ---------------- RSA ----------------
    sent_mean = normalize(np.stack([
        eeg[groups == g].mean(axis=0) for g in range(n_sent)
    ]))
    text_mean = normalize(np.stack([
        text[g] for g in range(n_sent)
    ]))
    iu = np.triu_indices(n_sent, k=1)
    rsa = float(spearmanr(
        (sent_mean @ sent_mean.T)[iu], (text_mean @ text_mean.T)[iu]
    ).statistic)

    cat_of_sent = np.array([cat_ids[groups == g][0] for g in range(n_sent)])
    cat_diag = category_diagnostics(
        eeg, text, sent_mean, text_mean, target, cat_ids, cat_of_sent
    )

    # ---------------- 每句指标 ----------------
    per_sentence = []
    for g, sentence in enumerate(unique_sentences):
        rows = np.where(groups == g)[0]
        sub = eeg[rows]
        view_cos = sub @ sub.T
        view_consistency = float(
            view_cos[~np.eye(len(rows), dtype=bool)].mean()
        ) if len(rows) > 1 else float("nan")
        correct_cos = float((sub @ text[g]).mean())
        masked = sim_all[rows].copy()
        masked[:, g] = -np.inf
        best_wrong = float(masked.max(axis=1).mean())
        per_sentence.append({
            "sentence": sentence,
            "n_chars": int(n_chars[rows][0]),
            "category": meta["category_name"].iloc[rows[0]],
            "n_views": len(rows),
            "top1": float((ranks[rows] == 1).mean()),
            "median_rank": float(np.median(ranks[rows])),
            "view_consistency": view_consistency,
            "cos_correct": correct_cos,
            "cos_best_wrong": best_wrong,
            "margin": correct_cos - best_wrong,
        })
    per_sentence_df = pd.DataFrame(per_sentence).sort_values("top1")

    # ---------------- 每 run loss（训练日志） ----------------
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs_df = None
    epochs_path = ckpt_path.parent / f"{args.subject}_train_epochs.csv"
    if epochs_path.exists():
        epochs_df = pd.read_csv(epochs_path)

    per_run_df = None
    steps_path = ckpt_path.parent / f"{args.subject}_train_steps.csv"
    if steps_path.exists():
        steps = pd.read_csv(steps_path)
        last = steps[steps["epoch"] > steps["epoch"].max() - 10]
        per_run_df = (
            last.groupby(["day", "run"])["loss"].mean().reset_index()
            .sort_values("loss")
        )
        per_run_df.to_csv(out_dir / "loss_by_run.csv", index=False)

    # ---------------- 出图 ----------------
    plot_diagnostics(
        out_dir / "fig_diagnostics.png", epochs_df, sim_trained, sim_random,
        by_length, {"raw": e2t["raw"], "run-center": e2t["run-center"],
                    "day-center": e2t["day-center"], "len-resid": e2t["len-resid"],
                    "cat-center": e2t["cat-center"]},
        random_top1, probe,
    )
    plot_simmatrix(out_dir / "fig_simmatrix.png", sent_mean)
    per_sentence_df.to_csv(out_dir / "per_sentence.csv", index=False, encoding="utf-8-sig")

    # ---------------- 汇总 ----------------
    final_loss = float(ckpt.get("train_loss", float("nan")))
    summary = {
        "ckpt": str(ckpt_path),
        "ckpt_epoch": ckpt.get("epoch"),
        "candidates": {
            "n_trial": int(len(meta)),
            "n_unique_sentence": n_sent,
            "random_e2t_top1": random_top1,
            "random_e2t_top5": random_top5,
        },
        "train_loss": {
            "final": final_loss,
            "ln_n_candidates": float(np.log(n_sent)),
            "effective_candidates": float(np.exp(final_loss)),
            "shrink_vs_random": float(n_sent / np.exp(final_loss)),
        },
        "e2t": e2t,
        "t2e": t2e,
        "similarity": sim_trained,
        "similarity_random": sim_random,
        "cross_run_eeg": cross_run,
        "cross_run_eeg_random": cross_run_random,
        "probe": probe,
        "by_length": by_length,
        "errors": {
            "n_wrong": int(wrong.sum()),
            "same_category_rate_wrong": same_cat_wrong,
            "same_category_rate_chance": chance_same_cat,
            "mean_length_gap": length_gap,
            "mean_length_gap_chance": chance_length_gap,
        },
        "rsa_spearman_with_labse": rsa,
        "category_diagnostics": cat_diag,
        "loss_by_run": None if per_run_df is None else per_run_df.to_dict("records"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- 打印 ----------------
    print("\n" + "=" * 78)
    print("一、训练 loss")
    print("=" * 78)
    print(
        f"  最终 train_loss = {final_loss:.4f}\n"
        f"  ln(252)         = {np.log(n_sent):.4f}（完全随机）\n"
        f"  e^loss          = {np.exp(final_loss):.2f} 个有效候选（相对 252 缩了 "
        f"{n_sent / np.exp(final_loss):.1f}×）"
    )
    if per_run_df is not None:
        print("\n  最后 10 epoch 每个 run 的平均 loss（低=该 run 更容易）：")
        for _, row in per_run_df.iterrows():
            print(f"    day{int(row['day'])} run{int(row['run'])}  {row['loss']:.4f}")

    print("\n" + "=" * 78)
    print("二、检索指标")
    print("=" * 78)
    print(f"  {'空间':<14}{'e2t@1':>9}{'e2t@5':>9}{'mean rank':>11}")
    for name, m in e2t.items():
        print(f"  {name:<14}{m['top1']:>9.4f}{m['top5']:>9.4f}{m['mean_rank']:>11.1f}")
    print(f"  {'random':<14}{random_top1:>9.4f}{random_top5:>9.4f}{'—':>11}")
    print(
        f"\n  t2e（文本→trial，4032 候选）: top1={t2e['top1']:.4f} "
        f"top5={t2e['top5']:.4f} mean_rank={t2e['mean_rank']:.1f}"
    )

    print("\n" + "=" * 78)
    print("三、run 是否还在表示里")
    print("=" * 78)
    print(
        f"  run 线性探针（16 类）  = {probe['run_accuracy']['mean']:.4f} "
        f"± {probe['run_accuracy']['std']:.4f}  (chance {1 / len(np.unique(run_ids)):.4f})"
    )
    print(
        f"  day 线性探针（4 类）   = {probe['day_accuracy']['mean']:.4f} "
        f"± {probe['day_accuracy']['std']:.4f}"
    )
    if "random_run_accuracy" in probe:
        print(
            f"  随机模型 run 探针      = {probe['random_run_accuracy']['mean']:.4f}  "
            f"（输入层面本来就有多少 run 信息）"
        )
    print("\n  EEG×EEG 相似度：")
    for name, value in sim_trained.items():
        ref = "" if sim_random is None else f"   随机模型 {sim_random[name]:+.4f}"
        print(f"    {name:<28}{value:+.4f}{ref}")
    print(
        "\n  读法：run-center 与 raw 几乎相同 -> run 分量不影响排序；"
        "\n        run-center 明显更高 -> run 在稀释句子信号；"
        "\n        run-center 明显更低 -> run 分量带有句子相关（位置/上下文）信息。"
    )

    print("\n" + "=" * 78)
    print("四、跨 run 稳定性（EEG→EEG，252 候选）")
    print("=" * 78)
    print(
        f"  训练后：top1={cross_run['top1']:.4f}  top5={cross_run['top5']:.4f}  "
        f"mean_rank={cross_run['mean_rank']:.1f}  （{cross_run['n_pairs']} 个 run 对）"
    )
    if cross_run_random is not None:
        print(
            f"  随机初始化：top1={cross_run_random['top1']:.4f}  "
            f"top5={cross_run_random['top5']:.4f}  "
            f"（随机基线 {cross_run_random['random_top1']:.4f}）"
        )
    print(
        "  读法：跨 run top1 高 -> 同一句在不同记录段的表征稳定；"
        "\n        若 e2t 高但跨 run 低 -> 更像按 run 分别记忆。"
    )

    print("\n" + "=" * 78)
    print("五、下一个捷径")
    print("=" * 78)
    print(
        f"  句长 ridge 探针：R²={probe['length_r2']['r2']:.4f}  "
        f"MAE={probe['length_r2']['mae']:.3f} 字  "
        f"ρ={probe['length_r2']['spearman']:+.4f}"
    )
    print(
        f"  run 内位置 ridge 探针：R²={probe['trial_index_r2']['r2']:.4f}  "
        f"ρ={probe['trial_index_r2']['spearman']:+.4f}"
    )
    print(
        f"  类别探针（{len(np.unique(cat_ids))} 类）："
        f"{probe['category_accuracy']['mean']:.4f} ± {probe['category_accuracy']['std']:.4f}"
    )
    print(f"  按句长 top-1（n_trial 为样本数）：")
    for row in by_length:
        print(
            f"    {row['n_chars']:>2} 字  n={row['n_trial']:>4}  "
            f"top1={row['top1']:.4f}  top5={row['top5']:.4f}  mean_rank={row['mean_rank']:.1f}"
        )
    print(
        f"\n  错误 top-1 与真句同类别比例 = {same_cat_wrong:.4f}"
        f"（随机基线 {chance_same_cat:.4f}）"
    )
    print(
        f"  错误 top-1 与真句平均句长差 = {length_gap:.3f} 字"
        f"（随机基线 {chance_length_gap:.3f}）"
    )
    print(f"\n  EEG 句子相似度 vs LaBSE 文本相似度：Spearman ρ={rsa:+.4f}")

    print("\n" + "=" * 78)
    print("六、类别 vs 语义")
    print("=" * 78)
    code_to_name = {}
    for code, name in zip(cat_ids, meta["category_name"].fillna("?")):
        code_to_name.setdefault(int(code), name)
    print("  类别内检索（候选 = 同类别句子；文本列是 LaBSE 的同类别上限）：")
    print(f"    {'类别':<22}{'句数':>5}{'e2t@1':>8}{'e2t@5':>8}{'文本@1':>8}{'随机@1':>9}")
    for code, row in cat_diag["within_category"].items():
        print(
            f"    {code_to_name.get(int(code), code):<22}{row['n_sentence']:>5}"
            f"{row['e2t_top1']:>8.4f}{row['e2t_top5']:>8.4f}"
            f"{row['text_top1']:>8.4f}{row['random_top1']:>9.4f}"
        )
    print(
        f"\n  类别内 RSA（大类）：ρ={cat_diag['rsa_within_big_categories']:+.4f}"
        f"\n  类别中心化后 RSA：大类 ρ={cat_diag['rsa_category_centered_big']:+.4f}  "
        f"全部 ρ={cat_diag['rsa_category_centered_all']:+.4f}"
    )
    tc = cat_diag["top5_same_category"]
    print(
        f"\n  top-5 命中中同类别的比例：EEG={tc['eeg']:.4f}  "
        f"LaBSE 文本={tc['text']:.4f}  随机={tc['chance']:.4f}"
    )

    print("\n" + "=" * 78)
    print("七、逐句")
    print("=" * 78)
    best = per_sentence_df.iloc[-5:][["sentence", "n_chars", "top1", "median_rank", "margin"]]
    worst = per_sentence_df.iloc[:5][["sentence", "n_chars", "top1", "median_rank", "margin"]]
    print("  最稳定：")
    for _, row in best.iterrows():
        print(f"    top1={row['top1']:.2f} rank={row['median_rank']:.0f} margin={row['margin']:+.3f}  {row['sentence']}")
    print("  最差：")
    for _, row in worst.iterrows():
        print(f"    top1={row['top1']:.2f} rank={row['median_rank']:.0f} margin={row['margin']:+.3f}  {row['sentence']}")

    print(f"\n[Output] {out_dir / 'summary.json'}")
    print(f"[Output] {out_dir / 'per_sentence.csv'}")
    print(f"[Output] {out_dir / 'fig_diagnostics.png'}")
    print(f"[Output] {out_dir / 'fig_simmatrix.png'}")


if __name__ == "__main__":
    main()
