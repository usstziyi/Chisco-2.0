"""``char_token_runbatch`` 三个绘图脚本共用的加载 / 编码 / 排序 / 绘图工具。

三个脚本：

- ``plot_eeg_matrix.py``      EEG × EEG 自相似矩阵
- ``plot_cross_matrix.py``    EEG × sentence 跨模态矩阵
- ``plot_text_matrix.py``     sentence × sentence 文本自相似矩阵

默认都取 **一个 (day, run) 的 252 条 trial**，因为 run-batch 实验里每个 run
恰好包含全部 252 句、每句一次。这样三张图都是 252×252，可以直接并排比较；
跨模态图的对角线正好是"第 i 条 trial ↔ 它自己的句子"。

本模块只依赖本目录的 dataset / model / text_encoder，不引用 ``char_token``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_runbatch.common import build_text_lookup
from char_token_runbatch.dataset import CharTokenEEGDataset, char_token_collate
from char_token_runbatch.model import CharTokenEEGEncoder
from char_token_runbatch.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs" / "char_token_runbatch"
TEXT_TABLE = PROJECT_ROOT / "datasets" / "preprocessed" / "all_text.tsv"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 参数 / 配置


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """三个脚本共用的参数。"""
    parser.add_argument("--ckpt", default=None, help="checkpoint；默认取 outputs/char_token_runbatch 下最新")
    parser.add_argument("--subject", default=None, help="被试；默认从 checkpoint 文件名推断")
    parser.add_argument("--day", type=int, nargs="+", default=None, help="day 编号；默认沿用训练参数")
    parser.add_argument("--event", default=None, choices=["inner", "reading"], help="默认沿用训练参数")
    parser.add_argument("--sample", type=int, default=None, help="采样率 Hz；默认沿用训练参数")
    parser.add_argument("--char-duration", type=float, default=None, help="每个字 token 时长（秒）")
    parser.add_argument("--batch-size", type=int, default=64, help="EEG 编码 batch 大小（默认 64）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=None, help="图输出目录；默认 checkpoint 同级 analysis/")
    parser.add_argument(
        "--order", default="raw", choices=["raw", "category", "sentence"],
        help=(
            "绘图顺序：raw=trial_index 顺序（默认）；category=按类别聚块；"
            "sentence=按句子聚在一起（同句相邻）"
        ),
    )
    return parser


def subject_from_ckpt(path: Path) -> str:
    stem = path.stem
    for suffix in ("_best_train", "_best", "_last"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_ckpt(ckpt: str | None, subject: str | None) -> Path:
    if ckpt is not None:
        path = Path(ckpt)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        return path

    name = f"{subject}_best_train.pt" if subject else "*_best_train.pt"
    candidates = sorted(OUT_ROOT.glob(f"*/{name}"), key=lambda p: p.parent.name)
    if not candidates:
        raise FileNotFoundError(f"{OUT_ROOT} 下没有 */{name}，请用 --ckpt 指定")
    return candidates[-1]


def resolve_config(args) -> dict:
    """合并命令行与 checkpoint 里的训练配置。"""
    ckpt_path = find_ckpt(args.ckpt, args.subject)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args") or {}

    return {
        "ckpt_path": ckpt_path,
        "ckpt": ckpt,
        "subject": args.subject or subject_from_ckpt(ckpt_path),
        "days": args.day or train_args.get("day") or [5, 6, 7, 8],
        "event": args.event or train_args.get("event") or "inner",
        "sample": args.sample or int(train_args.get("sample", 250)),
        "char_duration": (
            args.char_duration if args.char_duration is not None
            else float(train_args.get("char_duration", 0.4))
        ),
        "device": torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        "out_dir": Path(args.out_dir) if args.out_dir else ckpt_path.parent / "analysis",
    }


def days_tag(days) -> str:
    """把 day 列表压成文件名片段：连续写 ``5-8``，否则 ``5_6_7_8``。"""
    days = [int(d) for d in days]
    if days == list(range(days[0], days[-1] + 1)):
        return f"{days[0]}-{days[-1]}" if len(days) > 1 else str(days[0])
    return "_".join(str(d) for d in days)


# ---------------------------------------------------------------- 数据 / 模型


def build_dataset(cfg: dict, touch: bool = True) -> CharTokenEEGDataset:
    dataset = CharTokenEEGDataset(
        event=cfg["event"],
        sample=cfg["sample"],
        subject=cfg["subject"],
        days=cfg["days"],
        char_duration=cfg["char_duration"],
    )
    if touch:
        dataset[0]  # 读一条，确定 n_chans 并尽早暴露 FIF 问题
    return dataset


def load_model(ckpt: dict, dataset: CharTokenEEGDataset | None, device) -> CharTokenEEGEncoder:
    """载入 EEG 编码器；兼容训练时 token encoder 被 ChunkedTokenEncoder 包过一层。"""
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

    if dataset is not None:
        if dataset.n_chans != int(ckpt["n_chans"]):
            raise ValueError(
                f"数据通道数 {dataset.n_chans} 与 checkpoint 的 {ckpt['n_chans']} 不一致"
            )
        if dataset.token_samples != int(ckpt["token_samples"]):
            raise ValueError(
                f"token 长度 {dataset.token_samples} 与 checkpoint 的 {ckpt['token_samples']} 不一致"
            )
    return model


def load_meta(dataset: CharTokenEEGDataset) -> pd.DataFrame:
    """dataset.meta 补上 category_name / category_number（用于按类别排序）。"""
    meta = dataset.meta.copy()
    table = pd.read_csv(TEXT_TABLE, sep="\t")
    key = ["sample", "subject", "day", "run", "trial_index"]
    return meta.merge(
        table[key + ["category_name", "category_number"]],
        on=key, how="left", validate="many_to_one",
    )


@torch.no_grad()
def encode_eeg(model, dataset, positions, device, batch_size: int) -> np.ndarray:
    """按给定 trial 下标编码，返回 ``(N, D)`` 的 numpy 数组（已 L2 归一化）。"""
    loader = DataLoader(
        Subset(dataset, list(positions)),
        batch_size=batch_size, shuffle=False, collate_fn=char_token_collate,
    )
    parts = [model(tokens.to(device), mask.to(device)) for tokens, mask, _ in loader]
    return torch.cat(parts, dim=0).cpu().numpy()


def load_text_embeddings(dataset: CharTokenEEGDataset) -> dict:
    """LaBSE 编码全部唯一句，返回 ``{sentence: (D,) tensor}``（冻结，与 ckpt 无关）。"""
    print("[Text] 加载 LaBSE（冻结） ...")
    encoder = LaBSETextEncoder()
    lookup, _ = build_text_lookup(
        encoder, dataset.meta["sentence"].tolist(), torch.device("cpu")
    )
    return lookup


# ---------------------------------------------------------------- 选择 / 排序


def select_run(meta: pd.DataFrame, day: int | None, run: int | None) -> tuple[np.ndarray, tuple[int, int]]:
    """选一个 (day, run) 的全部 trial；默认取数据里第一个 run。"""
    available = list(dict.fromkeys(zip(meta["day"].astype(int), meta["run"].astype(int))))
    if not available:
        raise ValueError("meta 为空")

    day = int(day) if day is not None else available[0][0]
    runs_in_day = [r for d, r in available if d == day]
    run = int(run) if run is not None else runs_in_day[0]
    if (day, run) not in available:
        raise ValueError(f"(day, run)=({day}, {run}) 不在数据里；可选 {available}")

    rows = np.where((meta["day"].to_numpy() == day) & (meta["run"].to_numpy() == run))[0]
    return rows, (day, run)


def order_positions(meta: pd.DataFrame, positions, order: str) -> np.ndarray:
    """在给定 trial 子集内排序，返回排序后的原始 trial 下标。"""
    positions = np.asarray(positions, dtype=int)
    sub = meta.iloc[positions].copy()
    sub["_pos"] = np.arange(len(positions))

    if order == "raw":
        keys = ["day", "run", "trial_index"]
    elif order == "category":
        keys = ["category_name", "day", "run", "trial_index"]
    else:
        keys = ["sentence", "day", "run"]

    sub = sub.sort_values(keys + ["_pos"], kind="stable", na_position="last")
    return positions[sub["_pos"].to_numpy()]


def unique_sentence_rows(meta: pd.DataFrame) -> pd.DataFrame:
    """每个句子取首次出现的行，索引仍是 dataset 的 trial 下标。"""
    return meta.drop_duplicates("sentence", keep="first")


# ---------------------------------------------------------------- 相似度 / 中心化


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def center_by(embeddings: np.ndarray, groups) -> np.ndarray:
    """按组减均值再 L2 归一化。"""
    out = embeddings.copy()
    groups = np.asarray(groups)
    for key in np.unique(groups):
        rows = groups == key
        out[rows] -= out[rows].mean(axis=0, keepdims=True)
    return normalize(out)


def similarity_stats(sim: np.ndarray, sentences, runs, days) -> None:
    """打印同句 / 异句、同 run 跨天的同句 / 异句平均余弦。"""
    n = sim.shape[0]
    off = ~np.eye(n, dtype=bool)
    sent = np.asarray(sentences)
    run = np.asarray(runs)
    day = np.asarray(days)
    same_sent = sent[:, None] == sent[None, :]
    same_run_cross_day = (run[:, None] == run[None, :]) & (day[:, None] != day[None, :])

    def mean_of(mask):
        return float(sim[mask].mean()) if mask.any() else float("nan")

    print(
        f"  同句 mean={mean_of(off & same_sent):+.4f}  "
        f"异句 mean={mean_of(off & ~same_sent):+.4f}  "
        f"差={mean_of(off & same_sent) - mean_of(off & ~same_sent):+.4f}\n"
        f"  同 run 跨天：同句 mean={mean_of(same_run_cross_day & same_sent):+.4f}  "
        f"异句 mean={mean_of(same_run_cross_day & ~same_sent):+.4f}"
    )


# ---------------------------------------------------------------- 绘图


def plot_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    n_rows, n_cols = matrix.shape
    long_side = max(n_rows, n_cols)

    if long_side <= 400:
        figsize = (10.0, 8.8)
    elif long_side <= 1200:
        figsize = (11.0, 12.0)
    else:
        figsize = (12.0, 16.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix, cmap="viridis", aspect="auto",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
