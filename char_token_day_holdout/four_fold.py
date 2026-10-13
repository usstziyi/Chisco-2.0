"""char_token_day_holdout 的 leave-one-day-out 四折入口。

在 ``--days`` 指定的 day 池上做"留一天"交叉验证：每一折把其中一天整体留给 val，
其余天做 train，跑完整的训练 + 评测（直接复用 ``train.py`` 的 ``run_experiment``），
最后把所有折的 best 指标汇总成 mean ± std。

不改 ``train.py``：每折写进同一个时间戳目录下的子目录 ``val-day-DD/``，各自保存
checkpoint / CSV，根目录再写一份汇总 CSV；LaBSE 只加载一次，四折复用。

注意：折与折之间只换"被留出的那一天"，train / val 的句子集合仍完全重合，量化的
是"留哪天"的敏感性（跨天泛化的核心不确定性）；不要跨 para（句子池不同）混折。

用法::

    uv run python char_token_day_holdout/four_fold.py                  # para1: days 1 2 3 4
    uv run python char_token_day_holdout/four_fold.py --days 5 6 7 8   # para2
    uv run python char_token_day_holdout/four_fold.py --epochs 20 --n-sentences 32 --n-repeats 3
"""

from __future__ import annotations

import argparse
import copy
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

# 支持 `python char_token_day_holdout/four_fold.py` 直接执行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.text_encoder import LaBSETextEncoder
from char_token.train import dump_args_json
from char_token_day_holdout.train import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_day_holdout"

# 汇总表里展示的指标（越大越好/越小越好见 char_token/train.py 的说明）
METRIC_KEYS = [
    "val_loss", "eeg2text_top1", "eeg2text_top5", "text2eeg_top1", "text2eeg_top5",
]


def build_folds(days):
    """把 day 池展开成 leave-one-day-out 的折：返回 ``[(train_days, val_day), ...]``。"""
    days = sorted({int(d) for d in days})
    if len(days) < 2:
        raise ValueError(f"leave-one-out 至少需要 2 天，收到 {days}")
    return [([d for d in days if d != val_day], val_day) for val_day in days]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="char_token_day_holdout 的 leave-one-day-out 四折 + 自动汇总（复用 train.py）。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试，如 sub-01（默认）")
    parser.add_argument(
        "--days", type=int, nargs="+", default=[1, 2, 3, 4],
        help="做留一法的 day 池；默认 1 2 3 4（para1），para2 用 5 6 7 8",
    )
    parser.add_argument(
        "--event", default="inner", choices=["inner", "reading"],
        help="取 inner（默读，默认）或 reading（朗读）",
    )
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument("--char-duration", type=float, default=0.4, help="每个字 token 的时长（秒，默认 0.4）")
    parser.add_argument(
        "--n-sentences", type=int, default=18,
        help="每个训练 batch 里不同句子的数量（默认 18）",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=3,
        help="每个句子在每个 batch 里抽多少条 trial 当正样本（默认 3）",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="验证集 DataLoader 的 batch 大小（默认 32）")
    parser.add_argument("--epochs", type=int, default=30, help="每折训练轮数（默认 30）")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率（默认 1e-3）")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW 权重衰减（默认 1e-3）")
    parser.add_argument(
        "--lr-schedule", default="cosine", choices=["none", "cosine"],
        help="学习率调度：cosine 按 epoch 余弦衰减到 0（默认），none 恒定",
    )
    parser.add_argument(
        "--select-by", default="val_loss",
        choices=["val_loss", "eeg2text_top1", "eeg2text_top5", "text2eeg_top5"],
        help="用哪个指标选 best checkpoint（默认 val_loss，越小越好；其余越大越好）",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="对比学习温度系数（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0；各折共用）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args(argv)


def _mean_std(values):
    """返回 (mean, std)；std 用样本标准差，n<2 时为 0。"""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    args = parse_args()
    folds = build_folds(args.days)
    days_label = ",".join(f"{d:02d}" for d in sorted({int(d) for d in args.days}))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"[Four-Fold] device={device} subject={args.subject} days={days_label} "
        f"折数={len(folds)} ({len(folds)}-fold leave-one-day-out)\n"
        f"            temperature={args.temperature} "
        f"batch={args.n_sentences}句×{args.n_repeats}视图 epochs={args.epochs} "
        f"lr={args.lr} wd={args.weight_decay} schedule={args.lr_schedule} "
        f"select={args.select_by} seed={args.seed}"
    )

    # 所有折共用一个时间戳目录，各折写进 val-day-DD/ 子目录
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # LaBSE 冻结，只加载一次，各折复用
    print("[Text] 加载 LaBSE（冻结） ...")
    text_encoder = LaBSETextEncoder()

    rows = []
    for fold_idx, (train_days, val_day) in enumerate(folds, start=1):
        fold_args = copy.deepcopy(args)
        fold_args.train_days = train_days
        fold_args.val_days = [val_day]

        fold_dir = out_dir / f"fold{fold_idx}_val-day{val_day:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        dump_args_json(fold_args, fold_dir)

        print(
            f"\n{'#' * 88}\n"
            f"[Fold {fold_idx}/{len(folds)}] train day={train_days} -> val day={val_day}\n"
            f"{'#' * 88}"
        )
        best_score, best_metrics, info = run_experiment(
            fold_args, fold_dir, text_encoder, device
        )
        rows.append({
            "fold": fold_idx,
            "val_day": val_day,
            "train_days": " ".join(str(d) for d in train_days),
            "best_score": best_score,
            **{k: best_metrics[k] for k in METRIC_KEYS},
            "n_val": best_metrics["n_val"],
            "n_text_candidates": best_metrics["n_text_candidates"],
        })

    # ---------------- 汇总 ----------------
    header = (
        f"{'val-day':>8} {'val_loss':>10} {'e2t@1':>8} {'e2t@5':>8} "
        f"{'t2e@1':>8} {'t2e@5':>8}"
    )
    print(f"\n{'=' * 88}\n[四折汇总] best by {args.select_by}\n{'=' * 88}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{('day-%02d' % row['val_day']):>8} {row['val_loss']:>10.4f} "
            f"{row['eeg2text_top1']:>8.4f} {row['eeg2text_top5']:>8.4f} "
            f"{row['text2eeg_top1']:>8.4f} {row['text2eeg_top5']:>8.4f}"
        )
    print("-" * len(header))

    agg = {k: _mean_std([row[k] for row in rows]) for k in METRIC_KEYS}
    print(
        f"{'mean±std':>8} "
        f"{agg['val_loss'][0]:>10.4f} "
        f"{agg['eeg2text_top1'][0]:>8.4f} {agg['eeg2text_top5'][0]:>8.4f} "
        f"{agg['text2eeg_top1'][0]:>8.4f} {agg['text2eeg_top5'][0]:>8.4f}"
    )
    print(
        f"{'±std':>8} "
        f"{agg['val_loss'][1]:>10.4f} "
        f"{agg['eeg2text_top1'][1]:>8.4f} {agg['eeg2text_top5'][1]:>8.4f} "
        f"{agg['text2eeg_top1'][1]:>8.4f} {agg['text2eeg_top5'][1]:>8.4f}"
    )

    # 每折一行 + 一个 mean/std 行，写进汇总 CSV
    summary_path = out_dir / "four_fold_summary.csv"
    fields = [
        "fold", "val_day", "train_days", "best_score",
        *METRIC_KEYS, "n_val", "n_text_candidates",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "fold": "mean",
            "val_day": "-",
            "train_days": "-",
            "best_score": _mean_std([r["best_score"] for r in rows])[0],
            **{k: agg[k][0] for k in METRIC_KEYS},
        })
        writer.writerow({
            "fold": "std",
            "val_day": "-",
            "train_days": "-",
            "best_score": _mean_std([r["best_score"] for r in rows])[1],
            **{k: agg[k][1] for k in METRIC_KEYS},
        })

    print(
        f"\n[Summary] {summary_path}\n"
        f"[Output]  各折产物在 {out_dir}/fold*/ （best.pt / train_epochs.csv / train_steps.csv）"
    )


if __name__ == "__main__":
    main()
