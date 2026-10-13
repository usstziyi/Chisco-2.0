"""跨被试预测（zero-shot 跨被试检索）评测入口。

与 ``char_token/train.py`` 的唯一区别在 **评测的模型与数据来自不同被试**：

- ``--source`` 决定加载哪个被试的 best checkpoint（``{source}_best.pt``）；
- ``--target`` 决定用哪个被试的 EEG 数据做预测。

两者按位置一一配对，因此一条命令就能跑完任意方向，例如双向互测::

    uv run python char_token_cross_subject/predict.py \
        --source sub-01 sub-02 --target sub-02 sub-01

模型从未见过 target 的脑电，所以得到的是**零样本跨被试**检索成绩；文本候选池是
target 的句子（被试间共享同一批句子）。损失、检索口径、随机基线全部复用
``char_token.train.evaluate``，可与同被试的 val 指标直接对比。

``val_loss`` 的随机基线是 ``ln(候选数)``：它在整个候选池上一次性算 N×N 矩阵，EEG 与
文本完全不对齐时退化成 ``ln(N)``（N = 候选条数，不是 batch 大小）。实测值越接近这个
基线，说明跨被试迁移越弱。

用法::

    # 默认：用 sub-01_best.pt 预测 sub-02 的 day 1~4
    uv run python char_token_cross_subject/predict.py

    # 单侧：sub-01 -> sub-02（等价于默认）或反过来 sub-02 -> sub-01
    uv run python char_token_cross_subject/predict.py --source sub-01 --target sub-02
    uv run python char_token_cross_subject/predict.py --source sub-02 --target sub-01

    # 双向互测：sub-01 -> sub-02 与 sub-02 -> sub-01
    uv run python char_token_cross_subject/predict.py --source sub-01 sub-02 --target sub-02 sub-01

    # 显式指定权重目录 / 单个权重文件
    uv run python char_token_cross_subject/predict.py --ckpt-dir outputs/char_token/20261013_161958
    uv run python char_token_cross_subject/predict.py --source sub-01 --target sub-02 \
        --ckpt outputs/char_token/20261013_161958/sub-01_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 支持 `python char_token_cross_subject/predict.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import CharTokenEEGDataset, char_token_collate
from char_token.model import CharTokenEEGEncoder
from char_token.text_encoder import LaBSETextEncoder

# 复用 char_token 的同一份评测实现，保证口径一致
from char_token.train import build_text_lookup, dump_args_json, evaluate, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "outputs" / "char_token"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_cross_subject"


def parse_args():
    parser = argparse.ArgumentParser(
        description="跨被试零样本预测：用 source 的 best checkpoint 预测 target 的数据。"
    )
    parser.add_argument(
        "--source", nargs="+", default=["sub-01"],
        help="提供训练权重的被试（读 {source}_best.pt），与 --target 按位置一一配对；默认 sub-01",
    )
    parser.add_argument(
        "--target", nargs="+", default=["sub-02"],
        help="被预测的被试（数据来源），与 --source 按位置一一配对；默认 sub-02",
    )
    parser.add_argument(
        "--ckpt-dir", default=None,
        help="存放 {subject}_best.pt 的目录；默认在 outputs/char_token/*/ 里自动找最新的",
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="显式指定单个权重文件；仅允许单方向，会覆盖 --ckpt-dir",
    )
    parser.add_argument(
        "--day", type=int, nargs="+", default=[1, 2, 3, 4],
        help="target 的 day 编号（1~8），可传多个，如 --day 1 2 3 4；默认 1 2 3 4",
    )
    parser.add_argument(
        "--event", default="inner", choices=["inner", "reading"],
        help="取 inner（默读，默认）或 reading（朗读）",
    )
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument("--char-duration", type=float, default=0.4, help="每个字 token 的时长（秒，默认 0.4）")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch 大小（默认 32）")
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="对比学习温度；默认沿用 checkpoint 内的训练值",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args()


def find_ckpt(source: str, ckpt_dir: str | None) -> Path:
    """定位 ``{source}_best.pt``：给定目录优先，否则在 outputs/char_token/*/ 里取最新。"""
    name = f"{source}_best.pt"
    if ckpt_dir is not None:
        path = Path(ckpt_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"找不到权重: {path}")
        return path

    # 目录名是时间戳，按名字降序即"最新的在前"
    candidates = sorted(
        DEFAULT_CKPT_ROOT.glob(f"*/{name}"),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {DEFAULT_CKPT_ROOT} 下未找到 {name}，请用 --ckpt-dir 或 --ckpt 指定"
        )
    return candidates[0]


def run_pair(args, source, target, ckpt_path, text_encoder, out_dir, device):
    """用 ``source`` 的权重预测 ``target`` 的数据，返回该方向的指标字典。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    train_args = ckpt.get("args") or {}
    n_chans = int(ckpt["n_chans"])
    token_samples = int(ckpt["token_samples"])
    embedding_dim = int(ckpt["embedding_dim"])
    # 温度必须与训练一致，否则相似度尺度不同、检索排名会变
    temperature = (
        args.temperature if args.temperature is not None
        else float(train_args.get("temperature", 0.2))
    )

    print(f"\n{'=' * 88}\n[Pair] {source} -> {target}\n{'=' * 88}")
    print(
        f"[Ckpt] {ckpt_path}\n"
        f"       训练 epoch={ckpt.get('epoch')} n_chans={n_chans} "
        f"token_samples={token_samples} embedding_dim={embedding_dim} temperature={temperature}"
    )
    if train_args:
        print(
            f"       训练配置 subject={train_args.get('subject')} day={train_args.get('day')} "
            f"event={train_args.get('event')} select_by={train_args.get('select_by')}"
        )

    # ---------------- target 数据 ----------------
    dataset = CharTokenEEGDataset(
        event=args.event,
        sample=args.sample,
        subject=target,
        days=args.day,
        char_duration=args.char_duration,
    )
    dataset[0]  # 触发一次读取，确定 n_chans
    if dataset.n_chans != n_chans:
        raise ValueError(
            f"通道数不匹配：target 数据 {dataset.n_chans} vs checkpoint {n_chans}"
        )
    if dataset.token_samples != token_samples:
        raise ValueError(
            f"token 长度不匹配：target 数据 {dataset.token_samples} vs checkpoint {token_samples}"
        )

    days_label = ",".join(f"{d:02d}" for d in sorted(args.day))
    print(
        f"[Data] {target} day-{days_label} event={args.event} "
        f"trial {len(dataset)} 条，句子 {len(dataset.meta['sentence'].unique())} 条"
    )

    # ---------------- 模型 ----------------
    model = CharTokenEEGEncoder(
        n_chans=n_chans,
        token_samples=token_samples,
        embedding_dim=embedding_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=char_token_collate,
        num_workers=args.num_workers,
    )

    # ---------------- 文本 embedding（冻结，按 target 句子现算） ----------------
    lookup, emb_dim = build_text_lookup(
        text_encoder, dataset.meta["sentence"].tolist(), device
    )
    if emb_dim != embedding_dim:
        raise ValueError(
            f"文本 embedding 维度 {emb_dim} 与 checkpoint {embedding_dim} 不一致"
        )

    # ---------------- 检索评测（复用 char_token 的口径） ----------------
    metrics = evaluate(model, loader, lookup, device, temperature)

    n_val = metrics["n_val"]
    n_cand = metrics["n_text_candidates"]
    pos_per_anchor = metrics["pos_per_anchor"]
    # 三个随机基线：两个检索 top-k 的期望命中率 + val_loss 的随机水平。
    # val_loss 在整个候选池上一次性算 N×N 矩阵，EEG 与文本完全不对齐时退化成 ln(N)
    # （N = 候选条数 n_val，不是 batch 大小）。
    baseline = {
        "eeg2text_top1": 1 / n_cand,
        "eeg2text_top5": min(5, n_cand) / n_cand,
        "text2eeg_top1": min(pos_per_anchor / n_val, 1.0),
        "text2eeg_top5": min(5 * pos_per_anchor / n_val, 1.0),
        "val_loss": math.log(n_val),
    }
    print(
        f"\n[Result] {source} -> {target}\n"
        f"         eeg2text top-1={metrics['eeg2text_top1']:.4f} "
        f"top-5={metrics['eeg2text_top5']:.4f}；"
        f"text2eeg top-1={metrics['text2eeg_top1']:.4f} "
        f"top-5={metrics['text2eeg_top5']:.4f}；val_loss={metrics['val_loss']:.4f}\n"
        f"         eeg2text 候选 {n_cand} 句，随机基线 top-1={baseline['eeg2text_top1']:.4f} "
        f"top-5={baseline['eeg2text_top5']:.4f}\n"
        f"         text2eeg 候选 {n_val} 条（每 anchor {pos_per_anchor:.2f} 个同句），"
        f"随机基线 top-1={baseline['text2eeg_top1']:.4f} "
        f"top-5={baseline['text2eeg_top5']:.4f}\n"
        f"         val_loss 随机基线 = ln(候选数 {n_val}) ≈ {baseline['val_loss']:.4f}"
        f"（全库一次性算，非 ln(batch)；越接近说明越无跨被试迁移）"
    )

    result = {
        "source": source,
        "target": target,
        "ckpt": str(ckpt_path),
        "ckpt_epoch": ckpt.get("epoch"),
        "temperature": temperature,
        "baseline": baseline,
        **metrics,
    }
    (out_dir / f"{source}_to_{target}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    args = parse_args()
    set_seed(args.seed)

    if len(args.source) != len(args.target):
        raise SystemExit(
            f"--source({len(args.source)}) 与 --target({len(args.target)}) 数量必须一致"
        )
    if args.ckpt is not None and len(args.source) != 1:
        raise SystemExit("--ckpt 只能配单方向使用；多方向请用 --ckpt-dir 或自动查找")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    pairs = [f"{s}->{t}" for s, t in zip(args.source, args.target)]
    print(
        f"[Cross-Subject] device={device} pairs={pairs} day={args.day} "
        f"event={args.event} sample={args.sample} char_duration={args.char_duration} "
        f"seed={args.seed}"
    )

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    print("[Text] 加载 LaBSE（冻结） ...")
    text_encoder = LaBSETextEncoder()

    summary = {}
    for source, target in zip(args.source, args.target):
        ckpt_path = (
            Path(args.ckpt) if args.ckpt is not None
            else find_ckpt(source, args.ckpt_dir)
        )
        result = run_pair(
            args, source, target, ckpt_path, text_encoder, out_dir, device
        )
        summary[f"{source}->{target}"] = result

    # ---------------- 汇总 ----------------
    if len(summary) > 1:
        print(f"\n{'=' * 88}\n[Summary] 跨被试预测\n{'=' * 88}")
        header = f"{'pair':>18} {'e2t@1':>8} {'e2t@5':>8} {'t2e@1':>8} {'t2e@5':>8} {'val_loss':>10}"
        print(header)
        print("-" * len(header))
        for pair, m in summary.items():
            print(
                f"{pair:>18} {m['eeg2text_top1']:>8.4f} {m['eeg2text_top5']:>8.4f} "
                f"{m['text2eeg_top1']:>8.4f} {m['text2eeg_top5']:>8.4f} "
                f"{m['val_loss']:>10.4f}"
            )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[Output] {out_dir}")


if __name__ == "__main__":
    main()
