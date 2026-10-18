"""绘制 best.pt 在**预测集**上的 EEG 句嵌入自相似矩阵。

做什么
------
1. 载入某个被试的 best checkpoint（默认自动取 ``outputs/char_token/*/`` 下最新的
   ``{subject}_best.pt``）；
2. 用 checkpoint 里存下的训练参数（subject / day / event / sample / char_duration）
   重建数据集，并按训练时**完全一致**的句子级切分（``val_ratio`` / ``seed``）取出预测集；
3. 用模型把预测集**全体 trial** 编码成 L2 归一化的句嵌入 ``(N, D)``；
4. 两两余弦相似度得到 ``N×N`` 矩阵（EEG×EEG 自相似），画成热图并存 PNG；
5. 再按 **run** 把矩阵切片，**每个 run 单独一张图**（与全图共用同一色标，便于直接对比）；
6. 可选做 **run 中心化去相关**：按 ``(day, run)`` 或 ``run`` 号减去组质心后再算余弦，
   把 run 的公共偏移从 embedding 里去掉。

为什么看这张图
--------------
同一句在 day 1~4 反复出现，因此预测集里同一句会有多条 trial。若模型学到了跨天的
句子级不变签名，**同句 trial 会互相靠得近**，矩阵在按句子分组后应出现清晰的亮块；
若只是弱签名，亮块会很淡、与句间相似度拉不开差距。脚本同时打印"同句内 / 句间"的
平均相似度，量化这个差距。

run 中心化
----------
本数据里每个 run 是一段固定的句子块，且同一句每天都在同一个 run；因此 raw 矩阵的
run 亮块可能来自"内容块"而不是句子身份。``--center-by`` 提供两种中心化：

- ``day-run``：按 ``(day, run)`` 减质心，去掉当天当段的记录/电极/状态偏移；
- ``run``：按 run 号**跨天**减质心，去掉这个内容块的公共方向；这是检验"同句是否
  还有独立于 run 块的信息"的更严格版本。

中心化后输出文件名会带 ``_centerdayrun`` / ``_centerrun`` 后缀，并额外打印
"同 run 跨天：同句 vs 不同句"的相似度差值。

绘图顺序
--------
- ``sentence``（默认）：按句子分组，同句 trial 排在一起（组内保持预测集原顺序），
  亮块结构最直观；
- ``raw``：沿用预测集的原始物理顺序（subject→day→run→trial_index）。

用法::

    uv run python char_token/plot_sim_matrix.py
    uv run python char_token/plot_sim_matrix.py --ckpt outputs/char_token/20261018_144532/sub-01_best.pt
    uv run python char_token/plot_sim_matrix.py --subject sub-01 --order raw
    uv run python char_token/plot_sim_matrix.py --order raw --center-by day-run
    uv run python char_token/plot_sim_matrix.py --order raw --center-by run
    uv run python char_token/plot_sim_matrix.py --start 0 --end 200   # 只看子矩阵放大
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

# 支持 `python char_token/plot_sim_matrix.py` 直接执行
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import CharTokenEEGDataset, char_token_collate
from char_token.model import CharTokenEEGEncoder

# 复用训练的切分实现，保证预测集口径与训练日志完全一致
from char_token.train import split_by_sentence

# 图上标题含中文，指定可用的中文字体，避免显示成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "outputs" / "char_token"


def parse_args():
    parser = argparse.ArgumentParser(description="加载 best.pt，在预测集全体 trial 上画 EEG 句嵌入自相似矩阵。")
    parser.add_argument("--ckpt", default=None, help="checkpoint 路径；默认自动取最新")
    parser.add_argument("--subject", default=None, help="被试（如 sub-01）；默认从 checkpoint 推断")
    parser.add_argument("--day", type=int, nargs="+", default=None, help="day 编号；默认沿用训练参数")
    parser.add_argument("--event", default=None, help="inner / reading；默认沿用训练参数")
    parser.add_argument("--sample", type=int, default=None, help="采样率 Hz；默认沿用训练参数")
    parser.add_argument("--char-duration", type=float, default=None, help="每个字 token 时长（秒）")
    parser.add_argument("--val-ratio", type=float, default=None, help="验证集句子比例；默认沿用训练参数")
    parser.add_argument("--seed", type=int, default=None, help="切分种子；默认沿用训练参数")
    parser.add_argument("--batch-size", type=int, default=64, help="编码时的 batch 大小（默认 64）")
    parser.add_argument("--order", default="sentence", choices=["sentence", "raw"],help="绘图顺序：sentence 按句子分组（默认），raw 用预测集原始顺序")
    parser.add_argument(
        "--center-by", default="none", choices=["none", "day-run", "run"],
        help=(
            "run 中心化去相关：none 不处理（默认）；day-run 按 (day,run) 减质心，"
            "去记录/状态偏移；run 按 run 号跨天减质心，去内容块公共方向"
        ),
    )
    parser.add_argument("--start", type=int, default=0, help="绘图起始下标（默认 0）")
    parser.add_argument("--end", type=int, default=None, help="绘图结束下标，不含（默认到末尾）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=None, help="图输出目录；默认存到 checkpoint 同级目录")
    return parser.parse_args()


def find_ckpt(ckpt: str | None, subject: str | None) -> Path:
    """定位 checkpoint：给了路径就用它，否则在默认目录下取最新的 ``{subject}_best.pt``。"""
    if ckpt is not None:
        path = Path(ckpt)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        return path

    pattern = f"*/{subject}_best.pt" if subject else "*/*_best.pt"
    # 目录名是时间戳，按名字降序即"最新的在前"
    candidates = sorted(
        DEFAULT_CKPT_ROOT.glob(pattern), key=lambda p: p.parent.name, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {DEFAULT_CKPT_ROOT} 下未找到 {pattern}，请用 --ckpt 显式指定"
        )
    return candidates[0]


def subject_from_ckpt(path: Path) -> str:
    """从 ``{subject}_best.pt`` 文件名反推被试名。"""
    stem = path.stem  # "sub-01_best"
    return stem[: -len("_best")] if stem.endswith("_best") else stem


@torch.no_grad()
def encode_indices(model, dataset, indices, device, batch_size) -> torch.Tensor:
    """对给定索引逐 batch 前向，返回 (N, D) 的 L2 归一化 EEG 句嵌入。"""
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=char_token_collate,
        num_workers=0,
    )
    model.eval()
    parts = [model(tokens.to(device), mask.to(device)) for tokens, mask, _ in loader]
    return torch.cat(parts, dim=0)


def group_order(sentences) -> list[int]:
    """按句子分组排序：同句 trial 相邻（组内保持原顺序），组间按首次出现顺序。"""
    groups: dict[str, list[int]] = {}
    for pos, sentence in enumerate[Any](sentences):
        groups.setdefault(sentence, []).append(pos)
    return [pos for members in groups.values() for pos in members]


def center_embeddings(embeddings: torch.Tensor, groups) -> torch.Tensor:
    """按 ``groups`` 分组减去组质心，再重新 L2 归一化。

    ``groups`` 与 ``embeddings`` 的每一行一一对应。中心化把每个组的公共偏移
    （run 的记录条件 / 内容块方向）平移掉，只保留相对本组的偏差。
    """
    groups = np.asarray(groups)
    centered = embeddings.clone()
    for group in np.unique(groups):
        mask = torch.from_numpy(np.ascontiguousarray(groups == group)).to(centered.device)
        centered[mask] = centered[mask] - centered[mask].mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(centered, p=2, dim=-1)


def similarity_stats(sim, sentences, runs, days, title: str) -> None:
    """打印同句 / 句间，以及同 run 跨天内的同句 / 不同句平均相似度。

    第二条"同 run 跨天"才是排除 run/主题块混淆后的句子级检验：本数据里同一句
    每天都在同一个 run，所以只看全局同句 vs 句间会被 run 块结构严重污染。
    """
    n = sim.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    sent = np.asarray(sentences)
    run = np.asarray(runs)
    day = np.asarray(days)

    same_sent = sent[:, None] == sent[None, :]
    same_run_cross_day = (run[:, None] == run[None, :]) & (day[:, None] != day[None, :])

    def mean_of(values) -> float:
        return float(values.mean()) if values.size else float("nan")

    within = mean_of(sim[off_diag & same_sent])
    between = mean_of(sim[off_diag & ~same_sent])
    run_same = mean_of(sim[same_run_cross_day & same_sent])
    run_diff = mean_of(sim[same_run_cross_day & ~same_sent])
    n_same = int((off_diag & same_sent).sum() // 2)
    n_between = int((off_diag & ~same_sent).sum() // 2)
    n_run_same = int((same_run_cross_day & same_sent).sum() // 2)
    n_run_diff = int((same_run_cross_day & ~same_sent).sum() // 2)

    print(
        f"[Sim] {title}\n"
        f"      全体：同句 mean={within:+.4f}（{n_same} 对）  "
        f"句间 mean={between:+.4f}（{n_between} 对）  差={within - between:+.4f}\n"
        f"      同 run 跨天：同句 mean={run_same:+.4f}（{n_run_same} 对）  "
        f"不同句 mean={run_diff:+.4f}（{n_run_diff} 对）  差={run_same - run_diff:+.4f}"
    )


def plot_matrix(
    out_dir, matrix, filename, title, vmin, vmax=1.0,
    xlabel="trial", ylabel="trial", cbar_label="余弦相似度（EEG 句嵌入）",
) -> Path | None:
    """用 seaborn ``sns.heatmap`` 把矩阵画成热图并存 PNG。

    ``vmin`` / ``vmax`` 由调用方给定，便于多张图共用色标直接对比；轴标签与色标标签
    可自定义，方便自相似（EEG×EEG）与跨模态（EEG×文本）共用同一个绘图函数。
    """
    n = matrix.shape[0]
    if n == 0:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 句子越多画布越大，否则单元挤在一起看不清
    size = 8.0 if n <= 80 else min(20.0, 8.0 + (n - 80) * 0.05)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        # xticklabels=False,
        # yticklabels=False,
        square=True,
        linewidths=0,
        linecolor="white",
        cbar_kws={"shrink": 0.8,"label": cbar_label},
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_order_csv(out_dir, subject, tag, meta, val_idx, select) -> Path:
    """把**图上每一格**对应到原始 trial，导出 CSV 便于逐格对齐矩阵。

    行 = 绘图顺序（plot_index 0..N-1）；列含该格来自哪个句子（group_id / slot_in_group）、
    以及该 trial 的 day / run / trial_index / sentence。
    ``slot_in_group`` 仅在 ``--order sentence`` 下有含义（0..3 对应该句的 day 1..4）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{subject}_val_simmatrix_{tag}_order.csv"

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["plot_index", "group_id", "slot_in_group", "day", "run", "trial_index", "sentence"]
        )
        group_id = -1
        slot = 0
        prev = None
        for plot_index, pos in enumerate(select):
            row = meta.iloc[val_idx[pos]]
            sentence = str(row["sentence"])
            if sentence != prev:
                group_id += 1
                slot = 0
                prev = sentence
            writer.writerow(
                [plot_index, group_id, slot, int(row["day"]), int(row["run"]),
                 int(row["trial_index"]), sentence]
            )
            slot += 1
    return path


def main():
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = find_ckpt(args.ckpt, args.subject)
    # 先扫一遍取 n_chans 需要读模型，这里只读 checkpoint 元信息
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args") or {}

    subject = args.subject or subject_from_ckpt(ckpt_path)
    days = args.day if args.day is not None else (train_args.get("day") or [1])
    event = args.event or train_args.get("event") or "inner"
    sample = args.sample if args.sample is not None else int(train_args.get("sample", 250))
    char_duration = (
        args.char_duration if args.char_duration is not None
        else float(train_args.get("char_duration", 0.4))
    )
    val_ratio = args.val_ratio if args.val_ratio is not None else float(train_args.get("val_ratio", 0.2))
    seed = args.seed if args.seed is not None else int(train_args.get("seed", 0))

    n_chans = int(ckpt["n_chans"])
    token_samples = int(ckpt["token_samples"])
    embedding_dim = int(ckpt["embedding_dim"])

    # char_token 的 --subject 是 nargs="+"，checkpoint 里存的是 list，打印时统一成字符串
    train_subject = train_args.get("subject")
    if isinstance(train_subject, (list, tuple)):
        train_subject = ",".join(str(s) for s in train_subject)

    print(
        f"[Ckpt] {ckpt_path}\n"
        f"       epoch={ckpt.get('epoch')} n_chans={n_chans} "
        f"token_samples={token_samples} embedding_dim={embedding_dim}\n"
        f"       train config subject={train_subject} day={train_args.get('day')} "
        f"event={train_args.get('event')} val_ratio={train_args.get('val_ratio')} seed={train_args.get('seed')}"
    )

    # ---------------- 数据：与训练完全一致的句子级切分 ----------------
    dataset = CharTokenEEGDataset(
        event=event,
        sample=sample,
        subject=subject,
        days=days,
        char_duration=char_duration,
    )
    dataset[0]  # 触发一次读取，确定 n_chans
    if dataset.n_chans != n_chans:
        raise ValueError(f"数据集通道数 {dataset.n_chans} 与 checkpoint 的 {n_chans} 不一致")
    if dataset.token_samples != token_samples:
        raise ValueError(f"token 长度 {dataset.token_samples} 与 checkpoint 的 {token_samples} 不一致")

    _, val_set, info = split_by_sentence(dataset, val_ratio, seed)
    steps_label = ",".join(f"{d:02d}" for d in sorted(days))
    print(
        f"[Data] {subject} day-{steps_label} event={event} token_samples={dataset.token_samples}\n"
        f"       train {info['n_unique_train']} 句/{len(dataset) - len(val_set)} trial，"
        f"预测集 {info['n_unique_val']} 句/{len(val_set)} trial，句子重叠={info['sentence_overlap']}"
    )

    # ---------------- 模型 ----------------
    model = CharTokenEEGEncoder(
        n_chans=n_chans, token_samples=token_samples, embedding_dim=embedding_dim
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---------------- 编码预测集全体 trial ----------------
    val_idx = list(val_set.indices)
    embeddings = encode_indices(model, dataset, val_idx, device, args.batch_size)
    n = embeddings.shape[0]
    sentences = [dataset.meta["sentence"].tolist()[i] for i in val_idx]
    run_of_pos = [int(dataset.meta["run"].tolist()[i]) for i in val_idx]
    day_of_pos = [int(dataset.meta["day"].tolist()[i]) for i in val_idx]
    sizes = Counter(sentences)
    print(
        f"[Enc] 预测集 trial={n}，唯一句子={len(sizes)}，"
        f"每句 trial 数分布={dict(sorted(Counter(sizes.values()).items()))}"
    )

    # ---------------- 同句内 / 句间相似度诊断（原始 embedding） ----------------
    sim_raw = (embeddings @ embeddings.t()).float().cpu().numpy()
    similarity_stats(sim_raw, sentences, run_of_pos, day_of_pos, "原始 embedding")

    # ---------------- run 中心化去相关（可选） ----------------
    # day-run：去掉当天当段的记录/电极/状态偏移；
    # run：按 run 号跨天减质心，去掉内容块的公共方向（同句检验的更严格版本）。
    sim = sim_raw
    center_note = ""
    if args.center_by != "none":
        if args.center_by == "day-run":
            center_groups = [f"{d}-{r}" for d, r in zip(day_of_pos, run_of_pos)]
            center_label = "按(day,run)减质心"
        else:
            center_groups = run_of_pos
            center_label = "按run号跨天减质心"

        embeddings = center_embeddings(embeddings, center_groups)
        sim = (embeddings @ embeddings.t()).float().cpu().numpy()
        center_note = f"  中心化={center_label}"
        similarity_stats(
            sim, sentences, run_of_pos, day_of_pos,
            f"run 中心化（{center_label}）后",
        )

    # ---------------- 排序并截取 ----------------
    if args.order == "sentence":
        order = group_order(sentences)
        order_label = "按句子分组"
        tag = "bysentence"
    else:
        order = list(range(n))
        order_label = "原始顺序"
        tag = "raw"

    if args.center_by != "none":
        tag += "_center" + args.center_by.replace("-", "")

    start = max(0, args.start)
    end = n if args.end is None else min(args.end, n)
    select = order[start:end]
    zoom_note = "" if (start, end) == (0, n) else f"（索引 {start}:{end}）"
    if (start, end) != (0, n):
        tag += f"_idx{start}-{end}"
    sub = sim[np.ix_(select, select)]

    # ---------------- 绘图 ----------------
    out_dir = Path(args.out_dir) if args.out_dir is not None else ckpt_path.parent
    # 所有图共用同一色标（全矩阵最小值 ~ 1.0），保证"全图 / 单 run"之间可直接对比
    vmin = float(sim.min())

    title = (
        f"{subject} 预测集 EEG 句嵌入自相似矩阵（顺序={order_label}）\n"
        f"N={len(select)}  days={steps_label}  val_ratio={val_ratio:g}  seed={seed}{center_note}{zoom_note}"
    )
    path = plot_matrix(out_dir, sub, f"{subject}_val_simmatrix_{tag}.png", title, vmin)
    print(f"\n[Plot] 子矩阵 {len(select)}×{len(select)}（{order_label}{zoom_note}）")
    print(f"[Output] {path}")

    # ---------------- 每个 run 单独一张 ----------------
    # run 取自预测集每条 trial 的 meta；用完整 order（不受 --start/--end 影响），
    # 保证每张 run 图都覆盖该 run 的全部 trial。
    for run_id in sorted({run_of_pos[p] for p in order}):
        sel_run = [p for p in order if run_of_pos[p] == run_id]
        n_sent = len(dict.fromkeys(sentences[p] for p in sel_run))
        title_run = (
            f"{subject} 预测集 EEG 自相似矩阵（顺序={order_label}·run-{run_id}）\n"
            f"N={len(sel_run)}  句数={n_sent}  days={steps_label}  "
            f"val_ratio={val_ratio:g}  seed={seed}{center_note}"
        )
        path_run = plot_matrix(
            out_dir,
            sim[np.ix_(sel_run, sel_run)],
            f"{subject}_val_simmatrix_{tag}_run{run_id}.png",
            title_run,
            vmin,
        )
        print(f"[Plot] run-{run_id}: {len(sel_run)}×{len(sel_run)}（{n_sent} 句）")
        print(f"[Output] {path_run}")

    # 同步导出"图上每一格 -> 原始 trial"的映射，便于逐格对齐
    csv_path = save_order_csv(out_dir, subject, tag, dataset.meta, val_idx, select)
    print(f"[Output] {csv_path}")


if __name__ == "__main__":
    main()
