"""绘制 ``char_token_runbatch`` 的 **EEG × EEG** 句嵌入自相似矩阵。

默认取一个 ``(day, run)`` 的 252 条 trial（该 run 恰好包含全部 252 句、每句一次），
矩阵 252×252：

    格子 (i, j) = cos(EEG_i, EEG_j)

同一句在另外 3 天、3 个 run 也有 trial；``--scope all`` 可以画全部 4032 条，
但 4032×4032 的图会很大，通常先用单 run 看结构。

读图
----
- 按 ``--order raw``（默认）排序时，横纵轴都是 run 内的 trial_index，类别连续排列；
- ``--order sentence`` 会把同一句的 16 条 trial 聚在一起，块内亮 = 跨天/跨 run 稳定；
- ``--order category`` 按类别聚块，看类别结构。

用法::

    uv run python char_token_runbatch/plot_eeg_matrix.py
    uv run python char_token_runbatch/plot_eeg_matrix.py --order sentence
    uv run python char_token_runbatch/plot_eeg_matrix.py --run-day 6 --run-id 2
    uv run python char_token_runbatch/plot_eeg_matrix.py --scope all --order raw
    uv run python char_token_runbatch/plot_eeg_matrix.py --center-by day-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token_runbatch.plot_common import (
    add_common_args,
    build_dataset,
    center_by,
    days_tag,
    encode_eeg,
    load_meta,
    load_model,
    order_positions,
    plot_heatmap,
    resolve_config,
    select_run,
    similarity_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 EEG × EEG 自相似矩阵。")
    add_common_args(parser)
    parser.add_argument(
        "--scope", default="run", choices=["run", "all"],
        help="run=只画一个 (day, run) 的 252 条（默认）；all=全部 trial",
    )
    parser.add_argument("--run-day", type=int, default=None, help="run 所在 day；默认取训练 day 的第一个")
    parser.add_argument("--run-id", type=int, default=None, help="run 编号；默认取该 day 的第一个 run")
    parser.add_argument(
        "--center-by", default="none", choices=["none", "day", "day-run", "run"],
        help="中心化去公共偏移：none（默认）/ day / day-run / run（跨天）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config(args)
    dataset = build_dataset(cfg, touch=True)
    meta = load_meta(dataset)
    model = load_model(cfg["ckpt"], dataset, cfg["device"])

    if args.scope == "run":
        positions, (day, run) = select_run(meta, args.run_day, args.run_id)
        scope_tag = f"day{day}_run{run}"
    else:
        positions = np.arange(len(meta))
        scope_tag = f"all_day{days_tag(cfg['days'])}"

    positions = order_positions(meta, positions, args.order)
    eeg = encode_eeg(model, dataset, positions, cfg["device"], args.batch_size)

    center_note = ""
    if args.center_by != "none":
        if args.center_by == "day":
            groups = meta["day"].to_numpy()[positions]
        elif args.center_by == "day-run":
            groups = (
                meta["day"].astype(str) + "-" + meta["run"].astype(str)
            ).to_numpy()[positions]
        else:
            groups = meta["run"].to_numpy()[positions]
        eeg = center_by(eeg, groups)
        center_note = f"  中心化={args.center_by}"

    sim = eeg @ eeg.T
    sentences = meta["sentence"].to_numpy()[positions]
    runs = meta["run"].to_numpy()[positions]
    days = meta["day"].to_numpy()[positions]

    n_sent = len(set(sentences))
    print(
        f"[Data] {cfg['subject']} scope={scope_tag} trial={len(positions)} "
        f"唯一句={n_sent} order={args.order}{center_note}"
    )
    similarity_stats(sim, sentences, runs, days)

    tag = f"{scope_tag}_{args.order}"
    if args.center_by != "none":
        tag += f"_center{args.center_by.replace('-', '')}"
    title = (
        f"{cfg['subject']} EEG×EEG 自相似矩阵（{scope_tag}，order={args.order}）\n"
        f"N={len(positions)}  唯一句={n_sent}{center_note}"
    )
    path = plot_heatmap(
        sim,
        cfg["out_dir"] / f"{cfg['subject']}_eeg_eeg_{tag}.png",
        title,
        xlabel="trial",
        ylabel="trial",
        cbar_label="余弦相似度（EEG × EEG）",
        vmin=float(sim.min()),
        vmax=1.0,
    )
    print(f"[Output] {path}")


if __name__ == "__main__":
    main()
