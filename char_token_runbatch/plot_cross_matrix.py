"""绘制 ``char_token_runbatch`` 的 **EEG × sentence** 跨模态相似度矩阵。

行 = EEG trial，列 = 去重后的句子（LaBSE 冻结向量）：

    格子 (i, j) = cos(EEG_i, sentence_j)

默认取一个 ``(day, run)`` 的 252 条 trial。该 run 恰好含全部 252 句、每句一次，
所以矩阵是 252×252 方阵，**对角线就是"第 i 条 trial ↔ 它自己的句子"** 的配对相似度。
``--scope all`` 时行是全部 4032 条 trial、列仍是 252 句（4023×252 的长矩阵），
每行有 16 条 trial 指向同一列。

读图
----
- 对角线亮 = 该 trial 与自己的句子对得上；
- 按 ``--order raw`` 排序时横纵轴都是 trial_index，类别连续成块；
- 若把 ``--order sentence`` 打开，同句 trial 相邻，会看到沿对角线的 16×16 小块。

用法::

    uv run python char_token_runbatch/plot_cross_matrix.py
    uv run python char_token_runbatch/plot_cross_matrix.py --order sentence
    uv run python char_token_runbatch/plot_cross_matrix.py --run-day 5 --run-id 2
    uv run python char_token_runbatch/plot_cross_matrix.py --scope all --order raw
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
    days_tag,
    encode_eeg,
    load_meta,
    load_model,
    load_text_embeddings,
    order_positions,
    plot_heatmap,
    resolve_config,
    select_run,
    unique_sentence_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 EEG × sentence 跨模态相似度矩阵。")
    add_common_args(parser)
    parser.add_argument(
        "--scope", default="run", choices=["run", "all"],
        help="run=一个 (day, run) 的 252 条（默认）；all=全部 trial",
    )
    parser.add_argument("--run-day", type=int, default=None, help="run 所在 day；默认取训练 day 的第一个")
    parser.add_argument("--run-id", type=int, default=None, help="run 编号；默认取该 day 的第一个 run")
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

    # 行：trial；列：唯一句。两边用同一套排序规则，scope=run 时对角线自然对齐。
    positions = order_positions(meta, positions, args.order)
    unique = unique_sentence_rows(meta)
    col_positions = order_positions(meta, unique.index.to_numpy(), args.order)

    lookup = load_text_embeddings(dataset)
    col_sentences = meta["sentence"].to_numpy()[col_positions]
    text = np.stack([lookup[s] for s in col_sentences])

    eeg = encode_eeg(model, dataset, positions, cfg["device"], args.batch_size)
    sim = eeg @ text.T

    row_sentences = meta["sentence"].to_numpy()[positions]
    col_of = {s: i for i, s in enumerate(col_sentences)}
    target = np.array([col_of[s] for s in row_sentences])
    correct = sim[np.arange(len(sim)), target]
    rank = (sim > correct[:, None]).sum(axis=1) + 1

    print(
        f"[Data] {cfg['subject']} scope={scope_tag} trial={len(positions)} "
        f"唯一句={len(col_sentences)} order={args.order}"
    )
    print(
        f"[Retr] e2t@1={float((rank == 1).mean()):.4f}  "
        f"e2t@5={float((rank <= 5).mean()):.4f}  "
        f"mean_rank={float(rank.mean()):.1f}"
    )
    if len(positions) == len(col_sentences):
        print(f"[Sim] 对角线配对 mean={float(np.diag(sim).mean()):+.4f}")

    tag = f"{scope_tag}_{args.order}"
    title = (
        f"{cfg['subject']} EEG×sentence 跨模态相似度矩阵（{scope_tag}，order={args.order}）\n"
        f"行={len(positions)} trial  列={len(col_sentences)} 唯一句  "
        f"e2t@1={float((rank == 1).mean()):.3f}"
    )
    path = plot_heatmap(
        sim,
        cfg["out_dir"] / f"{cfg['subject']}_eeg_text_{tag}.png",
        title,
        xlabel="sentence（LaBSE，去重）",
        ylabel="trial（EEG）",
        cbar_label="余弦相似度（EEG × sentence）",
        vmin=float(sim.min()),
        vmax=float(sim.max()),
    )
    print(f"[Output] {path}")


if __name__ == "__main__":
    main()
