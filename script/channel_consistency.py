"""汇聚同一被试同一句话的所有 inner 段 EEG，并统计各通道的跨试次一致性。

数据来源：
    datasets/preprocessed/all_text.tsv              试次元数据（含 inner 切片位置）
    datasets/preprocessed/fif/<sr>hz/<subject>/*.fif  预处理后的 EEG

对每个通道 c，把 N 个试次的 inner 段组成 (N, T) 后计算两个一致性指标：

1. mean_pairwise_r —— 跨试次相关
   先把每个试次沿时间 z-score，再计算 N 个试次两两的 Pearson r，
   取上三角平均。取值 [-1, 1]，越大表示同句诱发的波形越一致。

2. snr —— 信噪比（ERP 一致性）
   evoked = 试次均值；residual = 试次 - evoked
   snr = var_t(evoked) / mean_{n,t}(residual^2)
   越大表示“诱发信号”相对“单试次噪声”越强。
   注意：不同 day/run 属于不同记录段，若各段增益不一致会抬高 residual，
   从而压低 snr；相关指标 mean_pairwise_r 对此不敏感。

用法：
    uv run python script/channel_consistency.py
    uv run python script/channel_consistency.py --subject sub-01 --sentence 附近还有其他的酒店吗
    uv run python script/channel_consistency.py --subject sub-01 --days 5      # 只用 day5 的试次
    uv run python script/channel_consistency.py --subject 1 --days 5 6 --sample-rate 250
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 图上标题含中文，指定可用的中文字体，避免显示成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import mne
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "preprocessed"
FIF_FOLDER = PREPROCESSED_FOLDER / "fif"
TEXT_TABLE = PREPROCESSED_FOLDER / "all_text.tsv"
OUTPUTS_FOLDER = PROJECT_ROOT / "outputs" / "consistency"

DEFAULT_SENTENCE = "附近还有其他的酒店吗"


def normalize_subject(subject: str) -> str:
    """'1' / '01' / 'sub-01' -> 'sub-01'。"""
    number = subject.strip().removeprefix("sub-")
    return f"sub-{int(number):02d}"


def select_trials(
    table: pd.DataFrame, subject: str, sentence: str, days: list[int] | None = None
) -> pd.DataFrame:
    subset = table[(table["subject"] == subject) & (table["sentence"] == sentence)]
    if days is not None:
        subset = subset[subset["day"].isin(days)]
    subset = subset.sort_values(["day", "run", "trial_index"]).reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"{subject} 中没有句子 {sentence!r}（days={days}）的试次")
    return subset


def load_inner_trials(
    subset: pd.DataFrame, sample_rate: int
) -> tuple[np.ndarray, list[str]]:
    """按 inner 切片读取每个试次，返回 (N, C, T) 与通道名。"""
    durations = subset["inner_dur_samples"].unique()
    if len(durations) != 1:
        raise ValueError(f"inner 段长度不一致：{durations.tolist()}，无法对齐")

    trials = []
    ch_names: list[str] | None = None
    for _, row in subset.iterrows():
        fif_path = (
            FIF_FOLDER
            / f"{sample_rate}hz"
            / str(row["subject"])
            / f"{row['subject']}_day-{int(row['day']):02d}_run-{int(row['run']):02d}_raw.fif"
        )
        if not fif_path.exists():
            raise FileNotFoundError(f"FIF 不存在：{fif_path}")

        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
        first_samp = int(raw.first_samp)
        picks = [
            i for i, ch_type in enumerate(raw.get_channel_types()) if ch_type == "eeg"
        ]
        if ch_names is None:
            ch_names = [raw.ch_names[i] for i in picks]

        start = int(row["inner_sample"]) - first_samp
        stop = start + int(durations[0])
        if start < 0 or stop > raw.n_times:
            raise ValueError(
                f"切片越界：[{start}, {stop}) 超出 FIF 长度 {raw.n_times}（{fif_path.name}）"
            )

        data = raw.get_data(picks=picks, start=start, stop=stop)
        trials.append(data)

    return np.stack(trials) * 1e6, ch_names  # V -> µV


def per_channel_consistency(trials: np.ndarray) -> pd.DataFrame:
    """trials: (N, C, T) -> 每通道的 mean_pairwise_r 与 snr。"""
    n_trials, n_chans, n_times = trials.shape

    # 跨试次相关：逐试次沿时间 z-score 后两两相关
    mean_t = trials.mean(axis=2, keepdims=True)
    std_t = trials.std(axis=2, keepdims=True) + 1e-12
    z_trials = (trials - mean_t) / std_t

    r_mean = np.empty(n_chans)
    for c in range(n_chans):
        corr = np.corrcoef(z_trials[:, c, :])  # (N, N)
        off_diag = corr[np.triu_indices(n_trials, k=1)]
        r_mean[c] = off_diag.mean()

    # 信噪比：诱发响应方差 / 单试次残差方差
    evoked = trials.mean(axis=0)  # (C, T)
    residual = trials - evoked
    signal_var = evoked.var(axis=1)  # (C,)
    noise_var = (residual**2).mean(axis=(0, 2))  # (C,)
    snr = signal_var / (noise_var + 1e-12)

    return pd.DataFrame(
        {
            "channel": [f"ch{i}" for i in range(n_chans)],
            "n_trials": n_trials,
            "mean_pairwise_r": r_mean,
            "snr": snr,
            "snr_db": 10 * np.log10(snr + 1e-12),
        }
    )


def plot_summary(
    values_r: np.ndarray,
    values_snr: np.ndarray,
    info,
    title: str,
    output: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title)

    rank_r = np.argsort(values_r)[::-1]
    axes[0, 0].bar(np.arange(len(rank_r)), values_r[rank_r])
    axes[0, 0].set_title(f"mean_pairwise_r（均值 {values_r.mean():.3f}）")
    axes[0, 0].set_xlabel("通道（按 r 降序）")
    axes[0, 0].set_ylabel("mean pairwise r")

    rank_snr = np.argsort(values_snr)[::-1]
    axes[1, 0].bar(np.arange(len(rank_snr)), values_snr[rank_snr])
    axes[1, 0].set_title(f"SNR（均值 {values_snr.mean():.3f}）")
    axes[1, 0].set_xlabel("通道（按 SNR 降序）")
    axes[1, 0].set_ylabel("snr")

    # topomap 依赖 FIF 里的电极坐标，缺失时优雅跳过
    for ax, values, name in (
        (axes[0, 1], values_r, "mean_pairwise_r"),
        (axes[1, 1], values_snr, "snr"),
    ):
        try:
            im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False)
            ax.set_title(f"{name} topomap")
            fig.colorbar(im, ax=ax, fraction=0.046)
        except Exception as exc:  # noqa: BLE001 - 仅用于可视化降级
            ax.axis("off")
            ax.set_title(f"{name} topomap 不可用：{exc}")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[Saved] {output}")


def main():
    parser = argparse.ArgumentParser(description="统计同一被试同一句话各通道的跨试次 EEG 一致性。")
    parser.add_argument("--subject", default="sub-01", help="被试 ID，如 sub-01 或 1")
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE, help="目标句子")
    parser.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=None,
        help="只使用这些 day（1~8），可多选；不传则使用全部 day",
    )
    parser.add_argument("--sample-rate", type=int, default=250, help="数据采样率 Hz（默认 250）")
    parser.add_argument("--out-dir", default=str(OUTPUTS_FOLDER), help="输出目录")
    args = parser.parse_args()

    subject = normalize_subject(args.subject)
    table = pd.read_csv(TEXT_TABLE, sep="\t")
    subset = select_trials(table, subject, args.sentence, args.days)
    print(f"[Trials] {subject} | {args.sentence!r} | days={args.days or '全部'} -> {len(subset)} 个试次")

    trials, ch_names = load_inner_trials(subset, args.sample_rate)
    print(f"[Data] trials shape = {trials.shape} (N, C, T)")

    df = per_channel_consistency(trials)
    df["channel"] = ch_names

    ranked_r = df.sort_values("mean_pairwise_r", ascending=False).reset_index(drop=True)
    print("\n=== 通道一致性（按 mean_pairwise_r 降序，前 10 / 后 10）===")
    print(pd.concat([ranked_r.head(10), ranked_r.tail(10)]).to_string(index=False))
    print(f"\n全通道：mean_pairwise_r 均值 = {df['mean_pairwise_r'].mean():.4f}，"
          f"最大 = {df['mean_pairwise_r'].max():.4f}（{df.loc[df['mean_pairwise_r'].idxmax(), 'channel']}）")
    print(f"全通道：SNR 均值 = {df['snr'].mean():.4f}，"
          f"最大 = {df['snr'].max():.4f}（{df.loc[df['snr'].idxmax(), 'channel']}）")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_{args.sentence}"
    if args.days:
        stem = f"{subject}_day-{'-'.join(map(str, sorted(args.days)))}_{args.sentence}"
    csv_path = out_dir / f"{stem}_channel_consistency.csv"
    ranked_r.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    # 复用 FIF 中已有电极坐标（取第一条试次的 run 文件）用于 topomap
    ref_row = subset.iloc[0]
    ref_raw = mne.io.read_raw_fif(
        FIF_FOLDER
        / f"{args.sample_rate}hz"
        / subject
        / f"{subject}_day-{int(ref_row['day']):02d}_run-{int(ref_row['run']):02d}_raw.fif",
        preload=False,
        verbose=False,
    )
    info = mne.pick_info(ref_raw.info, mne.pick_types(ref_raw.info, eeg=True))

    fig_path = out_dir / f"{stem}_channel_consistency.png"
    plot_summary(
        df["mean_pairwise_r"].to_numpy(),
        df["snr"].to_numpy(),
        info,
        f"{subject} | {args.sentence} | N={trials.shape[0]} | {trials.shape[1]} channels",
        fig_path,
    )


if __name__ == "__main__":
    main()
