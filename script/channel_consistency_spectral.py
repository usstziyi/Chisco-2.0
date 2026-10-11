"""汇聚同一被试同一句话的所有 inner 段 EEG，并**在频域**统计各通道的跨试次一致性。

与 ``channel_consistency.py`` 的区别：那里直接在时域波形上比较，
这里先把每个试次变换成功率谱（Welch PSD），再在频域上比较。

指标：
1. psd_r —— 宽带 PSD 谱形一致性
   每个通道把每个试次的 log10(PSD)（1~45 Hz，共 F 个频点）当特征向量，
   逐试次沿频率去均值/标准化后，计算 N 个试次两两的 Pearson r 取平均。
   越大表示“同句在不同时段的功率谱形状越一致”。

2. band_r —— 各频带（delta/theta/alpha/beta/gamma）的一致性
   同上，但只在频带内的频点上计算跨试次相关。
   band_power 同时给出该频带的平均功率（µV²/Hz），用于描述性参考。

注意：原始 log-PSD 的谱形一致性主要来自各试次共有的 1/f 非周期成分，
容易得到虚高的 r。加 --remove-aperiodic 会用 specparam 拟合每通道每试次的
非周期成分并在 log10 功率上减去，再重算上述一致性。

切片位置、试次筛选、--days 过滤均复用 ``channel_consistency``。

用法：
    uv run python script/channel_consistency_spectral.py
    uv run python script/channel_consistency_spectral.py --days 5
    uv run python script/channel_consistency_spectral.py --remove-aperiodic          # 去 1/f 后再算
    uv run python script/channel_consistency_spectral.py --subject sub-01 --days 5 --sentence 附近还有其他的酒店吗
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
from scipy.signal import welch
from specparam import SpectralModel

from channel_consistency import (
    FIF_FOLDER,
    OUTPUTS_FOLDER,
    TEXT_TABLE,
    load_inner_trials,
    normalize_subject,
    select_trials,
)

DEFAULT_SENTENCE = "附近还有其他的酒店吗"

# 频带划分（Hz，左闭右开）
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
FMIN, FMAX = 1.0, 45.0

# Welch：1 s 窗（250 Hz -> 1 Hz 频率分辨率），50% 重叠；
# nfft 零填充到更细的频率网格（~0.24 Hz 间隔），让低频带也有足够多的频点。
NPERSEG = 250
NOVERLAP = 125
NFFT = 1024
EPS = 1e-12


def compute_psd(trials: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """trials: (N, C, T) -> psd: (N, C, F)，freqs: (F,)。"""
    freqs, psd = welch(
        trials,
        fs=float(sample_rate),
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        axis=-1,
    )
    return freqs, psd


def remove_aperiodic(
    psd: np.ndarray, freqs: np.ndarray, bin_mask: np.ndarray
) -> np.ndarray:
    """用 specparam 拟合每通道每试次的非周期(1/f)成分并在 log10 功率上减去。

    返回与 psd 同形的“展平”对数谱（仅 bin_mask 内有值，其余为 0）。
    非周期模型为 fixed 模式：aperiodic_log10(f) = offset - exponent * log10(f)。
    """
    n_trials, n_chans, n_freqs = psd.shape
    flat = np.zeros_like(psd)
    fit_freqs = freqs[bin_mask]
    log_fit_freqs = np.log10(fit_freqs)

    model = SpectralModel(
        peak_width_limits=[1.0, 8.0],
        max_n_peaks=6,
        min_peak_height=0.1,
        aperiodic_mode="fixed",
        verbose=False,
    )

    r_squared = []
    for t in range(n_trials):
        for c in range(n_chans):
            model.fit(fit_freqs, psd[t, c, bin_mask], [FMIN, FMAX])
            results = model.results.get_results()
            offset, exponent = results.aperiodic_fit
            aperiodic_log = offset - exponent * log_fit_freqs
            flat[t, c, bin_mask] = np.log10(psd[t, c, bin_mask] + EPS) - aperiodic_log
            if results.metrics:
                r_squared.append(results.metrics.get("gof_rsquared", np.nan))

    print(f"[specparam] 拟合 {n_trials * n_chans} 个谱，"
          f"平均 R² = {np.nanmean(r_squared):.3f}")
    return flat


def mean_pairwise_r(feature: np.ndarray) -> float:
    """feature: (N, M) -> 逐个试次沿 M 去均值/标准化后，两两 Pearson r 的均值。"""
    z = (feature - feature.mean(axis=1, keepdims=True)) / (
        feature.std(axis=1, keepdims=True) + EPS
    )
    corr = np.corrcoef(z)  # (N, N)
    iu = np.triu_indices(feature.shape[0], k=1)
    return float(corr[iu].mean())


def per_channel_psd_r(log_psd: np.ndarray, bin_mask: np.ndarray) -> np.ndarray:
    """log_psd: (N, C, F) -> 每通道宽带谱形跨试次相关 (C,)。"""
    _, n_chans, _ = log_psd.shape
    return np.array(
        [mean_pairwise_r(log_psd[:, c, bin_mask]) for c in range(n_chans)]
    )


def band_consistency(
    log_psd: np.ndarray, psd: np.ndarray, freqs: np.ndarray, ch_names: list[str]
) -> pd.DataFrame:
    """逐通道逐频带：band_r（谱形一致性）与 band_power（平均功率 µV²/Hz）。"""
    rows = []
    for band, (lo, hi) in BANDS.items():
        bin_mask = (freqs >= lo) & (freqs < hi)
        n_bins = int(bin_mask.sum())
        for c, ch in enumerate(ch_names):
            band_r = mean_pairwise_r(log_psd[:, c, bin_mask])
            band_power = float(psd[:, c, bin_mask].mean())
            rows.append(
                {
                    "channel": ch,
                    "band": band,
                    "n_bins": n_bins,
                    "band_r": band_r,
                    "band_power_uv2_hz": band_power,
                }
            )
    return pd.DataFrame(rows)


def get_eeg_info(subset: pd.DataFrame, subject: str, sample_rate: int):
    """取 FIF 里的电极坐标，用于 topomap。"""
    ref_row = subset.iloc[0]
    ref_raw = mne.io.read_raw_fif(
        FIF_FOLDER
        / f"{sample_rate}hz"
        / subject
        / f"{subject}_day-{int(ref_row['day']):02d}_run-{int(ref_row['run']):02d}_raw.fif",
        preload=False,
        verbose=False,
    )
    return mne.pick_info(ref_raw.info, mne.pick_types(ref_raw.info, eeg=True))


def plot_broadband(values_r, info, title, output):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title)

    order = np.argsort(values_r)[::-1]
    axes[0].bar(np.arange(len(order)), values_r[order])
    axes[0].set_title(f"宽带 PSD 谱形 r（均值 {values_r.mean():.3f}）")
    axes[0].set_xlabel("通道（按 r 降序）")
    axes[0].set_ylabel("mean pairwise r")

    try:
        im, _ = mne.viz.plot_topomap(values_r, info, axes=axes[1], show=False)
        axes[1].set_title("宽带 PSD 谱形 r topomap")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
    except Exception as exc:  # noqa: BLE001 - 仅用于可视化降级
        axes[1].axis("off")
        axes[1].set_title(f"topomap 不可用：{exc}")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[Saved] {output}")


def plot_bands(band_df: pd.DataFrame, info, ch_names, title, output):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title)

    band_names = list(BANDS)
    for ax, band in zip(axes.flat, band_names):
        sub = band_df[band_df["band"] == band].set_index("channel").loc[ch_names]
        values = sub["band_r"].to_numpy()
        try:
            im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False)
            ax.set_title(f"{band} ({BANDS[band][0]:g}-{BANDS[band][1]:g}Hz) "
                         f"r均值={values.mean():.3f}")
            fig.colorbar(im, ax=ax, fraction=0.046)
        except Exception as exc:  # noqa: BLE001 - 仅用于可视化降级
            ax.axis("off")
            ax.set_title(f"{band} topomap 不可用：{exc}")

    ax = axes.flat[len(band_names)]
    means = [band_df[band_df["band"] == b]["band_r"].mean() for b in band_names]
    ax.bar(band_names, means)
    ax.set_title("各频带 r 的通道均值")
    ax.set_ylabel("mean pairwise r")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[Saved] {output}")


def main():
    parser = argparse.ArgumentParser(
        description="频域：统计同一被试同一句话各通道的跨试次 PSD 一致性。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试 ID，如 sub-01 或 1")
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE, help="目标句子")
    parser.add_argument(
        "--days", type=int, nargs="+", default=None,
        help="只使用这些 day（1~8），可多选；不传则使用全部 day",
    )
    parser.add_argument("--sample-rate", type=int, default=250, help="数据采样率 Hz（默认 250）")
    parser.add_argument(
        "--remove-aperiodic",
        action="store_true",
        help="用 specparam 去掉 1/f 非周期成分后再算一致性（默认关闭）",
    )
    parser.add_argument("--out-dir", default=str(OUTPUTS_FOLDER), help="输出目录")
    args = parser.parse_args()

    subject = normalize_subject(args.subject)
    table = pd.read_csv(TEXT_TABLE, sep="\t")
    subset = select_trials(table, subject, args.sentence, args.days)
    print(f"[Trials] {subject} | {args.sentence!r} | days={args.days or '全部'} -> {len(subset)} 个试次")
    if len(subset) < 2:
        raise ValueError("少于 2 个试次，无法计算跨试次一致性")

    trials, ch_names = load_inner_trials(subset, args.sample_rate)
    print(f"[Data] trials shape = {trials.shape} (N, C, T)")

    freqs, psd = compute_psd(trials, args.sample_rate)
    print(f"[PSD] freq grid {freqs[0]:.2f}~{freqs[-1]:.2f} Hz, n_freqs={len(freqs)}")

    bin_mask = (freqs >= FMIN) & (freqs <= FMAX)
    # 默认用原始 log10(PSD)；--remove-aperiodic 时先用 specparam 去掉 1/f
    feature = np.log10(psd + EPS)
    suffix = ""
    if args.remove_aperiodic:
        feature = remove_aperiodic(psd, freqs, bin_mask)
        suffix = "_flattened"

    psd_r = per_channel_psd_r(feature, bin_mask)
    df = pd.DataFrame({"channel": ch_names, "n_trials": len(subset), "psd_r": psd_r})
    ranked = df.sort_values("psd_r", ascending=False).reset_index(drop=True)

    print("\n=== 宽带 PSD 谱形一致性（按 psd_r 降序，前 10 / 后 10）===")
    print(pd.concat([ranked.head(10), ranked.tail(10)]).to_string(index=False))
    print(f"\n全通道：psd_r 均值 = {df['psd_r'].mean():.4f}，"
          f"最大 = {df['psd_r'].max():.4f}（{df.loc[df['psd_r'].idxmax(), 'channel']}）")

    band_df = band_consistency(feature, psd, freqs, ch_names)
    summary = (
        band_df.groupby("band", sort=False)
        .agg(n_bins=("n_bins", "first"), r均值=("band_r", "mean"), r最大=("band_r", "max"))
        .reindex(list(BANDS))
    )
    print("\n=== 各频带一致性（通道均值 / 最大）===")
    print(summary.to_string())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_{args.sentence}"
    if args.days:
        stem = f"{subject}_day-{'-'.join(map(str, sorted(args.days)))}_{args.sentence}"

    psd_csv = out_dir / f"{stem}_spectral_psd_consistency{suffix}.csv"
    band_csv = out_dir / f"{stem}_spectral_band_consistency{suffix}.csv"
    ranked.to_csv(psd_csv, index=False)
    band_df.to_csv(band_csv, index=False)
    print(f"[Saved] {psd_csv}")
    print(f"[Saved] {band_csv}")

    kind = "去1/f展平" if args.remove_aperiodic else "原始"
    info = get_eeg_info(subset, subject, args.sample_rate)
    plot_broadband(
        psd_r, info,
        f"{subject} | {args.sentence} | days={args.days or '全部'} | N={len(subset)} | "
        f"{kind}宽带 PSD（{FMIN:g}-{FMAX:g}Hz）",
        out_dir / f"{stem}_spectral_psd_consistency{suffix}.png",
    )
    plot_bands(
        band_df, info, ch_names,
        f"{subject} | {args.sentence} | days={args.days or '全部'} | N={len(subset)} | "
        f"{kind}各频带 PSD 一致性",
        out_dir / f"{stem}_spectral_band_consistency{suffix}.png",
    )


if __name__ == "__main__":
    main()
