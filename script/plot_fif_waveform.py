"""绘制预处理后 FIF 文件的一段 EEG 波形。

按 subject / day / run 定位文件：
    datasets/preprocessed/fif/<sample_rate>hz/sub-XX/sub-XX_day-XX_run-XX_raw.fif

只把 [tmin, tmax] 这一小段读进内存，不会加载整个文件。

用法：
    uv run python script/plot_fif_waveform.py --subject sub-01 --day 1 --run 1
    uv run python script/plot_fif_waveform.py --subject sub-01 --day 1 --run 1 --nchans 20 --tmin 300 --tmax 310

图片默认保存到 outputs/plot/<文件名>_<tmin>-<tmax>s.png。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "preprocessed"
OUTPUTS_FOLDER = PROJECT_ROOT / "outputs" / "eegplot"

DEFAULT_NCHANS = 10
DEFAULT_WINDOW = 10.0


def normalize_subject(subject: str) -> str:
    """'1' / '01' / 'sub-01' -> 'sub-01'。"""
    number = subject.strip().removeprefix("sub-")
    return f"sub-{int(number):02d}"


def resolve_fif_file(subject: str, day: int, run: int, sample_rate: float) -> Path:
    subject = normalize_subject(subject)
    fif_file = (
        PREPROCESSED_FOLDER
        / "fif"
        / f"{sample_rate:g}hz"
        / subject
        / f"{subject}_day-{int(day):02d}_run-{int(run):02d}_raw.fif"
    )
    if not fif_file.exists():
        raise FileNotFoundError(f"FIF 不存在: {fif_file}")
    return fif_file


def plot_eeg_window(
    fif_file: Path,
    n_channels: int = DEFAULT_NCHANS,
    tmin: float = 0.0,
    tmax: float | None = None,
    output: Path | None = None,
):
    """绘制 fif_file 中 [tmin, tmax] 秒、前 n_channels 个 EEG 通道的波形。"""
    raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)
    # 用最后一个采样点的时间作为上界：raw.duration 比 crop 允许的最大时间
    # 大 1 个采样间隔，直接用它会在 crop 时报错。
    max_time = float(raw.times[-1])

    tmin = max(float(tmin), 0.0)
    right = tmin + DEFAULT_WINDOW if tmax is None else float(tmax)
    tmax = min(right, max_time)
    if tmax <= tmin:
        raise ValueError(
            f"无效的绘制区间：[{tmin}, {tmax}]，文件时长 {max_time:.2f} s"
        )

    eeg_names = [
        name
        for name, ch_type in zip(raw.ch_names, raw.get_channel_types())
        if ch_type == "eeg"
    ]
    selected = eeg_names[:n_channels]
    if not selected:
        raise ValueError("文件里没有 EEG 通道。")

    # crop 后 raw.times 从 0 重新起算，但 annotations.onset 仍是相对记录起点的
    # 绝对时间，所以先把窗口内的事件 / 坏段记下来，绘图时再统一到绝对时间轴。
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    durations = np.asarray(raw.annotations.duration, dtype=float)
    descriptions = np.asarray(raw.annotations.description)
    in_window = (onsets >= tmin) & (onsets <= tmax)
    win_onsets = onsets[in_window]
    win_durations = durations[in_window]
    win_descriptions = [str(description) for description in descriptions[in_window]]
    is_bad = [description.lower().startswith("bad") for description in win_descriptions]

    raw.crop(tmin=tmin, tmax=tmax)
    raw.load_data()
    raw.pick(selected)

    data = raw.get_data() * 1e6  # V -> µV
    times = raw.times + tmin  # 还原为相对记录起点的绝对时间，与 annotations 对齐
    n_plot = len(raw.ch_names)

    peak = float(np.abs(data).max()) if data.size else 1.0
    separation = max(peak * 1.2, 1.0)

    fig, ax = plt.subplots(figsize=(14, max(3.0, 0.4 * n_plot)))
    for i in range(n_plot):
        ax.plot(times, data[i] + i * separation, lw=0.6, color="black")

    # 坏段用红色底色标出
    for onset, duration_, flag in zip(win_onsets, win_durations, is_bad):
        if flag:
            ax.axvspan(onset, onset + duration_, color="red", alpha=0.12)

    # 事件用蓝色竖线标出
    for onset, description, flag in zip(win_onsets, win_descriptions, is_bad):
        if flag:
            continue
        ax.axvline(onset, color="tab:blue", ls="--", lw=0.8, alpha=0.8)
        ax.text(
            onset,
            n_plot * separation,
            str(description),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
            color="tab:blue",
        )

    ax.set_yticks(np.arange(n_plot) * separation)
    ax.set_yticklabels(raw.ch_names, fontsize=8)
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(-separation, n_plot * separation + separation)
    ax.set_xlabel("Time (s)")
    ax.grid(True, axis="x", ls=":", lw=0.5, alpha=0.5)
    ax.set_title(
        f"{fif_file.stem}\n{tmin:.2f} - {tmax:.2f} s | {n_plot} EEG channels"
    )
    fig.tight_layout()

    if output is None:
        output = OUTPUTS_FOLDER / f"{fif_file.stem}_{tmin:.1f}-{tmax:.1f}s.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[Saved] {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="绘制预处理 FIF 中一段 EEG 波形。")
    parser.add_argument("--subject", required=True, help="被试 ID，如 sub-01 或 1")
    parser.add_argument("--day", type=int, required=True, help="day 编号（1~8）")
    parser.add_argument("--run", type=int, required=True, help="run 编号")
    parser.add_argument("--nchans", type=int, default=DEFAULT_NCHANS, help="绘制的 EEG 通道数（默认 10）")
    parser.add_argument("--tmin", type=float, default=0.0, help="起始时间（秒，默认 0）")
    parser.add_argument("--tmax", type=float, default=None, help=f"结束时间（秒，默认 tmin + {DEFAULT_WINDOW:g}s）")
    parser.add_argument("--sample-rate", type=float, default=250.0, help="数据采样率 Hz（默认 250）")
    parser.add_argument("--out", default=None, help="图片保存路径（默认 outputs/eegplot/）")
    args = parser.parse_args()

    fif_file = resolve_fif_file(args.subject, args.day, args.run, args.sample_rate)
    plot_eeg_window(
        fif_file,
        n_channels=args.nchans,
        tmin=args.tmin,
        tmax=args.tmax,
        output=args.out,
    )


if __name__ == "__main__":
    main()
