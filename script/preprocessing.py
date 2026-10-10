

"""Chisco-2.0 EEG 预处理脚本。

day 编号与数据的对应关系：
    day 1~4 -> ses-01~ses-04 的 para1
    day 5~8 -> ses-01~ses-04 的 para2

命令行示例：

    # 遍历某个被试的指定 days（每个 day 打印该天的 EDF 文件名）
    uv run python script/preprocessing.py --subject sub-01 --days 1 2 3 4 5 6 7 8

    # 只处理部分 days
    uv run python script/preprocessing.py --subject sub-01 --days 1 5

    # 不指定 --subject 时遍历 datasets/ds006317-download 下的全部被试
    uv run python script/preprocessing.py --days 1 2 3 4 5 6 7 8
"""

import argparse
import os
import re
import unicodedata
from pathlib import Path
from collections import Counter

import mne
import numpy as np
import pandas as pd

from pyprep.prep_pipeline import PrepPipeline
from mne_icalabel import label_components
from autoreject import get_rejection_threshold

from joblib import parallel_config



# ============================================================
# Paths / constants
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_FOLDER = PROJECT_ROOT / "datasets"
CHISCO2_FOLDER = DATASETS_FOLDER /"ds006317-download"
PREPROCESSED_FOLDER = DATASETS_FOLDER / "preprocessed"
MONTAGE_FILE = DATASETS_FOLDER / "montage.csv"
TEXTDATASET = DATASETS_FOLDER / "textdataset"




# Chisco-2.0 中这两个通道仍被标为 EEG，
# 但旧 Chisco montage.csv 中坐标为 (0, 0, 0)
INVALID_EEG_CHANNELS = ["10", "111"]

# 辅助通道
EOG_CHANNELS = ["VEO", "HEO"]
STIM_CHANNEL = "Trigger"

# Chisco-2.0 事件编码 -> annotations 描述
EVENT_DESCRIPTIONS = {
    65329: "Reading start",
    65379: "Inner-speech start",
    65381: "1.8 s rest start",
    65480: "long rest start",
    65481: "long rest end",
}

# 非任务坏段标注（描述以 BAD 开头，MNE 依 reject_by_annotation 自动排除）
BAD_LEAD_IN = "BAD_lead_in"
BAD_LONG_REST = "BAD_long_rest"
BAD_TAIL = "BAD_tail"

# 原 Chisco montage 中不应该作为 Chisco-2.0 scalp EEG 使用的条目
MONTAGE_EXCLUDE_CHANNELS = {
    "10", "11", "111", "110", "84", "85",
    "EKG", "EMG", "VEO", "HEO", "Trigger",
}

# day 编号 -> (session, task)
# day 1~4 对应 ses-01~ses-04 的 para1
# day 5~8 对应 ses-01~ses-04 的 para2
DAY_TO_SESSION_TASK = {
    1: ("ses-01", "para1"),
    2: ("ses-02", "para1"),
    3: ("ses-03", "para1"),
    4: ("ses-04", "para1"),
    5: ("ses-01", "para2"),
    6: ("ses-02", "para2"),
    7: ("ses-03", "para2"),
    8: ("ses-04", "para2"),
}


def load_chisco2_montage(montage_file: Path):
    """
    从 Chisco 1.0 montage.csv 构建 Chisco-2.0 montage。

    只保留真正有空间坐标的 122 个 scalp EEG 通道。
    """
    df = pd.read_csv(montage_file)

    required_columns = {"label", "x", "y", "z"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Montage must contain columns: {required_columns}")

    # 去掉空行
    df = df.dropna(subset=["label", "x", "y", "z"]).copy()
    df["label"] = df["label"].astype(str)

    # 删除 Chisco-2.0 不使用的空间通道
    df = df[~df["label"].isin(MONTAGE_EXCLUDE_CHANNELS)].copy()

    # 转换坐标
    for column in ["x", "y", "z"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # 检查是否还有 (0, 0, 0)
    zero_mask = np.isclose(df[["x", "y", "z"]].to_numpy(), 0.0).all(axis=1)
    if zero_mask.any():
        zero_channels = df.loc[zero_mask, "label"].tolist()
        raise ValueError(
            "Montage still contains zero-position EEG channels: "
            f"{zero_channels}"
        )

    if len(df) != 122:
        raise ValueError(
            f"Expected 122 EEG channels in montage, but got {len(df)}."
        )

    ch_pos = {
        row["label"]: np.array([row["x"], row["y"], row["z"]], dtype=float)
        for _, row in df.iterrows()
    }

    # 保留原 montage 的坐标语义。
    # 因为旧文件没有 nasion/LPA/RPA，
    # MNE 仍可能提示 fiducial warning，这是非致命的。
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="unknown")

    return montage


def get_bids_prefix(path: Path, suffix: str):
    """
    例如：

    sub-01_ses-01_task-para1_run-01_eeg.edf

    ->
    sub-01_ses-01_task-para1_run-01
    """
    name = path.name
    marker = f"_{suffix}"

    if marker not in name:
        raise ValueError(f"Cannot extract prefix from {path.name}")

    return name.split(marker)[0]


def pair_edf_and_events(eeg_dir: Path):
    """
    根据 BIDS 前缀严格匹配 EDF 和 events.tsv。

    不直接 zip(sorted(...))，
    防止少一个文件后所有文件发生错位。
    """
    edf_files = sorted(eeg_dir.glob("*_eeg.edf"))
    events_files = sorted(eeg_dir.glob("*_events.tsv"))

    edf_map = {get_bids_prefix(path, "eeg"): path for path in edf_files}
    events_map = {get_bids_prefix(path, "events"): path for path in events_files}

    edf_keys = set(edf_map)
    event_keys = set(events_map)

    missing_events = sorted(edf_keys - event_keys)
    missing_edf = sorted(event_keys - edf_keys)

    if missing_events:
        raise RuntimeError(
            "EDF without matching events.tsv:\n" + "\n".join(missing_events)
        )

    if missing_edf:
        raise RuntimeError(
            "events.tsv without matching EDF:\n" + "\n".join(missing_edf)
        )

    prefixes = sorted(edf_keys)

    return [(edf_map[prefix], events_map[prefix]) for prefix in prefixes]



def load_events_file(events_file: Path):
    """
    加载 Chisco-2.0 events.tsv。

    onset 列是相对记录起点的秒数（对应原始 1000 Hz 记录），
    sample 列对应原始 1000 Hz 采样点，这里都原样保留、不做换算。

    后续直接用 onset（秒）构建 annotations：
    onset 与采样率无关，重采样不需要任何手工换算。
    """
    events_df = pd.read_csv(events_file, sep="\t")

    required_columns = {"onset", "trial_type", "value", "sample"}
    missing = required_columns - set(events_df.columns)
    if missing:
        raise ValueError(f"{events_file.name} is missing columns: {missing}")

    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="raise")
    events_df["value"] = pd.to_numeric(events_df["value"], errors="raise")
    events_df["sample"] = pd.to_numeric(events_df["sample"], errors="raise")

    return events_df


def build_annotations(events_df):
    """
    用 events.tsv 的 onset（秒，对应原始 1000 Hz 记录）构建 mne.Annotations。

    onset 的单位是秒，与采样率无关：
    先 set_annotations，再 raw.resample 时，
    MNE 只改变数据点密度，annotations 的 onset 保持不变，
    因此事件时刻不会因为重采样而漂移。

    事件编码通过 EVENT_DESCRIPTIONS 映射为可读描述；
    若出现未登记的事件编码则直接报错，避免静默丢事件。
    """
    descriptions = events_df["value"].map(EVENT_DESCRIPTIONS)

    unknown = sorted(events_df.loc[descriptions.isna(), "value"].unique())
    if unknown:
        raise ValueError(
            f"Event codes without description mapping: {unknown}. "
            "Please add them to EVENT_DESCRIPTIONS."
        )

    return mne.Annotations(
        onset=events_df["onset"].to_numpy(dtype=float),
        duration=np.zeros(len(events_df), dtype=float),
        description=descriptions.tolist(),
    )

def build_bad_annotations(events_df, recording_end: float):
    """把任务前后的非任务段标记为坏段（描述以 BAD 开头，MNE 会据此排除）。

    - 前导段：[0, 第一个 Reading start)
    - 长休息：[long rest start, 紧随其后的第一个 Reading start)
    - 尾部  ：[最后一个 1.8 s rest start, 记录结束]
    """
    reading = events_df.loc[events_df["value"] == 65329, "onset"].to_numpy(dtype=float)
    rest = events_df.loc[events_df["value"] == 65381, "onset"].to_numpy(dtype=float)
    long_rest = events_df.loc[events_df["value"] == 65480, "onset"].to_numpy(dtype=float)

    onsets, durations, descriptions = [], [], []

    if reading[0] > 0:
        onsets.append(0.0)
        durations.append(float(reading[0]))
        descriptions.append(BAD_LEAD_IN)

    for start in long_rest:
        following = reading[reading > start]
        if len(following):
            onsets.append(float(start))
            durations.append(float(following[0] - start))
            descriptions.append(BAD_LONG_REST)

    if recording_end > rest[-1]:
        onsets.append(float(rest[-1]))
        durations.append(float(recording_end - rest[-1]))
        descriptions.append(BAD_TAIL)

    return np.asarray(onsets), np.asarray(durations), descriptions

def text_sentences_for(task: str, run_number: int) -> list[str]:
    """返回某个 task/run 对应的句子列表（按行顺序）。

    para1 的 run-0k 对应 text1-k，para2 的所有 run 对应 text2。
    """
    if task == "para1":
        stem = f"text1-{run_number}"
    elif task == "para2":
        stem = "text2"
    else:
        raise ValueError(f"未知 task: {task!r}")

    text_file = TEXTDATASET / f"{stem}.xlsx"
    if not text_file.exists():
        raise FileNotFoundError(f"文本文件不存在: {text_file}")

    return pd.read_excel(text_file)["句子"].astype(str).tolist()

def count_nonpunct_chars(sentence: str) -> int:
    """统计句子中非标点符号字符的个数（Unicode Punctuation 类别不算）。"""
    return sum(
        1 for char in sentence if unicodedata.category(char)[0] != "P"
    )

def write_text_sidecar(events_df, sentences, sidecar_path: Path):
    """把每个试次的句子与其 Reading / Inner-speech onset 写成 sidecar TSV。

    第 i 行句子对应第 i 个 65329 与第 i 个 65379，trial_index 从 0 开始。
    """
    reading = events_df.loc[events_df["value"] == 65329, "onset"].to_numpy(dtype=float)
    inner = events_df.loc[events_df["value"] == 65379, "onset"].to_numpy(dtype=float)

    if not (len(reading) == len(inner) == len(sentences)):
        raise RuntimeError(
            "试次数与文本数不匹配: "
            f"reading={len(reading)}, inner={len(inner)}, text={len(sentences)}"
        )

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trial_index": np.arange(len(sentences)),
            "reading_onset": reading,
            "inner_onset": inner,
            "sentence": sentences,
            "n_chars": [count_nonpunct_chars(sentence) for sentence in sentences],
        }
    ).to_csv(sidecar_path, sep="\t", index=False)

def to_str_list(values):
    """
    PyPREP 某些情况下可能返回 numpy.str_，
    统一转为 Python str。
    """
    if values is None:
        return []

    return [str(value) for value in values]


def get_eeg_channels(raw):
    """
    显式返回所有 EEG channel name。

    这里不依赖 pick_types 默认的 bad channel 排除行为。
    """
    return [
        ch_name
        for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types())
        if ch_type == "eeg"
    ]


def normalize_subject(subject: str) -> str:
    """统一 subject 格式。

    支持:
        "1"      -> "sub-01"
        "01"     -> "sub-01"
        "sub-01" -> "sub-01"
    """
    subject = subject.strip()
    if subject.startswith("sub-"):
        number = subject.removeprefix("sub-")
    else:
        number = subject
    try:
        return f"sub-{int(number):02d}"
    except ValueError as e:
        raise ValueError(
            f"非法 subject: {subject!r}，应类似 '01'、'1' 或 'sub-01'"
        ) from e

def all_subjects() -> list[str]:
    if not CHISCO2_FOLDER.exists():
        raise FileNotFoundError(f"数据目录不存在: {CHISCO2_FOLDER}")
    subjects = [p.name for p in CHISCO2_FOLDER.glob("sub-*") if p.is_dir()]
    def subject_number(name: str) -> int:
        match = re.fullmatch(r"sub-(\d+)", name)
        if match is None:
            return 10**9
        return int(match.group(1))
    return sorted(subjects, key=subject_number)

def eeg_files_for_day(subject: str, day: int) -> list[Path]:
    """返回某被试某天对应的 EDF 文件列表。

    day 1~4 -> ses-01~ses-04 的 para1
    day 5~8 -> ses-01~ses-04 的 para2
    """
    if day not in DAY_TO_SESSION_TASK:
        raise ValueError(
            f"非法 day: {day}，仅支持 {sorted(DAY_TO_SESSION_TASK)}"
        )

    session, task = DAY_TO_SESSION_TASK[day]
    eeg_dir = CHISCO2_FOLDER / subject / session / "eeg"
    if not eeg_dir.exists():
        raise FileNotFoundError(f"EEG 目录不存在: {eeg_dir}")

    pattern = f"{subject}_{session}_task-{task}_run-*_eeg.edf"
    return sorted(eeg_dir.glob(pattern))

def convert_edf_to_fif(
    edf_file: Path,
    events_file: Path,
    output_path: Path,
    montage,
    sentences,
    sample_rate: float = 250.0,
):
    """读取单个 EDF，设置通道类型/montage/annotations 并重采样，再存为 FIF。

    sentences: 该 run 的句子列表（按行顺序）。
    sample_rate: 目标采样率（Hz），默认 250；传 None 则保持原始采样率。

    是否跳过已存在的 FIF 由调用方（main）判断。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取原始 EDF
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    n_channels_raw = raw.info["nchan"]

    # 修正通道类型：VEO/HEO 是 EOG，不应参与 scalp montage
    eog_present = {ch: "eog" for ch in EOG_CHANNELS if ch in raw.ch_names}
    if eog_present:
        raw.set_channel_types(eog_present, on_unit_change="ignore")

    # 丢弃 montage 中不存在的无效 EEG 通道
    invalid_present = [ch for ch in INVALID_EEG_CHANNELS if ch in raw.ch_names]
    if invalid_present:
        raw.drop_channels(invalid_present)

    # 设置 montage：EEG 通道必须全部有坐标，否则报错
    raw.set_montage(montage, on_missing="raise", verbose=False)

    # annotations：onset 单位是秒，与采样率无关，先写入再重采样即可保持时刻不变
    events_df = load_events_file(events_file)
    event_annotations = build_annotations(events_df)

    # 非任务坏段（前导 / 长休息 / 尾部）也一并写入，autoreject 会据此排除对应 epoch
    bad_onsets, bad_durations, bad_descriptions = build_bad_annotations(
        events_df, recording_end=raw.n_times / raw.info["sfreq"]
    )
    onsets = np.concatenate([event_annotations.onset, bad_onsets])
    durations = np.concatenate([event_annotations.duration, bad_durations])
    descriptions = list(event_annotations.description) + bad_descriptions
    order = np.argsort(onsets, kind="stable")
    raw.set_annotations(
        mne.Annotations(
            onset=onsets[order],
            duration=durations[order],
            description=[descriptions[i] for i in order],
        )
    )

    # 文本 sidecar：FIF 只存信号+trigger，文本与试次 onset 单独存一份 TSV
    sidecar_path = output_path.with_name(
        output_path.name.replace("_raw.fif", "_text.tsv")
    )
    write_text_sidecar(events_df, sentences, sidecar_path)

    # 重采样
    if sample_rate is not None and raw.info["sfreq"] != sample_rate:
        raw.resample(sample_rate, verbose=False)

    # PyPREP：线噪去除 + 重参考 + bad 通道插值
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(50, raw.info["sfreq"] / 2, 50),
    }
    print("[PREP] Running PyPREP ...")
    prep = PrepPipeline(raw, prep_params, montage, ransac=False)
    prep.fit()
    raw = prep.raw.copy()

    noisy_original = to_str_list(prep.noisy_channels_original.get("bad_all", []))
    interpolated = to_str_list(prep.interpolated_channels)
    still_bad = to_str_list(prep.still_noisy_channels)
    print(f"[PREP] original bad : {noisy_original}")
    print(f"[PREP] interpolated : {interpolated}")
    print(f"[PREP] still bad    : {still_bad}")
    print(f"[PREP] info['bads']  : {raw.info['bads']}")

    # PyPREP 输出仍保留低频漂移（线噪去除时把 trend 加回了），
    # autoreject / ICA 之前需要显式对 EEG 做 1 Hz 高通
    raw.filter(l_freq=1.0, h_freq=None, picks="eeg", verbose=False)
    print("[Filter] 1 Hz high-pass applied to EEG.")

    # autoreject：在固定长度 EEG epoch 上估计拒绝阈值（供后续 ICA 拟合使用）
    # bads 会被 autoreject 自动排除，不需要提前插值
    fixed_epoch_duration = 2.0
    fixed_events = mne.make_fixed_length_events(
        raw,
        id=1,
        start=0,
        stop=None,
        duration=fixed_epoch_duration,
        overlap=0.0,
    )
    fixed_epochs = mne.Epochs(
        raw,
        events=fixed_events,
        event_id={"fixed": 1},
        tmin=0.0,
        tmax=fixed_epoch_duration - 1.0 / raw.info["sfreq"],
        baseline=None,
        picks="eeg",
        preload=True,
        reject=None,
        flat=None,
        verbose=False,
    )
    reject_threshold = get_rejection_threshold(
        fixed_epochs, ch_types=["eeg"], verbose=False
    )
    n_fixed_before = len(fixed_epochs)
    fixed_epochs.drop_bad(reject=reject_threshold, verbose=False)
    print(f"[Autoreject] threshold   : {reject_threshold}")
    print(f"[Autoreject] epochs kept : {len(fixed_epochs)}/{n_fixed_before}")

    # ICA：只在 EEG 上拟合（bads 自动排除），再用 ICLabel 剔除伪迹成分
    ica = mne.preprocessing.ICA(
        n_components=30,
        random_state=97,
        max_iter="auto",
        method="infomax",
        fit_params=dict(extended=True),
    )
    print("[ICA] Fitting on fixed-length epochs ...")
    ica.fit(fixed_epochs, picks="eeg", verbose=False)

    # ICLabel 需要与 ica 通道一致的实例；pick 出 ica 用到的通道
    ic_labels = label_components(raw.copy().pick(ica.ch_names), ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    print(f"[ICA] component labels : {dict(Counter(labels))}")
    print(f"[ICA] excluding idx    : {exclude_idx}")

    ica.apply(raw, exclude=exclude_idx)

    # ICA 之后，用已去伪迹的邻近通道插值 still-bad 通道
    if raw.info["bads"]:
        print(f"[ICA] interpolate bads : {raw.info['bads']}")
        raw.interpolate_bads(verbose=False)

    raw.save(output_path, overwrite=True, verbose=False)
    print(f"[Saved] {output_path}")
    print(f"[Saved] {sidecar_path}")

    qc = {
        "edf_file": edf_file.name,
        "events_file": events_file.name,
        "n_channels_raw": n_channels_raw,
        "n_eeg": len(get_eeg_channels(raw)),
        "sfreq": raw.info["sfreq"],
        "n_events": len(events_df),
        "n_event_65329": int((events_df["value"] == 65329).sum()),
        "n_event_65379": int((events_df["value"] == 65379).sum()),
        "n_event_65381": int((events_df["value"] == 65381).sum()),
        "original_bad_count": len(noisy_original),
        "original_bad": ";".join(noisy_original),
        "interpolated_count": len(interpolated),
        "interpolated": ";".join(interpolated),
        "still_bad_count": len(still_bad),
        "still_bad": ";".join(still_bad),
    }
    return qc

def main():
    parser = argparse.ArgumentParser(description="Chisco-2.0 EEG 预处理（PyPREP + ICA）。")
    parser.add_argument("--subject", help="被试 ID（例如 sub-01）")
    parser.add_argument("--days", type=int, nargs="+", required=True, help="day 编号（1~8），可传多个，如 --days 1 2 3")
    parser.add_argument("--sample-rate", type=float, default=250.0, help="重采样目标采样率 Hz（默认 250）")
    parser.add_argument("--n-jobs", type=int, default=1, help="并行任务数（默认 1）")
    args = parser.parse_args()

    subjects = ([normalize_subject(args.subject)] if args.subject else all_subjects())

    montage = load_chisco2_montage(MONTAGE_FILE)
    print(f"[Montage] {len(montage.ch_names)} EEG positions loaded.")

    for subject in subjects:
        output_dir = PREPROCESSED_FOLDER / "fif" / f"{args.sample_rate:g}hz" / subject
        qc_path = output_dir / "preproc_qc.csv"
        qc_rows = []

        for day in sorted(args.days):
            session, task = DAY_TO_SESSION_TASK[day]
            edf_files = eeg_files_for_day(subject, day)
            print(
                f"[{subject}] day {day} -> {session} / task-{task}: "
                f"{len(edf_files)} EDF files"
            )

            for edf_file in edf_files:
                events_file = edf_file.with_name(
                    edf_file.name.replace("_eeg.edf", "_events.tsv")
                )
                run_number = int(re.search(r"run-(\d+)", edf_file.name).group(1))
                output_path = (
                    output_dir
                    / f"{subject}_day-{day:02d}_run-{run_number:02d}_raw.fif"
                )

                # 过滤：该 run 的 FIF 已存在则跳过
                if output_path.exists():
                    print(f"[Skip] Existing FIF found: {output_path.name}")
                    continue

                sentences = text_sentences_for(task, run_number)
                qc = convert_edf_to_fif(
                    edf_file,
                    events_file,
                    output_path,
                    montage,
                    sentences,
                    args.sample_rate,
                )

                # 每处理一个文件就落一次 QC，防止中断后全丢
                qc_rows.append(qc)
                pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
                print(f"[QC] {qc_path}")




if __name__ == "__main__":
    main()