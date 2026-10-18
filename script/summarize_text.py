"""汇总 preprocessed 目录下所有 *_text.tsv 为一张总表。

列：sample / subject / day / run / trial_index /
    reading_onset / reading_sample / reading_dur_samples /
    inner_onset / inner_sample / inner_dur_samples / sentence /
    category_number / category_name / n_chars
输出：datasets/preprocessed/all_text.tsv

用法：
    uv run python script/summarize_text.py
"""

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "preprocessed"
OUTPUT_PATH = PREPROCESSED_FOLDER / "all_text.tsv"

JSON_FOLDER = PROJECT_ROOT / "datasets" / "json"
TEXTMAPS_PATH = JSON_FOLDER / "textmaps.json"
CLASSNUMBER_PATH = JSON_FOLDER / "classnumber.json"

# 每个字符的时长（秒）；reading/inner 的持续时间都按它换算
CHAR_DURATION = 0.4

# 从 sub-01_day-01_run-01_text.tsv 里提取 subject / day / run
NAME_PATTERN = re.compile(
    r"(?P<subject>sub-\d+)_day-(?P<day>\d+)_run-(?P<run>\d+)_text\.tsv$"
)

# 从路径里的 250hz 目录提取采样率
SAMPLE_RATE_PATTERN = re.compile(r"(?P<sample>\d+)hz")


def main():
    textmaps = json.loads(TEXTMAPS_PATH.read_text(encoding="utf-8"))
    classnames = json.loads(CLASSNUMBER_PATH.read_text(encoding="utf-8"))

    tsv_files = sorted(PREPROCESSED_FOLDER.rglob("*_text.tsv"))

    # 排除本脚本自己的输出文件
    tsv_files = [path for path in tsv_files if path != OUTPUT_PATH]

    if not tsv_files:
        print("没有找到 *_text.tsv 文件。")
        return

    frames = []
    for tsv_path in tsv_files:
        match = NAME_PATTERN.search(tsv_path.name)
        if match is None:
            print(f"[Skip] 文件名不符合命名规则: {tsv_path}")
            continue

        sample_match = SAMPLE_RATE_PATTERN.search(str(tsv_path))
        if sample_match is None:
            print(f"[Skip] 路径里找不到采样率: {tsv_path}")
            continue

        df = pd.read_csv(tsv_path, sep="\t")
        sample_rate = int(sample_match.group("sample"))
        df.insert(0, "sample", sample_rate)
        df.insert(1, "subject", match.group("subject"))
        df.insert(2, "day", int(match.group("day")))
        df.insert(3, "run", int(match.group("run")))

        # onset（秒）换算成该事件在 FIF 中的起始采样点索引
        reading_sample = (df["reading_onset"] * sample_rate).round().astype(int)
        inner_sample = (df["inner_onset"] * sample_rate).round().astype(int)
        # 事件持续采样点数：reading 按 n_chars 个字，inner 按 n_chars + 1 个字
        reading_dur = (df["n_chars"] * CHAR_DURATION * sample_rate).round().astype(int)
        inner_dur = (
            (df["n_chars"] + 1) * CHAR_DURATION * sample_rate
        ).round().astype(int)

        df.insert(
            df.columns.get_loc("reading_onset") + 1, "reading_sample", reading_sample
        )
        df.insert(
            df.columns.get_loc("reading_sample") + 1,
            "reading_dur_samples",
            reading_dur,
        )
        df.insert(
            df.columns.get_loc("inner_onset") + 1, "inner_sample", inner_sample
        )
        df.insert(
            df.columns.get_loc("inner_sample") + 1,
            "inner_dur_samples",
            inner_dur,
        )

        # 按句子查类别号码、类别名称，插到 sentence 列后面
        category_number = df["sentence"].map(textmaps)
        category_name = category_number.map(
            lambda number: classnames.get(str(int(number))) if pd.notna(number) else None
        )
        missing = category_number.isna()
        if missing.any():
            print(
                f"[Warn] {tsv_path.name} 有 {missing.sum()} 条句子未匹配到类别: "
                f"{sorted(df.loc[missing, 'sentence'].unique())}"
            )

        sentence_pos = df.columns.get_loc("sentence") + 1
        df.insert(sentence_pos, "category_number", category_number)
        df.insert(sentence_pos + 1, "category_name", category_name)

        frames.append(df)
        print(f"[Read] {tsv_path.name} ({len(df)} 行)")

    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values(
        ["sample", "subject", "day", "run", "trial_index"]
    ).reset_index(drop=True)

    output_path = OUTPUT_PATH
    summary.to_csv(output_path, sep="\t", index=False)
    print(f"\n[Saved] {output_path} 共 {len(summary)} 行")


if __name__ == "__main__":
    main()
