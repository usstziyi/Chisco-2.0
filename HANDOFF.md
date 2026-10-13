# HANDOFF · Chisco-2.0 EEG→文本 迁移交接文档

> 生成日期：2026-10-13
> 用途：把当前项目迁移到另一台电脑，并让接手的 agent/人快速恢复到可继续工作的状态。
> 本文件只描述**当前工作区实际存在的东西**；凡是不存在或被删除的，都会明确标注。

---

## 0. 一句话背景

把 Chisco-2.0（`ds006317`）的中文内语/朗读 EEG 解码成文本。技术路线：**字符级等长 token 化 →
共享权重 TSConv tokenizer → 2 层 Transformer → attention pooling → 与冻结 LaBSE 文本空间做对比学习**。
已完成真实信号、随机噪声对照、池化消融三组实验，并写出论文 Method 初稿。

---

## 1. 迁移清单（先看这节，避免踩坑）

### 1.1 必须迁移 / 不需要迁移

| 资产 | 路径 | 大小 | 迁移方式 |
|---|---|---|---|
| 代码 + 论文 + 脚本 | 仓库（见 §3） | < 1 MB | **git clone/push** |
| 原始 EDF 数据 | `datasets/ds006317-download/` | ≈ 26.5 GB | **被 gitignore，需单独拷贝** |
| 预处理 FIF 数据 | `datasets/preprocessed/` | ≈ 13.0 GB | **被 gitignore，需单独拷贝** |
| 训练产物 | `outputs/*_<时间戳>/` | 每个 ≈ 2.8 MB | **被 gitignore，需单独拷贝**（可选） |
| 虚拟环境 | `.venv/` | ≈ 1.0 GB | **不要拷贝**，到新机 `uv sync` 重建 |
| LaBSE 模型缓存 | `~/.cache/huggingface/hub/models--sentence-transformers--LaBSE` | ≈ 1.8 GB | **需单独拷贝或联网重下**（见 1.3） |

**合计需要搬运的数据约 40 GB**，代码部分不到 1 MB。

### 1.2 新机环境要求

- Python **3.12**（见 `.python-version`），用 `uv` 管理：`uv sync` 即可重建 `.venv`。
- **torch 版本按硬件重装**：本机是 **`torch 2.14.1+cpu`（无 CUDA）**。新机若有 GPU，
  需换成对应 CUDA 版本，否则依然只能跑 CPU（当前单 epoch 约 150 s，20 epoch × 4 组实验 ≈ 数小时）。
- 依赖清单见 `pyproject.toml` / `uv.lock`（已随仓库迁移）。

### 1.3 LaBSE 缓存的坑（最容易导致"跑不起来"）

`char_token/text_encoder.py` 里 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`，
首次运行会去 **hf-mirror.com** 下载 LaBSE（约 1.8 GB）。二选一：

1. 直接拷贝旧机缓存：把 `~/.cache/huggingface/hub/models--sentence-transformers--LaBSE` 复制到新机同位置；
2. 或让新机联网访问 hf-mirror.com 自动下载。

> 训练入口每次都会加载 LaBSE 预计算全部句子嵌入；缓存缺失会卡在下载。

---

## 2. 数据资产现状

| 项 | 值 |
|---|---|
| 总表 | `datasets/preprocessed/all_text.tsv`，**8528 条 trial**（含表头 8529 行） |
| FIF 文件 | `datasets/preprocessed/fif/250hz/sub-01/`，共 **32 个** `*_raw.fif`（day 01–08 × run 01–04） |
| 被试范围 | **仅 `sub-01`**（原始数据 `datasets/ds006317-download/` 走 BIDS 布局） |
| 采样率 | 250 Hz，**122** 个有效头皮 EEG 通道 |
| 每字时长 | 固定 **0.4 s**（250 Hz → 100 采样点/token） |
| day ↔ task | day 1–4 = ses-01~04 的 para1；day 5–8 = ses-01~04 的 para2 |

**关键数据特性**：句子只在**跨天**之间重复。`day 1–4` 下同一句子有 **4 个视图**，
这是对比学习"同句多视图正样本"的唯一来源（单天每句只录 1 次）。

预处理与汇总脚本：
- `script/preprocessing.py`：EDF → 250 Hz → PyPREP（线噪/重参考/插值）→ 1 Hz 高通 → autoreject 阈值
  → ICA(30, infomax) + ICLabel 剔除伪迹 → 存 FIF + `*_text.tsv` sidecar。
- `script/summarize_text.py`：汇总所有 sidecar → `all_text.tsv`（算 `*_sample`、`*_dur_samples`、`n_chars`）。

> 注意：`all_text.tsv` 是从 sidecar 汇总的产物；sidecar `*_text.tsv` 在 FIF 目录下。
> 若只拷 `all_text.tsv` 而没拷 FIF，数据集仍可构建元数据，但**取不到信号**（`__getitem__` 会报 FIF 不存在）。

---

## 3. 代码结构（git 已跟踪 35 个文件，分支 `token`）

```
char_token/            # 主实验：字符级 token 方案
  dataset.py           # CharTokenEEGDataset（切 token）+ char_token_collate（token 级 padding）
  model.py             # CharTokenEncoder（TSConv tokenizer）+ CharTokenEEGEncoder（+Transformer+attn pool）
  sampler.py           # SentenceGroupBatchSampler（n_sentences × n_repeats 分组采样）
  text_encoder.py      # LaBSETextEncoder（冻结，768 维）
  train.py             # 训练主循环 + 多正样本对称 InfoNCE + 检索评测 + CSV 日志（★被其余模块复用）

char_token_noise/      # 随机噪声对照（形状同构，数值换噪声）
  dataset.py           # NoiseTokenEEGDataset（按索引取种；refresh=never/epoch）
  train.py             # 噪声训练入口（复用 char_token 的损失/采样/评测）

char_token_pooling/    # 池化消融 + 长度探针
  model.py             # MeanPoolEEGEncoder（子类，仅覆写池化）+ build_encoder 工厂
  train.py             # 消融入口（--pooling attn mean 一次跑完）
  probe.py             # 长度探针：ridge 回归 e→n_tok + 按长度分层检索

script/                # 预处理与诊断
  preprocessing.py     # EDF→FIF（★注意：本文件有 1 处崩溃隐患，见 §6）
  summarize_text.py    # 汇总 all_text.tsv
  channel_consistency.py / channel_consistency_spectral.py   # 通道一致性分析（早期）
  plot_fif_waveform.py # FIF 波形画图

paper/
  Method.md            # ★论文 Method 节中文初稿（12 小节 + 3 附录）
  char_token_噪声对照实验报告.md  # 噪声对照实验报告（8 节）
```

### 3.1 重要：`train/` 目录已不存在

早期工作区曾有 `train/`（含 `train.py`、`dataloader/`、`eeg-encoder/TSConvAdaptiveEEGEncoder.py`、
`text-encoder/text_encoder_labse.py`），**当前已不在工作区，也未纳入 git**。

⚠️ 影响：**不能再把 `train/eeg-encoder/` 当作现成的内部 baseline 来引用**。
若论文需要一个 baseline，必须先确认这些文件能否找回，或改为重新实现 / 引用文献方法。

---

## 4. 已完成的工作

### 4.1 代码（全部可用，已冒烟验证）

- **数据管线**：预处理（PyPREP+ICA+ICLabel）→ 文本汇总 → `all_text.tsv`。
- **字符级 token 方案**：段长恒为 100 采样点整数倍（8528/8528 条可整除，零丢弃）；
  `(C, n·T) → (n, C, T)`，V→µV 缩放 1e6。
- **模型**：TSConv tokenizer（$T_1{=}92, T_2{=}40, kT_2{=}1280$）+ 位置编码 + 2 层 Transformer +
  attention pooling + L2 归一化；attn 版参数量 **723,680**。
- **损失**：多正样本对称 InfoNCE（正样本由句子 ID 定义，避免假负样本）。
- **采样**：`SentenceGroupBatchSampler`，保证同句多视图同框（`pos/anch = 4.00`）。
- **评测**：按句子 8:2 切分（train 899 句/3596 trial，val 225 句/900 trial，**句子重叠 = 0**）；
  `eeg2text` 与 `text2eeg` 双向 top-k（两方向候选池不同）。
- **噪声对照模块**、**池化消融模块**、**长度探针**。

### 4.2 实验结果（全部为 `sub-01`、`day 1–4`、`inner`、batch 32 句×4 视图、20 epoch、seed 0、CPU）

随机基线：`e2t@1 = 1/225 = 0.0044`，`e2t@5 = 5/225 = 0.0222`，`t2e@5 = 0.0222`，
`ln(128) = 4.852`，`ln(900) = 6.802`。

| 实验 | 输出目录 | 关键结果 |
|---|---|---|
| 真实信号 baseline | `outputs/char_token/20261012_190500/` | best `val_loss` **6.3631**(ep12)；`e2t@5` 峰值 **0.0878**；`t2e@5` 峰值 **0.1156** |
| 随机噪声对照 | `outputs/char_token_noise/20261012_195453/` | `train_loss` 全程 4.851–4.857（**≈ ln(128)=4.852**）；`val_loss` 全程 6.802–6.805（**≈ ln(900)=6.802**）；`e2t@5` 恒 **0.0222 = 随机** |
| 池化消融 attn | `outputs/char_token_pooling/attn_20261012_212316/` | best `val_loss` 6.3392(ep11)；`e2t@5` 峰值 0.0844；参数 **723,680** |
| 池化消融 mean | `outputs/char_token_pooling/mean_20261012_221521/` | best `val_loss` **6.3208**(ep10)；`e2t@5` 峰值 **0.0900**；参数 **657,504** |

每个目录含 `train_epochs.csv`（每轮一行）、`train_steps.csv`（每 step 一行）、`best.pt`。

长度探针（ridge 回归 `句嵌入 → n_tok`，train 拟合 / val 评测，零假设=打乱标签）：

| 模型 / 输入 | val R² | MAE(token) | Spearman ρ | 零假设 R² |
|---|---:|---:|---:|---:|
| attn + 真实 EEG | +0.124 | 1.31 | +0.373 | −0.083 |
| mean + 真实 EEG | **+0.290** | 1.18 | +0.570 | −0.052 |
| attn + 噪声（噪声模型） | **+0.986** | 0.14 | +0.970 | −0.072 |

### 4.3 论文

- `paper/Method.md`：Method 节完整中文初稿（§2.1 数据集 … §2.12 复现信息 + 符号表 / 五维自审 /
  claim–evidence 映射）。含 §2.11 长度泄漏诊断协议。
- `paper/char_token_噪声对照实验报告.md`：噪声对照 8 节报告。

### 4.4 三条已确立的结论（可作为论文主张）

1. **成绩依赖真实脑电**：噪声组精确钉在理论随机基线（`ln(128)` / `ln(900)` / `1/N`），
   真实组显著优于随机。
2. **排除了 token 计数/长度泄漏**：噪声组 token 形状与真实组完全一致，指标却无任何抬升。
3. **池化：表示层面有差、任务层面无差**：
   - mean-pool 句嵌入保留**更多**可线性读出的长度（R² 0.29 vs 0.12）；
   - 但端到端检索指标**统计上不可区分**（差异全在 0.11% 噪声底内）→ **不能声称 attn 更优**；
   - 噪声模型 R²≈0.99 却检索恰为随机 → **长度即使被完美编码也不足以抬高成绩**（给出上界）。

---

## 5. 未完成的工作（按优先级）

### P0 · 投稿前必须补

1. **多被试**：目前**只有 `sub-01`**。需对更多被试跑完整流程，主结果报均值 ± 标准差。
2. **训练充分性**：20 epoch 下真实组 `val_loss` 已在 ep12 见底回升、检索指标仍在缓慢爬升
   → 属**训练不足**。建议延长到 50–100 epoch + 早停。
3. **多种子误差棒**：所有已知差异都在 900 条 val 的噪声底（单条翻转 = 0.11%）内。
   需 seed 0/1/2 至少 3 次，才能对"attn vs mean 无差异"给出可信结论。
   - 附带修一个已知问题：`char_token_pooling/train.py` 顺序跑多变体时，
     `SentenceGroupBatchSampler` 的 `epoch` 状态**在两变体间累加**（attn 用 ep1–20，mean 用 ep21–40），
     两变体看到的 batch 顺序不同。多 seed 版本应一并修掉（`run_variant` 里重建 sampler）。

### P1 · 显著增强说服力

4. **Baseline 对比**：⚠️ 原 `train/eeg-encoder/` 已不存在（见 §3.1），需先确认能否找回，
   或改为重新实现 / 引用文献方法，否则论文缺"优于既有方法"的证据。
5. **其余消融**：
   - 采样器 `n_repeats ∈ {1, 2, 4}`（验证分组采样的必要性）；
   - token 时长 `0.2 / 0.4 s`（验证 0.4 s 切的合理性）；
   - `GroupNorm` vs `BatchNorm`（验证 padding 污染的论断，目前**只有设计论证**）；
   - 正样本定义（句子 ID vs 对角线），验证多正样本机制。
6. **统计检验**：所有指标配置换检验 / bootstrap 置信区间，不能只报点估计。
7. **更丰富的噪声对照**：相位打乱（保功率谱）、跨 trial 置换（保边际分布）、错配配对
   （真实 EEG 但打乱句子对应）。`--noise-refresh never` 那组也尚未纳入报告。

### P2 · 写作与工程

8. **论文其余章节**：Results/Experiments、Introduction、Abstract、Conclusion、Related Work
   （文献检索与综述应走文献技能，不在写作技能范围）。
9. **英文版**：Method 目前是中文，投稿需整节译写并统一术语。
10. **占位引用待补**：`Method.md` 中 3 处 `[待补引用]`（Chisco-2.0 数据集出处、
    "每字 0.4 s / inner 多一字"的事件定义依据、LaBSE 与 τ=0.2 的依据）。
    引用真实性与格式应走引用合规流程，不要凭记忆填。
11. **GPU 重跑**：迁移到有 GPU 的机器后，建议重跑全部实验并核对结论。

---

## 6. 已知坑与注意事项

1. **torch 是 CPU-only**：`2.14.1+cpu`，`torch.cuda.is_available() == False`。换机后按硬件重装。
2. **`script/preprocessing.py` 无需重跑**：本机预处理产物已存在，`main()` 遇已存在的 FIF 会自动跳过
   （`[Skip] Existing FIF found`）。若要重跑，注意它依赖 `datasets/ds006317-download/` 的 BIDS 布局与
   `datasets/montage.csv`，且 `--days` 为必填参数。
3. **工作区干净**：git 分支 `token`，HEAD `f249a5c`，`git status` 无改动。
   `outputs/` 除少数历史文件外**均被 gitignore**，搬运时勿漏。
4. **`ds006317-download/` 与 `preprocessed/` 被 gitignore**，git 迁移**不会**带上数据。
5. **单条 val 翻转 = 0.11%**：val 只有 900 条，`e2t@1` 极不稳定；模型选择默认按 `val_loss`
   而非 top-1，不要用 top-1 差异下结论。
6. **单种子不可复现细节**：`char_token_pooling/train.py` 在建模型前又调了一次 `set_seed`，
   RNG 消耗路径与 `char_token/train.py` 不同 → 即便同种子同配置，attn 变体也不会与主实验
   逐位一致（实测 6.3392 vs 6.3631）。这不是 bug，但报告时要如实说明是"单种子单次运行"。
7. **评测候选池口径**：`eeg2text` 候选池是**去重后的唯一句子**（225），`text2eeg` 是**全部 val EEG 样本**（900）。
   两方向候选池不同是刻意设计，比较指标时勿混用基线。
8. **`--noise-refresh` 语义**：`epoch`（默认，每轮换噪，噪声零可学习信息）/ `never`（固定噪声，
   模型可在训练集上"背噪声"，用于说明"低训练损失 ≠ 学到信号"）。
9. **数据整除约束**：`CharTokenEEGDataset` 只保留段长能被 `token_samples` 整除的 trial；
   改 `--char-duration` 需确认仍无丢弃（0.2 s 时 token 数翻倍，`max_tokens=32` 够用）。

---

## 7. 快速开始（新机复现）

```powershell
# 1) 重建环境（.venv 不迁移）
uv sync

# 2) 确认数据与缓存就位
Test-Path datasets\preprocessed\all_text.tsv        # 应为 True
Test-Path outputs\char_token_pooling\attn_20261012_212316\best.pt

# 3) 冒烟：数据集形状 + 单 epoch
uv run python char_token/dataset.py

# 4) 三条对照实验（同配置）
uv run python char_token/train.py        --day 1 2 3 4 --n-sentences 32 --n-repeats 4 --epochs 20 --seed 0
uv run python char_token_noise/train.py  --day 1 2 3 4 --n-sentences 32 --n-repeats 4 --epochs 20 --seed 0 --noise-mode gaussian --noise-std 1.0 --noise-seed 0 --noise-refresh epoch
uv run python char_token_pooling/train.py --day 1 2 3 4 --n-sentences 32 --n-repeats 4 --epochs 20 --pooling attn mean

# 5) 长度探针（用已有 checkpoint）
uv run python char_token_pooling/probe.py --pooling attn --input real
uv run python char_token_pooling/probe.py --pooling mean --input real
uv run python char_token_pooling/probe.py --pooling attn --input noise --ckpt outputs/char_token_noise/20261012_195453/best.pt
```

> 无 `uv` 时用 `.\.venv\Scripts\python.exe <脚本> <参数>`（等价）。

---

## 8. 给下一个 agent 的建议起手式

1. 先读 `paper/Method.md`（尤其 §2.4.3 池化、§2.10 噪声对照、§2.11 长度探针）与
   `paper/char_token_噪声对照实验报告.md`，即可掌握全部方法与结论。
2. 若目标是**投会议正文/期刊**：按 §5 的 P0 → P1 顺序补实验，**多被试 + 多 seed 是第一优先**。
3. 若目标是**先投 workshop/短文**：现有证据（三条结论）基本够用，可直接起草 Results 节。
4. 动任何代码前，先确认 `train/` 是否需要找回（§3.1），以及新机 torch 是否为 GPU 版。
