# Method（草稿）

> 说明：本稿为中文草稿，供后续译成英文投稿版使用。文中标注 `[待补引用]` 处需补齐文献/数据引用，
> 不得凭记忆填充。所有数值均取自本仓库现有实现与实际运行记录（见各节出处标注）。

---

## 2.1 数据集

实验使用 **Chisco-2.0** 数据集（本地目录 `datasets/ds006317-download`，`[待补引用]`）。该数据集为中文
脑电（EEG）范式，包含两类与文本相关的试次事件：

- **朗读（reading）**：被试出声朗读句子；
- **默读 / 内语（inner speech）**：被试默读句子。

每次记录采用 BIDS 组织，原始采样率 1000 Hz。数据包含 122 个有效头皮 EEG 通道（蒙特区取自 Chisco 1.0 的
`montage.csv`，仅保留具有空间坐标的 122 个位置；`10`、`111` 两个坐标无效的通道被剔除，
`EKG`、`EMG`、`VEO`、`HEO`、`Trigger` 不作为头皮 EEG 使用，其中 `VEO`/`HEO` 被重标为 EOG）。

day 与 session/task 的对应关系为：

| day | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| session | ses-01 | ses-02 | ses-03 | ses-04 | ses-01 | ses-02 | ses-03 | ses-04 |
| 文本集 | para1 | para1 | para1 | para1 | para2 | para2 | para2 | para2 |

其中 para1 的第 $k$ 个 run 对应文本文件 `text1-k`，para2 的所有 run 共享 `text2`。**因此同一批句子会在
day 1–4（或 day 5–8）之间跨会话重复出现**，这为后续的"同句多视图"正样本提供了数据基础（见 2.7）。

本稿报告的主实验使用 **`sub-01`、`sub-02` 两个被试、day 1–4、`inner`（默读）**条件；跨天留一
泛化见 2.12，跨被试零样本迁移见 2.13。

---

## 2.2 预处理

对每个 run 的 EDF 依次执行（实现见 `script/preprocessing.py`）：

1. **通道修正与蒙特区**：`VEO`/`HEO` 重标为 EOG；剔除无效 EEG 通道；设置 122 通道蒙特区
   （`on_missing="raise"`，保证任何缺坐标通道都会显式报错而非静默丢弃）。
2. **事件标注**：由 `events.tsv` 的 onset（秒，与采样率无关）构建 `mne.Annotations`，事件编码映射为
   `Reading start (65329)`、`Inner-speech start (65379)`、`1.8 s rest start (65381)`、
   `long rest (65480/65481)`；未登记的事件编码直接报错，避免静默丢事件。任务前导段、长休息段、尾部
   分别标注为 `BAD_lead_in` / `BAD_long_rest` / `BAD_tail`。
3. **重采样**：1000 Hz → **250 Hz**（先写 annotations 再重采样，事件时刻不漂移）。
4. **PyPREP**：线噪去除（50 Hz 及其谐波）、重参考（全 EEG 通道）、坏通道检测与插值（关闭 RANSAC）。
5. **1 Hz 高通**：仅作用于 EEG。PyPREP 在去线噪时把趋势项加回，因此在 autoreject / ICA 前显式高通。
6. **autoreject**：在固定 2.0 s 长度的 EEG epoch 上估计拒绝阈值。
7. **ICA 去伪迹**：`n_components=30`，infomax（extended），`random_state=97`，在固定长度 epoch 上拟合；
   用 **ICLabel** 自动标注成分，仅保留 `brain` / `other`，剔除其余成分；随后用邻近通道插值仍然坏掉的通道。
8. **落盘**：信号存为 FIF，文本与试次 onset 单独存为 `*_text.tsv` sidecar。

随后由 `script/summarize_text.py` 汇总所有 sidecar 为总表 `all_text.tsv`，按
`sample → subject → day → run → trial_index` 升序排列，并为每条试次计算：

- 起始采样点：`reading_sample = round(reading_onset × f_s)`，`inner_sample` 同理；
- **持续采样点数**：
  $$\text{reading\_dur} = n_{\text{char}} \times 0.4 \times f_s,\qquad
    \text{inner\_dur} = (n_{\text{char}} + 1) \times 0.4 \times f_s$$
  其中 $f_s = 250$ Hz，即**每个字固定占 0.4 s**；$n_{\text{char}}$ 为句子的**非标点字符数**
  （按 Unicode 类别剔除标点，即 `category[0] != 'P'`）。

> 该"每字 0.4 s、inner 段比 reading 段多一字"的时长约定与数据集事件定义一致
> （`reading/on` 至 `reading/off`、`inner/on` 至 `inner/off` 的间距）`[待补引用]`。

---

## 2.3 字符级等长 token 化

这是本文方法的出发点：**把"变长"从每个 token 的采样点数，转移到 token 的个数**。

设采样率对应的 token 长度为

$$T = \text{round}(d_{\text{char}} \cdot f_s) = \text{round}(0.4 \times 250) = 100 .$$

由于 2.2 中的段长均为 $0.4\,$s 的整数倍，**每条试次的段长必然可被 $T=100$ 整除**，因此可按字边界
无重叠切分：

$$(C,\, n_{\text{tok}} T) \;\xrightarrow{\text{reshape}}\; (n_{\text{tok}},\, C,\, T),$$

其中

$$n_{\text{tok}} = \frac{\text{dur}}{T} =
\begin{cases}
n_{\text{char}}, & \text{reading}\\
n_{\text{char}} + 1, & \text{inner}
\end{cases}$$

于是：

- 每个 token 的形状 $(C, T) = (122, 100)$ **恒定**，可复用同一套 token 编码器；
- 每条试次的 token 个数 $n_{\text{tok}}$ 可变 —— 在本数据上 $n_{\text{tok}} \in [5, 15]$。

切分后不丢弃任何样本：在本数据上 8528 条试次**全部**能被 100 整除（无残缺 token）。

**数值约定**：FIF 中信号以伏特存储，读入后统一乘以 $10^{6}$ 转换为**微伏（µV）**，与项目内其他数据集
约定一致。

**组 batch**：由于 token 形状已统一，collate 只需在 **token 数**这一维上 padding。对 batch
$\{(n_{\text{tok}}^{(i)}\}_{i=1}^{B}$，取 $n_{\max} = \max_i n_{\text{tok}}^{(i)}$，得到

$$\text{tokens} \in \mathbb{R}^{B \times n_{\max} \times C \times T},\qquad
  \text{mask} \in \{0,1\}^{B \times n_{\max}},$$

pad 位置填 0，`mask=1` 表示有效 token。padding 代价极小（最多十几个位置）。

---

## 2.4 EEG 编码器

整体流程：

$$\underbrace{(B, n, C, T)}_{\text{tokens}}
  \;\xrightarrow[\text{reshape}]{\text{共享权重}}\; (Bn, C, T)
  \;\xrightarrow{\text{token encoder}}\; (B, n, d)
  \;\xrightarrow{+\, \text{pos}}\; (B, n, d)$$
$$\xrightarrow{\text{Transformer}\times L}\; (B, n, d)
  \;\xrightarrow{\text{attn pool}}\; (B, d)
  \;\xrightarrow{\text{Linear}}\; (B, D)
  \;\xrightarrow{\text{L2 norm}}\; (B, D)$$

### 2.4.1 逐 token 编码器（共享权重）

token 编码器采用 NICE 风格的 **TSConv** 小 CNN（`CharTokenEncoder`）。输入 $(N, C, T)$ 先 reshape 为
$(N, 1, C, T)$，依次经过：

| 层 | 配置 | 输出形状（$N{=}Bn$） |
|---|---|---|
| 时间卷积 Conv2d | $1 \to k{=}32$，核 $(1, m_1{=}9)$ | $(N, 32, C, T_1)$ |
| 时间平均池化 AvgPool2d | 核 $(1, m_2{=}13)$，步长 $(1, s{=}2)$ | $(N, 32, C, T_2)$ |
| GroupNorm + ELU | $\text{num\_groups}{=}1$，通道 $k$ | $(N, 32, C, T_2)$ |
| 空间卷积 Conv2d | $k \to k$，核 $(C, 1)$ | $(N, 32, 1, T_2)$ |
| GroupNorm + ELU + Dropout | dropout $=0.3$ | $(N, 32, 1, T_2)$ |
| 线性投影 Linear | $k T_2 \to d$ | $(N, d)$ |

其中 $T_1 = T - m_1 + 1 = 92$，$T_2 = \lfloor (T_1 - m_2)/s \rfloor + 1 = 40$，
$k T_2 = 32 \times 40 = 1280$。

$$\mathbf{h}_j = W_{\text{proj}}\,\mathrm{flatten}\big(\mathrm{TSConv}(\mathbf{x}_j)\big),\quad j = 1,\dots,n_{\text{tok}}$$

**两个设计选择与理由**：

1. **权重共享**：所有字位置共用同一套 token 编码器权重（相当于 ViT 的 patch embedding）。若按字符位置
   各用一套权重，参数量随句子长度爆炸，且无法泛化到未见过的句子长度。
2. **GroupNorm 而非 BatchNorm**：collate 会向 batch 注入 pad 出来的**全零 token**；BatchNorm 会把这些
   零样本计入 batch 统计量，污染真实样本的归一化。GroupNorm（$\text{num\_groups}{=}1$）是**逐样本**归一化，
   完全不受 padding 影响。
3. 时间卷积核刻意取小（$m_1{=}9,\ m_2{=}13$）：输入仅 0.4 s，用大核会把时间信息压没。

### 2.4.2 位置编码与 Transformer

token 顺序即**字序**，位置信息必须显式给出。加入可学习位置编码
$\mathbf{p} \in \mathbb{R}^{1 \times M \times d}$（$M = \texttt{max\_tokens} = 32$，truncated normal，$\sigma=0.02$）：

$$\mathbf{z}_j = \mathbf{h}_j + \mathbf{p}_j .$$

随后经 $L=2$ 层 `TransformerEncoder`：

| 超参 | 值 |
|---|---|
| $d$（d_model） | 128 |
| 注意力头数 | 4 |
| 前馈维度 | 256 |
| dropout | 0.1 |
| 激活 | GELU |
| `norm_first` | True |

padding 通过 `key_padding_mask = ~mask` 屏蔽，使 pad token 不参与注意力。

**层数刻意取浅（$L=2$）**：token 数只有 5–15、数据量也小，层数一多极易过拟合。

### 2.4.3 注意力池化

用一个可学习 query $\mathbf{q} \in \mathbb{R}^{1 \times 1 \times d}$（truncated normal，$\sigma=0.02$）对有效
token 做加权汇总：

$$\mathbf{z}^{\text{pool}} = \mathrm{MHA}(\mathbf{q},\, \mathbf{H},\, \mathbf{H};\, \text{mask}) \in \mathbb{R}^{B \times 1 \times d}$$

**为什么用 attention pooling 而不是 mean-pool**：token 个数恰好等于字符数，mean-pool 的输出统计量会随
token 个数变化，理论上更容易让模型靠"平均了多少个 token"反推句子长度（**计数泄漏**）；可学习 query 的
加权汇总表达力更强，故本文默认采用 attention pooling。

但这一点**必须用实验说话，不能只凭直觉**（协议见 2.11，结果见池化消融）：

- **表示层面：成立。** 长度探针显示，mean-pool 的句嵌入比 attention pooling **携带更多可线性读出的
  长度信息**（ridge 探针 val $R^2 = 0.408$ vs $0.151$，Spearman $\rho = 0.667$ vs $0.431$，
  MAE $1.080$ vs $1.309$）。
- **任务层面：不成立。** 这一差异**没有**转化为检索优势：两种池化在 900 条 val 上的 `eeg2text` /
  `text2eeg` 指标统计上不可区分（差异全落在"单条翻转 $=0.11\%$"的噪声量级内）；按 `val_loss` 选出的
  best 也几乎重合（attention $6.350$ vs mean $6.355$，差 $0.005$）。**因此不能断言 attention pooling
  在端到端性能上更优。**
- **上界：仅靠长度不足以致胜。** 把同一 checkpoint 的输入 token 换成独立同分布噪声（保留 token 个数）后，
  长度仍被高度线性编码（attn 探针 $R^2 = 0.404$、mean $R^2 = 0.492$，均高于真实输入），但其检索指标
  精确停在随机基线（`eeg2text top-5` $= 0.024$ vs 基线 $0.022$）—— 说明即使长度被最大程度地编码，
  也不足以抬高检索成绩。

综上，本文将 attention pooling 定位为一种**更少保留长度冗余**的设计选择，而**不是性能优势的来源**。

最后经线性投影到共享嵌入维度 $D$ 并做 L2 归一化：

$$\mathbf{e} = \frac{W_{\text{out}}\,\mathbf{z}^{\text{pool}}}{\lVert W_{\text{out}}\,\mathbf{z}^{\text{pool}} \rVert_2}
\in \mathbb{R}^{D}, \qquad D = 768 .$$

本配置下编码器总参数量为 **723,680**（实测）。

---

## 2.5 文本编码器与共享嵌入空间

文本侧使用**冻结**的 **LaBSE**（`sentence-transformers/LaBSE`，`[待补引用]`）：

$$\mathbf{t} = \frac{\mathrm{LaBSE}(s)}{\lVert \mathrm{LaBSE}(s) \rVert_2} \in \mathbb{R}^{768}.$$

- 参数全程冻结（`requires_grad=False`），仅前向，不参与梯度更新；
- 所有**唯一句子**的文本嵌入在训练前**一次性预计算**为查表 $\{s \mapsto \mathbf{t}_s\}$，训练时仅做索引，
  不重复前向 LaBSE。

两侧输出均为单位向量，因此**点积即余弦相似度**：$\mathrm{sim}(\mathbf{e}, \mathbf{t}) = \mathbf{e}^\top \mathbf{t}$。

---

## 2.6 对比目标：多正样本对称 InfoNCE

给定一个 batch 的 EEG 嵌入 $\mathbf{E} \in \mathbb{R}^{B \times D}$、文本嵌入 $\mathbf{T} \in \mathbb{R}^{B \times D}$
与温度 $\tau$：

$$\mathbf{S} = \frac{\mathbf{E}\,\mathbf{T}^\top}{\tau} \in \mathbb{R}^{B \times B}.$$

正样本由**句子 ID** 决定，而非对角线：令 $g(i)$ 为第 $i$ 条样本的句子组号，

$$P_{ij} = \mathbb{1}\!\left[\,g(i) = g(j)\,\right].$$

定义多正样本交叉熵（MP-CE）：

$$\mathrm{MP\text{-}CE}(\mathbf{S}, P) = -\,\frac{1}{B}\sum_{i=1}^{B}
  \frac{1}{\sum_j P_{ij}} \sum_{j=1}^{B} P_{ij}\,
  \Big( S_{ij} - \log \sum_{k=1}^{B} \exp S_{ik} \Big),$$

最终损失为两个方向的对称平均：

$$\mathcal{L} = \tfrac{1}{2}\Big[\mathrm{MP\text{-}CE}(\mathbf{S}, P) + \mathrm{MP\text{-}CE}(\mathbf{S}^\top, P)\Big].$$

由于 $P$ 对称，两个方向共用同一套正样本掩码。

**性质**：

- 当一批中每个句子只出现一次时，$P$ 退化为单位阵，$\mathcal{L}$ 等价于标准 CLIP 式单正样本 InfoNCE；
- 当一批中同一句出现多次时（跨天多视图），同句的所有 trial **互为正样本**，不会被当作负样本互相推开
  （即避免**假负样本**）。

温度系数取 $\tau = 0.2$（`[待补引用/调参依据]`）。注意：$\tau$ 的调参须在负样本数量足够的前提下进行，
否则结果无参考价值。

---

## 2.7 多视图分组采样与数据划分

**按句子切分 train/val（8:2）**：以**句子**而非 trial 为切分单位，保证同一句子的所有 trial 只落在同一侧，
`sub-01` day 1–4 下 train/val 的句子集合**完全不相交**（重叠 = 0）。切分结果：
train 899 句 / 3596 条 trial，val 225 句 / 900 条 trial。

**分组 batch 采样器**（`SentenceGroupBatchSampler`）：每个 batch 抽取 `n_sentences = 32` 个**不同句子**，
每句再抽 `n_repeats = 4` 条 trial，故

$$B = n_{\text{sentences}} \times n_{\text{repeats}} = 128 .$$

动机：句子只在**跨天**之间重复。若改用随机 shuffle，同一句的多个视图几乎不可能落进同一个 batch
（实测 day 1–4 下每 anchor 的平均正样本数仅 1.03），多正样本机制等于空转。分组采样把它从"碰运气"
变成**确定性保证**：`pos/anch` 恒等于 `n_repeats = 4`。句子池按 `seed + epoch` 每轮打乱，保证可复现。
`n_repeats = 1` 可退化为随机 shuffle 的效果，用作消融对照。

---

## 2.8 训练配置

| 项目 | 设置 |
|---|---|
| 优化器 | AdamW |
| 学习率 | $1\times10^{-3}$ |
| 权重衰减 | $1\times10^{-3}$ |
| 学习率调度 | 按 epoch 余弦退火至 0 |
| batch 大小 | 128（32 句 × 4 视图） |
| 训练轮数 | 30 |
| 温度 $\tau$ | 0.2 |
| 随机种子 | 0 |
| 模型选择 | 按 `val_loss` 取最优 checkpoint |
| 硬件 | CUDA GPU（`device` 自动选择，无 GPU 时回退 CPU） |

**多被试协议**：对每个被试独立重复上述流程（本次为 `sub-01`、`sub-02`），报告各被试指标与随机基线；
跨天（留一天）泛化见 2.12，跨被试零样本迁移见 2.13。

**主结果**（day 1–4，`inner`，30 epoch；随机基线：`e2t@1`$=1/225=0.0044$、`e2t@5`$=5/225=0.0222$、
`t2e@1`$=4/900=0.0044$、`t2e@5`$=20/900=0.0222$、`val_loss`$=\ln 900=6.802$；指标取 `val_loss`
最优轮，括号为训练期峰值）：

| subject | best epoch | val_loss | e2t@1 | e2t@5 | t2e@1 | t2e@5 |
|---|---|---|---|---|---|---|
| sub-01 | 11 | **6.350** | 0.0111 | 0.0822（峰值 0.0933） | 0.0311 | 0.0978（峰值 0.1467） |
| sub-02 | 10 | **6.284** | 0.0144 | 0.0900（峰值 0.1156） | 0.0133 | 0.0800（峰值 0.1200） |

两被试的 `eeg2text top-5` 约为随机基线的 $3.7$–$4.0$ 倍、`text2eeg top-5` 约 $3.6$–$4.4$ 倍
（按峰值计更高），`val_loss` 均显著低于 $6.802$；但 epoch 12 之后 `val_loss` 开始回升、检索指标仍在
缓慢爬升，说明 30 epoch 仍未完全收敛。

---

## 2.9 评测协议

两个检索方向的**候选池不同**，这是有意为之：

- **EEG → Text（`eeg2text`）**：候选池 = **去重后的唯一句子**（val 上 225 句）。同一句的多个 trial 共享
  同一个文本向量，若不去重，同一句会占掉 4 个连续名次，`top-5` 实际只等于 top-2。
- **Text → EEG（`text2eeg`）**：候选池 = **全部 val EEG 样本**（900 条，彼此互不相同），命中判定为
  "前 $k$ 名中出现**任意同句**候选"。

指标：两方向的 `top-1` / `top-5`。

**随机基线**（务必随表报告）：

| 指标 | 随机基线 |
|---|---|
| `eeg2text top-1` | $1/225 = 0.0044$ |
| `eeg2text top-5` | $5/225 = 0.0222$ |
| `text2eeg top-1` | $4/900 = 0.0044$ |
| `text2eeg top-5` | $\min(5\times4/900,\,1) = 0.0222$ |

`val_loss` 在全 val 集上以同一分组 InfoNCE 计算；均匀分布下的理论值为 $\ln 900 = 6.802$。

> 注：$n_{\text{val}} = 900$ 时单条翻转即造成 $1/900 \approx 0.11\%$ 的指标波动，
> `eeg2text_top1` 噪声较大，故模型选择默认按 `val_loss` 而非 top-1。

---

## 2.10 随机噪声对照协议

**目的**：验证上述成绩来自**真实脑电信号**，而非与信号无关的旁路信息（架构、句子长度 / token 个数、
数据划分）。

**设计**：**只改变一个变量 —— EEG 数值的来源**，其余全部保持一致（token 形状、$n_{\text{tok}}$、
句子标签、train/val 划分、采样器、损失、评测口径、模型、文本侧）。把 token 的值替换为独立同分布随机噪声：

$$\tilde{\mathbf{x}}_i \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}) \in \mathbb{R}^{n_{\text{tok}} \times C \times T},
\qquad \sigma = \texttt{noise-std} = 1.0 .$$

**噪声按样本索引独立取种**：

$$\text{seed}_i = s_{\text{noise}} \cdot 10^{6} + i \;(+\; \text{epoch} \cdot 7919),$$

因此同一句跨天的多视图拿到的是**互不相关**的噪声（实测同句两视图噪声相关系数 $\approx -0.0012$），
多正样本之间不存在任何可学习的共同信号 —— 否则会人为制造出"假对齐"。同时，取种依赖索引而非全局 RNG
顺序，结果与 DataLoader 遍历顺序 / worker 数无关。

**两种刷新模式**：

- `--noise-refresh epoch`（默认）：种子含 `epoch`，**每个 epoch 重新采样**噪声。噪声携带**零可学习信息**，
  连"记住训练噪声"这条路也堵死，故 $train\_loss$ 必须停留在均匀分布基线 $\ln 128 = 4.852$。
- `--noise-refresh never`：噪声固定，模型**有能力**在训练集上把"噪声→句子"背下来（$train\_loss$ 可下降），
  但 val 上是未见过的噪声，指标仍应停在随机基线。用于说明**低训练损失不等于学到真实信号**。

**判读**：噪声组指标若落在随机基线附近（理想情形：$train\_loss = \ln B$、$val\_loss = \ln 900$、
top-$k$ = $1/N$），则支持"成绩来自真实信号"；若仍显著高于随机，则说明存在泄漏通道，需回头修正设计。

**实测**（`sub-01`，day 1–4，30 epoch，$B{=}128$，`--noise-refresh epoch`）：

| 指标 | 随机基线 | EEG 组 | 噪声组 |
|---|---|---|---|
| `train_loss` | $4.852=\ln 128$ | $3.45$（跌破基线） | $4.852$–$4.857$（$=$基线） |
| `val_loss` | $6.802=\ln 900$ | $6.350$ | $6.802$（$=$基线） |
| `eeg2text top-1` | $0.0044$ | $0.0111$（峰值 $0.0244$） | $0.0044$（恒定） |
| `eeg2text top-5` | $0.0222$ | $0.0822$（峰值 $0.0933$） | $0.0222$（恒定） |
| `text2eeg top-5` | $0.0222$ | $0.0978$（峰值 $0.1467$） | $0.0044$–$0.0533$（基线附近） |

噪声组的 $train\_loss$、$val\_loss$ 与 `eeg2text top-1/top-5` 全部精确钉在理论基线上（模型对噪声
零学习、输出近乎塌缩为常向量），而 EEG 组显著优于随机 —— 支持上述判读。

---

## 2.11 长度泄漏诊断协议（长度探针）

端到端检索指标在 900 条 val 上噪声过大（单条翻转 $=0.11\%$），不足以判定"计数泄漏"。因此直接检验
**句嵌入是否线性编码了 token 个数**：

> 注意 $n_{\text{tok}}$ 对模型而言**始终可得**（mask 每行求和即为它）。本协议问的不是"模型能否拿到长度"，
> 而是"池化之后的**句嵌入**是否还保留了长度"。

协议（实现见 `char_token_pooling/probe.py`）：

1. **长度探针**：冻住编码器，取 train / val 的句嵌入 $\mathbf{e}\in\mathbb{R}^{768}$，在 **train** 上拟合
   ridge 回归 $\mathbf{e}\mapsto n_{\text{tok}}$（特征标准化、目标中心化，闭式解），在 **val** 上报告
   $R^2$、MAE、Spearman $\rho$。以"打乱 train 标签后重拟合"为零假设对照（应 $\le 0$），
   以"直接预测 train 均值"为平凡下界（$R^2 = 0$）。
2. **按长度分层的检索**：口径与 2.9 的 `eeg2text` 完全一致，按 $n_{\text{tok}}$ 分桶报告 top-1 / top-5。
   若模型靠长度取巧，性能应随长度明显分档。
3. **噪声输入对照**：把同一 checkpoint 的输入 token 换成独立同分布随机噪声（保留 $n_{\text{tok}}$），
   其 $R^2$ 给出"不含信号内容、仅凭 token 结构"能被线性读出的长度上界。

**实测**（`sub-01`，day 1–4，ridge $\alpha=1$）：

| 输入 · 池化 | val $R^2$ | val MAE | Spearman $\rho$ | 零假设 $R^2$ |
|---|---|---|---|---|
| attn · 真实 EEG | $0.151$ | $1.309$ | $0.431$ | $-0.018$ |
| mean · 真实 EEG | $0.408$ | $1.080$ | $0.667$ | $-0.028$ |
| attn · 随机噪声输入 | $0.404$ | $1.075$ | $0.642$ | $-0.010$ |
| mean · 随机噪声输入 | $0.492$ | $0.987$ | $0.711$ | $+0.015$ |

结论：mean-pool 的句嵌入比 attention pooling **保留了更多可线性读出的长度**（$R^2$ 约高 $2.7\times$）；
噪声输入下两者 $R^2$ 进一步升高（$0.40$–$0.49$），说明这部分长度信息主要来自 token 的**个数/结构**
而非信号内容。但无论真实还是噪声输入，检索指标在噪声条件下都停在随机基线（`eeg2text top-5`
$=0.024$ vs 基线 $0.022$），即**长度可读 ≠ 检索制胜**。

---

## 2.12 跨天（留一天）泛化

**目的**：主实验（2.7）按**句子**随机切分，val 是**新句子**但来自**同一批天**；本节反过来把
**整天**划到 val，检验"同一句在未见过的某一天录到的信号"能否被识别。

**协议**（实现见 `char_token_day_holdout/`）：在 day 1–4 上做 leave-one-day-out 四折，每折
3 天 train / 1 天 val；模型、损失、采样、评测口径与主实验一致；每折 `n_sentences=32`、
`n_repeats=3`（$B{=}96$）、30 epoch 上限、seed 0，按 `val_loss` 选 best。

**关键前提**：para1 每天录的是**同一批 1124 句**，故 train / val 的句子集合**完全重合**——这里量化的
是**跨天稳定性**，与 2.7 的"新句子泛化"不可直接比较；每折 val 的候选池是 1124 句。

**结果**（`sub-01`，每折 $n_{\text{val}}=1124$；随机基线 `e2t@1`$=1/1124=0.00089$、
`e2t@5`$=5/1124=0.00445$、`val_loss` 基线 $=\ln 1124 = 7.025$）：

| val day | val_loss | e2t@1 | e2t@5 | t2e@1 | t2e@5 |
|---|---|---|---|---|---|
| day-01 | 7.030 | 0.0009 | 0.0053 | 0.0018 | 0.0053 |
| day-02 | 7.065 | 0.0000 | 0.0027 | 0.0000 | 0.0009 |
| day-03 | 7.038 | 0.0009 | 0.0044 | 0.0000 | 0.0098 |
| day-04 | 7.027 | 0.0009 | 0.0027 | 0.0009 | 0.0027 |
| **mean ± std** | **7.040 ± 0.017** | **0.0007 ± 0.0004** | **0.0038 ± 0.0013** | **0.0007 ± 0.0009** | **0.0047 ± 0.0039** |

**判读**：四折的 `val_loss`（$\approx 7.04$）与 `e2t@5`（均值 $0.0038$）都落在随机基线
（$7.025$ / $0.00445$）附近，甚至略低于随机——**没有出现跨天泛化**。即模型学到的是**句子特异**
的表征，不能在 session 之间对同一句的脑电形成稳定对齐。这与 2.13 的跨被试结果共同指向：当前方案的
泛化域仍局限在"训练时见过的那几天"。

---

## 2.13 跨被试零样本迁移

**目的**：检验在 A 被试上训练的编码器能否**零样本**识别 B 被试的脑电。

**协议**（实现见 `char_token_cross_subject/predict.py`）：`--source` 提供 A 的 best checkpoint，
`--target` 提供 B 的数据，两者按位置配对；文本候选池是 target 的句子（被试间共享同一批句子），
损失与检索口径复用 `char_token.train.evaluate`。报告四个方向：两个同被试（工程上界参照）与两个
跨被试。

**结果**（全员 day 1–4，$n_{\text{val}}=4496$，候选 1124 句，每 anchor $4$ 个同句正样本；
随机基线 `e2t@1`$=1/1124=0.00089$、`e2t@5`$=5/1124=0.00445$、`val_loss` 基线 $=\ln 4496 = 8.411$）：

| 方向 | val_loss | e2t@1 | e2t@5 | t2e@1 | t2e@5 |
|---|---|---|---|---|---|
| sub-01 → sub-01 | 7.542 | 0.0162 | 0.0707 | 0.0596 | 0.1797 |
| sub-02 → sub-02 | 7.592 | 0.0122 | 0.0532 | 0.0383 | 0.1121 |
| sub-01 → sub-02 | **8.563** | 0.0013 | 0.0058 | 0.0009 | 0.0053 |
| sub-02 → sub-01 | **8.568** | 0.0009 | 0.0047 | 0.0009 | 0.0036 |

> 同被试两行在**全数据集**（含其自身训练 trial）上评测，故高于 2.9 中 225 句 val 上的指标，仅作
> 工程上界参照；跨被试两行才是真正的零样本评测。

**判读**：跨被试两方向的 `val_loss`（$8.56$）**高于**随机基线 $8.411$，`e2t@5` / `t2e@5` 全部落在
随机基线（$0.00445$）附近——**当前编码器无法零样本跨被试迁移**。结合 2.12，泛化域同时受限于
个体与 session。

---

## 2.14 复现信息

| 组件 | 版本 |
|---|---|
| Python 环境 | 工程内 `.venv`（`uv` 管理） |
| torch | 2.14.1（CUDA 运行时；无 GPU 自动回退 CPU） |
| mne | 1.13.2 |
| numpy / pandas / scipy | 2.5.3 / 3.0.6 / 1.18.1 |
| sentence-transformers / transformers | 6.1.0 / 5.19.0 |
| pyprep / mne-icalabel / autoreject | 0.9.0 / 0.10.0 / 0.5.1 |

代码：`char_token/`（数据集 `dataset.py`、模型 `model.py`、采样 `sampler.py`、文本编码 `text_encoder.py`、
训练 `train.py`）、`char_token_noise/`（噪声对照）、`char_token_pooling/`（池化消融 `model.py`/`train.py`、
长度探针 `probe.py`）、`char_token_day_holdout/`（跨天留一 `split.py`/`train.py`/`four_fold.py`）、
`char_token_cross_subject/`（跨被试零样本 `predict.py`）、预处理 `script/preprocessing.py`、文本汇总
`script/summarize_text.py`。

---

## 附 A · 符号表

| 符号 | 含义 |
|---|---|
| $f_s$ | 采样率，250 Hz |
| $d_{\text{char}}$ | 每字时长，0.4 s |
| $T$ | 每 token 采样点数，$=100$ |
| $C$ | EEG 通道数，$=122$ |
| $n_{\text{char}}$ | 句子非标点字符数 |
| $n_{\text{tok}}$ | token 个数 |
| $n$ / $B$ | 句内 token 数 / batch 大小 |
| $d$ / $D$ | token 空间宽度（128）/ 共享嵌入维度（768） |
| $\tau$ | 对比学习温度，0.2 |
| $\mathbf{e},\mathbf{t}$ | EEG / 文本单位嵌入 |

## 附 B · 五维自审清单

| 维度 | 自评 |
|---|---|
| 贡献明确 | ✅ 字符级等长 token 化 + 抗泄漏评测协议 |
| 表述清晰 | ✅ 每小节先给形状/公式再给理由，符号表统一 |
| 实验强度 | ⚠️ 2 被试、30 epoch、单种子；val_loss 在 epoch 12 后回升，建议训到收敛 + 多种子 |
| 评估完整 | ⚠️ 已完成噪声对照、池化消融、长度探针、跨天留一、跨被试零样本；仍缺 baseline、其余消融（sampler / token 时长 / GroupNorm） |
| 方法合理 | ✅ 各设计选择均给出理由；池化一项**已按消融结果修正为"减少长度冗余"而非"性能优势"** |

## 附 C · Claim–Evidence 映射（Method 节相关）

| Claim | Evidence | Status |
|---|---|---|
| 字符化 token 化能无损处理变长 EEG | 8528 条试次 100% 可被 100 整除，无丢弃 | supported |
| GroupNorm 避免 padding 污染 | 代码实现 + 待补对照实验 | needs evidence |
| mean-pool 的句嵌入比 attention pooling 保留更多可线性读出的长度 | 长度探针：$R^2$ 0.408 vs 0.151，$\rho$ 0.667 vs 0.431，零假设 $\le 0$ | supported |
| attention pooling 在端到端检索上**优于** mean-pool | 同配置消融：指标不可区分（best `val_loss` 6.350 vs 6.355） | **refuted** |
| 仅靠长度编码不足以抬高检索成绩 | 噪声输入探针 $R^2 \approx 0.40$–$0.49$，检索恰为随机基线（0.024 vs 0.022） | supported |
| 多正样本对称 InfoNCE 避免假负样本 | 分组采样下 `pos/anch = 4.00` | supported（需消融） |
| 噪声对照可排除长度/架构旁路 | 噪声组 $train\_loss=\ln 128$、$val\_loss=\ln 900$、top-k$=1/N$ | supported |
| 主结果高于随机（2 被试） | `e2t@5` 0.082/0.090、`t2e@5` 0.098/0.080，约 $3.6$–$4.4\times$ 随机基线 | supported |
| 当前编码器无法跨天泛化（同句、留一天） | 四折 `val_loss` 7.040 vs 基线 7.025，`e2t@5` 0.0038 vs 0.0045 | **refuted**（负结果） |
| 当前编码器无法零样本跨被试迁移 | `val_loss` 8.56 > 基线 8.411，`e2t@5`/`t2e@5` 落在随机基线 | **refuted**（负结果） |