r"""长度探针：直接检验句嵌入是否线性编码了 token 个数（= 句子长度）。

背景
----
池化消融的端到端指标噪声太大（900 条 val 上单条翻转 = 0.11%），attn 与 mean 无法区分。
但"计数泄漏"这个假设本身可以用更**直接**的方式检验：模型池化后的句嵌入 $\mathbf{e}$ 里，
还能不能线性读出 token 个数 $n_{\\text{tok}}$？能读出 -> 长度信息被保留、存在被利用的可能；
读出不了 -> 该泄漏通道在表示层面就已经被截断。

注意：$n_{\\text{tok}}$ 对模型而言**始终可得**（mask 的第 i 行求和就是它）。本探针问的不是
"模型能否拿到长度"，而是"池化之后的**句嵌入**是否还保留了长度"。

两项分析
--------
1. **长度探针（ridge 回归）**：在 **train** 上拟合 $\mathbf{e} \\mapsto n_{\\text{tok}}$，在 **val**
   上评测 R² / MAE / Spearman ρ；并以"打乱 train 标签后重拟合"作为零假设对照。
   `n_tokens` 数值范围很窄（5~15），R² 才是主指标；同时报"预测均值"这一平凡下界（R²=0）。
2. **按长度分层的检索**：把 val 的 `eeg2text` top-1/top-5 按 $n_{\\text{tok}}$ 分桶统计。
   若模型靠长度取巧，性能会随长度分布明显分档。

用法::

    # 池化消融的两个 checkpoint（真实 EEG 输入）
    uv run python char_token_pooling/probe.py --pooling attn
    uv run python char_token_pooling/probe.py --pooling mean

    # 噪声对照：噪声模型的嵌入里是否还能读出长度（应为读不出）
    uv run python char_token_pooling/probe.py --pooling attn --input noise \\
        --ckpt outputs/char_token_noise/<时间戳>/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_token.dataset import CharTokenEEGDataset, char_token_collate
from char_token.text_encoder import LaBSETextEncoder
from char_token.train import build_text_lookup, gather_text, set_seed, split_by_sentence
from char_token_pooling.model import POOLING_MODES, build_encoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOLING_OUT_DIR = PROJECT_ROOT / "outputs" / "char_token_pooling"


# ---------------------------------------------------------------- 基础工具


def resolve_ckpt(pooling: str, ckpt: str | None) -> Path:
    """给定 pooling 时，自动选 ``outputs/char_token_pooling/<pooling>_*/best.pt`` 里最新的一个。"""
    if ckpt:
        path = Path(ckpt)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        return path

    candidates = sorted(
        POOLING_OUT_DIR.glob(f"{pooling}_*/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {POOLING_OUT_DIR} 下找不到 {pooling}_*/best.pt，请用 --ckpt 显式指定"
        )
    return candidates[-1]


def _ranks(x: torch.Tensor) -> torch.Tensor:
    """返回升序排名（0 起）。用两次 argsort 得到名次，等价于 scipy 的 rankdata(method='ordinal')。"""
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
    return ranks


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm() + 1e-12))


def ridge_fit(X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """闭式解 ridge：w = (XᵀX + αI)⁻¹ Xᵀy（X 已标准化、y 已中心化）。"""
    d = X.size(1)
    eye = torch.eye(d, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(X.T @ X + alpha * eye, X.T @ y)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------- 数据与编码


@torch.no_grad()
def encode_all(model, dataset, indices, device, batch_size, collate) -> torch.Tensor:
    """对给定索引集合逐 batch 前向，返回 (N, D) 的 L2 归一化句嵌入。"""
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    parts = []
    model.eval()
    for tokens, mask, _ in loader:
        parts.append(model(tokens.to(device), mask.to(device)))
    return torch.cat(parts, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="长度探针：句嵌入能否线性读出 token 个数。")
    parser.add_argument("--pooling", default="attn", choices=list(POOLING_MODES), help="用哪种池化的架构建模型")
    parser.add_argument("--ckpt", default=None, help="checkpoint 路径；默认取 pooling 目录下最新的 best.pt")
    parser.add_argument(
        "--input", default="real", choices=["real", "noise"],
        help="喂给模型的是真实 EEG 还是随机噪声（噪声对照组用）",
    )
    parser.add_argument("--subject", default="sub-01", help="被试，如 sub-01")
    parser.add_argument("--day", type=int, nargs="+", default=[1, 2, 3, 4], help="day 编号，默认 1 2 3 4")
    parser.add_argument("--event", default="inner", choices=["inner", "reading"], help="inner（默认）或 reading")
    parser.add_argument("--sample", type=int, default=250, help="采样率 Hz（默认 250）")
    parser.add_argument("--char-duration", type=float, default=0.4, help="每个字 token 的时长（秒）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集句子比例（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="切分与噪声种子（默认 0，需与训练一致）")
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge 正则强度（默认 1.0）")
    parser.add_argument("--batch-size", type=int, default=64, help="编码时的 batch 大小（默认 64）")
    parser.add_argument("--noise-seed", type=int, default=0, help="噪声种子（--input noise 时生效，默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = resolve_ckpt(args.pooling, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[Ckpt] {ckpt_path}")
    print(
        f"       训练时 pooling={ckpt.get('pooling', 'attn')} epoch={ckpt.get('epoch')} "
        f"n_chans={ckpt['n_chans']} token_samples={ckpt['token_samples']} "
        f"embedding_dim={ckpt['embedding_dim']}"
    )

    # ---------------- 数据：与训练完全一致的句子级切分 ----------------
    common = dict(
        event=args.event, sample=args.sample, subject=args.subject, days=args.day,
        char_duration=args.char_duration,
    )
    if args.input == "noise":
        from char_token_noise.dataset import NoiseTokenEEGDataset
        dataset = NoiseTokenEEGDataset(
            noise_mode="gaussian", noise_std=1.0, noise_seed=args.noise_seed,
            refresh="never", **common,
        )
    else:
        dataset = CharTokenEEGDataset(**common)
        dataset[0]  # 触发一次读取，确定 n_chans

    if dataset.n_chans != ckpt["n_chans"]:
        raise ValueError(f"数据集通道数 {dataset.n_chans} 与 checkpoint 的 {ckpt['n_chans']} 不一致")

    train_set, val_set, info = split_by_sentence(dataset, args.val_ratio, args.seed)
    print(
        f"[Data] input={args.input} {args.subject} day-{sorted(args.day)} event={args.event} "
        f"n_chans={dataset.n_chans} token_samples={dataset.token_samples}\n"
        f"       train {info['n_unique_train']} 句/{len(train_set)} trial，"
        f"val {info['n_unique_val']} 句/{len(val_set)} trial，句子重叠={info['sentence_overlap']}"
    )

    # ---------------- 模型（按 pooling 建架构 + 载权重） ----------------
    model = build_encoder(
        args.pooling,
        n_chans=dataset.n_chans,
        token_samples=dataset.token_samples,
        embedding_dim=ckpt["embedding_dim"],
    ).to(device)
    missing = model.load_state_dict(ckpt["model"], strict=True)
    print(f"[Model] pooling={args.pooling} 载入 {missing}")

    # ---------------- 编码 train / val ----------------
    collate = char_token_collate
    E_train = encode_all(model, dataset, list(train_set.indices), device, args.batch_size, collate)
    E_val = encode_all(model, dataset, list(val_set.indices), device, args.batch_size, collate)

    # 目标：token 个数（= 句子非标点字数；inner 段 +1）
    n_tok = dataset.meta["n_tokens"].to_numpy()
    y_train = torch.tensor(n_tok[train_set.indices], dtype=torch.float64, device=device)
    y_val = torch.tensor(n_tok[val_set.indices], dtype=torch.float64, device=device)

    print(
        f"[Enc] E_train={tuple(E_train.shape)} E_val={tuple(E_val.shape)}；"
        f"n_tok: train [{y_train.min():.0f},{y_train.max():.0f}] mean={y_train.mean():.3f}，"
        f"val [{y_val.min():.0f},{y_val.max():.0f}] mean={y_val.mean():.3f}"
    )

    # ================= 分析 1：长度探针（ridge 回归） =================
    Xtr = E_train.double()
    Xva = E_val.double()
    feat_mean, feat_std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-8)
    Xtr_s = (Xtr - feat_mean) / feat_std
    Xva_s = (Xva - feat_mean) / feat_std
    y_mean = y_train.mean()
    ytr_c = y_train - y_mean

    w = ridge_fit(Xtr_s, ytr_c, args.alpha)
    pred = Xva_s @ w + y_mean
    r2 = r2_score(y_val, pred)
    mae = float((y_val - pred).abs().mean())
    rho = _spearman(y_val, pred)

    # 零假设对照：打乱 train 标签后重拟合（应读不出任何信息，R²≈0 或为负）
    gen = torch.Generator().manual_seed(args.seed)
    ytr_shuf = ytr_c[torch.randperm(ytr_c.numel(), generator=gen, device=device)]
    w_null = ridge_fit(Xtr_s, ytr_shuf, args.alpha)
    r2_null = r2_score(y_val, Xva_s @ w_null + y_mean)

    print(f"\n{'=' * 76}\n[分析 1] 长度探针：e -> n_tok（ridge, alpha={args.alpha}）\n{'=' * 76}")
    print(f"  val R²            = {r2:+.4f}   （R²=0 即退化到只预测均值）")
    print(f"  val MAE           = {mae:.4f}   个 token")
    print(f"  val Spearman ρ    = {rho:+.4f}")
    print(f"  零假设对照 R²     = {r2_null:+.4f}   （打乱 train 标签，应 ≈0）")
    print(f"  平凡下界          = R² 0.0000（直接预测 train 均值 {y_mean:.3f}）")

    # ================= 分析 2：按长度分层的检索 =================
    # 口径与 char_token.train.evaluate 完全一致：候选池 = 去重后的唯一句子，用冻结 LaBSE 的
    # 文本嵌入排名，命中 = 前 k 名里出现该句。这样分层结果可直接与主实验指标对照。
    val_sentences = [dataset.meta["sentence"].tolist()[i] for i in val_set.indices]
    unique_sentences = sorted(set(val_sentences))
    sentence_to_col = {s: i for i, s in enumerate(unique_sentences)}
    target = torch.tensor([sentence_to_col[s] for s in val_sentences], device=device)

    text_encoder = LaBSETextEncoder()
    lookup_all, _ = build_text_lookup(text_encoder, dataset.meta["sentence"].tolist(), device)
    text_unique = gather_text(unique_sentences, lookup_all, device)

    ranked = (E_val @ text_unique.t()).argsort(dim=1, descending=True)
    hit = ranked.eq(target[:, None])
    n_cand = len(unique_sentences)
    n_tok_val = n_tok[val_set.indices].tolist()

    print(f"\n{'=' * 76}\n[分析 2] 按 n_tok 分层的 eeg2text 检索（口径同训练评测）\n{'=' * 76}")
    print(
        f"  全 val：top-1={float(hit[:, :1].any(1).float().mean()):.4f}  "
        f"top-5={float(hit[:, :5].any(1).float().mean()):.4f}"
    )
    print(f"  {'n_tok':>6} {'n_trial':>8} {'top1':>8} {'top5':>8}")
    for value in sorted(set(n_tok_val)):
        rows = torch.tensor([v == value for v in n_tok_val], device=device)
        n_rows = int(rows.sum())
        top1 = float(hit[rows][:, :1].any(dim=1).float().mean())
        top5 = float(hit[rows][:, :5].any(dim=1).float().mean())
        print(f"  {value:>6d} {n_rows:>8d} {top1:>8.4f} {top5:>8.4f}")

    print(
        f"\n  随机基线：top-1={1 / n_cand:.4f}  top-5={min(5, n_cand) / n_cand:.4f}"
        f"（候选 {n_cand} 句）"
    )
    print(
        "  读法：若模型靠长度取巧，各 n_tok 分桶的性能会随长度明显分档（长/短句更易命中）；"
        "若各桶贴近同一水平，则说明提升来自信号而非长度。"
    )


if __name__ == "__main__":
    main()