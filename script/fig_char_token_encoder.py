"""生成 CharTokenEncoder 结构示意图（论文用矢量图）。

运行::

    .venv\\Scripts\\python.exe script/fig_char_token_encoder.py

输出::

    paper/figs/fig_char_token_encoder.pdf
    paper/figs/fig_char_token_encoder.svg

图中数值全部取自 ``char_token/model.py`` 的默认配置，均为代码中的确定量::

    C = 122, T = 100, k = 32, m1 = 9, m2 = 13, s = 2, d_model = 128
    T1 = T - m1 + 1 = 92
    T2 = floor((T1 - m2) / s) + 1 = 40
    flatten = k * T2 = 1280

排版约定：中文用宋体、阶段名用黑体，数学符号用 STIX；配色为灰度，
保证黑白打印下仍可区分，唯一强调的环节（空间卷积）用深填充 + 粗边框标记。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# ---------------- 全局排版 ----------------
plt.rcParams.update(
    {
        "font.family": "SimSun",  # 中文正文用宋体
        "font.sans-serif": ["SimSun", "SimHei", "Microsoft YaHei"],
        "font.size": 8,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",  # 数学符号用 STIX，与宋体衬线协调
        "pdf.fonttype": 42,  # TrueType 嵌入，PDF 内文字可编辑
        "svg.fonttype": "path",  # SVG 文字转路径，任何环境都不会缺字体
    }
)

INK = "#000000"
EDGE = "#333333"
GRID = "#B4B4B4"
FILL = "#FFFFFF"
FILL_FOCUS = "#D8D8D8"

W, H = 720, 300  # 画布逻辑坐标
FIG_W = 6.3  # 目标宽度 (inch)，通栏约 16 cm
FIG_H = FIG_W * H / W

FS = 8  # 统一字号 (pt)


def _grid(ax, x0, y0, x1, y1, nv, nh):
    """在矩形内画淡灰网格，暗示张量是二维矩阵。

    注意 zorder=3：网格必须画在矩形填充（zorder=2）之上，否则会被白色填充盖住。
    """
    for i in range(1, nv):
        x = x0 + (x1 - x0) * i / nv
        ax.plot([x, x], [y0, y1], color=GRID, lw=0.6, zorder=3)
    for j in range(1, nh):
        y = y0 + (y1 - y0) * j / nh
        ax.plot([x0, x1], [y, y], color=GRID, lw=0.6, zorder=3)


def _tensor(ax, x, y, w, h, *, focus=False, nv=4, nh=5, inset=6):
    """画一个表示张量的矩形。focus=True 用深填充 + 粗边框标记重点环节。"""
    face = FILL_FOCUS if focus else FILL
    lw = 1.4 if focus else 1.0
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=face, edgecolor=EDGE, linewidth=lw, zorder=2)
    )
    if h < 20:  # 极扁的矩形（时间维保留、电极维压成 1）留白要小
        inset = 1.5
    _grid(ax, x + inset, y + inset, x + w - inset, y + h - inset, nv, nh)


def _label(ax, x, y, text, *, size=FS, color=INK, family=None):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        fontfamily=family,
        zorder=3,
    )


def _arrow(ax, x0, x1, y, *, focal=False):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=1.6 if focal else 1.0,
            color=EDGE,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def build_figure():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    gy0, gh, cy = 140, 130, 205  # 张量所在的竖直带与中心线

    # ---- 四个张量：宽度按时间维比例，高度按电极维表示 ----
    _tensor(ax, 50, gy0, 112, gh, nv=4, nh=5)  # 输入 (N, C, T)
    _tensor(ax, 268, gy0, 45, gh, nv=3, nh=5)  # 时间卷积 + 池化后 (N, k, C, T2)
    _tensor(ax, 453, cy - 5, 45, 10, focus=True, nv=6, nh=0)  # 空间卷积后 (N, k, 1, T2)
    _tensor(ax, 586, cy - 14, 88, 28, nv=5, nh=0)  # 输出向量 (N, d_model)

    # ---- 箭头 ----
    _arrow(ax, 162, 268, cy)  # 时间卷积 + 平均池化
    _arrow(ax, 313, 453, cy, focal=True)  # 空间卷积（本图重点）
    _arrow(ax, 498, 586, cy)  # 展平 + 线性投影

    # ---- 阶段名（黑体） ----
    _label(ax, 106, 288, "输入字 token", family="SimHei")
    _label(ax, 290, 288, "时间卷积 + 平均池化", family="SimHei")
    _label(ax, 475, 288, "空间卷积", family="SimHei")
    _label(ax, 630, 288, "展平 + 线性投影", family="SimHei")

    # ---- 张量形状 ----
    _label(ax, 106, 120, r"$(N,\ C,\ T)$")
    _label(ax, 290, 120, r"$(N,\ k,\ C,\ T_2)$")
    _label(ax, 475, 120, r"$(N,\ k,\ 1,\ T_2)$")
    _label(ax, 630, 120, r"$(N,\ d_{\mathrm{model}})$")

    # ---- 分隔线 ----
    ax.plot([40, 680], [98, 98], color=GRID, lw=0.8, linestyle=(0, (4, 3)))

    # ---- 维度与参数 ----
    _label(
        ax,
        360,
        72,
        r"$T_1 = T - m_1 + 1 = 92 \qquad T_2 = \lfloor (T_1 - m_2)/s \rfloor + 1 = 40$",
    )
    _label(
        ax,
        360,
        46,
        r"$\mathrm{flatten} = k \cdot T_2 = 1280 \qquad "
        r"\mathrm{Linear}: 1280 \rightarrow d_{\mathrm{model}} = 128$",
    )
    _label(
        ax,
        360,
        20,
        r"$k = 32,\quad m_1 = 9,\quad m_2 = 13,\quad s = 2,\quad C = 122,\quad T = 100$",
    )

    return fig


def main():
    out_dir = Path(__file__).resolve().parents[1] / "paper" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf = out_dir / "fig_char_token_encoder.pdf"
    svg = out_dir / "fig_char_token_encoder.svg"
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    print(f"written: {pdf}")
    print(f"written: {svg}")


if __name__ == "__main__":
    main()
