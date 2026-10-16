"""Plain-language overview figure (EN / zh-TW). Run: python figures/make_overview.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

import os, sys
# matplotlib only exposes the JP face of the .ttc; pass extracted TC .otf files via env var
for p in os.environ.get("TC_FONTS", "").split(":"):
    if p: fm.fontManager.addfont(p)

T = {
 "en": dict(font="DejaVu Sans", out="figures/overview_en.png",
  main=[("1  Raw motion signal", "A small motion sensor (IMU)\nreports acceleration and\nrotation, 200 times a second"),
        ("2  Self-calibrate", "The model corrects the\nsensor's own errors: scale,\ndrift, timing, axis direction"),
        ("3  Find \"down\"", "Works out which way gravity\npoints, i.e. how the\nrobot is tilted"),
        ("4  Read 40 seconds", "Cuts the signal into 1-second\nslices; a neural network\nreads 40 s of context"),
        ("5  Output speed", "One speed per second, in the\nrobot's own frame, handed on\nslice to slice by a physics rule")],
  head="Competition model (ranked: 20th of 131)",
  post_head="After the competition (not ranked)",
  post=[("Public method", "Add another author's\npublished post-processing"),
        ("Ground-robot fix", "Apply it at half strength\nto cars, dogs and humans"),
        ("Smoothing by trust", "Weight each second by how\nconfident the model is")]),
 "zh": dict(font="Noto Sans CJK TC", out="figures/overview_zh.png",
  main=[("1  原始 IMU 訊號", "小型動作感測器，每秒 200 次\n記錄加速度與角速度"),
        ("2  自動校正感測器", "修正感測器自身的誤差：\n尺度、偏差、時間差、軸向"),
        ("3  估計重力方向", "判斷重力指向哪裡，\n也就是機器人的傾斜姿態"),
        ("4  讀 40 秒上下文", "訊號切成每秒一窗，\n神經網路讀前後 40 秒"),
        ("5  輸出速度", "每秒一筆機體座標速度，\n以物理遞迴一窗接一窗傳遞")],
  head="競賽模型（計名次：131 隊中第 20）",
  post_head="賽後延伸（不計名次）",
  post=[("接上公開方法", "接上他人公開的\n後處理方法"),
        ("地面機器人修正", "對車、機器狗、人\n以半強度套用"),
        ("逐窗不確定度平滑", "越有把握的那一秒，\n平滑時權重越大")]),
}
INK, MUTE, FILL, ACC = "#1f2933", "#52606d", "#eef1f5", "#3e6a8a"

def draw(lang):
    c = T[lang]
    plt.rcParams["font.family"] = c["font"]
    fig = plt.figure(figsize=(16, 7.4), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1600); ax.set_ylim(0, 740); ax.axis("off")
    ax.text(40, 705, c["head"], fontsize=17, fontweight="bold", color=INK, va="center")
    w, h, gap, x0, y0 = 272, 230, 40, 40, 400
    for i, (t, d) in enumerate(c["main"]):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=12",
                                    fc=FILL, ec=ACC, lw=2))
        ax.text(x + w / 2, y0 + h - 48, t, ha="center", va="center", fontsize=14.5,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y0 + 80, d, ha="center", va="center", fontsize=12,
                color=MUTE, linespacing=1.55)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + w + 4, y0 + h / 2), (x + w + gap - 4, y0 + h / 2),
                         arrowstyle="-|>", mutation_scale=22, color=ACC, lw=2.2))
    # post-competition dashed region
    by, bh = 40, 270
    ax.add_patch(FancyBboxPatch((40, by), 1520, bh, boxstyle="round,pad=0,rounding_size=14",
                                fc="white", ec=MUTE, lw=1.8, ls=(0, (6, 4))))
    ax.text(64, by + bh - 34, c["post_head"], fontsize=15.5, fontweight="bold", color=INK, va="center")
    # connector from box 5 down into dashed region
    xc = x0 + 4 * (w + gap) + w / 2
    ax.add_patch(FancyArrowPatch((xc, y0 - 4), (xc, by + bh + 4), arrowstyle="-|>",
                 mutation_scale=22, color=MUTE, lw=2.2, ls=(0, (4, 3))))
    pw, ph, pg = 440, 150, 70
    px0 = 64 + (1472 - (3 * pw + 2 * pg)) / 2
    for i, (t, d) in enumerate(c["post"]):
        x = px0 + i * (pw + pg); y = by + 28
        ax.add_patch(FancyBboxPatch((x, y), pw, ph, boxstyle="round,pad=0,rounding_size=10",
                                    fc="white", ec=MUTE, lw=1.6, ls=(0, (6, 4))))
        ax.text(x + pw / 2, y + ph - 38, t, ha="center", va="center", fontsize=14.5,
                fontweight="bold", color=INK)
        ax.text(x + pw / 2, y + 52, d, ha="center", va="center", fontsize=12.5,
                color=MUTE, linespacing=1.5)
        if i < 2:
            ax.add_patch(FancyArrowPatch((x + pw + 4, y + ph / 2), (x + pw + pg - 4, y + ph / 2),
                         arrowstyle="-|>", mutation_scale=20, color=MUTE, lw=2))
    fig.savefig(c["out"], dpi=100, facecolor="white")

for l in T: draw(l)
