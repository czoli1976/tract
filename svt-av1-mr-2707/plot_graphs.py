#!/usr/bin/env python3
"""MR 2707 v2 graphs. Light mode, validated palette (dataviz reference instance)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import parse_log, bd_rate, ivf_frames

SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # validated categorical order
CLIPS = ["debugging", "wikipedia", "slides1", "spreadsheet"]
QPS = [23, 31, 43, 55]
TBRS = [750, 1500, 2500, 4000]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def series(clip, mode, enc):
    rates, psnrs = [], []
    pts = QPS if mode == "cqp" else TBRS
    for p in pts:
        f = (f"results/{clip}_q{p}_{enc}.log" if mode == "cqp"
             else f"results/cbrm_{clip}_{p}_{enc}.log")
        k, y, _ = parse_log(f)
        rates.append(k); psnrs.append(y)
    return rates, psnrs


import os
HAVE_CBR = all(os.path.exists(f"results/cbrm_{c}_{t}_{e}.log")
               for c in CLIPS for t in TBRS for e in ("base", "v2"))
os.makedirs("graphs", exist_ok=True)

# ---- g1: BD-rate per clip, CQP + CBR ----
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
bd_cqp, bd_cbr = [], []
for c in CLIPS:
    rb, pb = series(c, "cqp", "base"); rv, pv = series(c, "cqp", "v2")
    bd_cqp.append(bd_rate(rb, pb, rv, pv))
    if HAVE_CBR:
        rb, pb = series(c, "cbr", "base"); rv, pv = series(c, "cbr", "v2")
        bd_cbr.append(bd_rate(rb, pb, rv, pv))
    else:
        bd_cbr.append(float("nan"))
x = range(len(CLIPS)); w = 0.34
b1 = ax.bar([i - w / 2 for i in x], bd_cqp, w, color=S1, label="CQP", zorder=3)
b2 = ax.bar([i + w / 2 for i in x], bd_cbr, w, color=S2, label="CBR", zorder=3)
for bars in (b1, b2):
    for r in bars:
        v = r.get_height()
        ax.annotate(f"{v:+.1f}", (r.get_x() + r.get_width() / 2, v),
                    ha="center", va="top" if v < 0 else "bottom", fontsize=9, color=INK,
                    xytext=(0, -2 if v < 0 else 2), textcoords="offset points")
ax.axhline(0, color=BASE, lw=1, zorder=2)
ax.set_xticks(list(x)); ax.set_xticklabels([c.capitalize() for c in CLIPS], color=INK2)
ax.set_ylabel("BD-rate vs base (%)  ·  negative = better")
ax.set_title("v2 BD-rate per clip (Y-PSNR, 130 frames, p8 RTC scm1)", color=INK, fontsize=11)
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("graphs/g1_bdrate.png"); plt.close(fig)

# ---- g2: RD curves, 2x2 small multiples (CQP) ----
fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), dpi=160, sharex=False)
for ax, c in zip(axes.flat, CLIPS):
    rb, pb = series(c, "cqp", "base"); rv, pv = series(c, "cqp", "v2")
    ax.plot(rb, pb, "-o", color=S1, lw=2, ms=5, label="base", zorder=3)
    ax.plot(rv, pv, "-o", color=S2, lw=2, ms=5, label="v2", zorder=3)
    ax.set_title(c.capitalize(), fontsize=10, color=INK)
    ax.grid(axis="x", visible=False)
axes[0][0].legend(frameon=False, loc="lower right", fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("bitrate (kbps)")
for r in axes:
    r[0].set_ylabel("Y-PSNR (dB)")
fig.suptitle("Rate-distortion, CQP {23,31,43,55} (up/left = better)", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig("graphs/g2_rd.png"); plt.close(fig)

# ---- g3: demand-driven cost ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.6), dpi=160, width_ratios=[3, 2])
fb = ivf_frames("results/wikipedia_q31_base.ivf")
fv = ivf_frames("results/wikipedia_q31_v2.ivf")
idx = range(1, len(fb))  # skip keyframe (identical, off-scale)
a1.plot(list(idx), [fb[i][0] for i in idx], color=S1, lw=1.6, label="base", zorder=3)
a1.plot(list(idx), [fv[i][0] for i in idx], color=S2, lw=1.6, label="v2", zorder=3)
pref = 0
while pref + 1 < len(fb) and fb[pref] == fv[pref]:
    pref += 1
a1.axvspan(1, pref, color=GRID, alpha=0.5, zorder=1)
a1.annotate("static intro:\nv2 byte-identical", (pref / 2 + 1, max(f[0] for f in fb[1:]) * 0.82),
            fontsize=8.5, color=INK2, ha="center")
a1.set_xlabel("frame"); a1.set_ylabel("inter frame size (bytes)")
a1.set_title("Wikipedia q31: per-frame bytes", fontsize=10, color=INK)
a1.legend(frameon=False, fontsize=9)
a1.grid(axis="x", visible=False)
# wall-clock, 2 reps back-to-back (q31)
t = {"slides1": {"base": [9722, 9780], "v2": [9873, 10817]},
     "wikipedia": {"base": [6707, 6727], "v2": [6925, 7079]}}
x = range(2); w = 0.34
clips2 = ["slides1", "wikipedia"]
mb = [sum(t[c]["base"]) / 2000 for c in clips2]
mv = [sum(t[c]["v2"]) / 2000 for c in clips2]
a2.bar([i - w / 2 for i in x], mb, w, color=S1, zorder=3)
a2.bar([i + w / 2 for i in x], mv, w, color=S2, zorder=3)
for c, xi in zip(clips2, x):
    a2.scatter([xi - w / 2] * 2, [v / 1000 for v in t[c]["base"]], color=INK, s=12, zorder=4)
    a2.scatter([xi + w / 2] * 2, [v / 1000 for v in t[c]["v2"]], color=INK, s=12, zorder=4)
    d = (sum(t[c]["v2"]) / sum(t[c]["base"]) - 1) * 100
    a2.annotate(f"+{d:.0f}%", (xi + w / 2, max(t[c]["v2"]) / 1000), ha="center",
                va="bottom", fontsize=9, color=INK, xytext=(0, 3), textcoords="offset points")
a2.set_xticks(list(x)); a2.set_xticklabels([c.capitalize() for c in clips2], color=INK2)
a2.set_ylabel("encode wall-clock (s), q31, 2 reps")
a2.set_title("Encode time (indicative)", fontsize=10, color=INK)
a2.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("graphs/g3_cost.png"); plt.close(fig)

# ---- g4: dPSNR at matched QP ----
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
colors = [S1, S2, S3, S4]
for c, col in zip(CLIPS, colors):
    _, pb = series(c, "cqp", "base"); _, pv = series(c, "cqp", "v2")
    d = [v - b for v, b in zip(pv, pb)]
    ax.plot(QPS, d, "-o", color=col, lw=2, ms=6, zorder=3)
    ax.annotate(c.capitalize(), (QPS[-1], d[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=col, va="center")
ax.axhline(0, color=BASE, lw=1, zorder=2)
ax.set_xticks(QPS)
ax.set_xlabel("CQP"); ax.set_ylabel("ΔY-PSNR v2 − base (dB)")
ax.set_title("Quality gain at matched QP (never negative)", color=INK, fontsize=11)
ax.set_xlim(20, 62)
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("graphs/g4_dbgain.png"); plt.close(fig)

# CBR BD-rate table for the results doc
print("CBR BD-rates (tbr {750,1500,2500,4000}):")
for c, v in zip(CLIPS, bd_cbr):
    print(f"  {c}: {v:+.2f}%")
print("CQP BD-rates:", {c: round(v, 2) for c, v in zip(CLIPS, bd_cqp)})
