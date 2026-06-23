#!/usr/bin/env python3
"""Generate the RFC performance charts as dependency-free SVGs (measured M4 data)."""
import os

PAL = ["#5B8DEF", "#F5A623", "#2BB673"]  # f16 / scalar / simd (blue / orange / green)
W, H = 760, 460
ML, MR, MT, MB = 78, 26, 78, 70
X0, X1, Y0, Y1 = ML, W - MR, H - MB, MT


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def grouped_bar(title, subtitle, groups, series, ymax, yunit, colors, fname, vfmt="{:.0f}"):
    n_g, n_s = len(groups), len(series)
    gw = (X1 - X0) / n_g
    pad = gw * 0.18
    bw = (gw - 2 * pad) / n_s
    sv = []
    sv.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
    sv.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    sv.append(f'<text x="{ML}" y="34" font-size="21" font-weight="700" fill="#1a1a1a">{esc(title)}</text>')
    sv.append(f'<text x="{ML}" y="56" font-size="13.5" fill="#666">{esc(subtitle)}</text>')
    # y gridlines + labels
    nticks = 5
    for i in range(nticks + 1):
        yv = ymax * i / nticks
        y = Y0 - (Y0 - Y1) * i / nticks
        sv.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="#eee"/>')
        sv.append(f'<text x="{X0-10}" y="{y+4:.1f}" font-size="11.5" fill="#999" text-anchor="end">{vfmt.format(yv)}</text>')
    sv.append(f'<text x="20" y="{(Y0+Y1)/2:.0f}" font-size="12.5" fill="#666" transform="rotate(-90 20 {(Y0+Y1)/2:.0f})" text-anchor="middle">{esc(yunit)}</text>')
    # bars
    for gi, g in enumerate(groups):
        gx = X0 + gi * gw
        for si, (name, vals) in enumerate(series):
            v = vals[gi]
            bh = (Y0 - Y1) * (v / ymax)
            bx = gx + pad + si * bw
            by = Y0 - bh
            sv.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw-3:.1f}" height="{bh:.1f}" rx="2.5" fill="{colors[si]}"/>')
            sv.append(f'<text x="{bx+(bw-3)/2:.1f}" y="{by-5:.1f}" font-size="11" fill="#444" text-anchor="middle">{vfmt.format(v)}</text>')
        sv.append(f'<text x="{gx+gw/2:.1f}" y="{Y0+22:.0f}" font-size="13" fill="#333" text-anchor="middle">{esc(g)}</text>')
    # legend
    lx = X0
    ly = H - 22
    for si, (name, _) in enumerate(series):
        sv.append(f'<rect x="{lx}" y="{ly-10}" width="13" height="13" rx="2.5" fill="{colors[si]}"/>')
        sv.append(f'<text x="{lx+19}" y="{ly+1}" font-size="12.5" fill="#333">{esc(name)}</text>')
        lx += 26 + len(name) * 7.6
    sv.append("</svg>")
    open(fname, "w").write("\n".join(sv))
    print("wrote", fname)


here = os.path.dirname(os.path.abspath(__file__))
out = os.path.dirname(here)  # worktree root

# 1) KV-cache memory vs context (the headline)
grouped_bar(
    "KV-cache memory — Qwen3-1.7B (int4 = ¼ of f16, exact)",
    "705 MB reclaimed at 8K context; the f16 cache there ≈ the whole 4-bit model",
    ["1K", "2K", "4K", "8K"],
    [("f16", [115, 235, 470, 940]), ("int4", [29, 59, 118, 235])],
    1000, "megabytes", [PAL[0], PAL[2]],
    os.path.join(out, "rfc_kvquant_memory.svg"),
)

# 2) GPU attention kernel latency (M4, per head)
grouped_bar(
    "GPU attention kernel latency — Apple M4, per head (bit-exact)",
    "int4-SIMD vs f16:  1.41× @512   1.24× @2048   1.21× @4096   ·   3.8× smaller KV reads",
    ["T=512", "T=2048", "T=4096"],
    [("f16", [3.42, 13.77, 27.81]), ("int4 scalar", [3.11, 12.50, 24.72]), ("int4 SIMD", [2.43, 11.12, 22.91])],
    30, "microseconds", PAL,
    os.path.join(out, "rfc_kvquant_gpu_latency.svg"),
    vfmt="{:.1f}",
)

# 3) Causal-skip: latency scales with visible tokens (prefill, bit-exact)
grouped_bar(
    "Causal-skip — per-query latency ∝ visible tokens (prefill, bit-exact)",
    "decode tok/s unaffected; halves prompt attention (P²→P²/2); ~26% of prefill at 32K",
    ["valid 100% (2048)", "valid 50% (1024)", "valid 25% (512)"],
    [("int4 SIMD", [11.2, 4.6, 2.4])],
    12, "microseconds", [PAL[2]],
    os.path.join(out, "rfc_kvquant_causal_skip.svg"),
    vfmt="{:.1f}",
)
