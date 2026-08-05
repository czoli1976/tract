#!/usr/bin/env python3
"""Analyze the base-vs-v2 encode matrix: results table, BD-rate, frame-level diffs."""
import hashlib
import re
import sys


def ivf_frames(path):
    """Return list of (size, md5) per frame payload in an IVF file."""
    out = []
    with open(path, "rb") as f:
        hdr = f.read(32)
        assert hdr[:4] == b"DKIF", path
        while True:
            fh = f.read(12)
            if len(fh) < 12:
                break
            size = int.from_bytes(fh[0:4], "little")
            payload = f.read(size)
            out.append((size, hashlib.md5(payload).hexdigest()))
    return out


def parse_log(path):
    """Extract (kbps, psnr_y, fps) from an SvtAv1EncApp stat-report log."""
    text = open(path, errors="replace").read()
    kbps = psnr = fps = None
    m = re.search(r"([\d.]+)\s*kbps", text)
    if m:
        kbps = float(m.group(1))
    m = re.search(r"Average Speed:\s*([\d.]+)\s*fps", text)
    if m:
        fps = float(m.group(1))
    # stat-report data line follows the "Total Frames  Average QP  Y-PSNR ..." header;
    # its numbers are [frames, avg_qp, avg_y_psnr, ...]
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "Y-PSNR" in ln and "Average QP" in ln and i + 1 < len(lines):
            nums = re.findall(r"[\d.]+", lines[i + 1])
            if len(nums) >= 3:
                psnr = float(nums[2])
            break
    return kbps, psnr, fps


def bd_rate(r1, p1, r2, p2):
    """BD-rate of set2 vs set1 (negative = set2 needs less bitrate). Piecewise cubic (pchip) log-rate integration."""
    import math

    def pchip_int(xs, ys, lo, hi):
        # xs ascending (PSNR), ys log-bitrate; integrate ys over [lo,hi] with monotone cubic
        h = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        d = [(ys[i + 1] - ys[i]) / h[i] for i in range(len(h))]
        m = [0.0] * len(xs)
        m[0], m[-1] = d[0], d[-1]
        for i in range(1, len(xs) - 1):
            if d[i - 1] * d[i] <= 0:
                m[i] = 0.0
            else:
                w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
        total = 0.0
        for i in range(len(h)):
            a, b = max(lo, xs[i]), min(hi, xs[i + 1])
            if a >= b:
                continue
            # integrate cubic hermite on [xs[i], xs[i+1]] restricted to [a,b] numerically
            n = 64
            step = (b - a) / n
            s = 0.0
            for k in range(n + 1):
                x = a + k * step
                t = (x - xs[i]) / h[i]
                h00 = 2 * t**3 - 3 * t**2 + 1
                h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2
                h11 = t**3 - t**2
                y = h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]
                w = 1.0 if 0 < k < n else 0.5
                s += w * y
            total += s * step
        return total

    s1 = sorted(zip(p1, [math.log(x) for x in r1]))
    s2 = sorted(zip(p2, [math.log(x) for x in r2]))
    lo = max(s1[0][0], s2[0][0])
    hi = min(s1[-1][0], s2[-1][0])
    if lo >= hi:
        return float("nan")
    i1 = pchip_int([x for x, _ in s1], [y for _, y in s1], lo, hi)
    i2 = pchip_int([x for x, _ in s2], [y for _, y in s2], lo, hi)
    return (math.exp((i2 - i1) / (hi - lo)) - 1) * 100


if __name__ == "__main__":
    resdir = sys.argv[1] if len(sys.argv) > 1 else "results"
    qps = [23, 31, 43, 55]
    print(f"{'clip':<12} {'qp':>3} {'kbps base':>10} {'kbps v2':>10} {'dSize%':>7} "
          f"{'PSNR-Y base':>11} {'PSNR-Y v2':>10} {'dPSNR':>6} {'fps base':>8} {'fps v2':>7}")
    for clip in ["debugging", "wikipedia", "slides1", "spreadsheet"]:
        rb, pb, rv, pv = [], [], [], []
        for qp in qps:
            kb, yb, fb = parse_log(f"{resdir}/{clip}_q{qp}_base.log")
            kv, yv, fv = parse_log(f"{resdir}/{clip}_q{qp}_v2.log")
            rb.append(kb); pb.append(yb); rv.append(kv); pv.append(yv)
            print(f"{clip:<12} {qp:>3} {kb:>10.1f} {kv:>10.1f} {100*(kv-kb)/kb:>6.1f}% "
                  f"{yb:>11.2f} {yv:>10.2f} {yv-yb:>+6.2f} {fb:>8.1f} {fv:>7.1f}")
        print(f"{clip:<12} BD-rate (v2 vs base, Y-PSNR): {bd_rate(rb, pb, rv, pv):+.2f}%")
