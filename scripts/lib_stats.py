#!/usr/bin/env python3
"""Welch's t-test on bench samples.

Usage:  lib_stats.py b1 b2 ... bN -- p1 p2 ... pN

Prints means, stdevs, delta %, Welch t-statistic. |t| > 2 ≈ significant.
"""
import statistics as st
import sys


def main(args: list[str]) -> int:
    if "--" not in args:
        print("usage: lib_stats.py b1 b2 ... -- p1 p2 ...", file=sys.stderr)
        return 2
    sep = args.index("--")
    b = [float(x) for x in args[:sep]]
    p = [float(x) for x in args[sep + 1:]]
    if not b or not p:
        print("need at least 1 sample on each side", file=sys.stderr)
        return 2
    bm, pm = st.mean(b), st.mean(p)
    bs = st.stdev(b) if len(b) > 1 else 0.0
    ps = st.stdev(p) if len(p) > 1 else 0.0
    delta_pct = (pm - bm) / bm * 100 if bm != 0 else float('nan')
    pse = (bs**2 / len(b) + ps**2 / len(p)) ** 0.5
    t = (pm - bm) / pse if pse > 0 else float('nan')

    print(f"== Stats ==")
    print(f"  baseline (n={len(b)}): mean={bm:.3f} ms  stdev={bs:.3f}  min={min(b):.3f}  max={max(b):.3f}")
    print(f"  patched  (n={len(p)}): mean={pm:.3f} ms  stdev={ps:.3f}  min={min(p):.3f}  max={max(p):.3f}")
    print(f"  delta:    {delta_pct:+.2f}% (mean)")
    print(f"  Welch t:  {t:+.2f}   (|t| > 2 ≈ statistically significant)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
