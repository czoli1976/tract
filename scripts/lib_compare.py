#!/usr/bin/env python3
"""Compare two .npz files key-by-key. Prints max abs diff per key.

Exit non-zero if any key has nonzero diff (caller can use this for CI).
"""
import sys
import numpy as np


def main(path_a: str, path_b: str) -> int:
    a = np.load(path_a)
    b = np.load(path_b)
    keys_a = sorted(a.keys())
    keys_b = sorted(b.keys())
    if keys_a != keys_b:
        print(f"  KEY MISMATCH: {keys_a} vs {keys_b}", file=sys.stderr)
        return 2
    any_diff = False
    for k in keys_a:
        diff = np.abs(a[k].astype(np.float64) - b[k].astype(np.float64))
        max_d = float(diff.max())
        max_a = float(np.abs(a[k]).max()) if a[k].size > 0 else 0.0
        marker = "✗" if max_d != 0 else "✓"
        print(f"  {marker} {k}: shape={a[k].shape} dtype={a[k].dtype} max_abs_diff={max_d:.2e} max_abs_a={max_a:.2e}")
        if max_d != 0:
            any_diff = True
    return 1 if any_diff else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: lib_compare.py a.npz b.npz", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
