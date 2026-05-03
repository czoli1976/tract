# Bench data — `LazyIm2col` for grouped convolutions + threshold sweep

Companion data for [sonos/tract#2186](https://github.com/sonos/tract/pull/2186).

## Test environment

- **Hardware**: Apple M-series (arm64 macOS)
- **Native kernel**: AMX (`mmm_f32_32x4`)
- **Date**: 2026-05-03
- **Tract baseline**: `sonos/tract` main `41b7b027c`
- **Tract patched**: same base + the two commits in #2186
- **Bench command**: `tract --onnx-ignore-output-shapes --onnx-ignore-value-info <model> <inputs> -O bench --allow-random-input --max-time 4000 --warmup-time 500`
- **Methodology**: 3 interleaved runs per (config, kernel size); reported value is mean.

## Sweep — synthetic single-Conv ONNX

Input: `1 × in_ch × 100 × 96`, output channels = 10.

Two binaries built with the threshold forced:
- "eager" → `should_use_lazy` returns `false` (threshold 999)
- "lazy" → `should_use_lazy` returns `true` (threshold 1)

| Kernel volume | Kernel × group | Eager (ms) | Lazy (ms) | Δ |
|---|---|---:|---:|---:|
| 1  | 1×1, group=1 | 1.085 | 1.077 | **−0.7%** (noise; 1×1 has no spatial gather) |
| 4  | 2×2, group=1 | 1.543 | 1.168 | **−24%** |
| 5  | 5×1, **group=2** (DFN3 `df_convp.1` shape) | 5.201 | 3.129 | **−40%** |
| 6  | 3×2, group=1 | 1.934 | 1.336 | **−31%** |
| 9  | 3×3, group=1 | 2.067 | 1.620 | **−22%** |
| 16 | 4×4, group=1 | 4.566 | 2.260 | **−50%** |
| 25 | 5×5, group=1 | 5.255 | 4.249 | **−19%** |

Lazy wins for every kernel volume ≥ 4 in this sweep. The 1×1 case is structurally degenerate (no gather happens in either path).

The default `should_use_lazy` threshold of `> 5` (`>= 6`) is conservative; the data argues for `>= 2` to capture every measurable win without regression. The PR keeps the default unchanged and exposes the threshold via env var so users can opt in per platform.

## End-to-end DFN3 (T=100 frames = 1 s audio @ 48 kHz)

With `TRACT_LAZY_IM2COL_MIN_KERNEL=5` (enables the DFN3 case):

| Stage | Baseline | Patched + env | Δ |
|---|---:|---:|---:|
| enc | 11.27 ms | 9.51 ms | **−15.6%** |
| erb_dec | 19.90 ms | 20.03 ms | within noise (mostly DepthWiseConv path; no eager-Im2col Convs to convert) |
| df_dec | 11.17 ms | 10.14 ms | **−9.2%** |
| **composite** | **42.34 ms** | **39.68 ms** | **−6.3%** |
| **RTF** | 0.0423 (24× real-time) | 0.0397 (25× real-time) | −6.3% |

## Default behavior verification

With no env var set, the optimized graph for DFN3 stages is **byte-identical** between baseline (`41b7b027c`) and patched. Verified via `diff <(tract -O dump on baseline) <(tract -O dump on patched)` — empty diff for `enc`, `erb_dec`, and `df_dec`.

## Bit-exact verification

Across DFN3 stages with fixed-seed (`np.random.seed(42)`) inputs:

| Stage | Output | max_abs_diff (baseline vs patched + env) |
|---|---|---:|
| enc | e0, e1, e2, e3, emb, c0, lsnr | 0.0 |
| erb_dec | m | 0.0 |
| df_dec | coefs, 235 | 0.0 |

## Tests

- `cargo test --release -p tract-core --lib` → **231 passed, 0 failed**
- `cargo test --release -p test-onnx-core conv` → **372 passed, 48 ignored, 0 failed**
- `cargo test --release -p test-metal` → **21,898 passed, 5,030 ignored, 0 failed**
- `cargo fmt --check` → clean
