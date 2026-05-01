# Raw benchmark logs — K=1 broadcast Mul fix

These are the literal `Bench ran N times, X.XXX ms/i` lines I observed during testing, organized by model and platform. The aggregated tables in [RESULTS-OBSERVED.md](RESULTS-OBSERVED.md) are derived from these.

## Test environment

- **Hardware:** Apple M-series (arm64 macOS)
- **Native kernel:** AMX (`mmm_f32_32x4`)
- **WASM runtime:** `wasmtime 44.0.0`, target `wasm32-wasip1`, `RUSTFLAGS=-C target-feature=+simd128`
- **WASM kernel:** `wasm_f32_8x8`
- **Tract baseline:** `sonos/tract` main `b36f34e92` (initial runs) → `41b7b027c` (final rebase)
- **Tract patched:** same base + this PR's K=1 commit
- **Date:** 2026-05-01
- **Bench command form:** `tract --onnx-ignore-output-shapes --onnx-ignore-value-info <model> <inputs> -O bench --allow-random-input [--max-time MS --warmup-time MS]`

---

## DFN3 — Native (Apple M-series, AMX)

### erb_dec — interleaved 5×8s after rebase onto current main `41b7b027c`

```
baseline run 1: Bench ran 396 times, 20.222 ms/i.
patched  run 1: Bench ran 819 times,  9.766 ms/i.
baseline run 2: Bench ran 391 times, 20.467 ms/i.
patched  run 2: Bench ran 841 times,  9.507 ms/i.
baseline run 3: Bench ran 386 times, 20.767 ms/i.
patched  run 3: Bench ran 790 times, 10.121 ms/i.
baseline run 4: Bench ran 389 times, 20.595 ms/i.
patched  run 4: Bench ran 797 times, 10.037 ms/i.
baseline run 5: Bench ran 378 times, 21.183 ms/i.
patched  run 5: Bench ran 825 times,  9.694 ms/i.
```
Stats: baseline 20.65 ± 0.36 ms, patched 9.83 ± 0.25 ms. **−52.41%**.

### enc — single bench run

```
baseline: Bench ran 509 times, 9.830 ms/i.
patched:  Bench ran 516 times, 9.689 ms/i.
```
Δ −1.4% (noise; fix doesn't fire on enc).

### df_dec — 3 runs each

```
baseline run 1: Bench ran 457 times, 10.934 ms/i.
baseline run 2: Bench ran 447 times, 11.191 ms/i.
baseline run 3: Bench ran 451 times, 11.107 ms/i.
patched  run 1: Bench ran 454 times, 11.017 ms/i.
patched  run 2: Bench ran 464 times, 10.768 ms/i.
patched  run 3: Bench ran 433 times, 11.550 ms/i.
```
Stats: baseline 11.08 ± 0.13, patched 11.11 ± 0.39. Δ +0.3% (noise; fix doesn't fire on df_dec).

### erb_dec per-op profile (full main + cost output)

| Op | M | K | N | Baseline | Patched |
|---|---|---|---|---|---|
| `/convt1/ConvTranspose.einsum` | 1600 | 1 | 3 | 7.428 ms (33.0%, 41.358 MF/s) | 0.224 ms (2.2%, 1.371 GF/s) |
| `/convt2/0/ConvTranspose.einsum` | 800 | 1 | 3 | 3.930 ms (17.5%, 39.085 MF/s) | 0.109 ms (1.1%, 924.972 MF/s) |
| `/convt1/Conv.einsum` (other op) | — | — | — | 0.934 ms | 0.164 ms |

Total OptMatMul time across the model: baseline 18.286 ms (81.3% of erb_dec), patched 5.954 ms (58.8%).

---

## DFN3 — WASM (wasmtime 44, simd128)

### erb_dec — 3 runs each

```
baseline run 1: Bench ran 321 times, 15.590 ms/i.
baseline run 2: Bench ran 336 times, 14.885 ms/i.
baseline run 3: Bench ran 333 times, 15.032 ms/i.
patched  run 1: Bench ran 366 times, 13.649 ms/i.
patched  run 2: Bench ran 368 times, 13.584 ms/i.
patched  run 3: Bench ran 369 times, 13.551 ms/i.
```
Stats: baseline 15.17 ± 0.38, patched 13.59 ± 0.05. **−10.4%**.

### enc — single

```
baseline: Bench ran 306 times, 16.369 ms/i.
patched:  Bench ran 309 times, 16.218 ms/i.
```

### df_dec — single

```
baseline: Bench ran 411 times, 12.157 ms/i.
patched:  Bench ran 411 times, 12.165 ms/i.
```

### erb_dec per-op profile (WASM)

| Op | Baseline | Patched |
|---|---|---|
| `/convt1/ConvTranspose.einsum` | 1.149 ms (7.4%, 267.365 MF/s) | 0.316 ms (2.2%, 971.144 MF/s) |
| `/convt2/0/ConvTranspose.einsum` | 0.600 ms (3.9%, 256.014 MF/s) | 0.166 ms (1.2%, 924.972 MF/s) |

---

## DFN2 — Native (3 runs each)

### enc

```
baseline run 1: Bench ran 532 times,  9.408 ms/i.
baseline run 2: Bench ran 527 times,  9.486 ms/i.
baseline run 3: Bench ran 498 times, 10.036 ms/i.
patched  run 1: Bench ran 517 times,  9.675 ms/i.
patched  run 2: Bench ran 522 times,  9.576 ms/i.
patched  run 3: Bench ran 529 times,  9.456 ms/i.
```
Stats: baseline 9.64, patched 9.57. Noise.

### erb_dec

```
baseline run 1: Bench ran 245 times, 20.414 ms/i.
baseline run 2: Bench ran 242 times, 21.157 ms/i.
baseline run 3: Bench ran 192 times, 26.098 ms/i.
patched  run 1: Bench ran 437 times, 11.435 ms/i.
patched  run 2: Bench ran 494 times, 10.129 ms/i.
patched  run 3: Bench ran 500 times, 10.000 ms/i.
```
Stats: baseline 22.56, patched 10.52. **−53%**.

### df_dec

```
baseline run 1: Bench ran 468 times, 10.682 ms/i.
baseline run 2: Bench ran 488 times, 10.259 ms/i.
baseline run 3: Bench ran 478 times, 10.464 ms/i.
patched  run 1: Bench ran 480 times, 10.420 ms/i.
patched  run 2: Bench ran 463 times, 10.803 ms/i.
patched  run 3: Bench ran 466 times, 10.727 ms/i.
```
Stats: baseline 10.47, patched 10.65. Noise.

---

## GTCRN — Native (interleaved 5×10s)

```
baseline 1: Bench ran 858 times, 11.627 ms/i.
patched  1: Bench ran 890 times, 11.202 ms/i.
baseline 2: Bench ran 848 times, 11.766 ms/i.
patched  2: Bench ran 941 times, 10.602 ms/i.
baseline 3: Bench ran 879 times, 11.347 ms/i.
patched  3: Bench ran 948 times, 10.511 ms/i.
baseline 4: Bench ran 829 times, 12.023 ms/i.
patched  4: Bench ran 931 times, 10.707 ms/i.
baseline 5: Bench ran 873 times, 11.425 ms/i.
patched  5: Bench ran 945 times, 10.549 ms/i.
```
Stats: baseline 11.64 ± 0.27 ms, patched 10.71 ± 0.28 ms. **Δ −7.93%, Welch t = −5.27** (highly significant).

### Per-op profile (depthwise ConvTranspose ops)

| Op | Baseline | Patched | Speedup |
|---|---|---|---|
| `ConvTranspose_1098.einsum` (depthwise [3×3] group=16) | 0.422 ms (124 MF/s) | 0.033 ms (1.574 GF/s) | **12.8×** |
| `ConvTranspose_1335.einsum` (depthwise [3×3] group=16) | 0.194 ms (122.761 MF/s) | 0.014 ms (1.682 GF/s) | **13.9×** |
| `ConvTranspose_1572.einsum` (depthwise [3×3] group=16) | similar | 0.010 ms (1.424 GF/s) | similar |

---

## GTCRN — WASM (3 runs each)

```
baseline run 1: Bench ran 491 times, 10.127 ms/i.
baseline run 2: Bench ran 438 times, 11.358 ms/i.
baseline run 3: Bench ran 491 times, 10.133 ms/i.
patched  run 1: Bench ran 479 times, 10.390 ms/i.
patched  run 2: Bench ran 473 times, 10.521 ms/i.
patched  run 3: Bench ran 484 times, 10.281 ms/i.
```
Stats: baseline mean 10.54, patched mean 10.40. Δ −1.3% (noise; same WASM-vs-Native pattern as DFN3 — wasm 8x8 kernel has lower per-tile overhead than AMX so the K=1 case is less wasteful to begin with).

---

## Negative controls (Native — fix doesn't fire)

### HiFiGAN v2 (mel 80×100)

```
baseline run 1: Bench ran 19 times, 267.756 ms/i.
baseline run 2: Bench ran 19 times, 269.858 ms/i.
baseline run 3: Bench ran 17 times, 305.645 ms/i.
patched  run 1: Bench ran 19 times, 271.867 ms/i.
patched  run 2: Bench ran 19 times, 276.104 ms/i.
patched  run 3: Bench ran 18 times, 283.231 ms/i.
```
Δ −1.4% (noise). Optimized graph byte-identical (MD5 confirmed).

### HiFiGAN v3 (mel 80×100)

```
baseline run 1: Bench ran 15 times, 346.670 ms/i.
baseline run 2: Bench ran 15 times, 345.525 ms/i.
baseline run 3: Bench ran 15 times, 341.217 ms/i.
patched  run 1: Bench ran 15 times, 349.116 ms/i.
patched  run 2: Bench ran 15 times, 343.497 ms/i.
patched  run 3: Bench ran 15 times, 342.551 ms/i.
```
Δ +0.3% (noise). Graph byte-identical.

### Vocos (mel 100×100, 5×30s sequential)

```
baseline run 1: Bench ran 1610 times, 18.623 ms/i.
baseline run 2: Bench ran 1610 times, 18.630 ms/i.
baseline run 3: Bench ran 1598 times, 18.769 ms/i.
baseline run 4: Bench ran 1626 times, 18.442 ms/i.
baseline run 5: Bench ran 1600 times, 18.740 ms/i.
patched  run 1: Bench ran 1643 times, 18.255 ms/i.
patched  run 2: Bench ran 1600 times, 18.738 ms/i.
patched  run 3: Bench ran 1433 times, 20.924 ms/i.
patched  run 4: Bench ran 1487 times, 20.167 ms/i.
patched  run 5: Bench ran 1490 times, 20.146 ms/i.
```
Sequential mean: baseline 18.64 ± 0.13, patched 19.65 ± 1.11 → apparent +5.4%. **But the patched runs ran later and hit thermal throttling.**

Re-run interleaved 6 pairs (10s each) to control:

```
baseline 1: Bench ran 483 times, 20.709 ms/i.
patched  1: Bench ran 521 times, 19.189 ms/i.
baseline 2: Bench ran 469 times, 21.344 ms/i.
patched  2: Bench ran 535 times, 18.684 ms/i.
baseline 3: Bench ran 519 times, 19.259 ms/i.
patched  3: Bench ran 543 times, 18.405 ms/i.
baseline 4: Bench ran 510 times, 19.620 ms/i.
patched  4: Bench ran 540 times, 18.515 ms/i.
baseline 5: Bench ran 542 times, 18.438 ms/i.
patched  5: Bench ran 504 times, 19.860 ms/i.
baseline 6: Bench ran 507 times, 19.715 ms/i.
patched  6: Bench ran 537 times, 18.609 ms/i.
```
Interleaved mean: baseline 19.85, patched 18.88 → patched apparently faster. Graph is byte-identical between baseline and patched, so any apparent delta either way is pure thermal/scheduling noise. Methodology lesson: always interleave on a thermally-active laptop.

### MediaPipe Selfie (1×3×256×256)

```
baseline run 1: Bench ran 141 times, 35.671 ms/i.
baseline run 2: Bench ran 140 times, 35.836 ms/i.
patched  run 1: Bench ran 143 times, 35.154 ms/i.
patched  run 2: Bench ran 143 times, 35.087 ms/i.
```
Δ −1.8% (noise). Graph byte-identical.

### MODNet (1×3×512×512, 6×30s each)

```
baseline run 1: Bench ran 39 times, 775.961 ms/i.
baseline run 2: Bench ran 36 times, 849.707 ms/i.
baseline run 3: Bench ran 37 times, 812.934 ms/i.
baseline run 4: Bench ran 40 times, 764.146 ms/i.
baseline run 5: Bench ran 39 times, 780.107 ms/i.
baseline run 6: Bench ran 39 times, 788.103 ms/i.
patched  run 1: Bench ran 39 times, 784.307 ms/i.
patched  run 2: Bench ran 38 times, 789.853 ms/i.
patched  run 3: Bench ran 37 times, 829.784 ms/i.
patched  run 4: Bench ran 38 times, 799.632 ms/i.
patched  run 5: Bench ran 39 times, 774.483 ms/i.
patched  run 6: Bench ran 38 times, 795.583 ms/i.
```
Stats: baseline 795.16 ± 31.30 ms, patched 795.61 ± 18.93 ms. **Δ +0.06%, Welch t = 0.03** (literally indistinguishable). Graph byte-identical.

Initial 2-run sample had shown +3.7% — multi-run analysis collapsed it. Demonstrates importance of multi-run methodology.

### Demucs htdemucs_6s (waveform 1×2×343980 + spec 1×4×2048×336)

Optimized graph byte-identical (verified via `tract -O dump | md5`). Bench skipped — fix provably doesn't touch the graph, so any delta would be measurement noise.

---

## Negative controls (WASM)

### MediaPipe Selfie WASM (1×3×256×256)

```
baseline run 1: Bench ran 146 times, 34.338 ms/i.
patched  run 1: Bench ran 147 times, 34.180 ms/i.
```
Bit-exact, near-noise.

### MODNet WASM (1×3×512×512, 5×60s each)

```
baseline run 1: Bench ran 30 times, 2035.713 ms/i.
baseline run 2: Bench ran 30 times, 2065.637 ms/i.
baseline run 3: Bench ran 30 times, 2056.451 ms/i.
baseline run 4: Bench ran 33 times, 1854.972 ms/i.
baseline run 5: Bench ran 32 times, 1917.292 ms/i.
patched  run 1: Bench ran 32 times, 1895.092 ms/i.
patched  run 2: Bench ran 33 times, 1847.345 ms/i.
patched  run 3: Bench ran 31 times, 1966.164 ms/i.
patched  run 4: Bench ran 33 times, 1842.858 ms/i.
patched  run 5: Bench ran 32 times, 1893.573 ms/i.
```
Stats: baseline 1986 ± 94, patched 1889 ± 50. Δ −4.9%, Welch t = −2.0 (within wasmtime variance band; graph byte-identical).

---

## End-to-end DFN3 deep-filter binary (real WAV through full enhance() pipeline)

Setup: tract 0.22.1 baseline vs tract 0.22.1 + K=1 fix backport. DeepFilterNet `deep-filter` Rust binary built against each. Input: 27.29-second 48 kHz mono noisy WAV.

Wall-clock (`/usr/bin/time -p`):

```
baseline run 1: 3.58 s
baseline run 2: 3.27 s
baseline run 3: 3.11 s
patched  run 1: 1.95 s
patched  run 2: 1.96 s
patched  run 3: 1.97 s
```
Mean: baseline 3.32 s, patched 1.96 s. **Δ −41%, 1.69× faster.** RTF 0.122 → 0.072.

Wall time includes ~0.5 s model load + binary startup; in-loop inference speedup is larger than 41%.

### Output WAV bit-exact verification

| Test | Baseline MD5 | Patched MD5 | `cmp` exit | sample-bit diff |
|---|---|---|---|---|
| `f32_composite.wav` | (matches) | (matches) | 0 | 0 |
| `f32_composite_48k.wav` | `d50c1588f34fb98b4c921d9cd7fa28a0` | `d50c1588f34fb98b4c921d9cd7fa28a0` | 0 | **0 / 1,309,932** |
| `f32_composite_48k.wav --pf` | `36527d375a4d40557d0158f4d21a2881` | `36527d375a4d40557d0158f4d21a2881` | 0 | **0 / 1,309,932** |

Every output PCM sample is bit-for-bit identical between baseline and patched, with and without post-filter.

---

## Variance / methodology notes

- All DFN inputs use T=100 frames (1 s of audio @ 48 kHz, 10 ms hop).
- Single-pair samples on a thermally-active laptop biased ~2-5%. Always **interleave** baseline/patched and use **multi-run + Welch's t-test** for confidence.
- A 2-run sequential sample on MODNet showed +3.7%; 6-run interleaved with 30s budget collapsed to +0.06% (t=0.03).
- Vocos showed an apparent +5.4% on sequential runs that flipped to apparent −5% on interleaved. Optimized graph is byte-identical → both deltas are pure thermal noise.
- For "fix doesn't fire" cases, the strongest evidence is `md5 <(baseline -O dump) <(patched -O dump)` matching — that proves the rewrite is a no-op on those graphs.
