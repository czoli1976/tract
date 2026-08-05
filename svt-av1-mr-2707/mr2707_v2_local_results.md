# MR 2707 v2 — local verification results

Setup: v2 = rebased patch on master `13438c1`; base = master `13438c1`. AOM b2_scc
originals (Debugging, Wikipedia, Slides1, Spreadsheet — all 130 frames, full clips),
1080p, preset 8, `--rtc 1 --scm 1 --lp 1`, SIMD Release builds, 4-core Linux container.

## Correctness / scoping (all pass)

| check | result |
|---|---|
| `--scm 0` (Wikipedia, q31): v2 vs base | byte-identical |
| Frozen screen (Wikipedia frame 0 × 60): v2 vs base | byte-identical |
| Wikipedia static intro (12 inter frames before first motion) | byte-identical to base, per IVF payload |
| Keyframes, `--keyint 60` (frames 0/60/120, Debugging q31) | byte-identical payloads |
| First keyframe across all 16 matrix pairs | byte-identical |
| Determinism: same cmd twice; `--lp 1` vs `--lp 4` | all three streams md5-identical |

The static-intro and frozen-screen checks exercise the new MD-config idle gate on
real clip content: idle inter frames now signal exactly as baseline (palette level
dropped and `allow_screen_content_tools` re-derived).

## CQP (`--rc 0 --cqp {23,31,43,55}`, Y-PSNR, 130 frames)

| clip | BD-rate v2 vs base |
|---|---|
| Debugging | **−16.2%** |
| Wikipedia | **−17.1%** |
| Slides1 | **−2.0%** |
| Spreadsheet | **+1.1%** |

dPSNR at matched QP is ≥ 0 in all 16 points (+0.06 to +1.66 dB).

Per-QP detail:

| clip | qp | kbps base | kbps v2 | dSize | PSNR-Y base | PSNR-Y v2 | dPSNR |
|---|---|---|---|---|---|---|---|
| debugging | 23 | 1828.2 | 1538.4 | −15.9% | 54.98 | 55.14 | +0.16 |
| debugging | 31 | 1408.1 | 1217.7 | −13.5% | 50.97 | 51.59 | +0.62 |
| debugging | 43 | 838.9 | 777.1 | −7.4% | 44.20 | 45.39 | +1.19 |
| debugging | 55 | 426.0 | 459.6 | +7.9% | 38.15 | 39.81 | +1.66 |
| wikipedia | 23 | 2673.1 | 2237.8 | −16.3% | 50.83 | 50.89 | +0.06 |
| wikipedia | 31 | 1886.4 | 1564.0 | −17.1% | 48.09 | 48.28 | +0.19 |
| wikipedia | 43 | 1099.5 | 983.8 | −10.5% | 43.45 | 44.48 | +1.03 |
| wikipedia | 55 | 596.3 | 628.8 | +5.4% | 37.65 | 38.93 | +1.28 |
| slides1 | 23 | 3760.5 | 4022.0 | +7.0% | 52.33 | 52.59 | +0.26 |
| slides1 | 31 | 2515.1 | 2678.3 | +6.5% | 48.90 | 49.41 | +0.51 |
| slides1 | 43 | 1244.2 | 1340.3 | +7.7% | 43.41 | 44.24 | +0.83 |
| slides1 | 55 | 577.2 | 616.6 | +6.8% | 37.86 | 38.59 | +0.73 |
| spreadsheet | 23 | 5511.5 | 5476.6 | −0.6% | 48.64 | 48.71 | +0.07 |
| spreadsheet | 31 | 3827.0 | 3849.4 | +0.6% | 45.28 | 45.44 | +0.16 |
| spreadsheet | 43 | 2017.2 | 2234.4 | +10.8% | 39.86 | 40.29 | +0.43 |
| spreadsheet | 55 | 878.5 | 939.4 | +6.9% | 34.44 | 34.95 | +0.51 |

## CBR spot check (`--rc 2 --tbr {1000,2500}`)

| clip | tbr | kbps b/v2 | PSNR-Y base | PSNR-Y v2 | dPSNR |
|---|---|---|---|---|---|
| slides1 | 1000 | 883/902 | 37.74 | 38.60 | **+0.86** |
| slides1 | 2500 | 2191/2215 | 45.94 | 45.76 | **−0.18** |
| wikipedia | 1000 | 859/851 | 35.58 | 37.00 | +1.42 |
| wikipedia | 2500 | 1726/1617 | 44.53 | 45.37 | +0.84 |

The maintainer's "small loss on Slides1 under CBR" reproduces at the 2500 kbps
point; the 1000 kbps point gains.

## Encode time (2 reps, q31, wall-clock, noisy shared 4-core box — indicative only)

- Slides1: base 9722/9780 ms, v2 9873/10817 ms → **+2% to +11%**
- Wikipedia: base 6707/6727 ms, v2 6925/7079 ms → **+3% to +5%**

The maintainer's ~+15% on Slides1 is plausible; the MR text's blanket "+5–9%"
should be restated per-clip.

## Caveats vs the official refresh

- QP set {23, 31, 43, 55} chosen locally; MR's original QP list unknown.
- Spreadsheet +1.1% vs v1's reported +0.4%: consistent with the ≤0.5 pt effect
  the maintainer measured for removing the top-8-coverage gate, plus QP-set
  differences, but worth watching in the official numbers.
- No decoder-conformance run locally (no aomdec/dav1d in this container).
- Timing from a shared container; use a quiet machine for the published numbers.
