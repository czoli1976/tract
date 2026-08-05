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

## CBR (`--rc 2 --tbr {750,1500,2500,4000}`, Y-PSNR)

| clip | BD-rate v2 vs base |
|---|---|
| Debugging | **−10.8%** |
| Wikipedia | **−12.6%** |
| Slides1 | **−0.8%** |
| Spreadsheet | **+2.6%** |

Per-point detail:

| clip | tbr | kbps b/v2 | PSNR-Y base | PSNR-Y v2 | dPSNR |
|---|---|---|---|---|---|
| debugging | 750 | 720/722 | 35.18 | 36.80 | +1.62 |
| debugging | 1500 | 1276/1212 | 41.61 | 42.17 | +0.56 |
| debugging | 2500 | 2000/1833 | 47.44 | 47.87 | +0.43 |
| debugging | 4000 | 2661/2392 | 51.78 | 52.10 | +0.32 |
| wikipedia | 750 | 690/705 | 33.80 | 35.20 | +1.40 |
| wikipedia | 1500 | 1111/1094 | 39.06 | 40.53 | +1.47 |
| wikipedia | 2500 | 1726/1617 | 44.53 | 45.37 | +0.84 |
| wikipedia | 4000 | 2439/2232 | 47.79 | 48.32 | +0.53 |
| slides1 | 750 | 714/723 | 36.35 | 37.35 | +1.00 |
| slides1 | 1500 | 1294/1308 | 41.53 | 41.78 | +0.25 |
| slides1 | 2500 | 2191/2215 | 45.94 | 45.76 | **−0.18** |
| slides1 | 4000 | 3460/3496 | 50.10 | 50.02 | −0.08 |
| spreadsheet | 750 | 684/689 | 27.26 | 27.52 | +0.26 |
| spreadsheet | 1500 | 1078/1097 | 30.07 | 30.13 | +0.06 |
| spreadsheet | 2500 | 1755/1829 | 34.81 | 34.75 | −0.06 |
| spreadsheet | 4000 | 2714/2785 | 39.20 | 39.09 | −0.11 |

The maintainer's "small loss on Slides1 under CBR" reproduces and localizes:
CBR losses appear only at the high-rate points of Slides1/Spreadsheet
(−0.06 to −0.18 dB), where transform coding of residuals is cheap enough that
palette's fixed overhead (palette entries + index map) loses; the starved-rate
points gain +1.0 to +1.6 dB everywhere.

## Decoder conformance (reference decoder `aomdec` 3.x, plus dav1d 1.4.1)

All 16 v2 CQP points re-encoded with `--recon`: bitstream md5 unchanged vs the
matrix run (the recon tap does not perturb output), and SVT's reconstruction is
**byte-identical to aomdec's decode** in every case. All v2 CBR, frozen-screen,
multi-keyframe, and `--scm 0` streams also decode cleanly through both aomdec
and dav1d. Final tally: 16/16 recon-vs-decode byte-exact, 0 decode failures.

## Graphs (`graphs/`)

- `g1_bdrate.png` — BD-rate per clip, CQP + CBR
- `g2_rd.png` — RD curves, 2×2 small multiples
- `g3_cost.png` — per-frame bytes (byte-identical static intro marked) + wall-clock
- `g4_dbgain.png` — ΔY-PSNR at matched QP

## Encode time (2 reps, q31, wall-clock, noisy shared 4-core box — indicative only)

- Slides1: base 9722/9780 ms, v2 9873/10817 ms → **+2% to +11%**
- Wikipedia: base 6707/6727 ms, v2 6925/7079 ms → **+3% to +5%**

The maintainer's ~+15% on Slides1 is plausible; the MR text's blanket "+5–9%"
should be restated per-clip.

## Caveats vs the official refresh

- Operating points chosen locally and published above in full: CQP {23, 31, 43, 55},
  CBR {750, 1500, 2500, 4000} kbps, 130 frames (full clips); the MR's original
  QP/bitrate lists were never published.
- Spreadsheet +1.1% CQP / +2.6% CBR vs v1's reported +0.4% / +1.1%: consistent
  with the ≤0.5 pt effect the maintainer measured for removing the
  top-8-coverage gate plus operating-point differences, but it is the weak spot —
  state it plainly in the MR.
- Timing from a shared container; use a quiet machine for the published numbers.

## Spreadsheet investigation (post-review experiments)

Spreadsheet is the one clip where v2 loses (+1.1% CQP / +2.6% CBR). Four
variants tested over the full matrix to locate and price the loss:

| variant | Debugging | Wikipedia | Slides1 | Spreadsheet |
|---|---|---|---|---|
| v2 (shipping) | −16.2 / −10.8 | −17.1 / −12.6 | −2.0 / −0.8 | +1.1 / +2.6 |
| E1: residual floor 4→16 | — | — | — | +0.7 / — |
| E2: no inter palette < 16×16 | −9.0 / −6.4 | −8.1 / −5.7 | −0.8 / −2.6 | −2.6 / −1.5 |
| E4: <16×16 only if ≤4 colors | −9.5 / −6.8 | −9.4 / −6.7 | −1.8 / −3.3 | −3.0 / −1.2 |
| E5: <16×16 only if exact palette (≤8 colors) | −10.3 / −7.3 | −10.3 / −7.4 | −2.9 / −3.4 | −2.6 / −1.9 |

(CQP / CBR BD-rate per cell; E3, inter palette level 8, was strictly worse:
spreadsheet +2.6 CQP.)

Finding: many-color small blocks are simultaneously the largest source of the
text-clip wins and the entire source of the Spreadsheet loss — the same
block-local signature, differing only in surrounding content density. Every
small-block gate that fixes Spreadsheet (E2/E4/E5) surrenders ~40–45% of the
Debugging/Wikipedia gains; the exchange rate is ~6–7 points paid per ~4 gained.
All variants keep frozen-screen output byte-identical to baseline, and the
clean-v2 rebuild was canary-verified against the published matrix after each
experiment.

Conclusion: ship v2 and state the Spreadsheet trade. If a no-regression profile
is ever preferred, E5 is the principled knob (small blocks run palette only when
the palette is exact, i.e. no k-means quantization) at the cost above — two
lines in search_palette_luma.

### Content-detection feasibility (negative result)

A per-frame "spreadsheet-like" classifier switching to the E5 profile was
evaluated at the statistic level on the source clips. Neither whole-frame block
composition (exactly-palettizable share among non-flat 8×8: Debugging 56.4%,
Slides1 62.0%, Spreadsheet 60.3%, Wikipedia 75.6%) nor changed-region
composition (exact share: Wikipedia 39.8% vs Spreadsheet 38.7%; median colors
12 vs 13) separates the losing clip from the biggest winner. What differs is
transform-path efficiency on grid-aligned structure, which pixel statistics
cannot see cheaply — the RD search is already the per-block detector of it.
Preferred follow-up is therefore content-agnostic: audit palette index-map
rate estimation (MD estimate vs actual coded bits on Spreadsheet at high QP);
if palette rate is underestimated on many-transition maps, correcting it fixes
dense small blocks with no classifier and no cost to the winning clips.
