# Reproduction kit — `core/einsum`: lower K=1 contractions as broadcast Mul

Patch under review: [sonos/tract#2183](https://github.com/sonos/tract/pull/2183)

This kit gives you everything to (a) apply the patch to your tract checkout, (b) run the unit + property tests, (c) verify bit-exact correctness on real models, (d) reproduce the perf measurements with proper interleaving so thermal noise doesn't bias the comparison.

## What the patch does

In `core/src/ops/einsum/einsum_matmul.rs::detect_rule`, when an EinSum's contraction product is statically 1 (every k-axis size 1 in both inputs, or no k-axis), short-circuit to a broadcast `Mul` instead of going through `EinSumMatMul → OptMatMul`. The matmul kernel was running one FMA per output tile against fixed per-tile setup (clear, panel-load, store), so kernel preamble dominated runtime when there was no real contraction work to do.

Trigger: depthwise ConvTranspose with kernel having a 1-dim — common in audio enhancement decoders (DFN family, GTCRN, etc.). Hadamard / outer products in einsum form would also trigger via the existing `inject_k_axis` path catching itself on the next codegen pass.

## Files

| File | Purpose |
|---|---|
| `0001-core-einsum-lower-K-1-contractions-as-broadcast-Mul.patch` | The single commit; apply with `git am` against tract main @ b36f34e92 (or rebase) |
| `scripts/00_apply_patch.sh` | Set up baseline + patched checkouts, build both CLI binaries |
| `scripts/01_test_unit.sh` | Run einsum + matmul + quant unit tests on the patched build |
| `scripts/02_test_correctness_gtcrn.sh` | Bit-exact verification on GTCRN (small public ONNX, 344 KB) |
| `scripts/03_test_correctness_dfn3.sh` | Bit-exact verification on DFN3 (you provide ONNX) |
| `scripts/04_bench_native.sh` | Interleaved native bench with stats analysis |
| `scripts/05_diff_optimized_graph.sh` | Show the einsum→Mul rewrite in action via `tract -O dump` diff |
| `scripts/lib_compare.py` | Helper: bit-exact tensor comparison via numpy |
| `scripts/lib_stats.py` | Helper: Welch's t-test on bench samples |
| `models/get_gtcrn.sh` | Download GTCRN ONNX from its public GitHub repo |
| `RESULTS-OBSERVED.md` | The observed results from my own runs on Apple M-series + WASM (wasmtime 44) |

## Quick path (≈5 min compute, no external models needed)

```bash
# 1. Apply patch and build both binaries (assumes you have a tract checkout next door)
TRACT_BASELINE=/path/to/your/tract bash scripts/00_apply_patch.sh

# 2. Run unit tests
bash scripts/01_test_unit.sh

# 3. Bit-exact check + bench on GTCRN (downloads 344 KB model)
bash models/get_gtcrn.sh
bash scripts/02_test_correctness_gtcrn.sh
bash scripts/04_bench_native.sh gtcrn
```

Expected unit-test result: `235 passed, 0 failed` in tract-core lib (235 = 231 pre-existing + 4 new K=1 cases I added in `prefix_matmul::test`).

Expected correctness result: `max_abs_diff = 0.00e+00` on every output tensor of GTCRN.

Expected bench result: GTCRN native ~−8% with `Welch t < −5` (highly significant). The per-op `ConvTranspose_*.einsum` lines should drop from ~0.4 ms to ~0.03 ms (≈13× speedup).

## Full reproduction (DFN3, ≈15 min compute, requires DFN3 ONNX)

DFN3 ONNX models are at https://github.com/Rikorose/DeepFilterNet/releases (`DeepFilterNet3_onnx.tar.gz`). Extract, then:

```bash
DFN3_DIR=/path/to/extracted/DFN3 bash scripts/03_test_correctness_dfn3.sh
DFN3_DIR=/path/to/extracted/DFN3 bash scripts/04_bench_native.sh dfn3
```

Expected: bit-exact on enc, erb_dec, df_dec; erb_dec speedup ≈ 50% on M-series macOS (AMX kernel), ≈ 10% on x86/Linux (AVX2/AVX-512 kernel), ≈ 10% on WASM (8x8 SIMD kernel).

## Methodology notes

- **Use interleaved bench order** (`04_bench_native.sh` does this). Sequential "all baseline then all patched" runs are biased by thermal throttling — the patched runs come second and run hotter, making the patched binary look slower than it is. Verified by running Vocos sequentially (showed +9% "regression") then interleaved (showed within-noise) — the optimized graph is byte-identical, so any apparent delta is pure thermal.
- **Use multi-run + Welch's t-test** for confidence. A 2-run sample on MODNet showed +3.7% — multi-run analysis collapsed it to +0.06% with t=0.03 (indistinguishable from noise).
- **Verify "fix didn't fire" hypothesis** via `scripts/05_diff_optimized_graph.sh model.onnx`. If diff is empty, the optimized graph is identical and any perf delta must be noise.

## What's NOT in this kit (yet)

- **WAV-bit-exact end-to-end through the full DFN3 enhance() pipeline.** That requires backporting the K=1 fix to tract 0.22.1 (DFN's pinned version) and rebuilding the `deep-filter` Rust binary against it, then comparing baseline-WAV vs patched-WAV byte-by-byte. The fix is straightforward to backport (the einsum module exists in similar form in 0.22.1) — separate workstream.
- **Quantized matmul tests on real models.** The fix has a `q_params.is_none()` guard so quantized einsums aren't touched; the existing dequant path produces a non-q einsum that this rule then catches naturally on the next pass. This is exercised by `ops::matmul::quant::test::*` (passes) and `ops::quant::scale::*` (passes) but not yet end-to-end on a quantized model.

## Contact

For PR review questions: comment on https://github.com/sonos/tract/pull/2183. The author (czoli1976) is reachable via the PR.
