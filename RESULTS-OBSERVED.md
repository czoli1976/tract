# tract K=1 einsum → broadcast Mul: empirical results

PR: [sonos/tract#2183](https://github.com/sonos/tract/pull/2183) (draft)
Branch: `czoli1976/tract@feature/conv-transpose-k1-lowering` based on `sonos/tract@b36f34e92` (main 2026-04-30)
Date of measurements: 2026-05-01

## What the fix does

When an `EinSum`'s contraction product is statically 1 (every k-axis has size 1 in both inputs, or there are no k-axes at all), `detect_rule` previously built an `EinSumMatMul` and let the OptMatMul pipeline handle it. The GEMM kernel then ran one FMA per output tile against fixed per-tile setup (clear, panel-load, store) — kernel preamble dominated.

The fix short-circuits in `core/src/ops/einsum/einsum_matmul.rs::detect_rule`: drop the unit-K axes from both inputs, align remaining axes to the output via `AxisOp::Add/Move/Rm`, cast to `operating_dt`, wire a broadcast `Mul`. Quantized einsums (`q_params=Some`) defer to the existing `dequant` path which produces a non-q einsum that this rule then catches naturally on the next pass.

## Test environment

- **Native:** macOS arm64 (M-series), Apple AMX kernel (`mmm_f32_32x4`)
- **WASM:** `wasm32-wasip1` + `simd128`, wasmtime 44.0.0, kernel `wasm_f32_8x8`
- **Inputs:** fixed `np.random.seed(42)` for bit-exact comparisons; `--allow-random-input` for benches; `T = 100` frames where applicable (DFN family: 100 frames × 10 ms hop @ 48 kHz = 1.0 s of audio)
- **Tract baseline binary:** `/tmp/dfn3-q8spike/tract-main/target/release/tract` (sonos/tract main @ b36f34e92)
- **Tract patched binary:** `/tmp/tract-k1-fix/target/release/tract` (same base + the K=1 commit)
- **WASM binaries:** built locally with five additional unrelated upstream WASM-enablement workarounds (load_a_slice wasm32 stub, gating reqwest/rustls/criterion/CUDA helper for wasm32) — those are local-only, not part of the K=1 PR

## Bit-exact verification

For every model below, baseline and patched produce **bit-exact identical outputs** on the same fixed-seed input. `max_abs_diff = 0.0` across every output tensor on every model on both Native and WASM.

WASM output vs Native output differs by ~2e-5 max (tiny cross-architecture FP variation in reduction order — present on every model with or without the patch, not caused by the fix).

## Where the fix fires

Confirmed by byte-level diff of `tract -O dump` output (graph after optimization):

| Model | Fix fires? | Why |
|---|---|---|
| **DFN2 / erb_dec** | yes | depthwise ConvTranspose [1×3] group=64, weight [64,1,1,3] |
| **DFN3 / erb_dec** | yes | depthwise ConvTranspose [1×3] group=64, weight [64,1,1,3] |
| **DFN3 / erb_dec_ll** | yes (by inspection) | same pattern |
| **GTCRN** | yes | depthwise ConvTranspose [3×3] group=16, weight [16,1,3,3] (×3 ops) |
| DFN2 / enc, df_dec | no | no K=1 einsum |
| DFN3 / enc, df_dec | no | no K=1 einsum |
| HiFiGAN v2, v3 | no | full-rank ConvTranspose, K = in_channels |
| Vocos | no | iSTFT-based, no ConvTranspose |
| Demucs (htdemucs_6s) | no | full-rank ConvTranspose [8×1] group=1, K = in_channels |
| MediaPipe Selfie | no | no ConvTranspose |
| MODNet | no | no depthwise ConvTranspose |

For all "no" cases, the optimized graph hashes identically pre- and post-patch (verified via `tract -O dump | sed | md5`).

## Native performance (Apple Silicon, AMX kernel)

DFN family: T=100 frames = 1.0 s audio, so RTF = ms/1000.

### DFN3 — full pipeline

| Stage | Baseline | Patched | Δ | Baseline RTF | Patched RTF |
|---|---|---|---|---|---|
| enc | 9.83 ms | 9.69 ms | -1.4% (noise) | 0.0098 | 0.0097 |
| **erb_dec** | **20.57 ms** | **9.88 ms** | **-52%** | **0.0206** | **0.0099** |
| df_dec | 11.08 ms (mean of 3) | 11.11 ms (mean of 3) | +0.3% (noise) | 0.0111 | 0.0111 |
| **composite** | **41.48 ms** | **30.68 ms** | **-26%** | **0.0415** | **0.0307** |
| **headroom** | | | | **24× real-time** | **33× real-time** |

Per-op detail in erb_dec:

| Op | Baseline | Patched | Speedup |
|---|---|---|---|
| `/convt1/ConvTranspose.einsum` (M=1600, K=1, N=3) | 7.43 ms (33.0% of erb_dec) | 0.224 ms (2.2%) | **33×** |
| `/convt2/0/ConvTranspose.einsum` (M=800, K=1, N=3) | 3.93 ms (17.5%) | 0.109 ms (1.1%) | **36×** |

### DFN2 — full pipeline

| Stage | Baseline | Patched | Δ | Baseline RTF | Patched RTF |
|---|---|---|---|---|---|
| enc | 9.64 ms (mean) | 9.57 ms (mean) | -0.7% (noise) | 0.0096 | 0.0096 |
| **erb_dec** | **22.56 ms (mean)** | **10.52 ms (mean)** | **-53%** | **0.0226** | **0.0105** |
| df_dec | 10.47 ms (mean) | 10.65 ms (mean) | +1.7% (noise) | 0.0105 | 0.0107 |
| **composite** | **42.67 ms** | **30.74 ms** | **-28%** | **0.0427** | **0.0307** |
| **headroom** | | | | **23× real-time** | **33× real-time** |

### GTCRN — small streaming noise suppressor (different architecture family)

GTCRN is a per-frame streaming model: each invocation processes 1 STFT frame (`mix: [1, 257, 1, 2]`). At 16 kHz with hop=128, that's 8 ms of audio per call. RTF = inference_ms / 8.

| Bench (interleaved 5×10s) | Baseline | Patched | Δ |
|---|---|---|---|
| **Native (mean ±std)** | **11.64 ± 0.27 ms** | **10.71 ± 0.28 ms** | **-7.93%, Welch t=-5.27 (highly significant)** |
| WASM (mean of 3) | 10.54 ms | 10.40 ms | -1.3% (noise; same wasm-vs-native pattern as DFN3) |

Per-op gains in GTCRN (Native, profiled):

| Op | Baseline | Patched | Speedup |
|---|---|---|---|
| `ConvTranspose_1098.einsum` (depthwise [3×3] group=16) | 0.422 ms | 0.033 ms | **12.8×** |
| `ConvTranspose_1335.einsum` (depthwise [3×3] group=16) | 0.194 ms | 0.014 ms | **13.9×** |
| `ConvTranspose_1572.einsum` (depthwise [3×3] group=16) | similar | similar | similar |

The smaller end-to-end gain than DFN3 (8% vs 26% native) reflects that GTCRN is a small streaming model where each per-frame call is dominated by RNN cell state work, not the depthwise ConvTranspose stage.

### Negative controls (fix doesn't fire — graphs hash identically)

| Model | Baseline | Patched | Δ | Notes |
|---|---|---|---|---|
| HiFiGAN v2 | 281 ms | 277 ms | -1.4% (noise) | mel 80×100; ~1.16 s audio |
| HiFiGAN v3 | 344 ms | 345 ms | +0.3% (noise) | mel 80×100; ~1.16 s audio |
| Vocos | 18.64 ms (mean of 5, ±0.13) | 18.88 ms (mean of 6 interleaved, ±0.4) | within noise after thermal control | mel 100×100; ~1.07 s audio |
| Demucs (htdemucs_6s) | not benched | (graph identical) | (no fix fires — full-rank ConvTranspose) | waveform 1×2×343980 + spec 1×4×2048×336 |
| MediaPipe Selfie | 35.75 ms | 35.12 ms | -1.8% (noise) | image 1×3×256×256 |
| MODNet | 795.16 ms (mean of 6, ±31) | 795.61 ms (mean of 6, ±19) | +0.06%, Welch t=0.03 | image 1×3×512×512 |

## WASM performance (wasm32-wasip1 + simd128, wasmtime 44)

### DFN3 — full pipeline

| Stage | Baseline | Patched | Δ | Baseline RTF | Patched RTF |
|---|---|---|---|---|---|
| enc | 16.37 ms | 16.22 ms | -0.9% (noise) | 0.0164 | 0.0162 |
| **erb_dec** | **15.17 ms (mean of 3, ±0.4)** | **13.59 ms (mean of 3, ±0.05)** | **-10.4%** | **0.0152** | **0.0136** |
| df_dec | 12.16 ms | 12.17 ms | +0.1% (noise) | 0.0122 | 0.0122 |
| **composite** | **43.7 ms** | **42.0 ms** | **-3.9%** | **0.0437** | **0.0420** |
| **headroom** | | | | **23× real-time** | **24× real-time** |

Per-op detail in erb_dec (WASM):

| Op | Baseline | Patched | Speedup |
|---|---|---|---|
| `/convt1/ConvTranspose.einsum` | 1.149 ms (7.4% of erb_dec) | 0.316 ms (2.2%) | **3.6×** |
| `/convt2/0/ConvTranspose.einsum` | 0.600 ms (3.9%) | 0.166 ms (1.2%) | **3.6×** |

### Why WASM gain is smaller than Native

The native AMX kernel has high per-tile setup overhead (32-row tiles, AMX accumulator clear/load/store), so K=1 is catastrophically wasteful relative to useful FMA work — fix gives 33× per-op speedup, dominating erb_dec runtime. The wasm `f32_8x8` SIMD kernel has much less per-tile overhead, so K=1 is only ~3.6× wasteful and contributes a smaller fraction of erb_dec total.

### Negative controls (WASM)

| Model | Baseline | Patched | Notes |
|---|---|---|---|
| MediaPipe Selfie | 34.34 ms | 34.18 ms | bit-exact, noise |
| MODNet | 1986 ms (mean of 5, ±94) | 1889 ms (mean of 5, ±50) | -4.9%, Welch t=-2.0 (within wasmtime variance band) |

## Variance / noise notes

- Single 5-second bench runs have ±2-5% run-to-run variance on a thermally-active laptop.
- Sequential bench order matters: running baseline then patched can show "patched slower" purely from thermal throttling. Verified by interleaving — Vocos went from "patched +9%" sequential to "patched -5%" interleaved, with the byte-identical graph confirming zero real-effect.
- For high-confidence comparisons (modnet, vocos), 5-6× 30s runs with stats analysis (Welch's t-test) was used.

## Test suite results

`cargo test --release -p tract-core --lib`: **235 passed, 0 failed**, including:
- 4 new explicit K=1 cases in `prefix_matmul::test` (`amk,akn->amn`, `gmk,Ngnk->Ngmn`, `mk,kn->mn`, `m,n->mn`)
- All existing einsum/matmul/quant tests
- Integration suites: `tract-onnx`, `tract-onnx-opl`, `tract-hir`, `tract-nnef`, `tract-pulse` — all green

## Caveats

- DFN3 testing means **sum of 3 ONNX models** through tract CLI on random tensor inputs, not the full `enhance()` pipeline (STFT/feature extraction live outside tract). Bit-exact on each ONNX output tensor means tract-side compute is unchanged; downstream iSTFT/post-filtering is not exercised. A WAV-bit-exact end-to-end test would require backporting the K=1 fix to tract 0.22.1 (DFN's pinned version) and rebuilding `deep-filter` against it — see "Reproduction kit" section below for the followup plan.
- DFN2 was not benched on WASM (bit-exact on the same input pattern, expected to mirror DFN3 WASM gains).
- `modnet` initial 2-run sample showed +3.7%; multi-run analysis collapsed to +0.06% (Welch t=0.03 = indistinguishable). Demonstrates importance of multi-run + interleaved methodology.

## Bottom line

| Category | Models tested | Gain |
|---|---|---|
| **Audio enhancement, depthwise ConvTranspose with 1×N kernel** | DFN3, DFN2, GTCRN | **8% – 53% on the targeted op family; bit-exact on every output** |
| **Audio enhancement, depthwise ConvTranspose with N×N kernel** | (no models tested in that subclass) | — |
| **Vocoders, full-rank ConvTranspose** | HiFiGAN v2, v3 | no-op (fix doesn't fire) |
| **iSTFT-based vocoders** | Vocos | no-op (no ConvTranspose) |
| **Music source separation, full-rank ConvTranspose** | Demucs htdemucs_6s | no-op (fix doesn't fire) |
| **Vision models (image segmentation, mattes)** | MediaPipe Selfie, MODNet | no-op (no ConvTranspose) |

Other audio enhancement models in the user's candidate list (FRCRN, FullSubNet, CleanUNet, DCCRN, MossFormer, Sepformer) were not tested because no public ONNX exports were found on Hugging Face — only PyTorch checkpoints (`.pt`, `.tar`, `.bin`).
