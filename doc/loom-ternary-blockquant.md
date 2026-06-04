# Transferable speed idea from Loom: ternary (BitNet b1.58) block-quant weights

This note records the result of studying the [openfluke/loom](https://github.com/openfluke/loom)
project (the "M-POLY-VTD" engine and its research write-ups at
<https://openfluke.com/loom/research>) for a technique that is both *transferable to
tract* and *plausibly a speed win*.

## TL;DR

Loom's headline differentiators (cross-platform bit-exact determinism, 21 numerical
dtypes, Plan 9 asm kernels, WebGPU tiling, target-prop training) are mostly either
already covered by tract or irrelevant to an inference engine's hot path. tract's
`linalg` is far more heavily optimized than Loom's for the kernels they share.

**The one genuinely transferable, speed-relevant idea is Loom's `bitnet_cpu` kernel:
2-bit-packed ternary weights `{-1, 0, +1}` with an integer, multiply-free dot product
against int8-quantized activations** — i.e. BitNet b1.58. tract has *both halves of the
machinery this needs already in tree* but does not have the format itself, making it an
unusually clean graft.

## What Loom actually does (poly/bitnet_cpu.go)

- Weights are packed 2 bits each, 16 weights per `uint32`. Encoding: `-1 → 0`,
  `0 → 1`, `+1 → 2`.
- Activations are quantized to int8 with max-abs scaling: `xq = round(x * 127 / amax)`.
- The dot product is computed with **no multiplies** — each 2-bit code selects
  `+xq`, `0`, or `-xq` and accumulates into an `int32`:
  `sum += xq[i] * (code - 1)`.
- Output is rescaled once per row: `out = sum * weightScale * amax / 127`.
- Row-parallel over the thread pool above a work threshold.

This is the standard BitNet b1.58 recipe: ternary weights, int8 activations, integer
accumulation, single scalar rescale.

## Why it maps cleanly onto tract

tract already contains the two hard pieces; only the format and the glue are missing.

1. **A pluggable packed-weight abstraction** — the `BlockQuant` trait in
   `linalg/src/frame/block_quant/mod.rs`, with `q4_0.rs` and `q8_1.rs` as existing
   implementations. Adding a ternary format is "implement one more trait impl" plus
   registration, exactly the shape of `Q4_0`.

2. **Integer matmul kernels** — tract ships i8→i32 accumulating kernels:
   - `linalg/arm64/arm64simd/arm64simd_mmm_i32_8x8.S.j2`
   - `linalg/x86_64/fma/avx2_mmm_i32_8x8.S.j2`
   These already do the int8×int8 → int32 GEMM that a ternary path wants on the
   activation side.

3. **Confirmed gap**: a search for `ternary` / `bitnet` / `1.58` / `2bit` across
   `core/` and `linalg/` finds nothing (the only near-hit is an unrelated `Q2_29`
   softmax constant). So this is additive, not a duplicate.

## Where the speed actually comes from

Two independent levers, both of which matter for batch-1 LLM decode (the regime tract
increasingly targets via its `transformers`/`causal_llm` work):

- **Memory bandwidth.** Decode is weight-bandwidth bound. Today tract's block-quant is
  *purely a storage win*: `q4_0`'s pack path
  (`linalg/src/frame/block_quant/q4_0.rs`, the panel loops around lines 89–160)
  **dequantizes each block into an fp16/fp32 kernel panel before the matmul** — compute
  is still float. Ternary at ~1.6 bits/weight vs `q4_0`'s 4.5 bits/weight (18 bytes /
  32 weights) vs f16's 16 bits is a ~2.8× smaller weight stream than q4 and ~10×
  smaller than f16, translating roughly proportionally into faster decode.

- **Compute.** Unlike the current dequant-to-float path, the ternary dot is
  add/subtract-only and can target the existing i32 kernels, sidestepping FMA
  throughput limits and the per-panel dequant work.

The first lever is the safe, large one; the second is upside if the ternary panel feeds
the i32 kernel directly rather than re-expanding to float.

## Suggested integration path (smallest first)

1. **`TernaryBlockQuant` impl** in `linalg/src/frame/block_quant/`, mirroring `Q4_0`:
   `block_len = 32`, 2-bit codes packed 16/`u32` plus one f16 scale per block
   (~8.25 bytes/block ⇒ ~2.06 bits/weight). Implement `quant_block_*` / `dequant_block_*`
   and the `pack`/panel hooks. This alone gives the **bandwidth win** for free by reusing
   the existing dequant-to-fp panel path — no new kernel, low risk.
2. **Activation int8 quant + i32 panel** to unlock the **compute win**: expand a ternary
   panel to int8 `{-1,0,+1}` (or a dedicated sign-select micro-kernel) and route to
   `*_mmm_i32_8x8`, with a single per-row `scale = wscale * amax / 127` applied in the
   existing fused output pipeline (`linalg/src/frame/mmm/fuse.rs`).
3. **Model plumbing**: a `BlockQuant` weight-transform entry so ONNX/NNEF/GGUF ternary
   weights are recognized, analogous to `core/src/ops/matmul/de_block_quant.rs`.

Step 1 is a self-contained, testable unit (round-trip quant/dequant + a reference GEMM
equality test, like the existing block_quant tests). Steps 2–3 are where the real
latency win lands and should be benched against `q4_0` with `linalg/matmul-bench`.

## What is *not* worth porting

- **Determinism guarantees / 21 dtypes / DNA-engine / target-prop** — training-side and
  product-positioning features, not inference-speed levers.
- **MC tiling with hardware-autodetected tile size** — tract already solves tile/kernel
  selection more rigorously via its measured `linalg/cost_model` and `hwbench`, and
  hand-tuned per-ISA kernels. Loom's auto-detect is coarser than what tract has.
- **Plan 9 asm / WebGPU paths** — tract's NEON/FMA/AVX-512 asm and metal/cuda/gpu
  backends already exceed Loom's coverage here.

## One-line conclusion

Port **BitNet-b1.58 ternary weights as a new `BlockQuant` format**; it is the only Loom
idea that is simultaneously missing from tract, a real speed lever (≈2.8× smaller weight
stream than q4_0, multiply-free dot), and a near-drop-in given tract's existing
`BlockQuant` trait and i8→i32 kernels.
