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

## Addendum: the L1/L2 cache claim

Loom advertises "good L1/L2 cache optimisations in the kernels." Reading the source,
the substance is **one** thing, and it is real and *not* present in tract:

- `poly/tile_detection.go` **reads the machine's actual L1/L2/L3 sizes at runtime**
  (Linux `/sys/devices/system/cpu/cpu0/cache/index*/size`, macOS `sysctl
  hw.l1dcachesize`, Windows `wmic ... L2CacheSize,L3CacheSize`; fallbacks 32K/256K/8M)
  and sizes the tile so the working set fits L1, dtype-aware
  (`tileSize = L1 / (headDim * 2 * bytesPerWeight)`, clamped 8–256, 16-aligned).

What it is *not*: the consuming kernel (`poly/dense.go`) is a plain **single-level
square tile** (default 32) with batch-innermost loops — no panel packing, no register
micro-kernel, no prefetch. So Loom's "cache optimisation" = *runtime cache-size-driven
tile sizing*, bolted onto an otherwise naive loop.

How tract compares:

- tract is the *opposite* trade-off and far more sophisticated on the kernel side:
  BLIS-style **packed panels** (`pack.rs`, `panel_extract.rs`, `PackedFormat`),
  hand-tuned fixed-size register micro-kernels per ISA (`DynKernel<MR, NR>`, e.g. 8x8),
  a `no_prefetch`/prefetch hook in `mmm/kernel.rs`, and a **measured `cost_model`** for
  kernel selection.
- **But** tract does **not** read hardware cache sizes anywhere (a grep for
  `l1d`/`l2cache`/`sysconf`/sysfs cache paths finds nothing), and its matmul driver
  (`run_with_scratch_space_row_outer` / `_col_outer` in `mmm/mod.rs`) is a **flat 2-D
  sweep over register tiles** — `for ia in 0..m/mr { for ib in 0..n/nr { ker } }` — with
  the micro-kernel reducing the **entire K dimension in one call**. There is **no
  KC/MC/NC cache-blocking loop nest** (the classic Goto/BLIS L2/L3 blocking).

The transferable, on-point idea (distinct from ternary above):

> **Cache-size-aware K-blocking (KC tiling) for the matmul driver.** Loom's only real
> contribution here is the *cache-size detection* half; tract has everything else but
> currently leaves L2 A-panel reuse on the table for large-K matmuls.

Concretely: with `K=4096, f32`, an `mr=8` A-panel is `8*4096*4 ≈ 128 KB` and the `nr`
B-panel likewise — both far exceed a 32 KB L1, so in the row-outer loop the A-panel is
evicted by B streaming and **re-fetched from L2/L3 for every one of the `n/nr` column
blocks**. Blocking K into slabs `KC` sized so `mr*KC` (and `nr*KC`) fit L1/L2 — with `KC`
chosen from a detected/benchmarked cache size — keeps the A sub-panel resident and reused
across the whole N sweep, the textbook GEMM win. tract already measures memory bandwidth
in `linalg/src/hwbench/bandwidth.rs`, so a one-time cache-size probe (sysfs/sysctl) or a
`hwbench`-style auto-tuned `KC` would fit existing infrastructure.

Honest caveats: (1) this mainly helps **large-K, large-N** shapes (prefill, big
FFN/projection GEMMs); batch-1 **decode** is weight-bandwidth bound and benefits more
from the ternary idea above. (2) tract's authors are GEMM specialists and may have
concluded KC-blocking isn't worth it for their target shapes (weights packed once,
reused across tokens) — so this must be **proven with `linalg/matmul-bench` on large-K
shapes before committing**, not assumed.

## Addendum 2: calibrating against Loom's published benchmarks

`poly/asm/README.md` gives the only hard numbers (arm64/Metal, May 2026), as
**Go-reference vs Loom-ASM** speedups:

| dtype   | single-core | multi-core |
|---------|-------------|------------|
| Uint4   | 2.29×       | **3.55×**  |
| Uint8   | 2.46×       | 3.19×      |
| Ternary | ~2.0–2.2×   | ~3.2×      |
| FP4     | ~2.0–2.2×   | ~3.2×      |
| Float32 | ~1.11×      | ~parity    |
| Float64 | 0.85×       | (slower)   |

Two things these numbers **do** tell us, and one big thing they **don't**:

- **Supports the ternary thesis (direction, not magnitude):** the low-bit/integer paths
  get by far the largest wins (3.2–3.55× MC) while float is at parity — i.e. the payoff
  in Loom is concentrated exactly where tract has no offering today. It confirms low-bit
  integer kernels are where the headroom is.
- **Do NOT translate to a tract multiplier:** these compare Loom-asm to *Loom's own naive
  Go*. tract has no slow scalar baseline — it is already hand-tuned asm with packing and a
  cost model. So "3.55×" is "how far Loom's asm is ahead of Loom's Go," **not** a speedup
  tract would see. The honest tract expectation for ternary is the **memory-bandwidth**
  win (smaller weight stream on bandwidth-bound decode) plus modest compute savings from
  add-only accumulation — single-digit-percent-to-low-x on decode, to be measured, not a
  3× claim.

Two important corrections to the body above, learned from the docs:

1. **Loom's CPU low-bit path is *not* bit-packed** — `poly/asm/README.md` states it keeps
   "one quant byte per weight in RAM (`[]uint8` from morph)"; bit-packing into `[]uint32`
   is **GPU-only**. So on CPU Loom spends 8 bits/weight for ternary and gets its speedup
   purely from the integer kernel, leaving the bandwidth win on the table. tract's
   `BlockQuant` packing (q4_0 already packs to nibbles) would let a tract ternary format
   **beat Loom on the CPU bandwidth axis**, not just match it — the opportunity is larger
   for tract than Loom's own CPU realization.
2. **Loom's "L2/L3 cache locality" is mesh-level, not GEMM-level** — `docs/dispatch.md`
   ties it to blocking the 3D layer grid into "4×4×4 cell groups," i.e. traversal-order
   locality across *layers/cells*, not KC/MC blocking inside a matmul micro-kernel. So
   Loom does **not** actually demonstrate kernel cache-blocking that tract lacks; the
   KC-blocking idea in Addendum 1 is a tract-internal opportunity, and Loom's only genuine
   contribution to it remains the *runtime cache-size detection*, not a kernel design.

Net: the ternary block-quant remains the strongest transferable idea; the cache angle is
real but smaller and must be benchmarked. Neither should be sold with Loom's 3× figure.

## Addendum 3: the GPU doc — nothing transferable, tract is already ahead

`docs/gpu.md` lists five things it claims are "relevant to multi-backend engines like
tract." Checked against tract's `metal/` crate, **all five are already implemented**, so
there is no GPU speed idea to port:

| Loom GPU "idea"                     | tract status | evidence |
|-------------------------------------|--------------|----------|
| Single-command-buffer submit (BeginFrame/FlushFrame) | **already does it** | ops encode via `dispatch_eval` into one shared, lazily-created command buffer (`MetalStream::command_buffer` `get_or_insert_with`, `metal/src/context.rs`); production op path `eval_with_session` → `dispatch_eval` never waits (`metal/src/ops/gemm.rs:96`); commit happens once at readback (`to_host`), not per op |
| Pipeline cache                      | **already does it** | `cache_pipelines` + `cache_libraries` HashMaps, and pipelines are **preloaded** at startup (`preload_pipelines`, `context.rs`) |
| Buffer / activation pool reuse      | **already does it** | `DeviceSessionHandler` memory arena with `memory_sizing_hints` (`metal/src/lib.rs` `prepare_with_options`), plus `retain_tensor`/`retained_tensors` lifetime management |
| Bind-group caching                  | n/a for Metal | Metal binds args directly on the encoder; this is a WGPU-specific concern |
| Quantized native-vs-FP32 dispatch split | **more rigorous in tract** | handled at graph-optimization time via `BlockQuant` + the transform/optimize pipeline, not a runtime per-layer flag |

The per-kernel `stream.wait_until_completed()` calls that look like per-op syncs live only
in the **single-op `eval()` convenience wrappers** (e.g. `metal/src/kernels/nn/softmax.rs`
`eval`), used by tests/isolated execution — the graph runtime uses the non-waiting
`dispatch_eval`. So tract's Metal path is already "one submit per forward," exactly Loom's
headline pattern.

The only GPU items Loom has that tract doesn't are **WebGPU/WASM browser targeting** and
**runtime WGSL string generation** — portability/reach features, not inference-speed
levers. Conclusion: skip the GPU angle entirely.

## What is *not* worth porting

- **Determinism guarantees / 21 dtypes / DNA-engine / target-prop** — training-side and
  product-positioning features, not inference-speed levers.
- **Loom's square-tile CPU loop / GPU WGPU tiling** — tract's packed-panel + register
  micro-kernel design already dominates this; porting the loop itself is pointless.
  (The *one* salvageable sub-part — runtime cache-size detection to drive **KC**
  blocking — is broken out as its own recommendation in the addendum above.)
- **Plan 9 asm / WebGPU paths** — tract's NEON/FMA/AVX-512 asm and metal/cuda/gpu
  backends already exceed Loom's coverage here.

## One-line conclusion

Port **BitNet-b1.58 ternary weights as a new `BlockQuant` format**; it is the only Loom
idea that is simultaneously missing from tract, a real speed lever (≈2.8× smaller weight
stream than q4_0, multiply-free dot), and a near-drop-in given tract's existing
`BlockQuant` trait and i8→i32 kernels.
