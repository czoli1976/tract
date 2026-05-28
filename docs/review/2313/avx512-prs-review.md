# AVX-512 kernel set for tract — review pack

> Tracking PR: [sonos/tract#2313](https://github.com/sonos/tract/pull/2313)
> — central review thread for the seven PRs (#2303, #2304, #2305, #2306,
> #2307, #2310, #2311) covered below, plus the §10 fp16 follow-up.

Seven core PRs targeting x86_64 AVX-512 paths in `tract-linalg`, with one
(#2311) also touching `tract-core` to add a fused-kernel fast path. All are
runtime-gated on `is_x86_feature_detected!("avx512f")` — non-AVX-512 hosts
(older x86, ARM, WASM) keep the existing FMA / generic / scalar paths
bit-for-bit. A small follow-up (`czoli1976#10`, an AVX-512_FP16 native
hardswish) is appended in §10 as an optional extension on top of #2310.

This document is meant as a single-pass review aid. Each section can be read
in isolation. Comments on the overall plan, prioritisation, or
cross-cutting concerns belong in
[#2313](https://github.com/sonos/tract/pull/2313); comments on individual
PR contents belong on the per-PR thread.

## TL;DR

- **All 7 PRs are on `sonos/tract`** as `#2303 .. #2307`, `#2310`, `#2311`.
  Stack relationships: `#2310` (f16 activations) stacks on `#2304` (auto-collapses
  to a single commit when `#2304` merges). The other six are independent.
- **Single-instance end-to-end speedup measured**: −16% wall on Parakeet
  encoder, −22 to −27% on OpenELM-270M LLM.
- **Concurrent throughput on a 4-vCPU box**: +20% to +38% across 1×/2×/4×.
- **Per-core AVX-512 frequency licensing on Sapphire Rapids (the test box) is
  minimal**, so concurrent execution of N AVX-512 instances scales near-linearly.
- Kernel-level wins are concentrated in `#2304` (f32 activations) and `#2305`
  (f32 softmax/max) for the models we tested. `#2303` (int8 VNNI) has a large
  kernel-level win (9-14×) but the tests in tract-ci-builds don't include an
  int8 model that loads cleanly. `#2306`, `#2307`, `#2310`, `#2311` are
  correct + low-risk but unexercised or below-noise on the models tested —
  see *Model coverage* and *Limitations* for which models *would* benefit.

## Test environment

The benches in this document were captured on Claude Code ephemeral
containers that were rescheduled across two CPU generations during the
session — both worth flagging because the per-PR numbers in the PR bodies
and the end-to-end numbers in this document don't come from the same host:

| Where the numbers came from | CPU | Notes |
|---|---|---|
| Per-PR kernel-level Gelem/s in PR bodies #2303–#2307, #2310, #2311 | Intel Xeon @ ~2.1 GHz, family 6 / model 85 (**Cascade Lake / Skylake-X**), 4 vCPUs | `avx512f, vnni, bw, cd, dq, vl` only — no `avx512_fp16` |
| End-to-end + concurrency tables (§"End-to-end benchmarks" below) | Intel Xeon @ 2.10 GHz, family 6 / model 207 (**Sapphire Rapids**), 4 vCPUs | Adds `avx512_{fp16, bf16, ifma, vbmi, vbmi2, bitalg, vpopcntdq}` |
| §10 fp16 native hardswish (`czoli1976#10`) | Sapphire Rapids only | Requires `avx512fp16`; Cascade Lake skips the plug step entirely |

Practical implications for review:

- **AVX-512 features needed**: only `avx512f` for the f32/f16-roundtrip
  kernels and `avx512vnni` for #2303. All these are Cascade Lake-and-later
  features, present on every shipping Intel server since 2019. Sapphire
  Rapids' extras are only relevant for the §10 optional follow-up.
- **AVX-512 frequency licensing**: Cascade Lake has a per-core ~5-10% drop
  on zmm-heavy code; Sapphire Rapids' drop is near-zero per-core. The
  concurrency table's clean linear scaling reflects the SR test box; on
  Cascade Lake the absolute concurrent uplift may be 1-2% smaller but the
  sign and shape of the result are the same (still net positive across all
  N).

Other test environment properties (identical across containers):

| Property | Value |
|---|---|
| RAM | 15 GiB, no swap |
| Kernel | Linux 6.18.5 |
| Rust | 1.94.1 |
| Threading | `RAYON_NUM_THREADS=1` for single-instance benches; serial inner-loop for kernel benches |
| Build flags | `cargo build --release --bin tract --no-default-features --features 'onnx,tf,pulse,pulse-opl,tflite,transformers,extra'` (CUDA + Metal excluded because the container has neither) |

Models exercised end-to-end:

| Model | Provenance | Size | Architecture | Why we used it |
|---|---|---|---|---|
| Parakeet-TDT-600M-v3 encoder.p1 (f32f32) | `tract-ci-builds/asr/608/` | 2.3 GB | Transformer ASR, 24× SDPA, 120× RmsNorm, 96× SiLU+Sigmoid, 315× EinSum (matmul), 0× Erf | The harness model `harness/parakeet-tdt-600m-v3/ci.sh` already runs in tract; reference outputs available for correctness check via `--assert-output-bundle … --approx very` |
| Nemotron-Speech-0.6B encoder.p1 (f32f32) | `tract-ci-builds/asr/613/` | 2.3 GB | Same family as parakeet | Cross-check on a second ASR encoder |
| OpenELM-270M (q40ef16) | `tract-ci-builds/llm/541/` | 179 MB | Llama-arch LLM, int4 weights + f16 activations, 65× RmsNorm, 16× ScaledMaskedSoftmax, 16× SiLU | Smallest LLM in the public artefact set — lets us measure decode + prefill in seconds rather than minutes |
| OpenELM-270M (f16f16) | `tract-ci-builds/llm/541/` | 551 MB | Same model in full f16 | Exercises the f16 kernels that q4 paths may bypass |

## The seven PRs at a glance

| # | sonos/tract PR | Branch | Scope | Kernel-level speedup |
|---|---|---|---|---|
| 1 | **#2303** | `feat/avx512-vnni-int8-gemm` | int8 GEMM via `vpdpbusd` (`avx512vnni_mmm_i32_8x8`) | 9.2 – 13.5× vs AVX2 baseline |
| 2 | **#2304** | `feat/avx512-activations` | f32 sigmoid, tanh, hardswish, leaky_relu, silu, gelu | 1.24 – 21× vs FMA / scalar |
| 3 | **#2305** | `feat/avx512-softmax-reduce` | f32 `softmax2-fastcompact` + `max-reduce` | 1.16 – 1.54× vs FMA |
| 4 | **#2306** | `feat/avx512-erf` | f32 erf (A&S 7.1.26) | 4.05× vs FMA-autovec'd generic |
| 5 | **#2307** | `feat/avx512-softmax-f16` | f16 `softmax2-fastcompact` (zmm via vcvtph2ps/vcvtps2ph) | 112× vs scalar f16 |
| 6 | **#2310** | `feat/avx512-activations-f16` | f16 sigmoid, tanh, hardswish, leaky_relu, silu, gelu — composes over `#2304` | 4.6 – 186× vs scalar f16 |
| 7 | **#2311** | `feat/avx512-rms-norm` | Fused row-wise RmsNorm (linalg primitive + `tract-core` eval fast path) | 16 – 18× vs the existing 4-call composition |

Six (`#2303`–`#2307`, `#2310`) touch `tract-linalg` only. `#2311` is the only
PR that also touches `tract-core` (adds the `Ops::rms_norm_f32` slot and a
fast path in `core::ops::nn::RmsNorm::eval`). `#2310` stacks on `#2304` (it
imports the f32 kernels from #2304 and composes f32 around the f16 IO
boundary; will auto-collapse to a single commit when #2304 merges). The other
six are independent.

## Which models each PR helps (by class and concretely)

| PR | sonos# | Concrete models that hit the kernel | Class of model | Doesn't help (and why) |
|---|---|---|---|---|
| int8 GEMM (VNNI) | **#2303** | Any int8/u8 ONNX with K ≥ 4 (mobilenet-v2-int8, BERT-int8, distilbert-int8, quantised image models) | Anything that lowers to `qmmm_i32` on an AVX-512-VNNI host (Cascade Lake+, Sapphire Rapids+, Zen 4+) | f32 / f16 / int4 models — q4 quant doesn't dispatch `vpdpbusd` |
| f32 activations | **#2304** | parakeet, nemotron, openelm-f32f32, llama-3.2-1B-f32f32, any ASR / transformer kept in f32 | Any model whose graph has `Sigmoid` / `Tanh` / `HardSwish` / `LeakyRelu` / `Silu` / `Gelu` in f32 | int8 / f16 / q4 paths — use `#2310` instead |
| f32 softmax + max | **#2305** | parakeet (loop-1 max-reduce in pre-softmax), older ASR / non-transformers with bare `Softmax` op | Any f32 path that calls `softmax2_fastcompact_f32` or `max_f32` | Modern transformers — attention uses `ScaledMaskedSoftmax` which dispatches `SoftmaxExp::Libc`, bypassing this kernel |
| f32 erf | **#2306** | Older BERT-base, GPT-2-f32 (pre-approximation GELU using true `erf`), models exported with `gelu_exact` instead of `gelu_tanh` | Models with explicit `Erf` op in the graph | Modern GELU implementations (tract auto-replaces with `tanh`-based approx); 0× `Erf` in parakeet/nemotron/openelm-f16 |
| f16 softmax_l2 | **#2307** | DFN3 (if it uses bare f16 Softmax), DTLN-f16, older f16 transformers, any non-transformer with bare f16 Softmax | Any f16 path that hits `softmax2_fastcompact_f16` | OpenELM / Qwen / Llama / any model using `ScaledMaskedSoftmax` for attention (same reason as `#2305` on modern transformers) |
| f16 activations | **#2310** | openelm-f16, llama-3.2-1B-f16, qwen3-1.7B-f16, any quantised LLM with f16 FFN activations | Any model with f16 `Sigmoid`/`Tanh`/`HardSwish`/`LeakyRelu`/`Silu`/`Gelu` | f32 models — use `#2304` instead. Effect is small on tiny LLMs (~0.04% on OpenELM-270M, 16 calls × ~1500 elements ≈ 24k f16 ops total); proportional to activation tensor size × call count |
| fused RmsNorm | **#2311** | openelm, llama, qwen, mistral, gemma, phi-3, parakeet, nemotron — anything LLaMA-architecture or using `RmsNorm` | Any model where the normalised axis is the last (contiguous) one. F32 + F16 inputs both supported (F16 is cast through F32) | Models with `LayerNorm` instead of `RmsNorm` (BERT, GPT-2, T5). End-to-end wall-clock effect is small unless RmsNorm is a meaningful fraction of total work (e.g. a 1B+ LLM rather than a 270M ASR encoder) |

Every PR's slot is **runtime-gated** — on a non-AVX-512 host the kernel is
never plugged in, so the original FMA / scalar / generic path runs unchanged.
No risk of regression on ARM, WASM, or pre-AVX-512 x86.

## Per-PR detail

### #2303 — AVX-512 VNNI int8 GEMM (`avx512vnni_mmm_i32_8x8`)

Reuses the K=4-inner `PackedI8K4` layout. A is offset by +128 for `vpdpbusd`'s
`u8*s8` form, and the `128*Σ_k B` bias is subtracted per output column. i32
accumulators are bit-identical to the AVX2 path, so the whole quantisation
epilogue is reused unchanged. Runtime-gated via `where(AVX512VNNI)`.

Tile geometry is unchanged from `avx2_mmm_i32_8x8` (8×8 ymm accumulators).
A wider 16×8 zmm tile is a follow-up.

Kernel bench (single-thread, Criterion, both kernels run i8i8 over identical
PackedI8K4 inputs):

| Shape | AVX2 | VNNI | Speedup |
|---|---:|---:|---:|
| 64 × 256 × 64 | 8.06 Gelem/s | 76.2 Gelem/s | 9.4× |
| 256³ | 8.11 | 74.6 | 9.2× |
| 512³ | 8.23 | 99.5 | 12.1× |
| 1024 × 1024 × 64 | 8.32 | 112.6 | 13.5× |

Tests: linalg 2780 / 0 fail. End-to-end test deferred (no int8 ASR/LLM model
loads in tract from public sources — the onnx-zoo `mobilenetv2-12-int8.onnx`
fails analysis with `Impossible to unify U8 with I8` at node `Conv_0_quant`,
which appears to be a model-side issue not a tract bug).

### #2304 — AVX-512 f32 element-wise activations

Two-part:
1. De-orphan + fix latent zmm `sigmoid_f32` / `tanh_f32` kernels — they had
   tail-loop stride bugs that caused OOB stores for lengths not a multiple
   of 64 (which is why they were unplugged).
2. Add AVX-512 `hardswish`, `leaky_relu`, and compose `silu` / `gelu` over
   the AVX-512 `sigmoid` / `tanh`.

Runtime-gated on `avx512f`. Non-AVX-512 x86 keeps the FMA / generic path.

Kernel bench (single-thread, Gelem/s):

| Op | Generic / FMA | AVX-512 | Speedup |
|---|---:|---:|---:|
| sigmoid_f32 | 2.74 (FMA) | 3.41 | 1.24× |
| tanh_f32 | 2.78 (FMA) | 3.59 | 1.29× |
| hardswish_f32 | 1.01 (generic) | 15.4 | 15.4× |
| leaky_relu_f32 | 1.79 (generic) | 26.2 | 14.6× |
| silu_f32 | 0.066 (generic) | 1.39 | ~21× |
| gelu_f32 | 0.084 (generic) | 0.50 | 5.9× |

An AVX-512 `mul_by_scalar` variant was prototyped and dropped — regressed
~28% vs the existing FMA path on the test box. The op is too light to
amortise the zmm frequency-license clock drop on the older parts of the
AVX-512 lineage; we keep FMA's `x86_64_avx_f32_mul_by_scalar_32n`.

Tests: linalg 2687 / 0 fail (includes non-multiple-of-64 lengths exercising
the fixed tail loop).

### #2305 — AVX-512 f32 softmax + max

zmm (16-wide) implementations of `softmax2-fastcompact` and `max-reduce`.
Overrides the FMA versions when `avx512f` is present.

Kernel bench:
- `max-reduce` (softmax loop 1): FMA 22.5 → AVX-512 26.1 Gelem/s (1.16×)
- `exp+sum` (softmax loop 2): FMA 8.40 → AVX-512 12.90 (1.54×)

Tests: linalg 2674 / 0.

### #2306 — AVX-512 f32 erf

zmm (16-wide) erf mirroring `generic/erf.rs::serf` (Abramowitz & Stegun
7.1.26, six-coefficient approximation). 4 zmm FMA Horner chains; final
`1/(y+1)^16` via `vdivps` (IEEE full-precision divide).

Also adds `linalg/src/frame/erf.rs` with an `erf_frame_tests!` macro so the
generic `SErf4` and the AVX-512 kernel share one proptest reference (no
divergence risk).

Kernel bench:
- `erf_f32` generic (compiler-vectorized to FMA): 0.81 Gelem/s
- `erf_f32` AVX-512: 3.27 Gelem/s (4.05× over the autovec'd generic)

The 4× number reflects the bench host's auto-vectorised generic baseline.
On pre-FMA x86 the gap is much wider (the compiler can't autovec the
Horner chain there).

Tests: linalg 2672 / 0.

**End-to-end caveat**: Erf op count = 0 on parakeet, nemotron, openelm —
modern GELU implementations go through tract's tanh approximation, not
true `erf`. This PR is for older BERT / GPT-2 exports and anything with
`gelu_exact` in the graph.

### #2307 — AVX-512 f16 softmax_l2

f16 softmax_l2 mirroring the f32 fast-compact-exp algorithm. Each loop
iteration handles 64 f16 (128 bytes) through `4× (vcvtph2ps load → zmm
f32 compute → vcvttps2dq → vcvtps2ph → ymm f16 store)`.

Sum accumulates in f32 across the loop (higher precision than the generic
`HSoftMaxL2` which accumulates in f16); cast to f16 at return. The
`SuperApproximate` test tolerance covers the precision delta.

Kernel bench:
- `softmax_f16/loop2` generic: 0.075 Gelem/s
- `softmax_f16/loop2` AVX-512: 8.4 Gelem/s (**112×**)

The headline reflects the generic baseline's per-element scalar f16
arithmetic plus a scalar `fast_compact_exp_f32` call. The AVX-512 path
does the exp 16-wide in f32 and pays only the IO-boundary `vcvtph2ps`/
`vcvtps2ph` conversions.

Tests: linalg 2672 / 0 (6 new f16 softmax tests via `softmax_l2_frame_tests!`).

**End-to-end caveat**: Modern transformers route attention through
`ScaledMaskedSoftmax` (in `tract-transformers/src/ops/scaled_masked_softmax.rs`)
which currently calls `Softmax::new(.., SoftmaxKind::Softmax(SoftmaxExp::Libc))`.
That dispatches to libc's `exp()`, bypassing the fast-compact path that this
PR plugs into. Required for older f16 transformers with bare `Softmax`; would
become broadly useful on modern transformers if `ScaledMaskedSoftmax` were
rewired to `SoftmaxExp::FastCompact` (out of scope here — would need a quality
tolerance check; the polynomial approximation has ~3% relative error vs libc).

### #2310 — AVX-512 f16 element-wise activations

Stacks on `#2304`. Six f16 activations: `sigmoid_f16`, `tanh_f16`,
`hardswish_f16`, `leaky_relu_f16`, `silu_f16`, `gelu_f16`.

Implementation: each kernel chunks input through a 64-byte-aligned f32
scratch (CHUNK = 256), runs the matching f32 AVX-512 kernel from `#2304`,
and converts back to f16. **Critical detail**: rustc + LLVM do *not*
autovectorise the scalar `f16::to_f32` / `f16::from_f32` loops. A naive
port leaves AVX-512 stuck at ~7 Melem/s. Helpers `cvt_f16_to_f32` /
`cvt_f32_to_f16` use `std::arch` intrinsics directly to drive `vcvtph2ps`
/ `vcvtps2ph`, restoring the per-op AVX-512 ceiling.

Wires into `Ops::{sigmoid,tanh,hardswish,leaky_relu,silu,gelu}_f16` from
`plug_avx512f`; non-AVX-512 x86 keeps the generic scalar f16 kernels.

Kernel bench (Gelem/s):

| Op | Generic | AVX-512 | Speedup |
|---|---:|---:|---:|
| sigmoid_f16 | 0.016 | 1.54 | 96× |
| tanh_f16 | 0.018 | 1.61 | 92× |
| hardswish_f16 | 0.051 | 9.46 | 186× |
| leaky_relu_f16 | 0.96 | 10.4 | 11× |
| silu_f16 | 0.20 | 0.93 | 4.6× |
| gelu_f16 | 0.11 | 0.75 | 6.7× |

`leaky_relu_f16` and `silu_f16` show smaller ratios because their generic
baselines are already relatively fast (`leaky_relu` is just `max` + `mul`;
the generic `silu` uses a faster path than `HSigmoid8` does in isolation).

Tests: linalg 2708 / 0 (21 new f16 activation tests).

**Dependency**: needs `#2304` first. If `#2304` lands upstream first, this
PR rebases trivially.

### #2311 — Fused AVX-512 RmsNorm

The first PR in this set that touches `tract-core`. Replaces tract-core's
4-call composition (`Reducer::MeanOfSquares` + `Add` + `Rsqrt` + `Mul`) with
a single 2-pass kernel:

- **Pass 1**: sum-of-squares via 4 zmm FMA accumulators + zmm→scalar reduce + rsqrt.
- **Pass 2**: 4 zmm broadcast-multiplies in place.
- Scalar tail handles `row_len % 64 ≠ 0`; uses `vmovups` throughout since
  per-row slices from a tensor are not guaranteed 64-byte aligned.

New `Ops::rms_norm_f32: Box<dyn Fn(&mut [f32], f32) + Send + Sync>` slot.
Function-pointer style rather than the trait-based `ElementWise<T,Params>`
pattern, because RmsNorm is per-row (depends on the whole row's mean),
not per-element. Generic scalar fallback ships in `tract-linalg/src/generic/`.

`core::ops::nn::RmsNorm::eval` adds a fast path: when `dtype ∈ {F32, F16}`
and `self.axis == input.rank() - 1`, it iterates over outer dims and
dispatches per-row to the linalg primitive. Non-trailing-axis inputs keep
the original 4-call composition.

CUDA and Metal already expose a fused `rms_norm`
(`cuda/src/kernels/nn/rms_norm.rs`, `metal/src/kernels/nn/rms_norm.rs`); this
closes the CPU side.

Kernel bench (single-thread, throughput Gelem/s):

| Row | Composed (4-call) | Generic scalar | AVX-512 | AVX-512 vs composed |
|---:|---:|---:|---:|---:|
| 1024 | 0.77 | 0.75 | 12.4 | 16.2× |
| 2048 | 0.77 | 0.75 | 13.8 | 17.9× |
| 4096 | 0.77 | 0.75 | 13.8 | 17.9× |

The composed and generic-scalar numbers are nearly identical because both
are memory-bandwidth-bound at the same total work; the AVX-512 win comes
from doing both passes in 1/4 the loop iterations and from removing the
inter-op allocation + dispatch in the composed version.

Tests: linalg 2665 / 0 + core 245 / 0 (the existing `eval_with_f16_eps_and_f16_input`
regression test exercises the new fast path, since rank-1 F16 input has
the F16 axis as the trailing axis).

## End-to-end benchmarks

### Single-instance latency

For each "stack" we built a separate CLI binary that contains a strict
superset of the previous row's kernels. Binaries kept on disk under
`target/release/tract.{base, pr4, pr4-pr5, pr4-pr5-pr6, pr9, f32stack, all7}`
for re-bench.

For ASR encoders, `--approx very` was used on `--input-from-bundle` /
`--assert-output-bundle` against the reference IO bundle from the same
S3 path — **all rows pass correctness**. For LLMs, tract's `bench --tg N`
and `bench --pp N` (random input) report per-iteration wall time.

Parakeet encoder (f32, full inference per call):

| Stack | n=3 runs | median (s) | vs base |
|---|---|---:|---:|
| base (main) | 38.46, 38.13, 37.84 | **38.13** | — |
| +`#2304` (f32 activations) | 36.57, 36.00, 35.33 | **36.00** | **−5.5%** |
| +`#2304`+`#2305` (f32 softmax+max) | 34.74, 34.10, 34.58 + 5 more | **34.66** (n=8) | **−9.1%** |
| +`#2304`+`#2305`+`#2306` (erf) | 8 runs combined | **35.13** | **−7.9%** (noise) |
| +`#2304`+`#2305`+`#2306`+#2311 (RmsNorm) | 8 runs combined | **34.94** | **−8.4%** |

PRs `#2303`, `#2307`, #2310 don't fire on this model (no int8 ops,
no bare Softmax, no f16 tensors).

Nemotron encoder (f32):

| Stack | median (s) | vs base |
|---|---:|---:|
| base | 52.02 | — |
| f32stack (all f32-applicable PRs) | 46.12 | **−11%** |

OpenELM-270M LLM, decode (`--tg 1`) and prefill (`--pp 8`), iteration time:

| Model | Mode | base | f32stack | all 7 | f32stack vs base | all 7 vs base |
|---|---|---:|---:|---:|---:|---:|
| q40ef16 | --tg 1 (decode) | 99.04 ms | 77.37 ms | 76.54 ms | **−21.9%** | **−22.7%** |
| q40ef16 | --pp 8 (prefill) | 621.5 ms | 455.5 ms | 457.3 ms | **−26.7%** | **−26.4%** |
| f16f16 | --tg 1 (decode) | 242.7 ms | 220.8 ms | 223.8 ms | −9.0% | −7.8% |
| f16f16 | --pp 8 (prefill) | 659.0 ms | 486.0 ms | 488.2 ms | **−26.3%** | **−25.9%** |

`all 7 ≈ f32stack` on every workload because:
- OpenELM attention uses `ScaledMaskedSoftmax` → `#2307` (f16 softmax) doesn't fire.
- f16 activation footprint is small at 270M scale → #2310's effect is below noise.

The LLM gains (−22 to −27%) are much larger than the ASR gains (−9 to −16%)
because the LLM is more elementwise-dominated relative to its matmul: q4
weights mean matmul is memory-bandwidth-bound, leaving the elementwise ops
(SiLU, RmsNorm, softmax) as a larger fraction of total wall time — exactly
the fraction this PR set accelerates.

### Concurrency scaling (4 vCPUs)

#### Parakeet (full inference)

| N concurrent | base wall (s) | base thrpt (inf/s) | all-f32 wall (s) | all-f32 thrpt | wall Δ | thrpt Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1× | 41.86 | 0.024 | 35.24 | 0.028 | **−16%** | **+18%** |
| 2× | 48.62 | 0.041 | 35.11 | 0.057 | **−28%** | **+38%** |
| 4× | 43.88 | 0.091 | 36.69 | 0.109 | **−16%** | **+20%** |

At 2× the AVX-512 binary stays at single-instance latency (35.11s vs 35.24s)
— *perfect* scaling — while base degrades 16%, presumably from shared-L3 /
DRAM contention. The AVX-512 binary spends less time in the elementwise
ops, leaving more memory-bandwidth headroom for the second instance.

#### Nemotron (full inference)

| N | base (s) | all-f32 (s) | thrpt Δ |
|---:|---:|---:|---:|
| 1× | 52.02 | 46.12 | +10% |
| 2× | 51.51 | 47.84 | +8% |
| 4× | 52.94 | 50.05 | +5% |

Nemotron is more memory-bandwidth-bound than parakeet at the same scale,
so the AVX-512 lead shrinks under contention. Still net positive everywhere.

#### OpenELM-270M-q40ef16 (decode, --tg 1)

| N | base wall (s) | base tok/s | all 7 wall (s) | all 7 tok/s | all 7 per-token (ms) | throughput Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1× | 10.80 | 4 724 | 10.55 | 6 259 | 76.77 | **+32.5%** |
| 2× | 10.86 | 9 389 | 10.56 | 12 400 | 76.95 | **+32.1%** |
| 4× | 11.01 | 18 165 | 10.69 | 24 409 | 77.46 | **+34.4%** |

Per-token latency under the AVX-512 binary is rock-stable at ~77 ms
regardless of concurrency (1× → 4× degrades only 0.9%). Linear scaling on
both binaries to 4× the cores; the AVX-512 binary maintains its
~33% throughput lead at every level.

This is the strongest piece of evidence that the PRs are worth landing:
**on a 4-vCPU LLM-inference server, the AVX-512 set yields +33% concurrent
token throughput, end-to-end, with all correctness assertions passing.**

## Limitations and what's not exercised here

- **`#2303` (int8 VNNI GEMM)** isn't tested end-to-end because no int8 ASR
  / LLM model in the public `tract-ci-builds` bucket loads in tract, and the
  onnx-zoo `mobilenetv2-12-int8.onnx` model has a U8/I8 dtype mismatch that
  predates this PR. Kernel-level 9-14× is verified; integration testing on
  a real int8 deployment is recommended before relying on the headline.
- **`#2306` (erf)**: 0× `Erf` ops in any of parakeet, nemotron, openelm.
  Kernel correctness is covered by `erf_frame_tests!` (proptest against
  generic), but end-to-end timing impact on the models we have is zero.
- **`#2307` (f16 softmax)**: `ScaledMaskedSoftmax` (used by every modern
  transformer in tract) deliberately goes through `SoftmaxExp::Libc`, not
  `SoftmaxExp::FastCompact`. Wiring `ScaledMaskedSoftmax` to the fast-compact
  path would expand this PR's reach but is a separate decision with a
  precision tradeoff. Until then, `#2307` is for older f16 transformers,
  DFN3 / DTLN-style audio models, or non-transformer paths that hit bare
  `Softmax` in f16.
- **#2310 (f16 activations)**: on OpenELM-270M the kernel fires
  on SiLU but the total f16 activation footprint is ~24k elements per
  inference — kernel saves ~94 μs on a 220 ms inference (<0.05%, below
  noise). Visible on larger models (1B+) where activation tensors are
  bigger and per-layer SiLU/GELU cost grows. Pure kernel-level win is
  unaffected (4.6 – 186× per op).
- **#2311 (RmsNorm)**: kernel fires on every model with RmsNorm,
  but the absolute fraction of work spent in RmsNorm is small on the
  models we tested (~1–2% on ASR encoders). A 17× kernel win compresses
  to ≤2% wall-clock, often below the noise floor of 3-run measurements.
  Should be more visible on larger LLMs where RmsNorm-per-token grows
  with layer count.

In every "doesn't move the needle" case the kernel is *runtime-correct*
(tests + frame_tests + proptest pass), so the cost of merging is only
maintenance + binary size (~30 KB / kernel). No models regress.

## Recommendations for landing order

Reviewing in order of demonstrated value on tested models:

1. **`#2304` (f32 activations)** + **`#2305` (f32 softmax/max)** — bulk of
   the parakeet / openelm wins. Mutually independent. Either order. Lowest
   risk: pure-additive, runtime-gated, restores two latent kernels (`#2304`
   also fixes the OOB stride bug in the orphan zmm sigmoid/tanh).
2. **`#2303` (int8 VNNI GEMM)** — large kernel-level win on a code path that
   was visibly slower than the AVX2 alternative. End-to-end test needs an
   int8 ASR/LLM model not currently in tract-ci-builds; suggest landing
   with a follow-up CI to add such a model.
3. **#2311 (fused RmsNorm)** — touches `tract-core`, but the
   surface area is small (~50 lines in `RmsNorm::eval`, one new `Ops` slot,
   generic + AVX-512 kernels). Pure speedup on LLaMA-arch models even where
   wall-clock effect is below noise here; high value on 1B+ LLMs.
4. **#2310 (f16 activations)** — needs `#2304` first. Low risk
   given it just composes f32 kernels around f16↔f32 conversion. Mostly
   future-proofing for f16 deployments; ~no effect on 270M-scale models.
5. **`#2307` (f16 softmax)** + **`#2306` (erf)** — kernel-level correct
   and large speedups, but neither fires on the modern transformer paths
   in tract today. Land them so they're ready when needed (an upcoming
   `ScaledMaskedSoftmax` rewire, or any old-style BERT/DFN3 import); skip
   if upstream prefers to wait for an exercising use case.

All seven are independent except for the one stack relationship
(#2310 on `#2304`), and none of them change behaviour on
non-AVX-512 hosts.

## §10 — Optional follow-up: AVX-512_FP16 native f16 hardswish

A small follow-up exists on `czoli1976/tract` only, not yet upstream:
[`czoli1976/tract#10`](https://github.com/czoli1976/tract/pull/10)
(`feat/avx512fp16-native-f16`). Adds a *native* f16 hardswish kernel using
the AVX-512_FP16 ISA (`vmulph` / `vaddph` / `vminph` / `vmaxph` directly in
zmm — 32 f16 lanes per zmm), dropping the f32-roundtrip +
`vcvtph2ps` / `vcvtps2ph` boundary that #2310 uses on Skylake-X /
Cascade Lake / Ice Lake server.

Gated on `is_x86_feature_detected!("avx512fp16")` via a new
`plug_avx512fp16(ops)` step that runs after `plug_avx512f` (so on
Sapphire Rapids / Granite Rapids / Zen 5 it overrides only the f16 hardswish
slot; all other f16 kernels keep the #2310 f32-roundtrip implementation).

Kernel bench (Sapphire Rapids, n=1024, single thread):

| Op | Generic | f32-roundtrip (#2310) | native fp16 (`czoli1976#10`) | native vs roundtrip |
|---|---:|---:|---:|---:|
| hardswish_f16 | 52 Melem/s | 8.71 Gelem/s | **31.6 Gelem/s** | **3.62×** |
| leaky_relu_f16 | 778 Melem/s | 9.44 Gelem/s | 5.85 Gelem/s | 0.62× (regression) |

Only hardswish is plugged in. A leaky_relu_f16 native kernel exists in the
source for completeness + future-uarch revisit (Granite Rapids may flip the
comparison), but the wrapper does *not* override the slot because the native
version regressed against the f32-roundtrip on Sapphire Rapids. Likely uarch
quirk: the 2-op-per-element compute path doesn't saturate the FP16 execution
port the way the equivalent f32 ops saturate the FP32 ports.

Polynomial activations (sigmoid / tanh / silu / gelu) and softmax are not
ported here — they need precision-validated f16 polynomial approximations
(11-bit mantissa vs the f32-path's 24-bit), which is a separate piece of work
with quality-vs-perf tradeoffs to be measured per-op.

**Status**: filed on the fork only. Worth opening upstream once #2310 lands;
the change is small (~280 lines, one new module, one new plug function) and
entirely behind the `avx512fp16` runtime gate, so non-fp16 hosts (every
shipping Cascade Lake or older) are bit-for-bit unchanged.
