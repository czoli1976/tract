# Milestone 4 — Metal int4+Hadamard KV-attention kernel (DESIGN, UNVALIDATED)

> **Status: design + skeleton only.** This was authored without an on-GPU validation loop, so the
> kernel below is **not compiled or run**. It is the implementation plan + skeleton to be built and
> validated on an Apple GPU. The CPU reference it must match bit-for-(near)-bit is the **proven**
> `QuantKvInt4::attend` in `transformers/src/ops/kv_quant.rs` (tested, measured 1.13–1.50× over f32
> streaming, 7.5× smaller). Do not ship until the validation plan at the bottom passes.

## Why a dedicated kernel (recap)

The CPU path proved the algorithm. Banking the gains on the backend the app actually runs (Metal,
unified memory) needs the same scheme as a GPU kernel: the f16 baseline SDPA is well-optimized, so a
quantized kernel only wins if it (a) reads int4 from device memory and (b) is as well-vectorized.
The vendored MFA kernel from #2320 cannot be reused as-is — per #2321's note it uses `C` for both the
key-loop bound and the K address stride, and a quantized/strided K layout breaks that. This is an
owned `.metal` kernel.

## Data layout (matches the Rust cache)

Per (batch, kv_head), in the **Hadamard basis**, per-token symmetric int4:
- `k_packed[T][D/2]`, `k_scale[T]` (f32) — signed int4, 2 codes/byte (even→low nibble).
- `v_packed[T][D/2]`, `v_scale[T]` (f32).
- recent window (M5): newest `n_recent` tokens kept f16, scored exactly.
- `D` is a power of two (FWHT requirement); Qwen3-1.7B: D=128, 8 kv heads, 16 q heads.

## Kernel mapping

- **One threadgroup per (batch, q_head).** group_size = q_heads / kv_heads; kv_head = q_head/group.
- **Threads = D** (128) for the Q transform/quantize; reused as the token-parallel reduction team.
- Online (flash) softmax so we never store `T` scores in device memory.

```metal
// kv_int4_attention.metal  —  DESIGN DRAFT, NOT YET COMPILED/VALIDATED
#include <metal_stdlib>
using namespace metal;

// One threadgroup per (batch, q_head). THREADS == D (power of two, e.g. 128).
kernel void kv_int4_decode_attention(
    device const half*   q          [[buffer(0)]],   // [B, Hq, D]   (single decode query)
    device const uchar*  k_packed   [[buffer(1)]],   // [B, Hkv, T, D/2]
    device const float*  k_scale    [[buffer(2)]],   // [B, Hkv, T]
    device const uchar*  v_packed   [[buffer(3)]],
    device const float*  v_scale    [[buffer(4)]],
    device const half*   recent_kv  [[buffer(5)]],   // [B, Hkv, n_recent, 2, D] f16, exact
    constant uint4&      dims       [[buffer(6)]],   // (T, D, n_recent, group_size)
    constant float&      scale      [[buffer(7)]],
    device half*         out        [[buffer(8)]],   // [B, Hq, D]
    uint  tg   [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  nthreads [[threads_per_threadgroup]])
{
    const uint T = dims.x, D = dims.y, NR = dims.z, GROUP = dims.w;
    threadgroup float sh[256];          // FWHT / reductions (>= D)
    threadgroup int8_t qcodes[128];     // quantized rotated Q (int8)
    threadgroup float  qscale_sh;

    // 1) load Q row, FWHT in shared memory (butterfly, log2(D) stages, barrier per stage),
    //    reduce |max| -> qscale, quantize to int8 -> qcodes.  (each thread owns lane `lid`)
    //    [FWHT butterfly + max-reduction omitted in skeleton — see validation note]

    // 2) online-softmax pass over quantized tokens: each thread strides tokens t = lid, lid+nthreads…
    //    score_t = scale * qscale * k_scale[t] * dot_i8_i4(qcodes, k_packed[t]);
    //    maintain running (m, l) and an accumulator acc[D] in the Hadamard basis for P·V_int4.
    float m = -INFINITY, l = 0.0f;
    float acc[ /*D/threads tiling*/ 4 ] = {0};
    for (uint t = lid; t < T; t += nthreads) {
        int32_t dot = 0;
        device const uchar* kp = k_packed + (/*offset*/ 0) + t*(D/2);
        for (uint j = 0; j < D/2; ++j) {
            uchar b = kp[j];
            int lo = (int)(int8_t)(b << 4) >> 4;   // sign-extend low nibble
            int hi = (int)(int8_t)(b) >> 4;        // sign-extend high nibble
            dot += (int)qcodes[2*j]   * lo;
            dot += (int)qcodes[2*j+1] * hi;
        }
        float s = scale * qscale_sh * k_scale[t] * (float)dot;
        // online softmax update of (m, l) and rescale acc; unpack V_int4 into acc …
    }
    // 3) tree-reduce (m, l, acc) across threads; add the exact f16 recent-window contribution;
    //    inverse-FWHT acc (same butterfly); write out as half.
}
```

## Key implementation notes

1. **FWHT in threadgroup memory** — the radix-2 butterfly is `log2(D)` stages with a barrier each;
   for D=128 that's 7 stages on 128 threads. Same code path rotates Q (forward) and the output
   accumulator (inverse — H is its own inverse). This replaces the dense matmul; it is *the* reason
   the CPU small-T case went from 0.75× to 1.50×, and matters identically on GPU.
2. **Integer dot** — `dot_i8_i4` is the hot loop. Pack two int4 per byte as in Rust. Consider
   `simd_shuffle`/`dot()` on `char4`/`uchar4` once correctness is locked. i32 accumulate.
3. **Scales outside the reduction** — `scale * qscale * k_scale[t]` applied once per token, not per
   channel: the whole point of per-token + Hadamard (per-channel K could not do this).
4. **Recent window** — score `n_recent` f16 tokens exactly in the same online softmax, then their
   P·V adds directly to the un-rotated output (they are stored in the original basis).
5. **GQA** — `q_head/group` selects the kv head for the packed-buffer base offsets.

## Validation plan (must pass before shipping)

1. **Bit-faithfulness** — for random + planted-outlier (B,Hkv,T,D) cases, the kernel output must
   match `QuantKvInt4::attend` (the proven CPU path) to ≤ 1e-3 rel-dev. Dump the same fixtures from
   a Rust test, run both, diff. (FWHT ordering and nibble sign-extension are the likely first bugs.)
2. **End-to-end** — wire into the decode loop (Milestone 3) and confirm generated tokens match the
   f16 model within the int4 quality envelope on a real prompt.
3. **Perf** — measure decode tok/s vs the f16 SDPA at ctx = 512/2K/8K on M-series. Target: recover
   the bandwidth ceiling from the CPU bench (≈2× at 2K), since the GPU int4 read is ¼ the bytes.
   If it underperforms, profile for: shared-memory bank conflicts in the FWHT, un-coalesced
   `k_packed` reads, register spill on `acc`.

## Dependencies / ordering

- Needs **Milestone 3** (the op + decode-loop wiring) for the end-to-end + perf steps.
- Pairs with **#2320** (Metal SDPA infra: pipeline/buffer plumbing) and **#2348** (the int4·int8
  dot is the same primitive the W4A8 GEMV uses — share the unpack/dot helper if possible).
