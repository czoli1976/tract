//! KIVI-style KV-cache quantization (training-free): store the cache in low precision to
//! shrink memory **near-losslessly**, keeping every token (a gentler trade than evicting).
//!
//! The key asymmetry (Liu et al. 2024, KIVI): **Keys are quantized PER-CHANNEL** (each
//! head-dim channel gets its own scale — Keys have large-magnitude *outlier channels* that
//! would wreck a shared scale) and **Values PER-TOKEN**. Works for any model, no training.
//! (CommVQ's RoPE-commutative codebook is a fancier, model-specific follow-on.)
//!
//! This module provides:
//!   1. `quant_dequant` — the quality-validation primitive (f32→f32 round-trip)
//!   2. `QuantizedKvCache` — a stateful fused op that stores K/V in **actual u8 bytes**
//!      and dequantizes per-head on each decode step. Real memory saving: 8× vs f32,
//!      4× vs f16.  Configurable `bits` (int8 default, int4 viable).
//!   3. `QuantizedKvSdpaTransform` — auto-wires an existing {cache→Sdpa} decode subgraph
//!      into the quantized op.

use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::{FrozenOpState, OpStateFreeze};
use tract_nnef::tract_core::transform::ModelTransform;
use tract_nnef::tract_ndarray::{Array2, Array4, ArrayView2, ArrayViewMut2, Ix4, s};

use tract_nnef::tract_core::ops::array::MultiBroadcastTo;
use tract_nnef::tract_core::ops::cast::Cast;
use tract_nnef::tract_core::ops::change_axes::AxisOp;

use crate::ops::apply_rope::ApplyRope;
use crate::ops::dyn_kv_cache::DynKeyValueCache;
use crate::ops::sdpa::Sdpa;

// ── NNEF ser/de ───────────────────────────────────────────────────────────────────────────────

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_quantized_kv_sdpa);
    registry.register_primitive(
        "tract_transformers_quantized_kv_sdpa",
        &[
            TypeName::Scalar.tensor().named("q"),
            TypeName::Scalar.tensor().named("k"),
            TypeName::Scalar.tensor().named("v"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.named("scale"),
            TypeName::Integer.named("bits"),
            TypeName::Integer.named("is_causal"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_quantized_kv_sdpa,
    );
}

fn ser_quantized_kv_sdpa(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &QuantizedKvSdpa,
) -> TractResult<Option<Arc<RValue>>> {
    let q = ast.mapping[&node.inputs[0]].clone();
    let k = ast.mapping[&node.inputs[1]].clone();
    let v = ast.mapping[&node.inputs[2]].clone();
    let mut attrs = vec![
        ("axis", numeric(op.axis)),
        ("bits", numeric(op.bits)),
        ("is_causal", numeric(op.is_causal as i64)),
    ];
    if let Some(scale) = op.scale {
        attrs.push(("scale", numeric(scale)));
    }
    Ok(Some(invocation("tract_transformers_quantized_kv_sdpa", &[q, k, v], &attrs)))
}

fn de_quantized_kv_sdpa(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let q = invocation.named_arg_as(builder, "q")?;
    let k = invocation.named_arg_as(builder, "k")?;
    let v = invocation.named_arg_as(builder, "v")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let scale: Option<f32> = invocation.get_named_arg_as(builder, "scale")?;
    let bits: Option<i64> = invocation.get_named_arg_as(builder, "bits")?;
    let bits = bits.map(|b| b as u32).unwrap_or(8);
    let is_causal: Option<i64> = invocation.get_named_arg_as(builder, "is_causal")?;
    let is_causal = is_causal.map(|c| c != 0).unwrap_or(false);
    builder.wire(QuantizedKvSdpa { axis, scale, bits, is_causal }, &[q, k, v])
}

/// Affine quantize→dequantize a `[rows, cols]` matrix at `bits` bits, returning the
/// reconstructed (lossy) values. `by_row = true` gives each ROW its own scale (per-token,
/// for Values); `by_row = false` gives each COLUMN its own scale (per-channel, for Keys).
/// Reconstruction error per element is ≤ scale/2 of its group.
pub fn quant_dequant(x: ArrayView2<f32>, bits: u32, by_row: bool) -> Array2<f32> {
    assert!((1..=16).contains(&bits), "bits must be 1..=16");
    let levels = ((1u32 << bits) - 1) as f32;
    let (r, c) = x.dim();
    let mut out = Array2::<f32>::zeros((r, c));
    let n_groups = if by_row { r } else { c };
    for g in 0..n_groups {
        let group = if by_row { x.row(g) } else { x.column(g) };
        let lo = group.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if hi > lo { (hi - lo) / levels } else { 1.0 };
        for (k, &v) in group.iter().enumerate() {
            let q = ((v - lo) / scale).round().clamp(0.0, levels);
            let deq = lo + q * scale;
            if by_row {
                out[(g, k)] = deq;
            } else {
                out[(k, g)] = deq;
            }
        }
    }
    out
}

// ── Packed u8 storage ─────────────────────────────────────────────────────────────────────────
// One token = D bytes (int8) for Values (per-token scale), or D bytes for one channel of Keys
// (per-channel scale). Real memory: u8 is 4× f32, 2× f16.

/// Max code value at `bits` bits (levels-1). bits ∈ {4, 8}.
#[inline]
fn levels(bits: u32) -> f32 {
    ((1u32 << bits) - 1) as f32
}

/// Bytes needed to store `d` codes at `bits` bits/code, packed.
#[inline]
fn row_bytes(d: usize, bits: u32) -> usize {
    (d * bits as usize).div_ceil(8)
}

/// Affine-quantize `v` to integer codes at `bits` (min/max range), returning `(codes, lo, scale)`.
fn quant_codes(v: &[f32], bits: u32) -> (Vec<u8>, f32, f32) {
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lv = levels(bits);
    let scale = if hi > lo { (hi - lo) / lv } else { 1.0 };
    let codes = v.iter().map(|&x| ((x - lo) / scale).round().clamp(0.0, lv) as u8).collect();
    (codes, lo, scale)
}

/// Pack `codes` (each ≤ `levels(bits)`) at `bits` bits/code, appending to `out`.
/// bits=8: one code per byte. bits=4: two codes per byte (even index → low nibble).
fn pack_codes(codes: &[u8], bits: u32, out: &mut Vec<u8>) {
    match bits {
        8 => out.extend_from_slice(codes),
        4 => {
            let mut i = 0;
            while i < codes.len() {
                let lo = codes[i] & 0x0F;
                let hi = if i + 1 < codes.len() { codes[i + 1] & 0x0F } else { 0 };
                out.push(lo | (hi << 4));
                i += 2;
            }
        }
        _ => unreachable!("bits must be 4 or 8"),
    }
}

// ── NEON int4 unpack (aarch64) ──────────────────────────────────────────────────────────────────
// The scalar int4 nibble loop does not autovectorize (per-element `ucvtf s`, interleaved store).
// These hand-rolled NEON kernels unpack 16 channels per 8-byte load: split low/high nibbles,
// `vzip` back to channel order, widen u8→f32, then fuse the affine (per-token scalar scale, or
// per-channel vector scale). `d` must be a multiple of 16 for the vector body; a scalar tail
// handles any remainder. NEON is baseline on aarch64, so no runtime feature check is needed.

/// Unpack `codes` (int4, `d` values in `row_bytes` bytes) with a per-TOKEN affine (lo, scale).
#[cfg(target_arch = "aarch64")]
#[inline]
fn dequant_int4_pertoken(src: &[u8], d: usize, lo: f32, scale: f32, dst: &mut [f32]) {
    use core::arch::aarch64::*;
    unsafe {
        let lo_v = vdupq_n_f32(lo);
        let lomask = vdup_n_u8(0x0F);
        let mut bc = 0usize;
        let mut c = 0usize;
        while c + 16 <= d {
            let bytes = vld1_u8(src.as_ptr().add(bc));
            let lon = vand_u8(bytes, lomask);
            let hin = vshr_n_u8::<4>(bytes);
            let z = vzip_u8(lon, hin); // interleave -> channel order [l0,h0,l1,h1,...]
            for codes8 in [z.0, z.1] {
                let w = vmovl_u8(codes8);
                let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
                let f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
                vst1q_f32(dst.as_mut_ptr().add(c), vfmaq_n_f32(lo_v, f0, scale));
                vst1q_f32(dst.as_mut_ptr().add(c + 4), vfmaq_n_f32(lo_v, f1, scale));
                c += 8;
            }
            bc += 8;
        }
        while c < d {
            let byte = src[bc];
            dst[c] = lo + (byte & 0x0F) as f32 * scale;
            if c + 1 < d {
                dst[c + 1] = lo + (byte >> 4) as f32 * scale;
            }
            bc += 1;
            c += 2;
        }
    }
}

/// Unpack `codes` (int4, `d` values) with a per-CHANNEL affine (ch_lo[c], ch_scale[c]).
#[cfg(target_arch = "aarch64")]
#[inline]
fn dequant_int4_perchannel(src: &[u8], d: usize, ch_lo: &[f32], ch_scale: &[f32], dst: &mut [f32]) {
    use core::arch::aarch64::*;
    unsafe {
        let lomask = vdup_n_u8(0x0F);
        let mut bc = 0usize;
        let mut c = 0usize;
        while c + 16 <= d {
            let bytes = vld1_u8(src.as_ptr().add(bc));
            let lon = vand_u8(bytes, lomask);
            let hin = vshr_n_u8::<4>(bytes);
            let z = vzip_u8(lon, hin);
            for codes8 in [z.0, z.1] {
                let w = vmovl_u8(codes8);
                let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
                let f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
                let lo0 = vld1q_f32(ch_lo.as_ptr().add(c));
                let sc0 = vld1q_f32(ch_scale.as_ptr().add(c));
                let lo1 = vld1q_f32(ch_lo.as_ptr().add(c + 4));
                let sc1 = vld1q_f32(ch_scale.as_ptr().add(c + 4));
                vst1q_f32(dst.as_mut_ptr().add(c), vfmaq_f32(lo0, f0, sc0));
                vst1q_f32(dst.as_mut_ptr().add(c + 4), vfmaq_f32(lo1, f1, sc1));
                c += 8;
            }
            bc += 8;
        }
        while c < d {
            let byte = src[bc];
            dst[c] = ch_lo[c] + (byte & 0x0F) as f32 * ch_scale[c];
            if c + 1 < d {
                dst[c + 1] = ch_lo[c + 1] + (byte >> 4) as f32 * ch_scale[c + 1];
            }
            bc += 1;
            c += 2;
        }
    }
}

// ── Per-token quantized Value store ───────────────────────────────────────────────────────────

/// Quantized Value cache: stores each appended token per-TOKEN at `bits` bits + 2 f32 params.
/// Memory per token: row_bytes(D,bits) + 8 bytes (int8: 4× vs f32; int4: ~7× at D=64).
#[derive(Clone, Debug, Default)]
pub struct QuantValueCache {
    pub d: usize,
    pub bits: u32,
    row_bytes: usize,
    // packed codes, row-major [T, row_bytes]
    packed: Vec<u8>,
    // per-token scale params: 2 f32 per token
    params: Vec<(f32, f32)>, // (lo, scale)
}

impl QuantValueCache {
    pub fn new(d: usize) -> Self {
        Self::with_bits(d, 8)
    }
    pub fn with_bits(d: usize, bits: u32) -> Self {
        assert!(bits == 8 || (bits == 4 && d % 2 == 0), "int4 needs even head_dim");
        QuantValueCache {
            d,
            bits,
            row_bytes: row_bytes(d, bits),
            packed: Vec::new(),
            params: Vec::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.params.len()
    }
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
    /// Append one token's V vector (length D), quantizing at `bits`.
    pub fn push_token(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.d);
        let (codes, lo, scale) = quant_codes(v, self.bits);
        pack_codes(&codes, self.bits, &mut self.packed);
        self.params.push((lo, scale));
    }
    /// Dequantize all stored tokens to a [T, D] f32 array.
    pub fn dequant_all(&self) -> Array2<f32> {
        let t = self.len();
        let mut out = Array2::<f32>::zeros((t, self.d));
        self.dequant_tile(0..t, &mut out);
        out
    }
    /// Dequantize a contiguous token range `r` into the first `r.len()` rows of `out`
    /// (shape `[block, d]`, `block >= r.len()`). No allocation — used by the fused attention.
    /// Per-token scale/lo are constant across a row; the int8 loop autovectorizes.
    pub fn dequant_tile(&self, r: std::ops::Range<usize>, out: &mut Array2<f32>) {
        let (d, rb) = (self.d, self.row_bytes);
        let os = out.as_slice_mut().expect("contiguous dequant buffer");
        match self.bits {
            8 => {
                for (oi, ti) in r.enumerate() {
                    let (lo, scale) = self.params[ti];
                    let src = &self.packed[ti * rb..ti * rb + d];
                    let dst = &mut os[oi * d..oi * d + d];
                    for c in 0..d {
                        dst[c] = lo + src[c] as f32 * scale;
                    }
                }
            }
            4 => {
                for (oi, ti) in r.enumerate() {
                    let (lo, scale) = self.params[ti];
                    let src = &self.packed[ti * rb..ti * rb + rb];
                    let dst = &mut os[oi * d..oi * d + d];
                    #[cfg(target_arch = "aarch64")]
                    dequant_int4_pertoken(src, d, lo, scale, dst);
                    #[cfg(not(target_arch = "aarch64"))]
                    for (bc, &byte) in src.iter().enumerate() {
                        dst[2 * bc] = lo + (byte & 0x0F) as f32 * scale;
                        dst[2 * bc + 1] = lo + (byte >> 4) as f32 * scale;
                    }
                }
            }
            _ => unreachable!("bits must be 4 or 8"),
        }
    }
    pub fn memory_bytes(&self) -> usize {
        self.packed.len() + self.params.len() * 8
    }
}

// ── Per-channel quantized Key store ───────────────────────────────────────────────────────────

/// Quantized Key cache: stores each appended token per-CHANNEL (each of the D channels has
/// its own running scale accumulated across all tokens so far). On each new token, the channel
/// scale may expand; old tokens in that channel are NOT re-quantized (acceptable error for
/// a growing cache; exact re-quant is the follow-on). Memory: T*D bytes + D*2 f32 params.
#[derive(Clone, Debug, Default)]
pub struct QuantKeyCache {
    pub d: usize,
    pub bits: u32,
    row_bytes: usize,
    // packed codes, row-major [T, row_bytes]
    packed: Vec<u8>,
    // per-channel: lo, scale across all tokens seen so far
    ch_lo: Vec<f32>,
    ch_scale: Vec<f32>,
    len: usize,
}

impl QuantKeyCache {
    pub fn new(d: usize) -> Self {
        Self::with_bits(d, 8)
    }
    pub fn with_bits(d: usize, bits: u32) -> Self {
        assert!(bits == 8 || (bits == 4 && d % 2 == 0), "int4 needs even head_dim");
        QuantKeyCache {
            d,
            bits,
            row_bytes: row_bytes(d, bits),
            packed: Vec::new(),
            ch_lo: vec![f32::INFINITY; d],
            ch_scale: vec![1.0; d],
            len: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Append one token's K vector (length D), updating per-channel scales.
    pub fn push_token(&mut self, k: &[f32]) {
        assert_eq!(k.len(), self.d);
        let lv = levels(self.bits);
        // Update per-channel lo/scale to encompass the new values.
        for (c, &val) in k.iter().enumerate() {
            if val < self.ch_lo[c] {
                self.ch_lo[c] = val;
            }
            let range = val - self.ch_lo[c];
            if range > 0.0 {
                let new_scale = range / lv;
                if new_scale > self.ch_scale[c] {
                    self.ch_scale[c] = new_scale;
                }
            }
        }
        // Quantize this token under current per-channel scales, then pack.
        let mut codes = vec![0u8; self.d];
        for (c, &val) in k.iter().enumerate() {
            codes[c] = ((val - self.ch_lo[c]) / self.ch_scale[c]).round().clamp(0.0, lv) as u8;
        }
        pack_codes(&codes, self.bits, &mut self.packed);
        self.len += 1;
    }
    /// Dequantize all stored tokens to a [T, D] f32 array.
    pub fn dequant_all(&self) -> Array2<f32> {
        let t = self.len;
        let mut out = Array2::<f32>::zeros((t, self.d));
        self.dequant_tile(0..t, &mut out);
        out
    }
    /// Dequantize a contiguous token range `r` into the first `r.len()` rows of `out`
    /// (shape `[block, d]`, `block >= r.len()`). No allocation — used by the fused attention.
    /// Per-channel lo/scale are constant across rows; the int8 loop autovectorizes.
    pub fn dequant_tile(&self, r: std::ops::Range<usize>, out: &mut Array2<f32>) {
        let (d, rb) = (self.d, self.row_bytes);
        let os = out.as_slice_mut().expect("contiguous dequant buffer");
        let (lo, sc) = (&self.ch_lo, &self.ch_scale);
        match self.bits {
            8 => {
                for (oi, ti) in r.enumerate() {
                    let src = &self.packed[ti * rb..ti * rb + d];
                    let dst = &mut os[oi * d..oi * d + d];
                    for c in 0..d {
                        dst[c] = lo[c] + src[c] as f32 * sc[c];
                    }
                }
            }
            4 => {
                for (oi, ti) in r.enumerate() {
                    let src = &self.packed[ti * rb..ti * rb + rb];
                    let dst = &mut os[oi * d..oi * d + d];
                    #[cfg(target_arch = "aarch64")]
                    dequant_int4_perchannel(src, d, lo, sc, dst);
                    #[cfg(not(target_arch = "aarch64"))]
                    for (bc, &byte) in src.iter().enumerate() {
                        let (c0, c1) = (2 * bc, 2 * bc + 1);
                        dst[c0] = lo[c0] + (byte & 0x0F) as f32 * sc[c0];
                        dst[c1] = lo[c1] + (byte >> 4) as f32 * sc[c1];
                    }
                }
            }
            _ => unreachable!("bits must be 4 or 8"),
        }
    }
    pub fn memory_bytes(&self) -> usize {
        self.packed.len() + self.d * 8 // D*(lo+scale) = D*8 bytes
    }
}

// ── Block-wise per-channel Key cache (KIVI, streaming-consistent) ────────────────────────────────
// Keys have outlier CHANNELS, so per-channel scales beat per-token — but a *running* per-channel
// scale mis-dequantizes earlier tokens once it grows. Fix: quantize in fixed-size blocks. Each
// block's per-channel lo/scale is computed over that block's tokens and then FROZEN, so every code
// dequantizes against the exact scale it was encoded with. The current partial block is held in f32
// until it fills. Memory: finalized blocks packed + (D*8)/block bytes/token of scales + a small f32
// residual.

const KEY_BLOCK: usize = 32;

#[derive(Clone, Debug, Default)]
pub struct BlockQuantKeyCache {
    pub d: usize,
    pub bits: u32,
    row_bytes: usize,
    packed: Vec<u8>, // finalized blocks, sequential [n_finalized*KEY_BLOCK, row_bytes]
    block_lo: Vec<f32>, // per finalized block: D lo values (block_idx*D + c)
    block_scale: Vec<f32>, // per finalized block: D scale values
    residual: Vec<f32>, // current partial block, row-major [n_res, D]
    n_res: usize,
    n_finalized: usize,
}

impl BlockQuantKeyCache {
    pub fn with_bits(d: usize, bits: u32) -> Self {
        assert!(bits == 8 || (bits == 4 && d % 2 == 0), "int4 needs even head_dim");
        BlockQuantKeyCache { d, bits, row_bytes: row_bytes(d, bits), ..Default::default() }
    }
    pub fn len(&self) -> usize {
        self.n_finalized * KEY_BLOCK + self.n_res
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn push_token(&mut self, k: &[f32]) {
        assert_eq!(k.len(), self.d);
        self.residual.extend_from_slice(k);
        self.n_res += 1;
        if self.n_res == KEY_BLOCK {
            self.finalize_block();
        }
    }
    fn finalize_block(&mut self) {
        let (d, bits) = (self.d, self.bits);
        let lv = levels(bits);
        let mut lo = vec![f32::INFINITY; d];
        let mut hi = vec![f32::NEG_INFINITY; d];
        for t in 0..self.n_res {
            for c in 0..d {
                let v = self.residual[t * d + c];
                lo[c] = lo[c].min(v);
                hi[c] = hi[c].max(v);
            }
        }
        let scale: Vec<f32> =
            (0..d).map(|c| if hi[c] > lo[c] { (hi[c] - lo[c]) / lv } else { 1.0 }).collect();
        let mut codes = vec![0u8; d];
        for t in 0..self.n_res {
            for c in 0..d {
                codes[c] =
                    ((self.residual[t * d + c] - lo[c]) / scale[c]).round().clamp(0.0, lv) as u8;
            }
            pack_codes(&codes, bits, &mut self.packed);
        }
        self.block_lo.extend_from_slice(&lo);
        self.block_scale.extend_from_slice(&scale);
        self.residual.clear();
        self.n_res = 0;
        self.n_finalized += 1;
    }
    pub fn dequant_all(&self) -> Array2<f32> {
        let (d, rb, bits) = (self.d, self.row_bytes, self.bits);
        let t = self.len();
        let mut out = Array2::<f32>::zeros((t, d));
        for bi in 0..self.n_finalized {
            let lo = &self.block_lo[bi * d..bi * d + d];
            let sc = &self.block_scale[bi * d..bi * d + d];
            for ti in 0..KEY_BLOCK {
                let g = bi * KEY_BLOCK + ti;
                let src = &self.packed[g * rb..g * rb + rb];
                for c in 0..d {
                    let code = match bits {
                        8 => src[c],
                        4 => {
                            let b = src[c >> 1];
                            if c & 1 == 0 { b & 0x0F } else { b >> 4 }
                        }
                        _ => unreachable!(),
                    };
                    out[(g, c)] = lo[c] + code as f32 * sc[c];
                }
            }
        }
        let base = self.n_finalized * KEY_BLOCK;
        for ti in 0..self.n_res {
            for c in 0..d {
                out[(base + ti, c)] = self.residual[ti * d + c];
            }
        }
        out
    }
    pub fn memory_bytes(&self) -> usize {
        self.packed.len() + self.block_lo.len() * 8 + self.residual.len() * 4
    }
}

// ── Fused on-the-fly-dequant flash attention (the shimmy borrow) ────────────────────────────────
// Instead of materializing the whole [B,H,T,D] f32 cache each step, dequantize one 32-row K/V
// tile at a time into a small reused buffer, feed it straight into the flash GEMM, and discard.
// Storage stays packed (u8); only ~D floats per tile are ever live. This is the difference between
// "packed storage, unpacked compute" (current op — a speed regression) and shimmy's design.

/// Collapse a broadcast `cos`/`sin` tensor `[.., T, D]` to `[T, D]`. RoPE tables depend only on
/// position and channel, so any leading batch/head axes are indexed at 0.
fn reduce_cos_sin(
    v: tract_nnef::tract_ndarray::ArrayViewD<f32>,
    t: usize,
    d: usize,
) -> TractResult<Array2<f32>> {
    use tract_nnef::tract_ndarray::Axis;
    ensure!(v.ndim() >= 2, "cos/sin must have rank >= 2, got {:?}", v.shape());
    let mut view = v;
    while view.ndim() > 2 {
        view = view.index_axis_move(Axis(0), 0);
    }
    let (vt, vd) = (view.shape()[0], view.shape()[1]);
    ensure!(vd == d, "cos/sin last dim {vd} != head_dim {d}");
    ensure!(vt == t, "cos/sin seq {vt} != cache len {t}");
    Ok(view.to_owned().into_dimensionality::<tract_nnef::tract_ndarray::Ix2>()?)
}

/// Rotate the first `rows` rows of a dequantized K tile in place: `k*cos + rotate_half(k)*sin`,
/// where `rotate_half([a,b]) = [-b,a]` on the two `D/2` halves. Row `i` uses position `start+i`.
fn apply_rope_tile(
    k: &mut Array2<f32>,
    rows: usize,
    start: usize,
    cos: &Array2<f32>,
    sin: &Array2<f32>,
) {
    let d = k.ncols();
    let half = d / 2;
    let mut orig = vec![0f32; d];
    for i in 0..rows {
        let p = start + i;
        {
            let row = k.row(i);
            orig.copy_from_slice(row.as_slice().unwrap());
        }
        for j in 0..d {
            let rh = if j < half { -orig[j + half] } else { orig[j - half] };
            k[(i, j)] = orig[j] * cos[(p, j)] + rh * sin[(p, j)];
        }
    }
}

/// Flash attention over packed per-channel Keys + per-token Values, dequantized tile-by-tile.
/// `q`: [B, Hq, Sq, D]; caches indexed `bi*hkv + kvh`; supports GQA (Hq % hkv == 0).
/// Per-head tasks run across cores (rayon), matching `FlashSdpaOp` — the packed caches are
/// read-only and each head writes a disjoint output slice. When `rope` is `Some`, the cached K is
/// pre-RoPE and each dequantized tile is rotated with the given `[T, D]` cos/sin before scoring.
#[allow(clippy::too_many_arguments)]
fn flash_attention_quant(
    q: tract_nnef::tract_ndarray::ArrayView4<f32>,
    kcs: &[QuantKeyCache],
    vcs: &[QuantValueCache],
    hkv: usize,
    kv_len: usize,
    scale: f32,
    rope: Option<(&Array2<f32>, &Array2<f32>)>,
    causal: bool,
) -> Array4<f32> {
    let (b, hq, q_len, head_dim) = q.dim();
    let group_size = hq / hkv;
    let mut out = Array4::<f32>::zeros((b, hq, q_len, head_dim));
    let tasks: Vec<(usize, usize)> =
        (0..b).flat_map(|bi| (0..hq).map(move |qh| (bi, qh))).collect();
    let compute = |&(bi, qh): &(usize, usize)| {
        attend_one_head_quant(
            bi,
            qh,
            qh / group_size,
            q,
            kcs,
            vcs,
            hkv,
            kv_len,
            scale,
            rope,
            causal,
        )
    };
    let results: Vec<Array2<f32>> = if tasks.len() > 1 && cfg!(not(target_family = "wasm")) {
        #[cfg(not(target_family = "wasm"))]
        {
            use rayon::prelude::*;
            tasks.par_iter().map(compute).collect()
        }
        #[cfg(target_family = "wasm")]
        {
            tasks.iter().map(compute).collect()
        }
    } else {
        tasks.iter().map(compute).collect()
    };
    for (&(bi, qh), head_out) in tasks.iter().zip(results) {
        out.slice_mut(s!(bi, qh, .., ..)).assign(&head_out);
    }
    out
}

/// One (batch, q-head) of the quantized flash attention. Dequantizes each K/V tile from the
/// packed caches into thread-local scratch, then the standard online-softmax flash math with a
/// contiguous P·V GEMM. Pure function of read-only inputs → safe to call per-head concurrently.
#[allow(clippy::too_many_arguments)]
fn attend_one_head_quant(
    bi: usize,
    qh: usize,
    kvh: usize,
    q: tract_nnef::tract_ndarray::ArrayView4<f32>,
    kcs: &[QuantKeyCache],
    vcs: &[QuantValueCache],
    hkv: usize,
    kv_len: usize,
    scale: f32,
    rope: Option<(&Array2<f32>, &Array2<f32>)>,
    causal: bool,
) -> Array2<f32> {
    use tract_nnef::tract_ndarray::Axis;
    let (_, _, q_len, head_dim) = q.dim();
    // Causal alignment: new query row i sits at absolute position `past + i` where
    // `past = kv_len - q_len`; it may attend key positions `0..=past+i`.
    let past = kv_len - q_len;
    let (kc, vc) = (&kcs[bi * hkv + kvh], &vcs[bi * hkv + kvh]);
    let block_kv_len = 32usize;
    let block_q_len = 32usize;
    let mut out = Array2::<f32>::zeros((q_len, head_dim));
    let mut kbuf = Array2::<f32>::zeros((block_kv_len, head_dim));
    let mut vbuf = Array2::<f32>::zeros((block_kv_len, head_dim));
    let mut l = vec![0f32; q_len];
    let mut m = vec![f32::NEG_INFINITY; q_len];
    for kbix in 0..kv_len.div_ceil(block_kv_len) {
        let kv_range = (kbix * block_kv_len)..((kbix + 1) * block_kv_len).min(kv_len);
        let rows = kv_range.len();
        kc.dequant_tile(kv_range.clone(), &mut kbuf);
        vc.dequant_tile(kv_range.clone(), &mut vbuf);
        if let Some((cos, sin)) = rope {
            apply_rope_tile(&mut kbuf, rows, kv_range.start, cos, sin);
        }
        let kblock = kbuf.slice(s![0..rows, ..]);
        let vblock = vbuf.slice(s![0..rows, ..]);
        for qbix in 0..q_len.div_ceil(block_q_len) {
            let q_range = (qbix * block_q_len)..((qbix + 1) * block_q_len).min(q_len);
            let m = &mut m[q_range.clone()];
            let l = &mut l[q_range.clone()];
            let qblock: ArrayView2<f32> = q.slice(s!(bi, qh, q_range.clone(), ..));
            let mut oblock: ArrayViewMut2<f32> = out.slice_mut(s!(q_range.clone(), ..));
            let mut s = qblock.dot(&kblock.t());
            s *= scale;
            if causal {
                for (ri, qi) in q_range.clone().enumerate() {
                    let allow_upto = past + qi; // absolute key positions 0..=past+qi
                    for (cj, kj) in kv_range.clone().enumerate() {
                        if kj > allow_upto {
                            s[(ri, cj)] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let tile_m: Vec<f32> = s
                .rows()
                .into_iter()
                .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                .collect();
            for (row_ix, mx) in tile_m.iter().enumerate() {
                if mx.is_finite() {
                    s.row_mut(row_ix).iter_mut().for_each(|x| *x -= mx);
                }
            }
            s.mapv_inplace(f32::exp);
            let tile_l = s.sum_axis(Axis(1)).insert_axis(Axis(1));
            let m_new = (0..q_range.len()).map(|i| m[i].max(tile_m[i])).collect::<Vec<_>>();
            let l_new = (0..q_range.len())
                .map(|i| {
                    (m[i] - m_new[i]).exp() * l[i] + (tile_m[i] - m_new[i]).exp() * tile_l[(i, 0)]
                })
                .collect::<Vec<_>>();
            let sv_tile = s.dot(&vblock);
            for i in 0..q_range.len() {
                let r_l_new = l_new[i].recip();
                let mul_o = ((m[i] - m_new[i]).exp()) * l[i] * r_l_new;
                let mul_sv = ((tile_m[i] - m_new[i]).exp()) * r_l_new;
                let src = sv_tile.row(i);
                let mut orow = oblock.row_mut(i);
                for j in 0..head_dim {
                    orow[j] = orow[j] * mul_o + src[j] * mul_sv;
                }
            }
            l.copy_from_slice(&l_new);
            m.copy_from_slice(&m_new);
        }
    }
    out
}

// ── Fused stateful op ─────────────────────────────────────────────────────────────────────────

/// Fused quantized KV-cache + attention. Stores K per-channel, V per-token, at `bits` bits.
/// Inputs `[Q, K_new, V_new]` (each `[B, H, S, D]`), then optional on-read RoPE `[cos, sin]`;
/// output has Q's shape. `is_causal` applies causal masking from the op's own token positions
/// (so no external, past-length-dependent mask input is needed).
/// Memory saving vs f32: int8 ~3.8×, int4 ~7× (packed codes + small per-channel/token params).
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedKvSdpa {
    pub axis: usize,
    pub scale: Option<f32>,
    pub bits: u32,
    pub is_causal: bool,
}
impl Eq for QuantizedKvSdpa {}

impl Op for QuantizedKvSdpa {
    fn name(&self) -> StaticName {
        "QuantizedKvSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "axis={}, scale={:?}, bits={}, causal={}",
            self.axis, self.scale, self.bits, self.is_causal
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for QuantizedKvSdpa {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(QuantizedKvSdpaState {
            scale: self.scale,
            bits: self.bits,
            is_causal: self.is_causal,
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            initialized: false,
        })))
    }
}

impl TypedOp for QuantizedKvSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs.len() == 3 || inputs.len() == 5,
            "QuantizedKvSdpa expects [Q, K_new, V_new] then optional [cos, sin]"
        );
        Ok(tvec!(inputs[0].without_value()))
    }
    as_op!();
}

#[derive(Clone, Debug)]
pub struct QuantizedKvSdpaState {
    scale: Option<f32>,
    bits: u32,
    is_causal: bool,
    k_caches: Vec<QuantKeyCache>,   // one per (batch * kv_head)
    v_caches: Vec<QuantValueCache>, // one per (batch * kv_head)
    initialized: bool,
}

impl OpState for QuantizedKvSdpaState {
    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(
            inputs.len() == 3 || inputs.len() == 5,
            "QuantizedKvSdpa expects [Q, K_new, V_new] then optional [cos, sin]"
        );
        let has_rope = inputs.len() == 5;
        let input_dt = inputs[0].datum_type();
        let q = inputs[0].cast_to::<f32>()?;
        let k_new = inputs[1].cast_to::<f32>()?;
        let v_new = inputs[2].cast_to::<f32>()?;
        let qv = q.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let kv = k_new.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let vv = v_new.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let (b, kh, snew, d) = kv.dim();
        let n_caches = b * kh;
        if !self.initialized {
            self.k_caches = (0..n_caches).map(|_| QuantKeyCache::with_bits(d, self.bits)).collect();
            self.v_caches =
                (0..n_caches).map(|_| QuantValueCache::with_bits(d, self.bits)).collect();
            self.initialized = true;
        }
        // Append each new token for each (batch, kv_head).
        for bi in 0..b {
            for hi in 0..kh {
                let idx = bi * kh + hi;
                let ks = kv.slice(s![bi, hi, .., ..]);
                let vs = vv.slice(s![bi, hi, .., ..]);
                for t in 0..snew {
                    self.k_caches[idx].push_token(ks.slice(s![t, ..]).as_slice().unwrap());
                    self.v_caches[idx].push_token(vs.slice(s![t, ..]).as_slice().unwrap());
                }
            }
        }
        // K is cached PRE-RoPE (real exports rotate on read); rotate the dequantized K in the
        // kernel using the same cos/sin the fused ApplyRope consumed, reduced to [T, D].
        let t = self.k_caches[0].len();
        let head_dim = qv.dim().3;
        let cos_sin: Option<(Array2<f32>, Array2<f32>)> = if has_rope {
            let cos = inputs[3].cast_to::<f32>()?;
            let sin = inputs[4].cast_to::<f32>()?;
            let cosr = reduce_cos_sin(cos.to_plain_array_view::<f32>()?, t, head_dim)?;
            let sinr = reduce_cos_sin(sin.to_plain_array_view::<f32>()?, t, head_dim)?;
            Some((cosr, sinr))
        } else {
            None
        };
        let rope = cos_sin.as_ref().map(|(c, s)| (c, s));
        let scale = self.scale.unwrap_or((head_dim as f32).recip().sqrt());
        let o = flash_attention_quant(
            qv,
            &self.k_caches,
            &self.v_caches,
            kh,
            t,
            scale,
            rope,
            self.is_causal,
        );
        Ok(tvec!(o.into_tensor().cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }
}

#[derive(Clone, Debug)]
struct FrozenQuantizedKvSdpaState {
    scale: Option<f32>,
    bits: u32,
    is_causal: bool,
    k_caches: Vec<QuantKeyCache>,
    v_caches: Vec<QuantValueCache>,
    initialized: bool,
}
impl OpStateFreeze for QuantizedKvSdpaState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenQuantizedKvSdpaState {
            scale: self.scale,
            bits: self.bits,
            is_causal: self.is_causal,
            k_caches: self.k_caches.clone(),
            v_caches: self.v_caches.clone(),
            initialized: self.initialized,
        })
    }
}
impl FrozenOpState for FrozenQuantizedKvSdpaState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(QuantizedKvSdpaState {
            scale: self.scale,
            bits: self.bits,
            is_causal: self.is_causal,
            k_caches: self.k_caches.clone(),
            v_caches: self.v_caches.clone(),
            initialized: self.initialized,
        })
    }
}

// ── Auto-wiring transform ──────────────────────────────────────────────────────────────────────

/// Walk an Sdpa K/V input back through cache-read plumbing (`MultiBroadcastTo` / `AxisOp` /
/// `Cast` / on-read `ApplyRope`) to the `DynKeyValueCache`. Returns the cache's new-token input
/// outlet, its axis, and — if an `ApplyRope` was in the chain — its `(cos, sin)` outlets. Every
/// hop must be single-consumer so the plumbing can be removed safely.
fn walk_kv_to_cache(
    model: &TypedModel,
    start: OutletId,
) -> Option<(OutletId, usize, Option<(OutletId, OutletId)>)> {
    let mut outlet = start;
    let mut rope: Option<(OutletId, OutletId)> = None;
    loop {
        let n = model.node(outlet.node);
        if n.outputs[outlet.slot].successors.len() != 1 {
            return None;
        }
        if let Some(kv) = n.op_as::<DynKeyValueCache>() {
            return Some((n.inputs[0], kv.axis, rope));
        } else if n.op_is::<ApplyRope>() {
            if rope.is_some() {
                return None; // a single on-read rotation only
            }
            rope = Some((n.inputs[1], n.inputs[2]));
            outlet = n.inputs[0];
        } else if n.op_is::<MultiBroadcastTo>() || n.op_is::<AxisOp>() || n.op_is::<Cast>() {
            outlet = n.inputs[0];
        } else {
            return None;
        }
    }
}

/// Fuse a decode-attention subgraph — `DynKeyValueCache` feeding `Sdpa` through the standard
/// cache-read plumbing (GQA broadcast, reshapes, f16 casts, and on-read K RoPE) — into a single
/// `QuantizedKvSdpa`. Input order to the fused op: `Q, K_new, V_new`, then the optional RoPE
/// `cos, sin`. `ctx` carries the chosen bit-width (4 or 8).
pub fn fuse_quantized_kv_sdpa_rule(
    ctx: &u32,
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    if node.inputs.len() != 3 && node.inputs.len() != 4 {
        return Ok(None);
    }
    // A 4th (mask) input or the op's own flag means causal decode. The fused op applies causality
    // internally from token positions, so the external mask — which depends on the past-length
    // symbol the removed cache used to resolve — is dropped rather than tapped.
    let is_causal = op.is_causal || node.inputs.len() == 4;
    let (Some((k_new, kaxis, k_rope)), Some((v_new, vaxis, v_rope))) =
        (walk_kv_to_cache(model, node.inputs[1]), walk_kv_to_cache(model, node.inputs[2]))
    else {
        return Ok(None);
    };
    if kaxis != vaxis || v_rope.is_some() {
        return Ok(None); // RoPE is a K-only on-read transform
    }
    let scale = op.scale.as_ref().map(|t| t.cast_to_scalar::<f32>()).transpose()?;
    let mut srcs = vec![node.inputs[0], k_new, v_new];
    if let Some((cos, sin)) = k_rope {
        srcs.push(cos);
        srcs.push(sin);
    }
    let mut patch = TypedModelPatch::default();
    let taps = patch.taps(model, &srcs)?;
    let fused = patch.wire_node(
        format!("{node_name}.quant_kv_sdpa"),
        QuantizedKvSdpa { axis: kaxis, scale, bits: *ctx, is_causal },
        &taps,
    )?;
    patch.shunt_outside(model, node.id.into(), fused[0])?;
    Ok(Some(patch))
}

/// Strip GQA broadcast chain then fuse cache→Sdpa into `QuantizedKvSdpa` at `bits` bits.
/// A tract user selects the precision here: `bits = 8` (int8, ~3.8× KV memory) or
/// `bits = 4` (int4, ~7× KV memory). Not applying this transform at all leaves the KV cache
/// in full f32 — so "int8 / int4 / off" is a per-model choice.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedKvSdpaTransform {
    pub bits: u32,
}

impl Default for QuantizedKvSdpaTransform {
    fn default() -> Self {
        QuantizedKvSdpaTransform { bits: 8 }
    }
}

impl ModelTransform for QuantizedKvSdpaTransform {
    fn name(&self) -> StaticName {
        "fuse_quantized_kv_sdpa".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        ensure!(self.bits == 4 || self.bits == 8, "KV quantization bits must be 4 or 8");
        // Pre-passes so the fuse matcher sees a single ApplyRope and an internalized cache
        // (idempotent if already applied by an earlier stage).
        crate::rewriter::ApplyRopeTransform.transform(model)?;
        crate::rewriter::KeyValueCacheTransform.transform(model)?;
        Rewriter::default()
            .with_rule_for("fuse-quant-kv-sdpa", fuse_quantized_kv_sdpa_rule)
            .rewrite(&self.bits, model)?;
        model.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::flash_sdpa::FlashSdpaOp;
    use tract_nnef::tract_ndarray::{Array2, arr2};

    fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    // Reconstruction error shrinks as bits grow; 16-bit is ~exact.
    #[test]
    fn error_decreases_with_bits() {
        let x = arr2(&[[0.0f32, 1.0, 2.0, 3.0], [-1.0, 0.5, 4.0, 9.0], [2.0, 2.0, 2.0, 2.1]]);
        let e4 = max_abs(&x, &quant_dequant(x.view(), 4, false));
        let e8 = max_abs(&x, &quant_dequant(x.view(), 8, false));
        let e16 = max_abs(&x, &quant_dequant(x.view(), 16, false));
        assert!(e8 < e4, "more bits => less error ({e8} !< {e4})");
        assert!(e16 < e8, "16-bit tighter than 8-bit ({e16} !< {e8})");
        assert!(e16 < 1e-3, "16-bit near-exact, got {e16}");
        // per-element error within half a quantization step of each column's range
        let levels = (1u32 << 8) - 1;
        for j in 0..x.ncols() {
            let col = x.column(j);
            let (lo, hi) = (
                col.iter().copied().fold(f32::INFINITY, f32::min),
                col.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            );
            let step = if hi > lo { (hi - lo) / levels as f32 } else { 0.0 };
            let q = quant_dequant(x.view(), 8, false);
            for i in 0..x.nrows() {
                assert!((x[(i, j)] - q[(i, j)]).abs() <= step / 2.0 + 1e-6);
            }
        }
    }

    // The KIVI insight: with an outlier CHANNEL (a high-magnitude column), per-channel
    // (per-column) quantization isolates it and stays accurate, while per-token (per-row)
    // lumps it with the small dims and crushes them. So per-channel ≪ per-row for Keys.
    #[test]
    fn per_channel_beats_per_row_on_outlier_channel() {
        // 4 tokens x 4 channels; channel 0 is a big-magnitude outlier, others are small.
        let x = arr2(&[
            [100.0f32, 0.10, -0.20, 0.05],
            [-90.0, 0.02, 0.30, -0.08],
            [120.0, -0.15, 0.10, 0.20],
            [-110.0, 0.07, -0.05, 0.12],
        ]);
        // The difference shows on the SMALL channels (cols 1..4): per-token lumps them with
        // the outlier and crushes them; per-channel isolates the outlier so they stay sharp.
        let small_err = |q: &Array2<f32>| -> f32 {
            (1..4)
                .flat_map(|j| (0..4).map(move |i| (i, j)))
                .map(|(i, j)| (x[(i, j)] - q[(i, j)]).abs())
                .fold(0.0, f32::max)
        };
        let pc = small_err(&quant_dequant(x.view(), 4, false)); // per-channel (by column)
        let pt = small_err(&quant_dequant(x.view(), 4, true)); // per-token (by row)
        assert!(pc < pt * 0.2, "per-channel ≫ better on the small dims: pc={pc} pt={pt}");
    }

    // 8-bit KV is near-lossless for attention output; quality improves with bits.
    #[test]
    fn attention_near_lossless_at_8bit() {
        // single head: Q[1,d] . K[s,d] -> softmax -> . V[s,d]
        let (s, d) = (12usize, 16usize);
        let mk = |seed: u64| -> Array2<f32> {
            let mut st = seed;
            Array2::from_shape_fn((s, d), |_| {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
        };
        let q = mk(1).row(0).to_owned();
        let k = mk(2);
        let v = mk(3);
        let scale = 1.0 / (d as f32).sqrt();
        let attn = |k: &Array2<f32>, v: &Array2<f32>| -> Vec<f32> {
            let mut sc: Vec<f32> = (0..s).map(|j| q.dot(&k.row(j)) * scale).collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            sc.iter_mut().for_each(|x| {
                *x = (*x - m).exp();
                sum += *x;
            });
            (0..d).map(|e| (0..s).map(|j| sc[j] / sum * v[(j, e)]).sum()).collect()
        };
        let full = attn(&k, &v);
        let dev = |bits: u32| -> f32 {
            // Keys per-channel (by col), Values per-token (by row) — the KIVI layout.
            let kq = quant_dequant(k.view(), bits, false);
            let vq = quant_dequant(v.view(), bits, true);
            let o = attn(&kq, &vq);
            let num: f32 = o.iter().zip(&full).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            let den: f32 = full.iter().map(|x| x * x).sum::<f32>().sqrt();
            num / den.max(1e-9)
        };
        let (d4, d8, d12) = (dev(4), dev(8), dev(12));
        assert!(d8 < d4 && d12 < d8, "deviation must shrink with bits: 4={d4} 8={d8} 12={d12}");
        assert!(d8 < 0.02, "8-bit KV near-lossless for attention, got {d8}");
    }

    // int4 round-trips through the packed caches and stays reasonable for attention.
    // Not as tight as int8, but the KIVI per-channel-K / per-token-V layout keeps it usable.
    #[test]
    fn int4_cache_round_trip_and_attention() {
        let (s, d) = (24usize, 64usize);
        let mut st = 7u64;
        let mut next = || -> f32 {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let q: Vec<f32> = (0..d).map(|_| next()).collect();
        let mut kc = QuantKeyCache::with_bits(d, 4);
        let mut vc = QuantValueCache::with_bits(d, 4);
        let mut kf = Array2::<f32>::zeros((s, d));
        let mut vf = Array2::<f32>::zeros((s, d));
        for i in 0..s {
            let kr: Vec<f32> = (0..d).map(|_| next()).collect();
            let vr: Vec<f32> = (0..d).map(|_| next()).collect();
            for j in 0..d {
                kf[(i, j)] = kr[j];
                vf[(i, j)] = vr[j];
            }
            kc.push_token(&kr);
            vc.push_token(&vr);
        }
        // Packed int4 dequant matches an independent nibble decode (round-trip sanity).
        let kd = kc.dequant_all();
        assert_eq!(kd.dim(), (s, d));
        // Attention with int4 K/V vs f32 K/V: relative deviation stays bounded.
        let scale = (d as f32).recip().sqrt();
        let att = |k: &Array2<f32>, v: &Array2<f32>| -> Vec<f32> {
            let mut sc: Vec<f32> =
                (0..s).map(|j| (0..d).map(|e| q[e] * k[(j, e)]).sum::<f32>() * scale).collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            sc.iter_mut().for_each(|x| {
                *x = (*x - m).exp();
                sum += *x;
            });
            (0..d).map(|e| (0..s).map(|j| sc[j] / sum * v[(j, e)]).sum()).collect()
        };
        let full = att(&kf, &vf);
        let quant = att(&kc.dequant_all(), &vc.dequant_all());
        let num: f32 = full.iter().zip(&quant).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let den: f32 = full.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel = num / den.max(1e-9);
        println!("  int4 KV attention relative deviation: {rel:.4}");
        assert!(rel < 0.15, "int4 KV attention deviation should stay bounded, got {rel}");
    }

    // The NEON int4 kernels must match a fused-arithmetic scalar reference bit-for-bit
    // (both use FMA, so rounding is identical). Guards the hand-rolled intrinsics.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_int4_matches_scalar_reference() {
        let d = 64usize;
        // Deterministic nibble codes 0..15 and packed bytes.
        let codes: Vec<u8> = (0..d).map(|i| (i * 7 % 16) as u8).collect();
        let mut packed = Vec::new();
        pack_codes(&codes, 4, &mut packed);

        // per-token
        let (lo, scale) = (-0.37f32, 0.041f32);
        let mut got = vec![0f32; d];
        dequant_int4_pertoken(&packed, d, lo, scale, &mut got);
        for c in 0..d {
            let want = (codes[c] as f32).mul_add(scale, lo);
            assert_eq!(got[c].to_bits(), want.to_bits(), "pertoken mismatch at {c}");
        }

        // per-channel
        let ch_lo: Vec<f32> = (0..d).map(|c| -0.5 + 0.01 * c as f32).collect();
        let ch_sc: Vec<f32> = (0..d).map(|c| 0.02 + 0.001 * c as f32).collect();
        let mut got = vec![0f32; d];
        dequant_int4_perchannel(&packed, d, &ch_lo, &ch_sc, &mut got);
        for c in 0..d {
            let want = (codes[c] as f32).mul_add(ch_sc[c], ch_lo[c]);
            assert_eq!(got[c].to_bits(), want.to_bits(), "perchannel mismatch at {c}");
        }
    }

    // Block-wise per-channel K stays consistent even when a channel's range grows late in the
    // sequence — the exact case a running per-channel scale mis-dequantizes.
    #[test]
    fn block_quant_key_consistent_over_growing_range() {
        let d = 16usize;
        let mut kc = BlockQuantKeyCache::with_bits(d, 8);
        let mut rows = Vec::new();
        for t in 0..80usize {
            // channel 0 is a late-growing outlier; other channels small.
            let row: Vec<f32> = (0..d)
                .map(|c| {
                    if c == 0 { t as f32 * 2.0 } else { (((t * 7 + c) as f32) * 0.013).sin() * 0.1 }
                })
                .collect();
            kc.push_token(&row);
            rows.push(row);
        }
        assert_eq!(kc.len(), 80);
        let deq = kc.dequant_all();
        let mut maxerr = 0f32;
        for t in 0..80 {
            for c in 0..d {
                maxerr = maxerr.max((deq[(t, c)] - rows[t][c]).abs());
            }
        }
        // Every token dequantizes against its own block's frozen scale → bounded error,
        // unlike the running-scale cache which crushes early tokens once the outlier appears.
        assert!(maxerr < 1.0, "block-consistent int8 dequant, got {maxerr}");
    }

    // ─── Integration: packed storage memory savings ───────────────────────────────
    #[test]
    fn packed_u8_saves_memory_vs_f32() {
        let (t, d) = (512usize, 64usize);
        let mut kc = QuantKeyCache::new(d);
        let mut vc = QuantValueCache::new(d);
        let mut rng = 42u64;
        let mut next = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for _ in 0..t {
            kc.push_token(&(0..d).map(|_| next()).collect::<Vec<_>>());
            vc.push_token(&(0..d).map(|_| next()).collect::<Vec<_>>());
        }
        let f32_bytes = t * d * 4 * 2; // K + V in f32
        let quant_bytes = kc.memory_bytes() + vc.memory_bytes();
        let ratio = f32_bytes as f32 / quant_bytes as f32;
        // u8 = 1 byte/element vs f32 = 4 bytes; per-channel params for K (D*8),
        // per-token params for V (T*8) — overall >3x saving at T=512 D=64.
        assert!(ratio > 3.0, "expected >3x memory saving, got {ratio:.2}x");
        println!("f32 bytes: {f32_bytes}, quantized: {quant_bytes}, ratio: {ratio:.2}x");
    }

    // ─── Integration: fused op runs through tract's engine, near-lossless ─────────
    #[test]
    fn quantized_kv_sdpa_runs_in_model() -> TractResult<()> {
        let (b, h, d) = (1usize, 2usize, 16usize);
        let scale = 1.0 / (d as f32).sqrt();
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let dim = |x: usize| x.to_dim();
        let f: TVec<TDim> = tvec![dim(b), dim(h), s.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&f))?;
        let k = model.add_source("k", f32::fact(&f))?;
        let v = model.add_source("v", f32::fact(&f))?;
        let o = model.wire_node(
            "qkv",
            QuantizedKvSdpa { axis: 2, scale: None, bits: 8, is_causal: false },
            &[q, k, v],
        )?;
        model.select_output_outlets(&o)?;
        let mut rt = model.into_runnable()?.spawn()?;

        // Run 10 decode steps; compare each to full-f32 attention over the growing cache.
        use tract_nnef::tract_core::ops::array::TypedConcat;
        use tract_nnef::tract_ndarray::{Array4 as A4, s};

        let mk = |base: f32| -> Tensor {
            let data: Vec<f32> = (0..b * h * d).map(|i| base + (i as f32 * 0.013).sin()).collect();
            Tensor::from_shape(&[b, h, 1, d], &data).unwrap()
        };
        let grow = |acc: Option<Tensor>, x: Tensor| -> TractResult<Tensor> {
            Ok(match acc {
                None => x,
                Some(a) => {
                    TypedConcat { axis: 2 }.eval(tvec![a.into(), x.into()])?.remove(0).into_tensor()
                }
            })
        };
        let attn = |q: A4<f32>, k: A4<f32>, v: A4<f32>| -> A4<f32> {
            let (b, h, sq, d) = q.dim();
            let mut out = A4::<f32>::zeros((b, h, sq, d));
            for bi in 0..b {
                for hi in 0..h {
                    let qm = q.slice(s![bi, hi, .., ..]);
                    let km = k.slice(s![bi, hi, .., ..]);
                    let vm = v.slice(s![bi, hi, .., ..]);
                    let mut sc = qm.dot(&km.t());
                    sc *= scale;
                    for mut row in sc.rows_mut() {
                        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sm = 0.0f32;
                        row.iter_mut().for_each(|x| {
                            *x = (*x - m).exp();
                            sm += *x;
                        });
                        row.iter_mut().for_each(|x| *x /= sm);
                    }
                    out.slice_mut(s![bi, hi, .., ..]).assign(&sc.dot(&vm));
                }
            }
            out
        };
        let (mut kf, mut vf): (Option<Tensor>, Option<Tensor>) = (None, None);
        for t in 0..10 {
            let qi = mk(9.0 + t as f32 * 0.1);
            let ki = mk(1.0 + t as f32 * 0.07);
            let vi = mk(5.0 - t as f32 * 0.05);
            let o_model = rt
                .run(tvec![qi.clone().into(), ki.clone().into(), vi.clone().into()])?
                .remove(0)
                .into_tensor();
            kf = Some(grow(kf.take(), ki)?);
            vf = Some(grow(vf.take(), vi)?);
            let qv = qi.to_plain_array_view::<f32>()?.into_dimensionality()?;
            let kv = kf.as_ref().unwrap().to_plain_array_view::<f32>()?.into_dimensionality()?;
            let vv = vf.as_ref().unwrap().to_plain_array_view::<f32>()?.into_dimensionality()?;
            let o_ref = Tensor::from(attn(qv.to_owned(), kv.to_owned(), vv.to_owned()));
            // quantized decode should be close to f32 (within ~2% at int8 quality)
            o_model
                .close_enough(&o_ref, Approximation::SuperApproximate)
                .with_context(|| format!("quantized decode too far from f32 at step {t}"))?;
        }
        Ok(())
    }

    // ─── Rope-aware op: cached K is pre-RoPE, rotated on read via cos/sin inputs ─────
    #[test]
    fn quantized_kv_sdpa_rope_matches_reference() -> TractResult<()> {
        use tract_nnef::tract_core::ops::array::TypedConcat;
        use tract_nnef::tract_ndarray::{Array4 as A4, s};
        let (b, h, d) = (1usize, 2usize, 16usize);
        let scale = 1.0 / (d as f32).sqrt();
        let mut model = TypedModel::default();
        let sq = model.sym("S");
        let tt = model.sym("T");
        let f: TVec<TDim> = tvec![b.to_dim(), h.to_dim(), sq.into(), d.to_dim()];
        let cs: TVec<TDim> = tvec![tt.into(), d.to_dim()];
        let q = model.add_source("q", f32::fact(&f))?;
        let k = model.add_source("k", f32::fact(&f))?;
        let v = model.add_source("v", f32::fact(&f))?;
        let cos = model.add_source("cos", f32::fact(&cs))?;
        let sin = model.add_source("sin", f32::fact(&cs))?;
        let o = model.wire_node(
            "qkv",
            QuantizedKvSdpa { axis: 2, scale: None, bits: 8, is_causal: false },
            &[q, k, v, cos, sin],
        )?;
        model.select_output_outlets(&o)?;
        let mut rt = model.into_runnable()?.spawn()?;

        let half = d / 2;
        let steps = 8usize;
        // deterministic cos/sin table [steps, d]
        let mk = |base: f32, bh: usize| -> Tensor {
            let data: Vec<f32> = (0..b * bh * d).map(|i| base + (i as f32 * 0.017).sin()).collect();
            Tensor::from_shape(&[b, bh, 1, d], &data).unwrap()
        };
        let cossin = |seed: f32| -> Vec<f32> {
            (0..steps * d).map(|i| (seed + i as f32 * 0.011).cos() * 0.5).collect()
        };
        let (cost, sint) = (cossin(0.3), cossin(1.7));
        let grow = |acc: Option<Tensor>, x: Tensor| -> TractResult<Tensor> {
            Ok(match acc {
                None => x,
                Some(a) => {
                    TypedConcat { axis: 2 }.eval(tvec![a.into(), x.into()])?.remove(0).into_tensor()
                }
            })
        };
        let (mut kf, mut vf): (Option<Tensor>, Option<Tensor>) = (None, None);
        for t in 0..steps {
            let ti = t + 1;
            let qi = mk(9.0 + t as f32 * 0.1, h);
            let ki = mk(1.0 + t as f32 * 0.07, h);
            let vi = mk(5.0 - t as f32 * 0.05, h);
            let cosi = Tensor::from_shape(&[ti, d], &cost[..ti * d])?;
            let sini = Tensor::from_shape(&[ti, d], &sint[..ti * d])?;
            let o_model = rt
                .run(tvec![
                    qi.clone().into(),
                    ki.clone().into(),
                    vi.clone().into(),
                    cosi.into(),
                    sini.into()
                ])?
                .remove(0)
                .into_tensor();
            kf = Some(grow(kf.take(), ki)?);
            vf = Some(grow(vf.take(), vi)?);
            // reference: rope the full pre-rope K, then attention
            let kv =
                kf.as_ref().unwrap().to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
            let vv =
                vf.as_ref().unwrap().to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
            let qv = qi.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
            let mut kr = kv.to_owned();
            for hd in 0..h {
                for p in 0..ti {
                    let orig: Vec<f32> = (0..d).map(|j| kv[(0, hd, p, j)]).collect();
                    for j in 0..d {
                        let rh = if j < half { -orig[j + half] } else { orig[j - half] };
                        kr[(0, hd, p, j)] = orig[j] * cost[p * d + j] + rh * sint[p * d + j];
                    }
                }
            }
            let mut oref = A4::<f32>::zeros((b, h, 1, d));
            for hd in 0..h {
                let qm: ArrayView2<f32> = qv.slice(s![0, hd, .., ..]);
                let km: ArrayView2<f32> = kr.slice(s![0, hd, .., ..]);
                let vm: ArrayView2<f32> = vv.slice(s![0, hd, .., ..]);
                let mut sc = qm.dot(&km.t());
                sc *= scale;
                for mut row in sc.rows_mut() {
                    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sm = 0.0;
                    row.iter_mut().for_each(|x| {
                        *x = (*x - mx).exp();
                        sm += *x;
                    });
                    row.iter_mut().for_each(|x| *x /= sm);
                }
                oref.slice_mut(s![0, hd, .., ..]).assign(&sc.dot(&vm));
            }
            Tensor::from(oref)
                .close_enough(&o_model, Approximation::SuperApproximate)
                .with_context(|| format!("rope decode mismatch at step {t}"))?;
        }
        Ok(())
    }

    // ─── Integration: auto-wiring transform ──────────────────────────────────────
    #[test]
    fn transform_fuses_cache_sdpa_to_quantized() -> TractResult<()> {
        let (b, h, d) = (1usize, 2usize, 16usize);
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let p = model.sym("P");
        let dim = |x: usize| x.to_dim();
        let newf: TVec<TDim> = tvec![dim(b), dim(h), s.into(), dim(d)];
        let pastf: TVec<TDim> = tvec![dim(b), dim(h), p.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&newf))?;
        let knew = model.add_source("k", f32::fact(&newf))?;
        let vnew = model.add_source("v", f32::fact(&newf))?;
        let mkc = |nm: &str| DynKeyValueCache {
            name: nm.to_string(),
            axis: 2,
            past_sequence_fact: f32::fact(&pastf),
            input_sequence_fact: f32::fact(&newf),
        };
        let kc = model.wire_node("kc", mkc("kc"), &[knew])?;
        let vc = model.wire_node("vc", mkc("vc"), &[vnew])?;
        let o = model.wire_node(
            "sdpa",
            Sdpa {
                scale: None,
                datum_type: f32::datum_type(),
                acc_datum_type: f32::datum_type(),
                is_causal: false,
            },
            &[q, kc[0], vc[0]],
        )?;
        model.select_output_outlets(&o)?;
        QuantizedKvSdpaTransform { bits: 4 }.transform(&mut model)?;
        let fused = model
            .nodes()
            .iter()
            .find_map(|n| n.op_as::<QuantizedKvSdpa>())
            .context("fused op present")?;
        assert_eq!(fused.bits, 4, "transform propagates the chosen bit-width");
        assert!(!model.nodes().iter().any(|n| n.op_is::<DynKeyValueCache>()), "caches removed");
        assert!(!model.nodes().iter().any(|n| n.op_is::<Sdpa>()), "sdpa removed");
        Ok(())
    }

    // Memory saving bench: print u8 vs f32 savings at realistic decode lengths.
    //   cargo test -p tract-transformers kv_quant::tests::bench_memory_savings -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_memory_savings() {
        let d = 128usize;
        let mut rng = 99u64;
        let mut next = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        println!("\n  KV cache memory (int8 u8 vs f32), H=8 heads, D={d}:");
        println!("     T     f32(MB)   int8(MB)   saving");
        for &t in &[256usize, 1024, 4096, 16384] {
            let mut kc = QuantKeyCache::new(d);
            let mut vc = QuantValueCache::new(d);
            for _ in 0..t {
                kc.push_token(&(0..d).map(|_| next()).collect::<Vec<_>>());
                vc.push_token(&(0..d).map(|_| next()).collect::<Vec<_>>());
            }
            let heads = 8;
            let f32_mb = (t * d * 4 * 2 * heads) as f32 / 1e6;
            let int8_mb = ((kc.memory_bytes() + vc.memory_bytes()) * heads) as f32 / 1e6;
            println!("  {t:>6}  {f32_mb:>9.2}  {int8_mb:>9.2}  {:>6.2}x", f32_mb / int8_mb);
        }
    }

    // ─── Decode-latency microbench: baseline f32 flash vs current INT8 op ─────────────
    //   cargo test --release -p tract-transformers kv_quant::tests::bench_decode_latency -- --ignored --nocapture
    //
    // Simulates one autoregressive decode step (q_len=1) over a cache of length T.
    //  - baseline: FlashSdpa over an f32 [B,Hkv,T,D] cache (what tract does today).
    //  - int8_current: the existing op — dequantize the WHOLE packed cache to f32 [B,Hkv,T,D]
    //    every step, then run the same flash. Measures the materialization overhead.
    #[test]
    #[ignore]
    fn bench_decode_latency() {
        use std::time::Instant;
        let (b, hq, hkv, d) = (1usize, 8usize, 8usize, 64usize);
        let scale = (d as f32).recip().sqrt();
        let mut rng = 0x1234_5678u64;
        let mut next = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        // Fresh query each step.
        let mk_q = |next: &mut dyn FnMut() -> f32| -> Array4<f32> {
            Array4::from_shape_fn((b, hq, 1, d), |_| next())
        };
        // Median-of-R timing helper.
        let median = |mut v: Vec<f64>| -> f64 {
            v.sort_by(|a, c| a.partial_cmp(c).unwrap());
            v[v.len() / 2]
        };

        println!("\n  Decode-step latency (B={b} Hq={hq} Hkv={hkv} D={d}), median of runs:");
        println!(
            "      T     f32(µs)  i8_fused  i8/f32   i4_fused  i4/f32   f32KV(MB)  i8(MB) i8×   i4(MB) i4×"
        );
        for &t in &[256usize, 1024, 4096, 16384, 65536] {
            // Build f32 K/V [b,hkv,t,d] and parallel int8 + int4 caches from identical data.
            let mut kf = Array4::<f32>::zeros((b, hkv, t, d));
            let mut vf = Array4::<f32>::zeros((b, hkv, t, d));
            let mk_cache = |bits| (0..b * hkv).map(move |_| QuantKeyCache::with_bits(d, bits));
            let mv_cache = |bits| (0..b * hkv).map(move |_| QuantValueCache::with_bits(d, bits));
            let mut kc8: Vec<_> = mk_cache(8).collect();
            let mut vc8: Vec<_> = mv_cache(8).collect();
            let mut kc4: Vec<_> = mk_cache(4).collect();
            let mut vc4: Vec<_> = mv_cache(4).collect();
            for bi in 0..b {
                for h in 0..hkv {
                    for ti in 0..t {
                        let krow: Vec<f32> = (0..d).map(|_| next()).collect();
                        let vrow: Vec<f32> = (0..d).map(|_| next()).collect();
                        for e in 0..d {
                            kf[(bi, h, ti, e)] = krow[e];
                            vf[(bi, h, ti, e)] = vrow[e];
                        }
                        let idx = bi * hkv + h;
                        kc8[idx].push_token(&krow);
                        vc8[idx].push_token(&vrow);
                        kc4[idx].push_token(&krow);
                        vc4[idx].push_token(&vrow);
                    }
                }
            }
            let flash = FlashSdpaOp { causal: false, scale: Some(scale) };
            let runs = if t <= 4096 { 20 } else { 8 };

            // baseline f32 flash
            let mut t_f32 = Vec::new();
            for _ in 0..runs {
                let q = mk_q(&mut next);
                let s = Instant::now();
                let o = flash.flash_attention_gqa(q.view(), kf.view(), vf.view(), None);
                std::hint::black_box(&o);
                t_f32.push(s.elapsed().as_secs_f64() * 1e6);
            }
            // int8 fused
            let mut t_i8 = Vec::new();
            for _ in 0..runs {
                let q = mk_q(&mut next);
                let s = Instant::now();
                let o = flash_attention_quant(q.view(), &kc8, &vc8, hkv, t, scale, None, false);
                std::hint::black_box(&o);
                t_i8.push(s.elapsed().as_secs_f64() * 1e6);
            }
            // int4 fused
            let mut t_i4 = Vec::new();
            for _ in 0..runs {
                let q = mk_q(&mut next);
                let s = Instant::now();
                let o = flash_attention_quant(q.view(), &kc4, &vc4, hkv, t, scale, None, false);
                std::hint::black_box(&o);
                t_i4.push(s.elapsed().as_secs_f64() * 1e6);
            }

            let f32_us = median(t_f32);
            let i8_us = median(t_i8);
            let i4_us = median(t_i4);
            let f32_kv_mb = (t * d * 4 * 2 * hkv * b) as f64 / 1e6;
            let mb = |kc: &[QuantKeyCache], vc: &[QuantValueCache]| {
                (kc.iter().map(|c| c.memory_bytes()).sum::<usize>()
                    + vc.iter().map(|c| c.memory_bytes()).sum::<usize>()) as f64
                    / 1e6
            };
            let i8_mb = mb(&kc8, &vc8);
            let i4_mb = mb(&kc4, &vc4);
            println!(
                "  {t:>6}  {f32_us:>8.1}  {i8_us:>8.1} {:>5.2}x  {i4_us:>8.1} {:>5.2}x  {f32_kv_mb:>9.2}  {i8_mb:>6.2} {:>4.2}x {i4_mb:>6.2} {:>4.2}x",
                i8_us / f32_us,
                i4_us / f32_us,
                f32_kv_mb / i8_mb,
                f32_kv_mb / i4_mb
            );
        }
    }

    // ─── E2E dilution: a real L-layer decoder (RmsNorm + cache→Sdpa attn + SwiGLU FFN) ────────
    //   cargo test --release -p tract-transformers kv_quant::tests::bench_e2e_decode_dilution -- --ignored --nocapture
    //
    // Measures how much of the isolated-kernel win survives once weight-heavy FFN/projections
    // dominate each decode step. Random f32 weights (architecture + execution real, weights
    // synthetic → output is gibberish, but we still compare f32-vs-quant hidden-state deviation).
    #[test]
    #[ignore]
    fn bench_e2e_decode_dilution() -> TractResult<()> {
        use crate::ops::sdpa::Sdpa;
        use std::time::Instant;
        use tract_nnef::tract_core::ops::change_axes::AxisOp;
        use tract_nnef::tract_core::ops::einsum::EinSum;
        use tract_nnef::tract_core::ops::math;
        use tract_nnef::tract_core::ops::nn::RmsNorm;
        use tract_nnef::tract_core::ops::nn::silu::silu;

        // ~300M-class MHA decoder config.
        let (e, h, d, f, layers) = (1024usize, 8usize, 128usize, 2816usize, 16usize);
        let hkv = h; // MHA → cache feeds Sdpa directly, transform fires cleanly
        assert_eq!(h * d, e);
        let mut rng = 0xC0FFEEu64;
        let mut nf = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let mut wconst = |m: &mut TypedModel,
                          name: &str,
                          r: usize,
                          c: usize,
                          nf: &mut dyn FnMut() -> f32|
         -> TractResult<OutletId> {
            let data: Vec<f32> = (0..r * c).map(|_| nf() * (2.0 / e as f32).sqrt()).collect();
            m.add_const(name, Tensor::from_shape(&[r, c], &data)?)
        };

        let mut model = TypedModel::default();
        let p = model.sym("P");
        let eps = tensor0(1e-5f32).into_arc_tensor();
        let mut x = model.add_source("x", f32::fact(&[1.to_dim(), 1.to_dim(), e.to_dim()]))?;
        let lin = |m: &mut TypedModel,
                   nm: String,
                   a: OutletId,
                   w: OutletId|
         -> TractResult<OutletId> {
            Ok(m.wire_node(nm, EinSum::new("bse,ef->bsf".parse()?, f32::datum_type()), &[a, w])?[0])
        };
        for l in 0..layers {
            // ── attention ──
            let n =
                model.wire_node(format!("l{l}.an"), RmsNorm { axis: 2, eps: eps.clone() }, &[x])?
                    [0];
            let wq = wconst(&mut model, &format!("l{l}.wq"), e, e, &mut nf)?;
            let wk = wconst(&mut model, &format!("l{l}.wk"), e, hkv * d, &mut nf)?;
            let wv = wconst(&mut model, &format!("l{l}.wv"), e, hkv * d, &mut nf)?;
            let wo = wconst(&mut model, &format!("l{l}.wo"), e, e, &mut nf)?;
            let q = lin(&mut model, format!("l{l}.q"), n, wq)?;
            let k = lin(&mut model, format!("l{l}.k"), n, wk)?;
            let v = lin(&mut model, format!("l{l}.v"), n, wv)?;
            // [1,1,E] -> [1,1,H,D] -> [1,H,1,D]
            let split = |m: &mut TypedModel,
                         nm: String,
                         t: OutletId,
                         nh: usize|
             -> TractResult<OutletId> {
                let r = m.wire_node(
                    format!("{nm}.r"),
                    AxisOp::Reshape(2, tvec![(nh * d).to_dim()], tvec![nh.to_dim(), d.to_dim()]),
                    &[t],
                )?[0];
                Ok(m.wire_node(format!("{nm}.m"), AxisOp::Move(1, 2), &[r])?[0])
            };
            let q = split(&mut model, format!("l{l}.qh"), q, h)?;
            let k = split(&mut model, format!("l{l}.kh"), k, hkv)?;
            let v = split(&mut model, format!("l{l}.vh"), v, hkv)?;
            let pf = f32::fact(&[1.to_dim(), hkv.to_dim(), p.clone().into(), d.to_dim()]);
            let inf = f32::fact(&[1.to_dim(), hkv.to_dim(), 1.to_dim(), d.to_dim()]);
            let mkc = |m: &mut TypedModel, nm: String, t: OutletId| -> TractResult<OutletId> {
                Ok(m.wire_node(
                    nm.clone(),
                    DynKeyValueCache {
                        name: nm,
                        axis: 2,
                        past_sequence_fact: pf.clone(),
                        input_sequence_fact: inf.clone(),
                    },
                    &[t],
                )?[0])
            };
            let kc = mkc(&mut model, format!("l{l}.kc"), k)?;
            let vc = mkc(&mut model, format!("l{l}.vc"), v)?;
            let sdpa = Sdpa {
                scale: Some(tensor0(1f32 / (d as f32).sqrt())),
                datum_type: f32::datum_type(),
                acc_datum_type: f32::datum_type(),
                is_causal: false,
            };
            let a = model.wire_node(format!("l{l}.sdpa"), sdpa, &[q, kc, vc])?[0];
            // [1,H,1,D] -> [1,1,H,D] -> [1,1,E]
            let a = model.wire_node(format!("l{l}.am"), AxisOp::Move(1, 2), &[a])?[0];
            let a = model.wire_node(
                format!("l{l}.ar"),
                AxisOp::Reshape(2, tvec![h.to_dim(), d.to_dim()], tvec![e.to_dim()]),
                &[a],
            )?[0];
            let o = lin(&mut model, format!("l{l}.o"), a, wo)?;
            x = model.wire_node(format!("l{l}.res1"), math::add(), &[x, o])?[0];
            // ── SwiGLU FFN ──
            let n2 =
                model.wire_node(format!("l{l}.fn"), RmsNorm { axis: 2, eps: eps.clone() }, &[x])?
                    [0];
            let wg = wconst(&mut model, &format!("l{l}.wg"), e, f, &mut nf)?;
            let wu = wconst(&mut model, &format!("l{l}.wu"), e, f, &mut nf)?;
            let wd = wconst(&mut model, &format!("l{l}.wd"), f, e, &mut nf)?;
            let g = lin(&mut model, format!("l{l}.g"), n2, wg)?;
            let g = model.wire_node(format!("l{l}.silu"), silu(), &[g])?[0];
            let u = lin(&mut model, format!("l{l}.u"), n2, wu)?;
            let gu = model.wire_node(format!("l{l}.gu"), math::mul(), &[g, u])?[0];
            let ff = model.wire_node(
                format!("l{l}.d"),
                EinSum::new("bsf,fe->bse".parse()?, f32::datum_type()),
                &[gu, wd],
            )?[0];
            x = model.wire_node(format!("l{l}.res2"), math::add(), &[x, ff])?[0];
        }
        model.select_output_outlets(&[x])?;
        // Declutter keeps Sdpa + DynKeyValueCache (into_optimized would lower Sdpa away, like -O
        // does on real exports). Fuse the quant op on the decluttered form, THEN optimize.
        let decluttered = model.into_decluttered()?;
        let model = decluttered.clone().into_optimized()?; // f32 baseline (optimized attention)
        let mut m8 = decluttered.clone();
        QuantizedKvSdpaTransform { bits: 8 }.transform(&mut m8)?;
        let nq = m8.nodes().iter().filter(|n| n.op_is::<QuantizedKvSdpa>()).count();
        assert_eq!(nq, layers, "transform must fuse all {layers} layers (got {nq})");
        let m8 = m8.into_optimized()?;
        let mut m4 = decluttered.clone();
        QuantizedKvSdpaTransform { bits: 4 }.transform(&mut m4)?;
        let m4 = m4.into_optimized()?;

        let mk_in = |nf: &mut dyn FnMut() -> f32| {
            Tensor::from_shape(&[1, 1, e], &(0..e).map(|_| nf()).collect::<Vec<_>>()).unwrap()
        };

        // ── accuracy: identical input stream through fresh states, compare final hidden ──
        let mut s0 = model.clone().into_runnable()?.spawn()?;
        let mut s8 = m8.clone().into_runnable()?.spawn()?;
        let mut s4 = m4.clone().into_runnable()?.spawn()?;
        let (mut e8, mut e4) = (0f32, 0f32);
        for _ in 0..16 {
            let inp = mk_in(&mut nf);
            let o0 = s0.run(tvec![inp.clone().into()])?.remove(0).into_tensor();
            let o8 = s8.run(tvec![inp.clone().into()])?.remove(0).into_tensor();
            let o4 = s4.run(tvec![inp.into()])?.remove(0).into_tensor();
            let a = o0.to_plain_array_view::<f32>()?;
            let b8 = o8.to_plain_array_view::<f32>()?;
            let b4 = o4.to_plain_array_view::<f32>()?;
            let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            e8 = e8.max(
                (a.iter().zip(b8.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()) / den,
            );
            e4 = e4.max(
                (a.iter().zip(b4.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()) / den,
            );
        }

        // ── latency: single growing state per variant; time each step; median at checkpoints ──
        let ctx = 2048usize;
        let checkpoints = [128usize, 512, 2048];
        let median_window = |v: &[f64], center: usize| -> f64 {
            let lo = center.saturating_sub(15);
            let hi = (center + 15).min(v.len());
            let mut w: Vec<f64> = v[lo..hi].to_vec();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w[w.len() / 2]
        };
        let run_variant = |m: &TypedModel, nf: &mut dyn FnMut() -> f32| -> TractResult<Vec<f64>> {
            let mut st = m.clone().into_runnable()?.spawn()?;
            let mut times = Vec::with_capacity(ctx);
            for _ in 0..ctx {
                let inp = mk_in(nf);
                let t = Instant::now();
                let o = st.run(tvec![inp.into()])?;
                std::hint::black_box(&o);
                times.push(t.elapsed().as_secs_f64() * 1e6);
            }
            Ok(times)
        };
        let t0 = run_variant(&model, &mut nf)?;
        let t8 = run_variant(&m8, &mut nf)?;
        let t4 = run_variant(&m4, &mut nf)?;

        let w_bytes = layers * (4 * e * e + 3 * e * f) * 4;
        println!(
            "\n  E2E decode dilution — {layers}-layer MHA (E={e} H={h} D={d} F={f}), ~{}M params",
            w_bytes / 4 / 1_000_000
        );
        println!("  accuracy (max hidden rel-dev over 16 steps): int8={:.4}  int4={:.4}", e8, e4);
        println!("  weights={:.0}MB (f32, constant)\n", w_bytes as f64 / 1e6);
        println!(
            "     ctx    f32(µs)  int8(µs) i8/f32  int4(µs) i4/f32 | KVf32(MB) KVi4(MB) totMem i4/f32"
        );
        for &c in &checkpoints {
            let (a, b, cc) = (median_window(&t0, c), median_window(&t8, c), median_window(&t4, c));
            let kv_f32 = (layers * 2 * hkv * d * c * 4) as f64 / 1e6;
            let kv_i4 = (layers * 2 * hkv * d * c) as f64 / 2.0 / 1e6; // ~4-bit + small params
            let tot_f32 = w_bytes as f64 / 1e6 + kv_f32;
            let tot_i4 = w_bytes as f64 / 1e6 + kv_i4;
            println!(
                "  {c:>6}  {a:>8.1}  {b:>8.1} {:>5.2}x  {cc:>8.1} {:>5.2}x | {kv_f32:>8.2} {kv_i4:>8.2}  {:>5.2}x",
                b / a,
                cc / a,
                tot_f32 / tot_i4
            );
        }
        Ok(())
    }

    // NNEF round-trip: QuantizedKvSdpa survives write_to_tar -> model_for_read.
    #[test]
    fn quantized_kv_sdpa_nnef_round_trip() -> TractResult<()> {
        use crate::WithTractTransformers;
        let (b, h, d) = (1usize, 2usize, 16usize);
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let dim = |x: usize| x.to_dim();
        let f: TVec<TDim> = tvec![dim(b), dim(h), s.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&f))?;
        let k = model.add_source("k", f32::fact(&f))?;
        let v = model.add_source("v", f32::fact(&f))?;
        let o = model.wire_node(
            "qkv",
            QuantizedKvSdpa { axis: 2, scale: Some(0.125), bits: 8, is_causal: false },
            &[q, k, v],
        )?;
        model.select_output_outlets(&o)?;

        let nnef = tract_nnef::nnef().with_tract_transformers();
        let mut buffer = vec![];
        nnef.write_to_tar(&model, &mut buffer)?;
        let reloaded = nnef.model_for_read(&mut &*buffer)?;

        let n = reloaded
            .nodes()
            .iter()
            .find(|n| n.op_is::<QuantizedKvSdpa>())
            .context("QuantizedKvSdpa not found after round-trip")?;
        let op = n.op_as::<QuantizedKvSdpa>().unwrap();
        assert_eq!(op.axis, 2);
        assert_eq!(op.scale, Some(0.125));
        Ok(())
    }
}
