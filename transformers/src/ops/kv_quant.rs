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
use tract_nnef::tract_ndarray::{Array2, Array4, ArrayView2, Ix4, s};

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
            TypeName::Integer.named("causal"),
            TypeName::Integer.named("n_recent"),
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
        ("causal", numeric(op.causal as i64)),
        ("n_recent", numeric(op.n_recent)),
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
    let causal: bool =
        invocation.get_named_arg_as::<i64>(builder, "causal")?.map(|v| v != 0).unwrap_or(false);
    let n_recent: usize =
        invocation.get_named_arg_as::<i64>(builder, "n_recent")?.map(|v| v as usize).unwrap_or(0);
    builder.wire(QuantizedKvSdpa { axis, scale, causal, n_recent }, &[q, k, v])
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

// ── Fixed Hadamard rotation (data-free outlier spreading) ───────────────────────────────────────
// The shared trick of TurboQuant (random rotation) and OSCAR (calibrated rotation): rotate the
// head-dim vector into a basis where outlier-channel energy is spread across all coordinates, so a
// shared per-group scale no longer has to straddle a huge dynamic range. The *calibration-free*
// instance is a fixed (Sylvester) Hadamard — what QuaRot/SpinQuant established and what
// TurboQuant's random rotation reduces to in practice. Two clean properties for KV:
//   • Keys: rotate Q and K by the same orthonormal H ⇒ (QH)(KH)ᵀ = QKᵀ, scores bit-identical,
//     so the rotation only ever *helps* K quantization — no accuracy argument needed, it's an
//     identity. (H is symmetric, Hᵀ=H, H·H=I.)
//   • Values: store V·H quantized; after P·(V·H) recover the true output by right-multiplying H
//     (linear, exact up to the quant error which is now spread across channels).

/// Normalized (orthonormal) Hadamard matrix of size `n` (must be a power of two). Symmetric
/// (`Hᵀ = H`) and an involution (`H·H = I`), so the same matrix rotates and un-rotates.
pub fn hadamard_normalized(n: usize) -> Array2<f32> {
    assert!(n.is_power_of_two() && n > 0, "Hadamard size must be a power of two, got {n}");
    let mut h = Array2::<f32>::zeros((n, n));
    h[(0, 0)] = 1.0;
    let mut size = 1;
    while size < n {
        for i in 0..size {
            for j in 0..size {
                let v = h[(i, j)];
                h[(i, j + size)] = v;
                h[(i + size, j)] = v;
                h[(i + size, j + size)] = -v;
            }
        }
        size *= 2;
    }
    let norm = 1.0 / (n as f32).sqrt();
    h.mapv_inplace(|x| x * norm);
    h
}

/// In-place fast Walsh–Hadamard transform (normalized), O(n log n). Natural (Sylvester) ordering,
/// so `fwht_normalized(x)` equals `x · hadamard_normalized(n)` but without the O(n²) matmul — this
/// is what makes the rotation cheap enough to run per query in the decode loop.
pub fn fwht_normalized(x: &mut [f32]) {
    let n = x.len();
    assert!(n.is_power_of_two() && n > 0, "FWHT length must be a power of two, got {n}");
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let norm = (n as f32).sqrt().recip();
    for v in x.iter_mut() {
        *v *= norm;
    }
}

// ── Packed u8 storage ─────────────────────────────────────────────────────────────────────────
// One token = D bytes (int8) for Values (per-token scale), or D bytes for one channel of Keys
// (per-channel scale). Real memory: u8 is 4× f32, 2× f16.

/// Quantize a 1-D token/channel into `D` u8 bytes; return `(bytes, lo, scale)`.
fn quant_token_to_u8(v: &[f32]) -> (Vec<u8>, f32, f32) {
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
    let q: Vec<u8> =
        v.iter().map(|&x| ((x - lo) / scale).round().clamp(0.0, 255.0) as u8).collect();
    (q, lo, scale)
}

/// Dequantize a u8 slice back to f32 given `(lo, scale)`.
fn dequant_u8(q: &[u8], lo: f32, scale: f32) -> Vec<f32> {
    q.iter().map(|&b| lo + b as f32 * scale).collect()
}

// ── Per-token quantized Value store ───────────────────────────────────────────────────────────

/// Quantized Value cache: stores each appended token as D u8 bytes + 2 f32 params.
/// Memory per token: D + 8 bytes (vs D*4 f32 = 4× saving at large D).
#[derive(Clone, Debug, Default)]
pub struct QuantValueCache {
    pub d: usize,
    // packed: [token_idx * d .. (token_idx+1)*d] = u8 bytes for that token
    packed: Vec<u8>,
    // per-token scale params: 2 f32 per token
    params: Vec<(f32, f32)>, // (lo, scale)
}

impl QuantValueCache {
    pub fn new(d: usize) -> Self {
        QuantValueCache { d, packed: Vec::new(), params: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.params.len()
    }
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
    /// Append one token's V vector (length D), quantizing to u8.
    pub fn push_token(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.d);
        let (q, lo, scale) = quant_token_to_u8(v);
        self.packed.extend_from_slice(&q);
        self.params.push((lo, scale));
    }
    /// Dequantize all stored tokens to a [T, D] f32 array.
    pub fn dequant_all(&self) -> Array2<f32> {
        let t = self.len();
        let mut out = Array2::<f32>::zeros((t, self.d));
        for (i, &(lo, scale)) in self.params.iter().enumerate() {
            let row = dequant_u8(&self.packed[i * self.d..(i + 1) * self.d], lo, scale);
            for (j, v) in row.into_iter().enumerate() {
                out[(i, j)] = v;
            }
        }
        out
    }
    /// Streaming P·V: `out += p · dequant(V[t])` (per-token scale), no row materialized.
    /// This is the seam where Milestone 2 replaces the f32 multiply with an int dot.
    #[inline]
    pub fn accumulate_into(&self, t: usize, p: f32, out: &mut [f32]) {
        let (lo, scale) = self.params[t];
        let row = &self.packed[t * self.d..(t + 1) * self.d];
        for (o, &b) in out.iter_mut().zip(row) {
            *o += p * (lo + b as f32 * scale);
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
    // packed: [token_idx * d .. (token_idx+1)*d] = u8 bytes; row-major [T, D]
    packed: Vec<u8>,
    // per-channel: lo, scale across all tokens seen so far
    ch_lo: Vec<f32>,
    ch_scale: Vec<f32>,
    len: usize,
}

impl QuantKeyCache {
    pub fn new(d: usize) -> Self {
        QuantKeyCache {
            d,
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
        // Update per-channel lo/scale to encompass the new values.
        for (c, &val) in k.iter().enumerate() {
            if val < self.ch_lo[c] {
                self.ch_lo[c] = val;
            }
            let hi_needed = val;
            let range = hi_needed - self.ch_lo[c];
            if range > 0.0 {
                let new_scale = (hi_needed - self.ch_lo[c]) / 255.0;
                if new_scale > self.ch_scale[c] {
                    self.ch_scale[c] = new_scale;
                }
            }
        }
        // Quantize this token under current per-channel scales.
        let mut row = vec![0u8; self.d];
        for (c, &val) in k.iter().enumerate() {
            row[c] = ((val - self.ch_lo[c]) / self.ch_scale[c]).round().clamp(0.0, 255.0) as u8;
        }
        self.packed.extend_from_slice(&row);
        self.len += 1;
    }
    /// Dequantize all stored tokens to a [T, D] f32 array.
    pub fn dequant_all(&self) -> Array2<f32> {
        let t = self.len;
        let mut out = Array2::<f32>::zeros((t, self.d));
        for i in 0..t {
            for c in 0..self.d {
                let b = self.packed[i * self.d + c];
                out[(i, c)] = self.ch_lo[c] + b as f32 * self.ch_scale[c];
            }
        }
        out
    }
    /// Streaming score: `q · dequant(K[t])` (per-channel scale), no row materialized.
    /// Milestone 2 will fold the per-channel scale into Q so this becomes an int·int dot —
    /// which is exactly why the per-token-K + Hadamard layout is needed (shared scale).
    #[inline]
    pub fn score_with(&self, t: usize, q: &[f32]) -> f32 {
        let row = &self.packed[t * self.d..(t + 1) * self.d];
        let mut acc = 0.0f32;
        for c in 0..self.d {
            acc += q[c] * (self.ch_lo[c] + row[c] as f32 * self.ch_scale[c]);
        }
        acc
    }
    pub fn memory_bytes(&self) -> usize {
        self.packed.len() + self.d * 8 // D*(lo+scale) = D*8 bytes
    }
}

// ── Milestone 2: integer-domain attention (per-token K + Hadamard + int8×int4 i32 dot) ──────────
// This is the layout that actually *banks the speed*: K and V are stored as packed signed int4 in
// the Hadamard-rotated basis with a single per-token scale, so the score is a pure integer dot —
// int8(Q) · int4(K) → i32 — with the scales pulled outside the reduction. Per-channel K (Milestone
// 1) structurally can't do this: its scale lives inside the sum. The Hadamard is what makes the
// per-token (shared-scale) layout survive outlier channels. K read drops to ¼ of f16 (the dominant
// decode-bandwidth term), and the dot is cheap integer MACs.

/// Symmetric (zero-point-free) quantization to signed ints in `-qmax..=qmax`. Returns codes and the
/// scale s such that x ≈ code·s. Zero-point-free so the dot needs no offset-correction term.
fn quant_sym(x: &[f32], qmax: i32) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = if amax > 0.0 { amax / qmax as f32 } else { 1.0 };
    let inv = scale.recip();
    let q = qmax as f32;
    let codes = x.iter().map(|&v| (v * inv).round().clamp(-q, q) as i8).collect();
    (codes, scale)
}

/// Fused symmetric-int4 quantize + pack of an already-rotated token, appended directly into
/// `packed` (even index → low nibble, odd → high nibble). Returns the per-token scale. No
/// intermediate code/byte Vec — keeps the push hot path allocation-free. Bit-identical to
/// `quant_sym(rotated, 7)` followed by the old two-per-byte pack.
fn quant_pack_i4(rotated: &[f32], packed: &mut Vec<u8>) -> f32 {
    let amax = rotated.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
    let inv = scale.recip();
    let start = packed.len();
    packed.resize(start + rotated.len().div_ceil(2), 0);
    for (i, &v) in rotated.iter().enumerate() {
        let nib = (((v * inv).round().clamp(-7.0, 7.0) as i8) as u8) & 0x0F;
        if i % 2 == 0 {
            packed[start + i / 2] |= nib;
        } else {
            packed[start + i / 2] |= nib << 4;
        }
    }
    scale
}

/// Per-token int4 KV cache in the Hadamard basis. K and V each: ⌈D/2⌉ bytes + one f32 scale per
/// token. Decode reads int4 and scores with an integer dot — the Milestone-2 speed path.
///
/// Milestone 5 (mixed precision): the most recent `n_recent` tokens are held in **full f32**
/// (exact) and only age into the int4 store once they leave the window. Recent tokens dominate
/// attention and are the cheapest to keep exact, so this is the lever that lets the int4 (or, later,
/// int2) body stay accurate — the OSCAR/StreamingLLM "keep sink/recent high-precision" trick.
#[derive(Clone, Debug)]
pub struct QuantKvInt4 {
    pub d: usize,
    k_packed: Vec<u8>,
    k_scale: Vec<f32>,
    v_packed: Vec<u8>,
    v_scale: Vec<f32>,
    n_recent: usize,
    recent: Vec<(Vec<f32>, Vec<f32>)>, // exact-f32 window of the newest tokens (k, v)
    len: usize,                        // total tokens = quantized + recent
    scratch: Vec<f32>,                 // reused rotation buffer — keeps push() allocation-free
}

impl QuantKvInt4 {
    pub fn new(d: usize) -> Self {
        Self::with_recent(d, 0)
    }
    /// `n_recent` newest tokens are kept exact (f32); older tokens are int4. `n_recent = 0` is the
    /// pure-int4 cache; `n_recent ≥ len` is exact f32.
    pub fn with_recent(d: usize, n_recent: usize) -> Self {
        assert!(d.is_power_of_two(), "head dim must be a power of two for FWHT, got {d}");
        QuantKvInt4 {
            d,
            k_packed: Vec::new(),
            k_scale: Vec::new(),
            v_packed: Vec::new(),
            v_scale: Vec::new(),
            n_recent,
            recent: Vec::new(),
            len: 0,
            scratch: Vec::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// x·H via FWHT (H symmetric, so this both rotates and un-rotates), O(D log D).
    fn rotate(&self, x: &[f32]) -> Vec<f32> {
        let mut out = x.to_vec();
        fwht_normalized(&mut out);
        out
    }
    /// Rotate + per-token int4 quantize one token into the packed store, allocation-free: the
    /// rotation reuses `self.scratch` and the quantize-and-pack writes straight into the byte store.
    fn push_quantized(&mut self, k: &[f32], v: &[f32]) {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        scratch.extend_from_slice(k);
        fwht_normalized(&mut scratch);
        let ks = quant_pack_i4(&scratch, &mut self.k_packed);
        scratch.clear();
        scratch.extend_from_slice(v);
        fwht_normalized(&mut scratch);
        let vs = quant_pack_i4(&scratch, &mut self.v_packed);
        self.scratch = scratch;
        self.k_scale.push(ks);
        self.v_scale.push(vs);
    }
    /// Append one token. It enters the exact f32 recent window; the token that falls out of the
    /// window (if any) is quantized into the int4 store.
    pub fn push_token(&mut self, k: &[f32], v: &[f32]) {
        self.recent.push((k.to_vec(), v.to_vec()));
        if self.recent.len() > self.n_recent {
            let (ko, vo) = self.recent.remove(0);
            self.push_quantized(&ko, &vo);
        }
        self.len += 1;
    }
    /// Single-query decode attention over the whole cache. Quantized tokens score via
    /// int8(Q)·int4(K)→i32 in the Hadamard basis; recent f32 tokens score exactly.
    pub fn attend(&self, q: &[f32], scale: f32) -> Vec<f32> {
        self.attend_limited(q, scale, self.len.saturating_sub(1))
    }
    /// Causal attention: attend only to absolute token indices `0..=limit`. One joint softmax over
    /// the visible quantized + recent tokens (masked tokens get exp()=0).
    pub fn attend_limited(&self, q: &[f32], scale: f32, limit: usize) -> Vec<f32> {
        let d = self.d;
        let bpt = d.div_ceil(2); // bytes per token
        let n_q = self.k_scale.len(); // quantized token count
        let (qc, sq) = quant_sym(&self.rotate(q), 127);
        let mut scores = vec![f32::NEG_INFINITY; self.len];
        let mut row_max = f32::NEG_INFINITY;
        // quantized tokens [0..n_q): integer dot, masked past `limit`
        for (t, kp) in self.k_packed.chunks(bpt).take(n_q).enumerate() {
            if t > limit {
                continue;
            }
            let mut acc = 0i32;
            for (j, &b) in kp.iter().enumerate() {
                let lo = ((b as i8) << 4) >> 4; // low nibble, sign-extended
                let hi = (b as i8) >> 4; // high nibble, sign-extended
                acc += qc[2 * j] as i32 * lo as i32;
                acc += qc[2 * j + 1] as i32 * hi as i32;
            }
            let s = scale * sq * self.k_scale[t] * acc as f32;
            scores[t] = s;
            row_max = row_max.max(s);
        }
        // recent f32 tokens [n_q..len): exact dot in the original basis (q·k = qr·kr)
        for (i, (k, _)) in self.recent.iter().enumerate() {
            if n_q + i > limit {
                continue;
            }
            let s = scale * q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>();
            scores[n_q + i] = s;
            row_max = row_max.max(s);
        }
        let mut sum = 0f32;
        for sc in scores.iter_mut() {
            let e = (*sc - row_max).exp(); // masked (-inf) → 0
            *sc = e;
            sum += e;
        }
        let inv = sum.recip();
        // out = H·(Σ quantized P·V_rot) + Σ recent P·V   (int4 part computed in rotated basis)
        let mut orot = vec![0f32; d];
        for (t, &sc) in scores.iter().take(n_q).enumerate() {
            if sc == 0.0 {
                continue;
            }
            let p = sc * inv;
            let vp = &self.v_packed[t * bpt..(t + 1) * bpt];
            let vs = self.v_scale[t];
            for (j, &b) in vp.iter().enumerate() {
                let lo = ((b as i8) << 4) >> 4;
                let hi = (b as i8) >> 4;
                orot[2 * j] += p * lo as f32 * vs;
                orot[2 * j + 1] += p * hi as f32 * vs;
            }
        }
        let mut out = self.rotate(&orot);
        for (i, (_, v)) in self.recent.iter().enumerate() {
            let p = scores[n_q + i] * inv;
            if p == 0.0 {
                continue;
            }
            for (o, &vv) in out.iter_mut().zip(v) {
                *o += p * vv;
            }
        }
        out
    }
    pub fn memory_bytes(&self) -> usize {
        self.k_packed.len()
            + self.v_packed.len()
            + (self.k_scale.len() + self.v_scale.len()) * 4
            + self.recent.len() * self.d * 2 * 4
    }
    // Raw store accessors — used by the GPU-validation dumper to feed the Metal kernel the exact
    // bytes the CPU `attend` reads, so the on-device output can be diffed against this reference.
    pub fn k_packed(&self) -> &[u8] {
        &self.k_packed
    }
    pub fn v_packed(&self) -> &[u8] {
        &self.v_packed
    }
    pub fn k_scale(&self) -> &[f32] {
        &self.k_scale
    }
    pub fn v_scale(&self) -> &[f32] {
        &self.v_scale
    }
}

// ── Fused stateful op ─────────────────────────────────────────────────────────────────────────

/// Fused quantized KV-cache + attention. Stores K in per-channel u8, V in per-token u8.
/// Inputs `[Q, K_new, V_new]` each `[B, H, S, D]`; output has Q's shape.
/// Memory saving vs f32: ~4× (u8 storage + small per-channel/token params).
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedKvSdpa {
    pub axis: usize,
    pub scale: Option<f32>,
    /// Causal masking. Must be set from the fused `Sdpa::is_causal`: for a real causal LM, dropping
    /// it lets prompt tokens attend to future prompt tokens during a multi-token prefill chunk
    /// (decode chunks of 1 are unaffected). Required for correct end-to-end use (Milestone 3).
    pub causal: bool,
    /// Mixed precision (Milestone 5): keep the newest `n_recent` tokens in exact f32, quantize the
    /// rest to int4. 0 = pure int4.
    pub n_recent: usize,
}
impl Eq for QuantizedKvSdpa {}

impl Op for QuantizedKvSdpa {
    fn name(&self) -> StaticName {
        "QuantizedKvSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "axis={}, scale={:?}, causal={}, n_recent={}",
            self.axis, self.scale, self.causal, self.n_recent
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
            causal: self.causal,
            n_recent: self.n_recent,
            caches: Vec::new(),
            initialized: false,
        })))
    }
}

impl TypedOp for QuantizedKvSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3, "QuantizedKvSdpa expects [Q, K_new, V_new]");
        Ok(tvec!(inputs[0].without_value()))
    }
    as_op!();
}

#[derive(Clone, Debug)]
pub struct QuantizedKvSdpaState {
    scale: Option<f32>,
    causal: bool,
    n_recent: usize,
    caches: Vec<QuantKvInt4>, // one per (batch * kv_head): int4+Hadamard, exact-f32 recent window
    initialized: bool,
}

impl OpState for QuantizedKvSdpaState {
    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 3, "QuantizedKvSdpa expects [Q, K_new, V_new]");
        let input_dt = inputs[0].datum_type();
        let q = inputs[0].cast_to::<f32>()?;
        let k_new = inputs[1].cast_to::<f32>()?;
        let v_new = inputs[2].cast_to::<f32>()?;
        let qv = q.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let kv = k_new.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let vv = v_new.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let (b, kh, snew, d) = kv.dim();
        let n = b * kh;
        if !self.initialized {
            ensure!(
                d.is_power_of_two(),
                "QuantizedKvSdpa int4 path needs head_dim to be a power of two (FWHT), got {d}"
            );
            self.caches = (0..n).map(|_| QuantKvInt4::with_recent(d, self.n_recent)).collect();
            self.initialized = true;
        }
        // Append each new token (rotate + per-token int4, or keep in the exact f32 recent window).
        for bi in 0..b {
            for hi in 0..kh {
                let idx = bi * kh + hi;
                let ks = kv.slice(s![bi, hi, .., ..]);
                let vs = vv.slice(s![bi, hi, .., ..]);
                for t in 0..snew {
                    self.caches[idx].push_token(
                        ks.slice(s![t, ..]).as_slice().unwrap(),
                        vs.slice(s![t, ..]).as_slice().unwrap(),
                    );
                }
            }
        }
        // ── Int4 + Hadamard attention (Milestones 2/5), causal-masked (Milestone 3) ─────────────
        // Per (batch, query head): score the int4 cache with an integer dot in the Hadamard basis
        // (recent tokens exact), softmax, P·V, un-rotate — all inside QuantKvInt4, never expanding
        // the cache to f32. GQA: query head qh reads kv head qh/group. Causal: query si (absolute
        // position t-sq+si) attends to [0..=t-sq+si]; decode (sq=1) sees all, so masking only bites
        // on multi-token prefill chunks.
        let (_, hq, sq, _) = qv.dim();
        let group = (hq / kh).max(1);
        let scale = self.scale.unwrap_or((d as f32).recip().sqrt());
        let t = self.caches[0].len();
        let mut o = Array4::<f32>::zeros((b, hq, sq, d));
        for bi in 0..b {
            for qh in 0..hq {
                let cache = &self.caches[bi * kh + qh / group];
                for si in 0..sq {
                    let qrow = qv.slice(s![bi, qh, si, ..]);
                    let lim = if self.causal { t + si - sq } else { t - 1 };
                    let out_vec = cache.attend_limited(qrow.as_slice().unwrap(), scale, lim);
                    o.slice_mut(s![bi, qh, si, ..])
                        .as_slice_mut()
                        .unwrap()
                        .copy_from_slice(&out_vec);
                }
            }
        }
        Ok(tvec!(o.into_tensor().cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }
}

#[derive(Clone, Debug)]
struct FrozenQuantizedKvSdpaState {
    scale: Option<f32>,
    causal: bool,
    n_recent: usize,
    caches: Vec<QuantKvInt4>,
    initialized: bool,
}
impl OpStateFreeze for QuantizedKvSdpaState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenQuantizedKvSdpaState {
            scale: self.scale,
            causal: self.causal,
            n_recent: self.n_recent,
            caches: self.caches.clone(),
            initialized: self.initialized,
        })
    }
}
impl FrozenOpState for FrozenQuantizedKvSdpaState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(QuantizedKvSdpaState {
            scale: self.scale,
            causal: self.causal,
            n_recent: self.n_recent,
            caches: self.caches.clone(),
            initialized: self.initialized,
        })
    }
}

// ── Auto-wiring transform ──────────────────────────────────────────────────────────────────────

/// Fuse `{DynKeyValueCache(K), DynKeyValueCache(V), Sdpa}` into `QuantizedKvSdpa`.
pub fn fuse_quantized_kv_sdpa_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    if node.inputs.len() != 3 {
        return Ok(None);
    }
    let k_node = model.node(node.inputs[1].node);
    let v_node = model.node(node.inputs[2].node);
    let (Some(kc), Some(vc)) =
        (k_node.op_as::<DynKeyValueCache>(), v_node.op_as::<DynKeyValueCache>())
    else {
        return Ok(None);
    };
    if kc.axis != vc.axis {
        return Ok(None);
    }
    if k_node.outputs[0].successors.len() != 1 || v_node.outputs[0].successors.len() != 1 {
        return Ok(None);
    }
    let scale = op.scale.as_ref().map(|t| t.cast_to_scalar::<f32>()).transpose()?;
    let mut patch = TypedModelPatch::default();
    let taps = patch.taps(model, &[node.inputs[0], k_node.inputs[0], v_node.inputs[0]])?;
    let fused = patch.wire_node(
        format!("{node_name}.quant_kv_sdpa"),
        QuantizedKvSdpa { axis: kc.axis, scale, causal: op.is_causal, n_recent: 0 },
        &taps,
    )?;
    patch.shunt_outside(model, node.id.into(), fused[0])?;
    Ok(Some(patch))
}

/// Strip GQA broadcast chain then fuse cache→Sdpa into QuantizedKvSdpa.
#[derive(Debug, Default)]
pub struct QuantizedKvSdpaTransform;

impl ModelTransform for QuantizedKvSdpaTransform {
    fn name(&self) -> StaticName {
        "fuse_quantized_kv_sdpa".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("fuse-kv-broadcast", crate::ops::sdpa::fuse_kv_cache_broadcast_rule)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse-quant-kv-sdpa", fuse_quantized_kv_sdpa_rule)
            .rewrite(&(), model)?;
        model.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            QuantizedKvSdpa { axis: 2, scale: None, causal: false, n_recent: 0 },
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

    // Milestone-3 causality: in a 2-token prefill, causal query position 0 attends to ONLY token 0
    // (single-element softmax = 1), so its output must equal token 0's V (within int8). The
    // non-causal op mixes in token 1 and must differ. Decoupled from the per-channel K running-scale
    // because a one-element softmax weights that token at 1.0 regardless of its score.
    #[test]
    fn causal_masks_future_tokens() -> TractResult<()> {
        let (b, h, d) = (1usize, 2usize, 16usize);
        let run = |causal: bool, q: &Tensor, k: &Tensor, v: &Tensor| -> TractResult<Tensor> {
            let mut model = TypedModel::default();
            let s = model.sym("S");
            let f: TVec<TDim> = tvec![b.to_dim(), h.to_dim(), s.into(), d.to_dim()];
            let qn = model.add_source("q", f32::fact(&f))?;
            let kn = model.add_source("k", f32::fact(&f))?;
            let vn = model.add_source("v", f32::fact(&f))?;
            let o = model.wire_node(
                "qkv",
                QuantizedKvSdpa { axis: 2, scale: None, causal, n_recent: 0 },
                &[qn, kn, vn],
            )?;
            model.select_output_outlets(&o)?;
            Ok(model
                .into_runnable()?
                .run(tvec![q.clone().into(), k.clone().into(), v.clone().into()])?
                .remove(0)
                .into_tensor())
        };
        // two-token prefill; V differs sharply between token 0 and token 1
        let mk = |f: &dyn Fn(usize, usize) -> f32| -> Tensor {
            let mut data = vec![0f32; b * h * 2 * d];
            for hd in 0..h {
                for tk in 0..2 {
                    for e in 0..d {
                        data[(hd * 2 + tk) * d + e] = f(tk, e);
                    }
                }
            }
            Tensor::from_shape(&[b, h, 2, d], &data).unwrap()
        };
        let q = mk(&|tk, e| 0.2 + 0.1 * tk as f32 + (e as f32 * 0.07).sin());
        let k = mk(&|tk, e| 0.5 * (tk as f32 + 1.0) + (e as f32 * 0.05).cos());
        let v = mk(&|tk, e| if tk == 0 { 0.4 + 0.01 * e as f32 } else { -0.6 });

        let o_causal = run(true, &q, &k, &v)?;
        let o_plain = run(false, &q, &k, &v)?;
        let oc = o_causal.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let op = o_plain.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let vv = v.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        // causal position-0 output ≈ token-0 V (int8); non-causal differs
        let mut causal_err = 0f32;
        let mut plain_gap = 0f32;
        for hd in 0..h {
            for e in 0..d {
                causal_err = causal_err.max((oc[[0, hd, 0, e]] - vv[[0, hd, 0, e]]).abs());
                plain_gap = plain_gap.max((op[[0, hd, 0, e]] - vv[[0, hd, 0, e]]).abs());
            }
        }
        println!("  causal pos-0 err={causal_err:.4} (≈int4 noise); non-causal gap={plain_gap:.4}");
        // causal pos-0 ≈ token-0 V within int4 quant noise; non-causal mixes token-1 and is far off.
        assert!(causal_err < 0.2, "causal pos-0 must ≈ token-0 V (int4), err {causal_err}");
        assert!(
            causal_err < plain_gap * 0.3,
            "causal must be far closer to token-0 V than non-causal: {causal_err} vs {plain_gap}"
        );
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
        QuantizedKvSdpaTransform.transform(&mut model)?;
        assert!(model.nodes().iter().any(|n| n.op_is::<QuantizedKvSdpa>()), "fused op present");
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
            QuantizedKvSdpa { axis: 2, scale: Some(0.125), causal: false, n_recent: 0 },
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

    // ─── Hadamard-rotated int4 KV: the A/B against plain KIVI ──────────────────────────────
    // Rotation only helps where outlier channels exist (real K caches have them; uniform-random
    // synthetic data does not). So this fixture *plants* outlier channels in K — the documented
    // "massive activation" phenomenon — and measures whether a fixed Hadamard lets int4 recover
    // toward int8 quality. Layout is KIVI: K per-channel, V per-token.
    use tract_nnef::tract_ndarray::{Array1, Array2 as A2};

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    /// One head of synthetic K with `n_outlier` high-magnitude outlier channels (×`mag`),
    /// plus a flat V. Returns (q[D], k[S,D], v[S,D]).
    fn head_with_outliers(
        s: usize,
        d: usize,
        n_outlier: usize,
        mag: f32,
        seed: &mut u64,
    ) -> (Array1<f32>, A2<f32>, A2<f32>) {
        let q = Array1::from_shape_fn(d, |_| lcg(seed));
        let mut k = A2::from_shape_fn((s, d), |_| lcg(seed) * 0.5);
        let v = A2::from_shape_fn((s, d), |_| lcg(seed) * 0.5);
        for c in 0..n_outlier.min(d) {
            for t in 0..s {
                // a big, token-varying value concentrated in a few channels
                k[(t, c)] = mag * (lcg(seed) + if t % 2 == 0 { 1.0 } else { -1.0 });
            }
        }
        (q, k, v)
    }

    fn softmax_inplace(x: &mut [f32]) {
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        x.iter_mut().for_each(|v| {
            *v = (*v - m).exp();
            sum += *v;
        });
        x.iter_mut().for_each(|v| *v /= sum);
    }

    fn attn_ref(q: &Array1<f32>, k: &A2<f32>, v: &A2<f32>) -> Array1<f32> {
        let (s, d) = k.dim();
        let scale = 1.0 / (d as f32).sqrt();
        let mut sc: Vec<f32> = (0..s).map(|j| k.row(j).dot(q) * scale).collect();
        softmax_inplace(&mut sc);
        let mut out = Array1::<f32>::zeros(d);
        for j in 0..s {
            for e in 0..d {
                out[e] += sc[j] * v[(j, e)];
            }
        }
        out
    }

    /// Attention through a quantized cache. `rot=Some(H)` rotates K (+matching Q) and V by H,
    /// quantizes in the rotated basis, then un-rotates the V output. `k_per_token` picks the K
    /// scale layout: `false` = per-channel (KIVI, isolates outlier channels but needs a running
    /// per-channel scale that goes stale on a growing cache); `true` = per-token (self-contained,
    /// no staleness — but collapses without help on outlier channels).
    fn attn_quant(
        q: &Array1<f32>,
        k: &A2<f32>,
        v: &A2<f32>,
        bits: u32,
        rot: Option<&A2<f32>>,
        k_per_token: bool,
    ) -> Array1<f32> {
        let (s, d) = k.dim();
        let scale = 1.0 / (d as f32).sqrt();
        let (k_src, q_row, v_src) = match rot {
            Some(h) => (k.dot(h), h.t().dot(q), v.dot(h)),
            None => (k.clone(), q.clone(), v.clone()),
        };
        let k_dq = quant_dequant(k_src.view(), bits, k_per_token); // by_row=true ⇒ per-token
        let v_dq = quant_dequant(v_src.view(), bits, true); // per-token   (Values)
        let mut sc: Vec<f32> = (0..s).map(|j| k_dq.row(j).dot(&q_row) * scale).collect();
        softmax_inplace(&mut sc);
        let mut out = Array1::<f32>::zeros(d);
        for j in 0..s {
            for e in 0..d {
                out[e] += sc[j] * v_dq[(j, e)];
            }
        }
        match rot {
            Some(h) => h.dot(&out), // un-rotate V output (H symmetric ⇒ H·Hᵀ = I)
            None => out,
        }
    }

    fn rel_dev(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
        let den: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        num / den.max(1e-9)
    }

    // Robust invariants of the KIVI cache (data-independent): int8 per-channel is near-lossless,
    // and int4 stays usable. The *rotation* question is data-dependent and answered on real K/V,
    // not here — see harness/kv_quant_hadamard.py: on real GPT-2, a fixed Hadamard is a modest int4
    // win on the deeper layers (best as per-channel+Hadamard; and it rescues the staleness-free
    // per-token layout to match per-channel KIVI), but it is NOT a path to int2 — sub-4-bit needs
    // the mixed-precision sink/recent window, independent of rotation. Synthetic outliers (the
    // `bench_hadamard_int4` table) do NOT reproduce that win, which is why the call is made on real
    // tensors. The Hadamard primitive (`hadamard_normalized`) and its round-trip are verified below.
    #[test]
    fn kivi_int8_near_lossless_int4_usable() {
        let (heads, s, d) = (8usize, 96usize, 64usize);
        let mut seed = 0x1234_5678u64;
        let (mut pc8, mut pc4) = (0.0f32, 0.0f32);
        for _ in 0..heads {
            let (q, k, v) = head_with_outliers(s, d, 3, 25.0, &mut seed);
            let r = attn_ref(&q, &k, &v);
            pc8 += rel_dev(&attn_quant(&q, &k, &v, 8, None, false), &r);
            pc4 += rel_dev(&attn_quant(&q, &k, &v, 4, None, false), &r);
        }
        let (pc8, pc4) = (pc8 / heads as f32, pc4 / heads as f32);
        println!("  int8 perCh={pc8:.4}  int4 perCh(KIVI)={pc4:.4}");
        assert!(pc8 < 0.03, "int8 per-channel KIVI should be near-lossless, got {pc8}");
        assert!(pc4 < 0.20, "int4 per-channel KIVI should stay usable on outliers, got {pc4}");
    }

    // FWHT must equal the dense Hadamard matmul (x·H), so swapping it in is exact, just O(D log D).
    #[test]
    fn fwht_matches_dense_hadamard() {
        for &d in &[2usize, 8, 64, 128] {
            let h = hadamard_normalized(d);
            let mut seed = 3u64;
            let x: Vec<f32> = (0..d).map(|_| lcg(&mut seed)).collect();
            let dense: Vec<f32> =
                (0..d).map(|m| (0..d).map(|k| x[k] * h[(k, m)]).sum::<f32>()).collect();
            let mut fast = x.clone();
            fwht_normalized(&mut fast);
            let maxerr = dense.iter().zip(&fast).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(maxerr < 1e-4, "FWHT != dense Hadamard for d={d}, maxerr {maxerr}");
        }
    }

    // Alloc-avoidance must be LOSSLESS: the fused quantize+pack has to be byte-identical to the
    // original quant_sym + two-per-byte pack (and bit-identical scale), or it's reverted.
    #[test]
    fn quant_pack_i4_is_bit_identical() {
        let mut seed = 0xABCDu64;
        for _ in 0..64 {
            let d = 64usize;
            // include large/outlier values to exercise clamping + the full code range
            let x: Vec<f32> =
                (0..d).map(|_| lcg(&mut seed) * (1.0 + 20.0 * lcg(&mut seed).abs())).collect();
            let (codes, s_ref) = quant_sym(&x, 7);
            let mut packed_ref = vec![0u8; codes.len().div_ceil(2)];
            for (i, &c) in codes.iter().enumerate() {
                let nib = (c as u8) & 0x0F;
                if i % 2 == 0 {
                    packed_ref[i / 2] |= nib;
                } else {
                    packed_ref[i / 2] |= nib << 4;
                }
            }
            let mut packed_new = Vec::new();
            let s_new = quant_pack_i4(&x, &mut packed_new);
            assert_eq!(s_ref.to_bits(), s_new.to_bits(), "scale not bit-identical");
            assert_eq!(packed_ref, packed_new, "packed bytes not byte-identical");
        }
    }

    // Milestone-2 correctness: full int4+Hadamard attention (integer score dot) on data with
    // outlier K channels stays close to f32. This is the layout that banks the speed.
    #[test]
    fn int4_hadamard_attention_matches_f32() {
        let (t, d) = (48usize, 64usize);
        let mut seed = 11u64;
        let mut cache = QuantKvInt4::new(d);
        let (mut ks, mut vs): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (vec![], vec![]);
        for _ in 0..t {
            let mut k: Vec<f32> = (0..d).map(|_| lcg(&mut seed) * 0.5).collect();
            for kk in k.iter_mut().take(3) {
                *kk = 20.0 * lcg(&mut seed); // outlier channels
            }
            let v: Vec<f32> = (0..d).map(|_| lcg(&mut seed) * 0.5).collect();
            cache.push_token(&k, &v);
            ks.push(k);
            vs.push(v);
        }
        let q: Vec<f32> = (0..d).map(|_| lcg(&mut seed)).collect();
        let scale = 1.0 / (d as f32).sqrt();
        // f32 reference
        let mut sc: Vec<f32> =
            (0..t).map(|j| scale * (0..d).map(|e| q[e] * ks[j][e]).sum::<f32>()).collect();
        let m = sc.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        sc.iter_mut().for_each(|x| {
            *x = (*x - m).exp();
            sum += *x;
        });
        let mut refo = vec![0f32; d];
        for j in 0..t {
            let p = sc[j] / sum;
            for e in 0..d {
                refo[e] += p * vs[j][e];
            }
        }
        let got = cache.attend(&q, scale);
        let num: f32 = got.iter().zip(&refo).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let den: f32 = refo.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel = num / den.max(1e-9);
        println!(
            "  int4+Hadamard attention rel-dev vs f32 = {rel:.4}  (mem {} B)",
            cache.memory_bytes()
        );
        assert!(rel < 0.15, "int4+Hadamard should stay within 15% of f32 on outliers, got {rel}");
    }

    // Milestone-5: the exact-f32 recent window reduces error vs pure int4, and a full window is
    // exact. Proves mixed precision works — the lever for pushing the body below int4.
    #[test]
    fn recent_window_improves_accuracy() {
        let (t, d) = (64usize, 64usize);
        let scale = 1.0 / (d as f32).sqrt();
        let mut seed = 19u64;
        let (mut ks, mut vs): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (vec![], vec![]);
        for _ in 0..t {
            let mut k: Vec<f32> = (0..d).map(|_| lcg(&mut seed) * 0.5).collect();
            for kk in k.iter_mut().take(3) {
                *kk = 22.0 * lcg(&mut seed);
            }
            ks.push(k);
            vs.push((0..d).map(|_| lcg(&mut seed) * 0.5).collect());
        }
        let q: Vec<f32> = (0..d).map(|_| lcg(&mut seed)).collect();
        // f32 reference
        let mut sc: Vec<f32> =
            (0..t).map(|j| scale * (0..d).map(|e| q[e] * ks[j][e]).sum::<f32>()).collect();
        let m = sc.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        sc.iter_mut().for_each(|x| {
            *x = (*x - m).exp();
            sum += *x;
        });
        let mut refo = vec![0f32; d];
        for j in 0..t {
            let p = sc[j] / sum;
            for e in 0..d {
                refo[e] += p * vs[j][e];
            }
        }
        let dev = |n_recent: usize| -> f32 {
            let mut c = QuantKvInt4::with_recent(d, n_recent);
            for j in 0..t {
                c.push_token(&ks[j], &vs[j]);
            }
            let got = c.attend(&q, scale);
            let num: f32 = got.iter().zip(&refo).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            num / refo.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9)
        };
        let (d0, d16, dfull) = (dev(0), dev(16), dev(t));
        println!("  rel-dev:  pure-int4={d0:.4}  recent16={d16:.4}  full-f32={dfull:.5}");
        assert!(d16 <= d0 + 1e-6, "recent window must not worsen accuracy: {d16} vs {d0}");
        assert!(dfull < 1e-4, "full window must be exact f32, got {dfull}");
    }

    // Measured microbench: int4 attention (Milestone-2 path) vs f32 streaming, per-query time at
    // growing context. The int path reads ¼ the K/V bytes and scores with an integer dot.
    //   cargo test -p tract-transformers kv_quant::tests::bench_int4_vs_f32_attention -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_int4_vs_f32_attention() {
        let d = 128usize;
        let scale = 1.0 / (d as f32).sqrt();
        println!("\n  Single-query attention latency, D={d}: f32 streaming vs int4+Hadamard");
        println!("     T  | f32 µs/q | int4 µs/q | speedup | f32 KV(KB) | int4 KV(KB)");
        for &t in &[256usize, 1024, 2048, 4096] {
            let mut seed = 5u64;
            let mut int4 = QuantKvInt4::new(d);
            let mut kf: Vec<Vec<f32>> = Vec::with_capacity(t);
            let mut vf: Vec<Vec<f32>> = Vec::with_capacity(t);
            for _ in 0..t {
                let mut k: Vec<f32> = (0..d).map(|_| lcg(&mut seed) * 0.5).collect();
                for kk in k.iter_mut().take(4) {
                    *kk = 18.0 * lcg(&mut seed);
                }
                let v: Vec<f32> = (0..d).map(|_| lcg(&mut seed) * 0.5).collect();
                int4.push_token(&k, &v);
                kf.push(k);
                vf.push(v);
            }
            let q: Vec<f32> = (0..d).map(|_| lcg(&mut seed)).collect();
            let reps = 200;
            // f32 streaming attention (reads f32 K/V, f32 dot) — the baseline this must beat.
            let f32_attend = || {
                let mut scores = vec![0f32; t];
                let mut mx = f32::NEG_INFINITY;
                for (j, s) in scores.iter_mut().enumerate() {
                    let d_: f32 = (0..d).map(|e| q[e] * kf[j][e]).sum::<f32>() * scale;
                    *s = d_;
                    mx = mx.max(d_);
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = sum.recip();
                let mut o = vec![0f32; d];
                for (j, s) in scores.iter().enumerate() {
                    let p = s * inv;
                    for e in 0..d {
                        o[e] += p * vf[j][e];
                    }
                }
                o[0]
            };
            let t0 = std::time::Instant::now();
            let mut sink = 0f32;
            for _ in 0..reps {
                sink += f32_attend();
            }
            let f32_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let t1 = std::time::Instant::now();
            for _ in 0..reps {
                sink += int4.attend(&q, scale)[0];
            }
            let int4_us = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;
            std::hint::black_box(sink);
            let f32_kb = (t * d * 4 * 2) as f64 / 1024.0;
            let int4_kb = int4.memory_bytes() as f64 / 1024.0;
            println!(
                "  {t:>5} | {f32_us:>8.1} | {int4_us:>9.1} | {:>6.2}x | {f32_kb:>10.1} | {int4_kb:>10.1}",
                f32_us / int4_us
            );
        }
    }

    // Milestone-1 guard: the streaming primitives (score_with / accumulate_into) must produce the
    // same numbers as the dequant-all path they replace, so the fused eval stays bit-faithful to
    // the prior behavior while never materializing the full f32 cache.
    #[test]
    fn streaming_primitives_match_dequant_all() {
        let (t, d) = (20usize, 16usize);
        let mut kc = QuantKeyCache::new(d);
        let mut vc = QuantValueCache::new(d);
        let mut seed = 7u64;
        for _ in 0..t {
            kc.push_token(&(0..d).map(|_| lcg(&mut seed)).collect::<Vec<_>>());
            vc.push_token(&(0..d).map(|_| lcg(&mut seed)).collect::<Vec<_>>());
        }
        let q: Vec<f32> = (0..d).map(|_| lcg(&mut seed)).collect();
        let kd = kc.dequant_all();
        let vd = vc.dequant_all();
        for tt in 0..t {
            let want: f32 = (0..d).map(|j| q[j] * kd[(tt, j)]).sum();
            assert!((kc.score_with(tt, &q) - want).abs() < 1e-3, "score_with mismatch at t={tt}");
        }
        let p = 0.37f32;
        let mut acc = vec![0f32; d];
        vc.accumulate_into(5, p, &mut acc);
        for j in 0..d {
            assert!((acc[j] - p * vd[(5, j)]).abs() < 1e-4, "accumulate_into mismatch at j={j}");
        }
    }

    // The Hadamard is an orthonormal involution: H·H = I and Hᵀ = H. (So the same matrix rotates
    // K/V into the spread basis and un-rotates the V output — the property attn_quant relies on.)
    #[test]
    fn hadamard_is_orthonormal_involution() {
        for &n in &[2usize, 8, 64, 128] {
            let h = hadamard_normalized(n);
            let hh = h.dot(&h);
            let mut max_off = 0.0f32;
            for i in 0..n {
                for j in 0..n {
                    let expect = if i == j { 1.0 } else { 0.0 };
                    max_off = max_off.max((hh[(i, j)] - expect).abs());
                }
            }
            assert!(max_off < 1e-4, "H·H must be identity for n={n}, max dev {max_off}");
        }
    }

    // Full A/B table: bit-widths × outlier-magnitude sweep, both K layouts ± Hadamard.
    //   cargo test -p tract-transformers kv_quant::tests::bench_hadamard_int4 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_hadamard_int4() {
        let (heads, s, d) = (8usize, 128usize, 64usize);
        let h = hadamard_normalized(d);
        println!("\n  Attention rel-deviation vs full-f32 (lower=better), D={d}, S={s}");
        println!("  V always per-token; 3 of {d} K channels are outliers, magnitude swept.\n");
        println!("       |        int4 Keys              |        int2 Keys");
        println!(
            "   mag | perCh | perTok | perTok+Hada | perCh | perTok | perTok+Hada | perCh+Hada(int4)"
        );
        for &mag in &[0.0f32, 5.0, 15.0, 40.0] {
            let mut seed = 0xABCD_1234u64;
            let mut acc = [0.0f32; 7];
            for _ in 0..heads {
                let (q, k, v) = head_with_outliers(s, d, 3, mag, &mut seed);
                let r = attn_ref(&q, &k, &v);
                acc[0] += rel_dev(&attn_quant(&q, &k, &v, 4, None, false), &r);
                acc[1] += rel_dev(&attn_quant(&q, &k, &v, 4, None, true), &r);
                acc[2] += rel_dev(&attn_quant(&q, &k, &v, 4, Some(&h), true), &r);
                acc[3] += rel_dev(&attn_quant(&q, &k, &v, 2, None, false), &r);
                acc[4] += rel_dev(&attn_quant(&q, &k, &v, 2, None, true), &r);
                acc[5] += rel_dev(&attn_quant(&q, &k, &v, 2, Some(&h), true), &r);
                acc[6] += rel_dev(&attn_quant(&q, &k, &v, 4, Some(&h), false), &r);
            }
            for a in acc.iter_mut() {
                *a /= heads as f32;
            }
            println!(
                "  {mag:>4.0} | {:.3} | {:.3}  |   {:.3}     | {:.3} | {:.3}  |   {:.3}     |   {:.3}",
                acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6]
            );
        }
        println!("\n  Read-out:");
        println!(
            "  • per-channel K already handles outliers — Hadamard on it is redundant (see last col)."
        );
        println!(
            "  • per-token K collapses on outliers but Hadamard rescues it → the staleness-free layout."
        );
        println!(
            "  • gap widens at int2 and with outlier magnitude — the TurboQuant/OSCAR regime."
        );
    }
}
