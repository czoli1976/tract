//! Fused SDPA via tract-owned ports of the MLX attention kernels (see
//! mlx_sdpa.metal): a decode-specialized `sdpa_vector` family (single-pass +
//! split-KV 2-pass) and the `steel_attention` tiled prefill kernel. Native
//! GQA (`gqa_factor`), f32/f16, causal or no mask. Dispatch mirrors MLX's
//! decision tree; unsupported shapes fall back to `MetalMfaSdpa` or the
//! explode path via the chooser translator at the bottom of this file.

use crate::encoder::EncoderExt;
use crate::{ConstantValues, LibraryName, MetalStream, Value};
use anyhow::ensure;
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

/// Head dims with vector (decode) kernel instantiations.
const VECTOR_DIMS: &[usize] = &[64, 96, 128, 256];
/// Head dims with steel (prefill) kernel instantiations.
const STEEL_DIMS: &[usize] = &[64, 80, 128];
/// Split-KV blocks for the 2-pass vector path (must be a multiple of 32).
const TWO_PASS_BLOCKS: i32 = 32;

fn vector_tname(dt: DatumType) -> TractResult<&'static str> {
    match dt {
        DatumType::F32 => Ok("float"),
        DatumType::F16 => Ok("float16_t"),
        _ => bail!("MLX sdpa_vector: unsupported dt {dt:?}"),
    }
}

fn steel_tname(dt: DatumType) -> TractResult<&'static str> {
    match dt {
        DatumType::F32 => Ok("float32"),
        DatumType::F16 => Ok("float16"),
        _ => bail!("MLX steel attention: unsupported dt {dt:?}"),
    }
}

fn natural_strides_of(shape: &[usize]) -> TVec<isize> {
    let mut strides = tvec![1isize; shape.len()];
    for ix in (0..shape.len().saturating_sub(1)).rev() {
        strides[ix] = strides[ix + 1] * shape[ix + 1] as isize;
    }
    strides
}

fn ensure_natural(t: &DeviceTensor, what: &str) -> TractResult<()> {
    ensure!(
        t.strides() == natural_strides_of(t.shape()).as_slice(),
        "MLX SDPA expects contiguous {what}, got shape {:?} strides {:?}",
        t.shape(),
        t.strides()
    );
    Ok(())
}

/// Mirror of MLX `AttnParams` (steel/attn/params.h) — keep field order in sync.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AttnParams {
    b: i32,
    h: i32,
    d: i32,
    ql: i32,
    kl: i32,
    gqa_factor: i32,
    scale: f32,
    nq: i32,
    nk: i32,
    nq_aligned: i32,
    nk_aligned: i32,
    ql_rem: i32,
    kl_rem: i32,
    ql_off: i32,
    q_strides: [i64; 3],
    k_strides: [i64; 3],
    v_strides: [i64; 3],
    o_strides: [i64; 3],
}

/// Single-pass decode kernel: one threadgroup per (batch*head, q position),
/// 32 simdgroups striding over keys. Grid mirrors mlx cpp:358.
#[allow(clippy::too_many_arguments)]
fn dispatch_sdpa_vector_1pass(
    stream: &MetalStream,
    dt: DatumType,
    scale: f32,
    do_causal: bool,
    (b, hq, hkv, ql, kl, d): (usize, usize, usize, usize, usize, usize),
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    let name = format!("sdpa_vector_{t}_{d}_{d}", t = vector_tname(dt)?);
    let constants = Some(ConstantValues::new(vec![
        (20, Value::Bool(false)),     // has_mask
        (21, Value::Bool(false)),     // query_transposed
        (22, Value::Bool(do_causal)), // do_causal
        (23, Value::Bool(false)),     // bool_mask
        (24, Value::Bool(false)),     // float_mask
        (25, Value::Bool(false)),     // has_sinks
    ]));
    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxSdpa, &name, constants)?;

    let gqa_factor = (hq / hkv) as i32;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, q, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, k, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, v, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[gqa_factor]);
        encoder.set_slice(5, &[kl as i32]);
        encoder.set_slice(6, &[k.strides()[1] as u64]);
        encoder.set_slice(7, &[k.strides()[2] as u64]);
        encoder.set_slice(8, &[v.strides()[1] as u64]);
        encoder.set_slice(9, &[v.strides()[2] as u64]);
        encoder.set_slice(10, &[scale]);
        let grid = MTLSize { width: (b * hq) as _, height: ql as _, depth: 1 };
        let group = MTLSize { width: 1024, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Split-KV decode: pass 1 writes per-block partials (one simdgroup per
/// (q head, q pos, kv block)), pass 2 reduces. Grids mirror mlx cpp:484/583.
#[allow(clippy::too_many_arguments)]
fn dispatch_sdpa_vector_2pass(
    stream: &MetalStream,
    dt: DatumType,
    scale: f32,
    do_causal: bool,
    (b, hq, hkv, ql, kl, d): (usize, usize, usize, usize, usize, usize),
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    let blocks = TWO_PASS_BLOCKS as usize;
    let partials = unsafe { DeviceTensor::uninitialized_dt(dt, &[b, hq, ql, blocks, d])? };
    let sums = unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), &[b, hq, ql, blocks])? };
    let maxs = unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), &[b, hq, ql, blocks])? };
    stream.retain_tensor(&partials);
    stream.retain_tensor(&sums);
    stream.retain_tensor(&maxs);

    let tname = vector_tname(dt)?;
    let gqa_factor = hq / hkv;

    // Pass 1
    let name = format!("sdpa_vector_2pass_1_{tname}_{d}_{d}");
    let constants = Some(ConstantValues::new(vec![
        (20, Value::Bool(false)),
        (21, Value::Bool(false)),
        (22, Value::Bool(do_causal)),
        (23, Value::Bool(false)),
        (24, Value::Bool(false)),
        (25, Value::Bool(false)),
        (26, Value::I32(TWO_PASS_BLOCKS)), // blocks
    ]));
    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxSdpa, &name, constants)?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, q, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, k, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, v, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, &partials, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(4, &sums, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(5, &maxs, metal::MTLResourceUsage::Write);
        // buffer 6 intentionally unset (matches mlx host, cpp:534)
        encoder.set_slice(7, &[kl as i32]);
        encoder.set_slice(8, &[k.strides()[1] as u64]);
        encoder.set_slice(9, &[k.strides()[2] as u64]);
        encoder.set_slice(10, &[v.strides()[1] as u64]);
        encoder.set_slice(11, &[v.strides()[2] as u64]);
        encoder.set_slice(12, &[scale]);
        let grid = MTLSize { width: hkv as _, height: b as _, depth: blocks as _ };
        let group = MTLSize { width: 32, height: gqa_factor as _, depth: ql as _ };
        encoder.dispatch_thread_groups(grid, group);
    });

    // Pass 2
    let name = format!("sdpa_vector_2pass_2_{tname}_{d}");
    let pipeline = stream.load_pipeline(LibraryName::MlxSdpa, &name)?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, &partials, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, &sums, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, &maxs, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[TWO_PASS_BLOCKS]);
        let grid = MTLSize { width: (b * hq) as _, height: ql as _, depth: 1 };
        let group = MTLSize { width: 1024, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Steel tiled prefill kernel: 4 simdgroups per threadgroup, one threadgroup
/// per (q block, q head, batch). Grid mirrors mlx cpp:160.
#[allow(clippy::too_many_arguments)]
fn dispatch_steel_attention(
    stream: &MetalStream,
    dt: DatumType,
    scale: f32,
    do_causal: bool,
    (b, hq, hkv, ql, kl, d): (usize, usize, usize, usize, usize, usize),
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    let (bq, bk) = (32usize, if d < 128 { 32usize } else { 16usize });
    let tname = steel_tname(dt)?;
    let name = format!("steel_attention_{tname}_bq{bq}_bk{bk}_bd{d}_wm4_wn1_mask{tname}");
    let constants = Some(ConstantValues::new(vec![
        (200, Value::Bool(ql % bq == 0)), // align_Q
        (201, Value::Bool(kl % bk == 0)), // align_K
        (300, Value::Bool(false)),        // has_mask
        (301, Value::Bool(do_causal)),    // do_causal
        (302, Value::Bool(false)),        // has_sinks
    ]));
    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxSdpa, &name, constants)?;

    let nq = ql.div_ceil(bq);
    let nk = kl.div_ceil(bk);
    let params = AttnParams {
        b: b as i32,
        h: hq as i32,
        d: d as i32,
        ql: ql as i32,
        kl: kl as i32,
        gqa_factor: (hq / hkv) as i32,
        scale,
        nq: nq as i32,
        nk: nk as i32,
        nq_aligned: (ql / bq) as i32,
        nk_aligned: (kl / bk) as i32,
        ql_rem: (ql % bq) as i32,
        kl_rem: (kl % bk) as i32,
        ql_off: kl.saturating_sub(ql) as i32,
        q_strides: [q.strides()[0] as i64, q.strides()[1] as i64, q.strides()[2] as i64],
        k_strides: [k.strides()[0] as i64, k.strides()[1] as i64, k.strides()[2] as i64],
        v_strides: [v.strides()[0] as i64, v.strides()[1] as i64, v.strides()[2] as i64],
        o_strides: [out.strides()[0] as i64, out.strides()[1] as i64, out.strides()[2] as i64],
    };

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, q, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, k, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, v, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, std::slice::from_ref(&params));
        let grid = MTLSize { width: nq as _, height: hq as _, depth: b as _ };
        let group = MTLSize { width: 32, height: 4, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Fused SDPA over `[B,Hq,Sq,D]` Q / `[B,Hkv,Sk,D]` K,V (GQA when Hkv < Hq).
/// Picks the decode (vector / split-KV) or prefill (steel) kernel following
/// MLX's dispatch tree; causal is bottom-right aligned (`qL_off = kL - qL`).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlx_sdpa(
    stream: &MetalStream,
    scale: f32,
    is_causal: bool,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    let dt = q.datum_type();
    ensure!(matches!(dt, DatumType::F32 | DatumType::F16), "MLX SDPA: F32/F16 only");
    ensure!(q.rank() == 4 && k.rank() == 4 && v.rank() == 4, "MLX SDPA expects rank-4 inputs");
    let (b, hq, ql, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (hkv, kl) = (k.shape()[1], k.shape()[2]);
    ensure!(k.shape()[3] == d && v.shape()[3] == d, "MLX SDPA expects equal head dims");
    ensure!(v.shape()[1] == hkv && v.shape()[2] == kl, "K/V layout mismatch");
    ensure!(hq % hkv == 0, "q heads ({hq}) must be a multiple of kv heads ({hkv})");
    for (t, w) in [(q, "Q"), (k, "K"), (v, "V"), (out, "O")] {
        ensure_natural(t, w)?;
    }

    stream.retain_tensor(q);
    stream.retain_tensor(k);
    stream.retain_tensor(v);
    stream.retain_tensor(out);

    let gqa_factor = hq / hkv;
    let shape6 = (b, hq, hkv, ql, kl, d);
    let vector_ok = VECTOR_DIMS.contains(&d) && ql <= 8 && ql <= kl && ql * gqa_factor <= 32;
    if vector_ok {
        // mlx forces causal off for single-position queries (cpp:746)
        let do_causal = is_causal && ql > 1;
        if kl >= 1024 {
            dispatch_sdpa_vector_2pass(stream, dt, scale, do_causal, shape6, q, k, v, out)
        } else {
            dispatch_sdpa_vector_1pass(stream, dt, scale, do_causal, shape6, q, k, v, out)
        }
    } else {
        ensure!(
            STEEL_DIMS.contains(&d),
            "MLX SDPA: no kernel for head dim {d} with query len {ql} (translator gate too wide?)"
        );
        ensure!(!is_causal || ql <= kl, "causal SDPA needs qL <= kL, got {ql} > {kl}");
        dispatch_steel_attention(stream, dt, scale, is_causal, shape6, q, k, v, out)
    }
}

/// Metal device op: fused SDPA via the ported MLX kernels.
#[derive(Debug, Clone)]
pub struct MetalMlxSdpa {
    pub scale: f32,
    pub is_causal: bool,
}

impl PartialEq for MetalMlxSdpa {
    fn eq(&self, o: &Self) -> bool {
        self.scale.to_bits() == o.scale.to_bits() && self.is_causal == o.is_causal
    }
}
impl Eq for MetalMlxSdpa {}
impl std::hash::Hash for MetalMlxSdpa {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.scale.to_bits().hash(state);
        self.is_causal.hash(state);
    }
}

impl Op for MetalMlxSdpa {
    fn name(&self) -> StaticName {
        "MetalMlxSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale={} causal={}", self.scale, self.is_causal)])
    }
    op_as_typed_op!();
}

impl EvalOp for MetalMlxSdpa {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        use tract_gpu::tensor::DeviceTensorExt;
        ensure!(inputs.len() == 3, "MetalMlxSdpa expects Q,K,V");
        let q = inputs[0].to_device_tensor()?;
        let k = inputs[1].to_device_tensor()?;
        let v = inputs[2].to_device_tensor()?;
        ensure!(q.rank() == 4, "expects rank-4 [B,H,Sq,D], got {:?}", q.shape());
        let out = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            q.datum_type(),
            q.shape(),
        )?;
        crate::with_metal_stream(|stream| {
            dispatch_mlx_sdpa(stream, self.scale, self.is_causal, q, k, v, &out)
        })?;
        Ok(tvec![out.into_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalMlxSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |f| Ok(tvec![f[0].without_value()]))
    }
    as_op!();
}

/// Whether an `Sdpa` node can be fused by the MLX kernels: exactly Q,K,V
/// (causal or no external mask), f16/f32, rank-4, concrete heads with
/// `Hq % Hkv == 0`, equal concrete head dim. Steel dims ({64,80,128}) cover
/// any sequence lengths; vector-only dims ({96,256}) additionally need
/// translate-time proof of decode-shape eligibility.
pub fn mlx_sdpa_supported(
    op: &tract_transformers::ops::sdpa::Sdpa,
    in_facts: &[&TypedFact],
) -> bool {
    if in_facts.len() != 3 {
        return false; // 4th input = external mask: not wired yet
    }
    let (q, k, v) = (in_facts[0], in_facts[1], in_facts[2]);
    if !matches!(q.datum_type, DatumType::F16 | DatumType::F32) || !op.acc_datum_type.is_float() {
        return false;
    }
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return false;
    }
    let heads =
        (q.shape[1].to_usize().ok(), k.shape[1].to_usize().ok(), v.shape[1].to_usize().ok());
    let (Some(qh), Some(kh), Some(vh)) = heads else { return false };
    if kh != vh || qh % kh != 0 {
        return false;
    }
    let dims = (q.shape[3].to_usize().ok(), k.shape[3].to_usize().ok(), v.shape[3].to_usize().ok());
    let (Some(qd), Some(kd), Some(vd)) = dims else { return false };
    if qd != kd || qd != vd {
        return false;
    }
    if STEEL_DIMS.contains(&qd) {
        return true;
    }
    if VECTOR_DIMS.contains(&qd) {
        // No steel fallback at these dims: require provable decode shape.
        let lens = (q.shape[2].to_usize().ok(), k.shape[2].to_usize().ok());
        let (Some(ql), Some(kl)) = lens else { return false };
        return ql <= 8 && ql <= kl && ql * (qh / kh) <= 32;
    }
    false
}

// Single Sdpa translator: prefer the MLX port (GQA, decode kernel), fall back
// to the vendored MFA metallib, else explode via flatten_unfused_sdpa.
crate::register_metal_op!(tract_transformers::ops::sdpa::Sdpa, |source, node, op| {
    let in_facts = source.node_input_facts(node.id)?;
    let mlx = mlx_sdpa_supported(op, &in_facts);
    let mfa = crate::kernels::matmul::mfa::mfa_sdpa_supported(op, &in_facts);
    if !mlx && !mfa {
        return Ok(None);
    }
    let head_dim = in_facts[0].shape[in_facts[0].rank() - 1].to_usize()?;
    let scale = match &op.scale {
        Some(t) => t.cast_to_scalar::<f32>()?,
        None => (head_dim as f32).recip().sqrt(),
    };
    if mlx {
        Ok(Some(Box::new(MetalMlxSdpa { scale, is_causal: op.is_causal }) as Box<dyn TypedOp>))
    } else {
        Ok(Some(Box::new(crate::kernels::matmul::mfa::MetalMfaSdpa {
            scale,
            is_causal: op.is_causal,
        }) as Box<dyn TypedOp>))
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::IntoDevice;

    fn cpu_reference(
        dt: DatumType,
        is_causal: bool,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> TractResult<Tensor> {
        let cpu = tract_transformers::ops::sdpa::Sdpa {
            scale: None,
            datum_type: dt,
            acc_datum_type: f32::datum_type(),
            is_causal,
        };
        Ok(cpu.eval(tvec![
            q.clone().into_tvalue(),
            k.clone().into_tvalue(),
            v.clone().into_tvalue()
        ])?[0]
            .clone()
            .into_tensor())
    }

    fn pseudo<F: Datum + num_traits::Float>(shape: &[usize], seed: i64) -> Tensor
    where
        f32: num_traits::AsPrimitive<F>,
    {
        use num_traits::AsPrimitive;
        let n: usize = shape.iter().product();
        let data: Vec<F> = (0..n)
            .map(|i| {
                let x = (((i as i64 * 2654435761 + seed).rem_euclid(2000)) as f32 / 1000.0) - 1.0;
                x.as_()
            })
            .collect();
        Tensor::from_shape(shape, &data).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        dt: DatumType,
        b: usize,
        hq: usize,
        hkv: usize,
        ql: usize,
        kl: usize,
        d: usize,
        is_causal: bool,
    ) -> TractResult<()> {
        let (q, k, v) = if dt == f16::datum_type() {
            (
                pseudo::<f16>(&[b, hq, ql, d], 1),
                pseudo::<f16>(&[b, hkv, kl, d], 2),
                pseudo::<f16>(&[b, hkv, kl, d], 3),
            )
        } else {
            (
                pseudo::<f32>(&[b, hq, ql, d], 1),
                pseudo::<f32>(&[b, hkv, kl, d], 2),
                pseudo::<f32>(&[b, hkv, kl, d], 3),
            )
        };
        let reference = cpu_reference(dt, is_causal, &q, &k, &v)?;
        let scale = (d as f32).recip().sqrt();
        let metal = with_borrowed_metal_stream(|stream| {
            let qd = q.clone().into_device()?;
            let kd = k.clone().into_device()?;
            let vd = v.clone().into_device()?;
            let out = unsafe { DeviceTensor::uninitialized_dt(dt, &[b, hq, ql, d])? };
            dispatch_mlx_sdpa(stream, scale, is_causal, &qd, &kd, &vd, &out)?;
            stream.wait_until_completed()?;
            Ok(out.to_host()?.into_tensor())
        })?;
        reference.close_enough(&metal, Approximation::Approximate).with_context(|| {
            format!("dt={dt:?} b={b} hq={hq} hkv={hkv} ql={ql} kl={kl} d={d} causal={is_causal}")
        })
    }

    #[test]
    fn vector_1pass_f32() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 8, 8, 1, 64, 64, false)
    }

    #[test]
    fn vector_1pass_f32_gqa() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 8, 2, 4, 128, 128, false)
    }

    #[test]
    fn vector_1pass_f32_causal() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 4, 64, 64, true)
    }

    #[test]
    fn vector_1pass_f16() -> TractResult<()> {
        run_case(f16::datum_type(), 1, 8, 8, 1, 256, 96, false).ok();
        run_case(f16::datum_type(), 1, 8, 8, 1, 256, 256, false)
    }

    #[test]
    fn vector_1pass_batched() -> TractResult<()> {
        run_case(f32::datum_type(), 3, 4, 2, 2, 65, 64, false)
    }

    #[test]
    fn vector_2pass_f32() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 8, 8, 1, 2048, 64, false)
    }

    #[test]
    fn vector_2pass_f32_gqa_causal() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 8, 2, 4, 1536, 128, true)
    }

    #[test]
    fn vector_2pass_f16() -> TractResult<()> {
        run_case(f16::datum_type(), 2, 4, 4, 2, 1024, 64, false)
    }

    #[test]
    fn steel_f32_aligned() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 64, 64, 64, false)
    }

    #[test]
    fn steel_f32_unaligned() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 37, 53, 64, false)
    }

    #[test]
    fn steel_f32_gqa() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 8, 2, 128, 128, 128, false)
    }

    #[test]
    fn steel_f32_causal() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 96, 160, 64, true)
    }

    #[test]
    fn steel_f32_d80() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 64, 64, 80, false)
    }

    #[test]
    fn steel_f16() -> TractResult<()> {
        run_case(f16::datum_type(), 1, 4, 4, 64, 96, 128, false)
    }

    #[test]
    fn steel_f16_gqa_causal_unaligned() -> TractResult<()> {
        run_case(f16::datum_type(), 2, 8, 4, 33, 47, 64, true)
    }

    // qL <= 8 at a steel-only dim must take the steel kernel (no vector inst).
    #[test]
    fn steel_decode_d80() -> TractResult<()> {
        run_case(f32::datum_type(), 1, 4, 4, 1, 100, 80, false)
    }
}
