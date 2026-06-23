//! int4 + Hadamard KV-cache decode attention, ported into tract-metal (Milestone 4 / follow-up #2).
//!
//! The kernel source (`nn/kv_int4.metal`) is validated **bit-exact** vs the CPU `QuantKvInt4::attend`
//! on an Apple M4 (`harness/kv_gpu_validate.swift`: scalar + SIMD, rel-dev 1e-6; SIMD 1.2–1.4× over
//! the f16 baseline; causal-skip via `valid` is bit-exact and latency ∝ valid). This module dispatches
//! it through the crate's own `MetalStream`/`DeviceTensor` machinery; the test below runs it on-device
//! and diffs against the CPU reference — the integration gate.
//!
//! Remaining (the larger part of #2): a stateful `MetalQuantizedKvSdpa` op holding the int4 cache in
//! GPU-resident `DeviceTensor` buffers (grown/appended on device across decode steps via the
//! Metal-backend `OpState`) + a CPU→Metal rewrite rule + the P-position prefill dispatch (each query
//! at position i dispatched with `valid = i+1`, the causal-skip).

use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::{MTLResourceUsage, MTLSize};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

/// Dispatch one (batch, query-head) of int4+Hadamard attention: score the int4 cache with an integer
/// dot, softmax, P·V, un-rotate — single query, attending to the first `valid` tokens (causal-skip).
///   q: [D] f32 · k_packed/v_packed: [T·D/2] u8 · k_scale/v_scale: [T] f32 · out: [D] f32
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attend(
    stream: &MetalStream,
    q: &DeviceTensor,
    k_packed: &DeviceTensor,
    k_scale: &DeviceTensor,
    v_packed: &DeviceTensor,
    v_scale: &DeviceTensor,
    d: usize,
    valid: usize,
    scale: f32,
    out: &DeviceTensor,
) -> TractResult<()> {
    for t in [q, k_packed, k_scale, v_packed, v_scale, out] {
        stream.retain_tensor(t);
    }
    let t = k_scale.len();
    let pipeline = stream.load_pipeline(LibraryName::KvInt4, "kv_int4_attend_simd")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, q, MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, k_packed, MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, k_scale, MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, v_packed, MTLResourceUsage::Read);
        encoder.set_metal_tensor(4, v_scale, MTLResourceUsage::Read);
        encoder.set_slice(5, &[t as u32, d as u32, valid as u32, 0u32]); // uint3 TD (16-byte slot)
        encoder.set_slice(6, &[scale]);
        encoder.set_metal_tensor(7, out, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: d as u64, height: 1, depth: 1 },
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::kv_quant::QuantKvInt4;

    // End-to-end through tract-metal: build the int4 cache on CPU (the proven path), upload its packed
    // buffers as DeviceTensors, dispatch the kernel on the GPU, diff vs CPU `attend` — and the causal
    // case vs `attend_limited`. (The Swift harness already proved the MSL bit-exact; this proves it
    // runs correctly through the crate's buffer/dispatch infrastructure.)
    #[test]
    fn kv_int4_dispatch_matches_cpu() -> TractResult<()> {
        let (t, d) = (48usize, 64usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let mut seed = 11u64;
        let mut lcg = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let mut cache = QuantKvInt4::new(d);
        for _ in 0..t {
            let mut k: Vec<f32> = (0..d).map(|_| lcg() * 0.5).collect();
            for kk in k.iter_mut().take(3) {
                *kk = 20.0 * lcg(); // outlier channels
            }
            let v: Vec<f32> = (0..d).map(|_| lcg() * 0.5).collect();
            cache.push_token(&k, &v);
        }
        let q: Vec<f32> = (0..d).map(|_| lcg()).collect();

        with_borrowed_metal_stream(|stream| {
            let q_dev = Tensor::from_shape(&[d], &q)?.into_device()?;
            let kp =
                Tensor::from_shape(&[cache.k_packed().len()], cache.k_packed())?.into_device()?;
            let vp =
                Tensor::from_shape(&[cache.v_packed().len()], cache.v_packed())?.into_device()?;
            let ks = Tensor::from_shape(&[t], cache.k_scale())?.into_device()?;
            let vs = Tensor::from_shape(&[t], cache.v_scale())?.into_device()?;

            for valid in [t, t / 2] {
                let reference =
                    Tensor::from_shape(&[d], &cache.attend_limited(&q, scale, valid - 1))?;
                let out = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[d])? };
                dispatch_attend(stream, &q_dev, &kp, &ks, &vp, &vs, d, valid, scale, &out)?;
                let got = out.to_host()?.into_tensor();
                reference
                    .close_enough(&got, Approximation::Approximate)
                    .with_context(|| format!("GPU dispatch != CPU attend at valid={valid}"))?;
            }
            Ok(())
        })
    }
}
