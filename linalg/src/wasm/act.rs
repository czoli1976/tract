#[cfg(target_feature = "relaxed-simd")]
use crate::frame::element_wise::ElementWiseKer;

// Relaxed-SIMD activation kernels (f32, FMA path).
//
// `f32x4_relaxed_madd(a, b, c)` computes `a * b + c`. On hosts with hardware
// FMA (all ARM64, x86_64 with FMA3) it lowers to a single fused, single-
// rounded instruction. On hosts without, it falls back to mul+add — hence
// "relaxed". The result is therefore not bit-deterministic across all hosts,
// but it is at least as accurate as the separate mul+add (FMA does fewer
// roundings).
//
// For sigmoid/tanh polynomial evaluation, the 14 muladds in the Horner chain
// fuse cleanly. Measured ~1.65x over the baseline-simd128 explicit kernel and
// over LLVM auto-vec'd scalar on V8. silu, gelu and erf reuse the same
// Horner-with-FMA pattern (silu/gelu share the sigmoid/tanh Padé
// polynomials; erf runs the Abramowitz & Stegun chain).
//
// Gated on `target_feature = "relaxed-simd"` because `f32x4_relaxed_madd`
// requires the relaxed-simd proposal to be enabled at compile time.
// ---------------------------------------------------------------------------

#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Debug)]
pub struct WasmSigmoid4Relaxed;

#[cfg(target_feature = "relaxed-simd")]
impl ElementWiseKer<f32> for WasmSigmoid4Relaxed {
    fn name() -> &'static str {
        "wasm_relaxed_simd"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        // Coefficients match generic/sigmoid.rs::ssigmoid bit-for-bit.
        // Output may differ by ≤1 ulp from scalar on FMA hosts (more accurate).
        const LOW: f32 = -18.6;
        const HIGH: f32 = -LOW;

        const ALPHA_13: f32 = -4.433153405e-18;
        const ALPHA_11: f32 = 1.169974371e-14;
        const ALPHA_9: f32 = -1.875289645e-11;
        const ALPHA_7: f32 = 4.257889523e-8;
        const ALPHA_5: f32 = 0.00004811817576;
        const ALPHA_3: f32 = 0.008163842030;
        const ALPHA_1: f32 = 0.2499999971;

        const BETA_6: f32 = 3.922935744e-6;
        const BETA_4: f32 = 0.001524872358;
        const BETA_2: f32 = 0.1159886749;
        const BETA_0: f32 = 1.0;

        unsafe {
            let lo = f32x4_splat(LOW);
            let hi = f32x4_splat(HIGH);

            let a13 = f32x4_splat(ALPHA_13);
            let a11 = f32x4_splat(ALPHA_11);
            let a9 = f32x4_splat(ALPHA_9);
            let a7 = f32x4_splat(ALPHA_7);
            let a5 = f32x4_splat(ALPHA_5);
            let a3 = f32x4_splat(ALPHA_3);
            let a1 = f32x4_splat(ALPHA_1);

            let b6 = f32x4_splat(BETA_6);
            let b4 = f32x4_splat(BETA_4);
            let b2 = f32x4_splat(BETA_2);
            let b0 = f32x4_splat(BETA_0);

            let half = f32x4_splat(0.5);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let x = f32x4_min(hi, f32x4_max(lo, v));
                let x2 = f32x4_mul(x, x);

                // Horner numerator with FMA: pn = x2 * pn + a_n
                let pn = a13;
                let pn = f32x4_relaxed_madd(x2, pn, a11);
                let pn = f32x4_relaxed_madd(x2, pn, a9);
                let pn = f32x4_relaxed_madd(x2, pn, a7);
                let pn = f32x4_relaxed_madd(x2, pn, a5);
                let pn = f32x4_relaxed_madd(x2, pn, a3);
                let pn = f32x4_relaxed_madd(x2, pn, a1);
                let pn = f32x4_mul(pn, x);

                // Horner denominator with FMA
                let qn = b6;
                let qn = f32x4_relaxed_madd(x2, qn, b4);
                let qn = f32x4_relaxed_madd(x2, qn, b2);
                let qn = f32x4_relaxed_madd(x2, qn, b0);

                let r = f32x4_add(f32x4_div(pn, qn), half);
                v128_store(p as *mut v128, r);
                p = p.add(4);
            }
        }
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Debug)]
pub struct WasmTanh4Relaxed;

#[cfg(target_feature = "relaxed-simd")]
impl ElementWiseKer<f32> for WasmTanh4Relaxed {
    fn name() -> &'static str {
        "wasm_relaxed_simd"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        const LOW: f32 = -8.9;
        const HIGH: f32 = 8.9;

        const ALPHA_13: f32 = -8.488492677e-14;
        const ALPHA_11: f32 = 5.277853000e-11;
        const ALPHA_9: f32 = -2.022500419e-8;
        const ALPHA_7: f32 = 0.00001115424833;
        const ALPHA_5: f32 = 0.003103950131;
        const ALPHA_3: f32 = 0.1308400453;
        const ALPHA_1: f32 = 0.9999999934;

        const BETA_6: f32 = 0.0002546136580;
        const BETA_4: f32 = 0.02449515379;
        const BETA_2: f32 = 0.4641733162;
        const BETA_0: f32 = 1.0;

        unsafe {
            let lo = f32x4_splat(LOW);
            let hi = f32x4_splat(HIGH);

            let a13 = f32x4_splat(ALPHA_13);
            let a11 = f32x4_splat(ALPHA_11);
            let a9 = f32x4_splat(ALPHA_9);
            let a7 = f32x4_splat(ALPHA_7);
            let a5 = f32x4_splat(ALPHA_5);
            let a3 = f32x4_splat(ALPHA_3);
            let a1 = f32x4_splat(ALPHA_1);

            let b6 = f32x4_splat(BETA_6);
            let b4 = f32x4_splat(BETA_4);
            let b2 = f32x4_splat(BETA_2);
            let b0 = f32x4_splat(BETA_0);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let x = f32x4_min(hi, f32x4_max(lo, v));
                let x2 = f32x4_mul(x, x);

                let pn = a13;
                let pn = f32x4_relaxed_madd(x2, pn, a11);
                let pn = f32x4_relaxed_madd(x2, pn, a9);
                let pn = f32x4_relaxed_madd(x2, pn, a7);
                let pn = f32x4_relaxed_madd(x2, pn, a5);
                let pn = f32x4_relaxed_madd(x2, pn, a3);
                let pn = f32x4_relaxed_madd(x2, pn, a1);
                let pn = f32x4_mul(pn, x);

                let qn = b6;
                let qn = f32x4_relaxed_madd(x2, qn, b4);
                let qn = f32x4_relaxed_madd(x2, qn, b2);
                let qn = f32x4_relaxed_madd(x2, qn, b0);

                let r = f32x4_div(pn, qn);
                v128_store(p as *mut v128, r);
                p = p.add(4);
            }
        }
    }
}

/// SiLU (swish): `silu(x) = x * sigmoid(x)`, with the sigmoid evaluated by the
/// same Padé polynomial and ±18.6 clamp as `WasmSigmoid4Relaxed`.
///
/// The multiplying factor is `max(x, -18.6)` — clamped below so the negative
/// tail saturates at `-18.6 * sigmoid(-18.6) ≈ -1.55e-7` instead of growing
/// linearly with `x * sigmoid(-18.6)`, and left unclamped above because
/// `silu(x) → x` as `x → +inf`. Mirrors `arm64simd_silu_f32_4n_fused`.
#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Debug)]
pub struct WasmSilu4Relaxed;

#[cfg(target_feature = "relaxed-simd")]
impl ElementWiseKer<f32> for WasmSilu4Relaxed {
    fn name() -> &'static str {
        "wasm_relaxed_simd"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        // Sigmoid coefficients, matching generic/sigmoid.rs::ssigmoid.
        const LOW: f32 = -18.6;
        const HIGH: f32 = -LOW;

        const ALPHA_13: f32 = -4.433153405e-18;
        const ALPHA_11: f32 = 1.169974371e-14;
        const ALPHA_9: f32 = -1.875289645e-11;
        const ALPHA_7: f32 = 4.257889523e-8;
        const ALPHA_5: f32 = 0.00004811817576;
        const ALPHA_3: f32 = 0.008163842030;
        const ALPHA_1: f32 = 0.2499999971;

        const BETA_6: f32 = 3.922935744e-6;
        const BETA_4: f32 = 0.001524872358;
        const BETA_2: f32 = 0.1159886749;
        const BETA_0: f32 = 1.0;

        unsafe {
            let lo = f32x4_splat(LOW);
            let hi = f32x4_splat(HIGH);

            let a13 = f32x4_splat(ALPHA_13);
            let a11 = f32x4_splat(ALPHA_11);
            let a9 = f32x4_splat(ALPHA_9);
            let a7 = f32x4_splat(ALPHA_7);
            let a5 = f32x4_splat(ALPHA_5);
            let a3 = f32x4_splat(ALPHA_3);
            let a1 = f32x4_splat(ALPHA_1);

            let b6 = f32x4_splat(BETA_6);
            let b4 = f32x4_splat(BETA_4);
            let b2 = f32x4_splat(BETA_2);
            let b0 = f32x4_splat(BETA_0);

            let half = f32x4_splat(0.5);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let f = f32x4_max(lo, v);
                let x = f32x4_min(hi, f);
                let x2 = f32x4_mul(x, x);

                let pn = a13;
                let pn = f32x4_relaxed_madd(x2, pn, a11);
                let pn = f32x4_relaxed_madd(x2, pn, a9);
                let pn = f32x4_relaxed_madd(x2, pn, a7);
                let pn = f32x4_relaxed_madd(x2, pn, a5);
                let pn = f32x4_relaxed_madd(x2, pn, a3);
                let pn = f32x4_relaxed_madd(x2, pn, a1);
                let pn = f32x4_mul(pn, x);

                let qn = b6;
                let qn = f32x4_relaxed_madd(x2, qn, b4);
                let qn = f32x4_relaxed_madd(x2, qn, b2);
                let qn = f32x4_relaxed_madd(x2, qn, b0);

                let sig = f32x4_add(f32x4_div(pn, qn), half);
                let r = f32x4_mul(f, sig);
                v128_store(p as *mut v128, r);
                p = p.add(4);
            }
        }
    }
}

/// Tanh-form GELU (pow=3) matching tract's `GeluApproximate`:
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))`,
/// with the tanh evaluated by the same Padé polynomial and ±8.9 clamp as
/// `WasmTanh4Relaxed`. Single pass: argument, tanh and the final
/// `0.5 * x * (1 + tanh)` are fused per 4-lane block, mirroring
/// `arm64simd_gelu_f32_4n_fused`.
#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Debug)]
pub struct WasmGelu4Relaxed;

#[cfg(target_feature = "relaxed-simd")]
impl ElementWiseKer<f32> for WasmGelu4Relaxed {
    fn name() -> &'static str {
        "wasm_relaxed_simd"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        // 0.044715 * sqrt(2/pi): the tanh argument is evaluated as
        // x * (SQRT_2_OVER_PI + COEF_X3 * x²).
        const COEF_X3: f32 = 0.03567739613;

        // Tanh coefficients, matching generic/tanh.rs::stanh.
        const LOW: f32 = -8.9;
        const HIGH: f32 = 8.9;

        const ALPHA_13: f32 = -8.488492677e-14;
        const ALPHA_11: f32 = 5.277853000e-11;
        const ALPHA_9: f32 = -2.022500419e-8;
        const ALPHA_7: f32 = 0.00001115424833;
        const ALPHA_5: f32 = 0.003103950131;
        const ALPHA_3: f32 = 0.1308400453;
        const ALPHA_1: f32 = 0.9999999934;

        const BETA_6: f32 = 0.0002546136580;
        const BETA_4: f32 = 0.02449515379;
        const BETA_2: f32 = 0.4641733162;
        const BETA_0: f32 = 1.0;

        unsafe {
            let k0 = f32x4_splat(SQRT_2_OVER_PI);
            let k3 = f32x4_splat(COEF_X3);

            let lo = f32x4_splat(LOW);
            let hi = f32x4_splat(HIGH);

            let a13 = f32x4_splat(ALPHA_13);
            let a11 = f32x4_splat(ALPHA_11);
            let a9 = f32x4_splat(ALPHA_9);
            let a7 = f32x4_splat(ALPHA_7);
            let a5 = f32x4_splat(ALPHA_5);
            let a3 = f32x4_splat(ALPHA_3);
            let a1 = f32x4_splat(ALPHA_1);

            let b6 = f32x4_splat(BETA_6);
            let b4 = f32x4_splat(BETA_4);
            let b2 = f32x4_splat(BETA_2);
            let b0 = f32x4_splat(BETA_0);

            let half = f32x4_splat(0.5);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let v2 = f32x4_mul(v, v);
                let u = f32x4_mul(v, f32x4_relaxed_madd(k3, v2, k0));
                let x = f32x4_min(hi, f32x4_max(lo, u));
                let x2 = f32x4_mul(x, x);

                let pn = a13;
                let pn = f32x4_relaxed_madd(x2, pn, a11);
                let pn = f32x4_relaxed_madd(x2, pn, a9);
                let pn = f32x4_relaxed_madd(x2, pn, a7);
                let pn = f32x4_relaxed_madd(x2, pn, a5);
                let pn = f32x4_relaxed_madd(x2, pn, a3);
                let pn = f32x4_relaxed_madd(x2, pn, a1);
                let pn = f32x4_mul(pn, x);

                let qn = b6;
                let qn = f32x4_relaxed_madd(x2, qn, b4);
                let qn = f32x4_relaxed_madd(x2, qn, b2);
                let qn = f32x4_relaxed_madd(x2, qn, b0);

                let th = f32x4_div(pn, qn);
                let half_v = f32x4_mul(half, v);
                let r = f32x4_relaxed_madd(half_v, th, half_v);
                v128_store(p as *mut v128, r);
                p = p.add(4);
            }
        }
    }
}

/// Error function, Abramowitz & Stegun 7.1.26, mirroring `generic/erf.rs::serf`:
/// `erf(x) = copysign(1 - (1 + p(|x|))⁻¹⁶, x)` with the degree-6 polynomial
/// run as an FMA Horner chain and the 16th power as four squarings. The
/// rational tail keeps `1 - r ∈ [0, 1)`, so the sign is re-applied by OR-ing
/// the argument's sign bit back in.
#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Debug)]
pub struct WasmErf4Relaxed;

#[cfg(target_feature = "relaxed-simd")]
impl ElementWiseKer<f32> for WasmErf4Relaxed {
    fn name() -> &'static str {
        "wasm_relaxed_simd"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        const A1: f32 = 0.0705230784;
        const A2: f32 = 0.0422820123;
        const A3: f32 = 0.0092705272;
        const A4: f32 = 0.0001520143;
        const A5: f32 = 0.0002765672;
        const A6: f32 = 0.0000430638;

        unsafe {
            let a1 = f32x4_splat(A1);
            let a2 = f32x4_splat(A2);
            let a3 = f32x4_splat(A3);
            let a4 = f32x4_splat(A4);
            let a5 = f32x4_splat(A5);
            let a6 = f32x4_splat(A6);

            let one = f32x4_splat(1.0);
            let sign_mask = i32x4_splat(i32::MIN);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let sign = v128_and(v, sign_mask);
                let t = f32x4_abs(v);

                let y = a6;
                let y = f32x4_relaxed_madd(t, y, a5);
                let y = f32x4_relaxed_madd(t, y, a4);
                let y = f32x4_relaxed_madd(t, y, a3);
                let y = f32x4_relaxed_madd(t, y, a2);
                let y = f32x4_relaxed_madd(t, y, a1);
                let y = f32x4_mul(y, t);

                let u = f32x4_add(y, one);
                let u2 = f32x4_mul(u, u);
                let u4 = f32x4_mul(u2, u2);
                let u8 = f32x4_mul(u4, u4);
                let u16 = f32x4_mul(u8, u8);

                let r = f32x4_sub(one, f32x4_div(one, u16));
                let r = v128_or(r, sign);
                v128_store(p as *mut v128, r);
                p = p.add(4);
            }
        }
    }
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
#[macro_use]
mod test_wasm_sigmoid_relaxed {
    sigmoid_frame_tests!(true, f32, crate::wasm::WasmSigmoid4Relaxed);
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
#[macro_use]
mod test_wasm_tanh_relaxed {
    tanh_frame_tests!(true, f32, crate::wasm::WasmTanh4Relaxed);
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
#[macro_use]
mod test_wasm_silu_relaxed {
    silu_frame_tests!(true, f32, crate::wasm::WasmSilu4Relaxed);
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
#[macro_use]
mod test_wasm_gelu_relaxed {
    gelu_frame_tests!(true, f32, crate::wasm::WasmGelu4Relaxed);
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
#[macro_use]
mod test_wasm_erf_relaxed {
    erf_frame_tests!(true, f32, crate::wasm::WasmErf4Relaxed);
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
mod microbench_activations {
    //! Microbench: WASM SIMD activations vs the generic scalar fallback.
    //! Sizes mirror typical RNN/transformer hidden dims (256, 512, 1024).
    //!
    //! Run with:
    //!   RUSTFLAGS='-C target-feature=+simd128' \
    //!     CARGO_TARGET_WASM32_WASIP1_RUNNER='wasmtime --env RUST_TEST_NOCAPTURE=1 --' \
    //!     cargo test --release --target=wasm32-wasip1 -p tract-linalg \
    //!     wasm::microbench_activations::microbench -- --nocapture --ignored
    use crate::frame::element_wise::ElementWiseKer;
    use std::time::Instant;

    fn ns_per_call<K: ElementWiseKer<f32>>(buf: &mut [f32], iters: usize) -> f64 {
        // Warmup
        for _ in 0..50 {
            K::run(buf, ());
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            K::run(buf, ());
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9
    }

    fn bench_pair<Scalar: ElementWiseKer<f32>, Simd: ElementWiseKer<f32>>(
        label: &str,
        op: &str,
        n: usize,
        iters: usize,
    ) {
        // Same input for both kernels — rebuild between to avoid post-clamp
        // saturation mucking up the measurement.
        let make = || (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.5).collect::<Vec<f32>>();

        let mut buf = make();
        let scalar = ns_per_call::<Scalar>(&mut buf, iters);
        let mut buf = make();
        let simd = ns_per_call::<Simd>(&mut buf, iters);

        eprintln!(
            "{label} {op} n={n} iters={iters}: scalar={scalar:.0} ns simd={simd:.0} ns ({:.2}x)",
            scalar / simd,
        );
    }

    fn bench(label: &str, n: usize, iters: usize) {
        use crate::generic;
        use crate::wasm;
        bench_pair::<generic::sigmoid::SSigmoid4, wasm::WasmSigmoid4Relaxed>(
            label, "sigmoid", n, iters,
        );
        bench_pair::<generic::tanh::STanh4, wasm::WasmTanh4Relaxed>(label, "tanh", n, iters);
        bench_pair::<generic::silu::SSiLU4, wasm::WasmSilu4Relaxed>(label, "silu", n, iters);
        bench_pair::<generic::gelu::SGelu4, wasm::WasmGelu4Relaxed>(label, "gelu", n, iters);
        bench_pair::<generic::erf::SErf4, wasm::WasmErf4Relaxed>(label, "erf", n, iters);
    }

    #[test]
    #[ignore]
    fn microbench() {
        eprintln!("=== WASM SIMD activations: scalar vs simd ===");
        bench("hidden=256", 256, 5_000);
        bench("hidden=512", 512, 3_000);
        bench("hidden=1024", 1024, 2_000);
    }
}
