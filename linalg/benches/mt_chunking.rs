//! Multi-threaded MMM chunking bench + correctness check.
//!
//! Validates that the cache-adaptive `chunk_grid` (which only changes how tiles
//! are partitioned across rayon workers) produces bit-identical results to the
//! single-thread walk, and A/Bs the cache-adaptive sizing against the ggml
//! baseline *interleaved within one process* (budget toggled at runtime via
//! `set_mm_chunk_l2_budget`) to cancel thermal drift.
//!
//!   cargo bench -p tract-linalg --features multithread-mm --bench mt_chunking
//!
//! Threads via TRACT_BENCH_THREADS (default 4); adaptive budget via
//! TRACT_MM_CHUNK_L2_BYTES (default 512 KiB).

use std::time::Instant;
use tract_data::internal::*;
use tract_linalg::mmm::{set_mm_chunk_l2_budget, AsInputValue, FusedSpec, MatMatMul, MMMInputValue};
use tract_linalg::multithread::{Executor, multithread_tract_scope, set_threading_panel_threshold};
use DatumType::F32;

const ADAPTIVE: usize = 512 * 1024;
const BASELINE: usize = usize::MAX; // huge => chunk clamps to ggml base (unmodified)

fn threads() -> usize {
    std::env::var("TRACT_BENCH_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4)
}

fn pack(
    mmm: &dyn MatMatMul,
    m: usize,
    k: usize,
    n: usize,
) -> (Box<dyn MMMInputValue>, Box<dyn MMMInputValue>) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let at = Tensor::from_shape(&[m, k], &a).unwrap();
    let bt = Tensor::from_shape(&[k, n], &b).unwrap();
    let p = &mmm.packings()[0];
    (p.0.prepare_one(&at, 1, 0).unwrap(), p.1.prepare_one(&bt, 0, 1).unwrap())
}

unsafe fn run_into(
    mmm: &dyn MatMatMul,
    m: usize,
    n: usize,
    pa: &dyn MMMInputValue,
    pb: &dyn MMMInputValue,
    c: &mut Tensor,
) {
    unsafe {
        mmm.run(
            m,
            n,
            &[
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(pa),
                    b: AsInputValue::Borrowed(pb),
                    packing: 0,
                },
                FusedSpec::Store(mmm.c_view(Some(0), Some(1)).wrap(&c.view_mut())),
            ],
        )
        .unwrap();
    }
}

fn check(exec: &Executor, m: usize, k: usize, n: usize) {
    let mmm = tract_linalg::ops().mmm(F32, Some(m), Some(k), Some(n)).unwrap();
    let (pa, pb) = pack(&*mmm, m, k, n);
    let mut c_st = Tensor::zero::<f32>(&[m, n]).unwrap();
    let mut c_mt = Tensor::zero::<f32>(&[m, n]).unwrap();
    unsafe { run_into(&*mmm, m, n, &*pa, &*pb, &mut c_st) };
    multithread_tract_scope(exec.clone(), || unsafe { run_into(&*mmm, m, n, &*pa, &*pb, &mut c_mt) });
    let eq = c_st == c_mt;
    eprintln!("  m={m} k={k} n={n}: MT vs ST {}", if eq { "BIT-EXACT" } else { "MISMATCH!!" });
    assert!(eq, "MT diverged from ST for m={m} k={k} n={n}");
}

// One timed measurement (avg us over `iters`) at the currently-set budget.
fn time_once(
    exec: &Executor,
    mmm: &dyn MatMatMul,
    m: usize,
    n: usize,
    pa: &dyn MMMInputValue,
    pb: &dyn MMMInputValue,
    c: &mut Tensor,
    iters: usize,
) -> f64 {
    multithread_tract_scope(exec.clone(), || {
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe { run_into(mmm, m, n, pa, pb, c) };
        }
        t0.elapsed().as_secs_f64() / iters as f64 * 1e6
    })
}

// Interleaved A/B: per rep, time adaptive then baseline (cancels drift); min of each.
fn ab(exec: &Executor, m: usize, k: usize, n: usize, iters: usize, reps: usize) {
    let mmm = tract_linalg::ops().mmm(F32, Some(m), Some(k), Some(n)).unwrap();
    let (pa, pb) = pack(&*mmm, m, k, n);
    let mut c = Tensor::zero::<f32>(&[m, n]).unwrap();
    // warmup
    for _ in 0..10 {
        unsafe { run_into(&*mmm, m, n, &*pa, &*pb, &mut c) };
    }
    let (mut a_min, mut b_min) = (f64::MAX, f64::MAX);
    for _ in 0..reps {
        set_mm_chunk_l2_budget(ADAPTIVE);
        a_min = a_min.min(time_once(exec, &*mmm, m, n, &*pa, &*pb, &mut c, iters));
        set_mm_chunk_l2_budget(BASELINE);
        b_min = b_min.min(time_once(exec, &*mmm, m, n, &*pa, &*pb, &mut c, iters));
    }
    eprintln!(
        "  m={m} k={k} n={n}: adaptive={a_min:.1} baseline={b_min:.1} us  speedup={:.3}x",
        b_min / a_min
    );
}

fn main() {
    set_threading_panel_threshold(0); // always thread, even for smaller shapes
    let nth = threads();
    let exec = Executor::multithread(nth);
    eprintln!("=== MT MMM chunking ({nth} threads, adaptive budget = {} KiB) ===", ADAPTIVE / 1024);

    eprintln!("-- correctness (MT vs single-thread) --");
    for &(m, k, n) in
        &[(256, 256, 256), (128, 1024, 128), (512, 64, 512), (64, 2048, 64), (300, 700, 200)]
    {
        check(&exec, m, k, n);
    }

    eprintln!("-- A/B adaptive vs ggml baseline (us/call, min-of-reps, interleaved) --");
    for &(m, k, n, it, rep) in &[
        (512, 512, 512, 200, 12),
        (256, 2048, 256, 200, 12),
        (1024, 256, 1024, 100, 12),
        (384, 4096, 384, 50, 12),
        (2048, 256, 2048, 30, 12),
        (512, 8192, 512, 20, 12),
    ] {
        ab(&exec, m, k, n, it, rep);
    }
}
