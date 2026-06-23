//! Dump a single-query int4+Hadamard attention test case for the Metal kernel to validate against
//! (bit-exact vs the proven CPU `QuantKvInt4::attend`) and to time int4 vs an f16 baseline on GPU.
//! Layout (little-endian):
//!   u32 T, u32 D, f32 scale, f32[D] q, f32[T] k_scale, f32[T] v_scale,
//!   u8[T*D/2] k_packed, u8[T*D/2] v_packed, f32[D] ref_out,
//!   f32[T*D] k_orig, f32[T*D] v_orig   (originals, for the f16 baseline)
//! Usage: dump_kv_int4 <path> [T]
use std::io::Write;
use tract_transformers::ops::kv_quant::QuantKvInt4;

fn main() {
    let d = 128usize;
    let t: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut seed = 0x00C0_FFEEu64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let mut cache = QuantKvInt4::new(d); // n_recent = 0: pure int4 path
    let (mut k_orig, mut v_orig): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    for _ in 0..t {
        let mut k: Vec<f32> = (0..d).map(|_| lcg() * 0.5).collect();
        for kk in k.iter_mut().take(4) {
            *kk = 18.0 * lcg(); // outlier channels — the Hadamard's reason to exist
        }
        let v: Vec<f32> = (0..d).map(|_| lcg() * 0.5).collect();
        cache.push_token(&k, &v);
        k_orig.extend_from_slice(&k);
        v_orig.extend_from_slice(&v);
    }
    let q: Vec<f32> = (0..d).map(|_| lcg()).collect();
    let refo = cache.attend(&q, scale);

    let mut buf: Vec<u8> = Vec::new();
    let push_f32 = |buf: &mut Vec<u8>, xs: &[f32]| {
        for &x in xs {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    };
    buf.extend_from_slice(&(t as u32).to_le_bytes());
    buf.extend_from_slice(&(d as u32).to_le_bytes());
    buf.extend_from_slice(&scale.to_le_bytes());
    push_f32(&mut buf, &q);
    push_f32(&mut buf, cache.k_scale());
    push_f32(&mut buf, cache.v_scale());
    buf.extend_from_slice(cache.k_packed());
    buf.extend_from_slice(cache.v_packed());
    push_f32(&mut buf, &refo);
    push_f32(&mut buf, &k_orig);
    push_f32(&mut buf, &v_orig);
    // causal reference: attend to only the first ⌈T/2⌉ tokens (the prefill causal-skip case)
    let causal_valid = t.div_ceil(2);
    let refo_causal = cache.attend_limited(&q, scale, causal_valid - 1);
    buf.extend_from_slice(&(causal_valid as u32).to_le_bytes());
    push_f32(&mut buf, &refo_causal);

    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/kv_int4_testcase.bin".into());
    std::fs::File::create(&path).unwrap().write_all(&buf).unwrap();
    eprintln!(
        "wrote {} bytes to {path}  (T={t} D={d}); ref_out[0..4]={:?}",
        buf.len(),
        &refo[0..4]
    );
}
