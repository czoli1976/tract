//! Decode-loop benchmark: measure real tok/s and KV-cache MB vs context length on a real model,
//! then derive the int8/int4 KV-quant gain *empirically* from the measured per-token time slope.
//!
//! At batch-1 decode the per-token attention cost is a GEMV over the cache (memory-bound: it reads
//! all of K and V each step), so per-token latency grows ~linearly with context and the slope is
//! dominated by KV bandwidth. Fitting step_ms = a + b·ctx isolates that slope; quantizing the cache
//! to int4 cuts the bytes read along it by 4× → ceiling projection a + 0.25·b·ctx.
use anyhow::Result;
use causal_llm::{CausalLlmModel, CausalLlmModelConfig};
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Measure decode tok/s and KV-cache MB vs context, and project KV-quant gains")]
struct Args {
    #[arg(short, long)]
    tokenizer: String,
    #[arg(short, long)]
    model: String,
    /// Number of tokens to decode
    #[arg(short, default_value = "2048")]
    n: usize,
    #[arg(long)]
    force_cpu: bool,
    #[arg(default_value = "The history of computing began in the nineteenth century when")]
    prompt: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let conf = CausalLlmModelConfig { force_cpu: args.force_cpu };
    let llm = CausalLlmModel::from_paths_and_conf(&args.tokenizer, &args.model, conf)?;
    let mut state = llm.spawn()?;
    state.append_text(&args.prompt)?;

    // Prefill the prompt (timed separately — it's compute-bound, not the decode regime).
    let t0 = Instant::now();
    state.generate_next_token()?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    let prompt_len = state.seq.len().saturating_sub(1);
    let dt = state.kv_cache_dt()?;
    let seq0 = state.kv_seq_len()?;
    let bytes_per_tok_per_ctx =
        if seq0 > 0 { state.kv_cache_bytes()? as f64 / seq0 as f64 } else { 0.0 };

    eprintln!(
        "# model loaded | prompt_tokens={prompt_len} prefill_ms={prefill_ms:.1} kv_dt={dt:?} \
         kv_bytes/token={:.0}",
        bytes_per_tok_per_ctx
    );
    println!("# step  ctx  step_ms  tok/s  kv_MB");

    let mut samples: Vec<(f64, f64)> = Vec::with_capacity(args.n); // (ctx, step_ms)
    for i in 0..args.n {
        let t = Instant::now();
        state.generate_next_token()?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let ctx = state.kv_seq_len()? as f64;
        samples.push((ctx, ms));
        if i % 128 == 0 || i + 1 == args.n {
            let kv_mb = state.kv_cache_bytes()? as f64 / 1e6;
            println!("{i:>5}  {ctx:>4.0}  {ms:>7.2}  {:>5.1}  {kv_mb:>5.1}", 1e3 / ms);
        }
    }

    // ── Linear fit step_ms = a + b·ctx (least squares), discarding warmup ───────────────────────
    // The first decode steps include one-time warmup (kernel compile, cold caches); skip them so
    // the slope reflects steady-state KV bandwidth, not startup.
    let warmup = 16.min(samples.len() / 4);
    let fit: &[(f64, f64)] = &samples[warmup..];
    let n = fit.len() as f64;
    let (sx, sy): (f64, f64) = fit.iter().fold((0.0, 0.0), |(ax, ay), (x, y)| (ax + x, ay + y));
    let (sxx, sxy): (f64, f64) =
        fit.iter().fold((0.0, 0.0), |(axx, axy), (x, y)| (axx + x * x, axy + x * y));
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx); // ms per token-of-context (KV slope)
    let a = (sy - b * sx) / n; // context-independent floor (weights/FFN/compute)

    let max_ctx = samples.last().map(|s| s.0).unwrap_or(0.0);
    let toks = |ctx: f64, frac: f64| 1e3 / (a + frac * b * ctx); // tok/s with KV slope scaled by frac
    eprintln!(
        "\n# fit (ctx {:.0}..{max_ctx:.0}, {n:.0} steps): step_ms = {a:.2} + {b:.4}·ctx \
         (a=compute/weights floor, b=KV-read slope)",
        fit.first().map(|s| s.0).unwrap_or(0.0)
    );
    println!("\n  ctx | f16 tok/s | int8 ceil | int4 ceil | int4 speedup | KV MB f16 -> int4");
    let kv_mb = |ctx: f64, frac: f64| bytes_per_tok_per_ctx * ctx * frac / 1e6;
    for &ctx in &[512.0_f64, 1024.0, 2048.0, 4096.0, 8192.0] {
        let tag = if ctx > max_ctx { "extrap" } else { "meas'd" };
        println!(
            "  {ctx:>5.0} |  {:>6.1}   |  {:>6.1}   |  {:>6.1}   |    {:>4.2}x     |  {:>6.1} -> {:>5.1}  [{tag}]",
            toks(ctx, 1.0),
            toks(ctx, 0.5),
            toks(ctx, 0.25),
            toks(ctx, 0.25) / toks(ctx, 1.0),
            kv_mb(ctx, 1.0),
            kv_mb(ctx, 0.25),
        );
    }
    eprintln!(
        "\n# ceilings assume the int-domain attention dot (no f32 re-expand); the current\n\
         # QuantizedKvSdpa dequants the whole cache each step, so it realizes the MB column but\n\
         # not the tok/s column until the W4A8 path lands."
    );
    Ok(())
}
