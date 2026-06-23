//! True incremental-decode tok/s for the int4 KV cache vs f16, on CPU.
//!  • int4: facade `transformers_detect_all` + `fuse_quantized_kv_sdpa` → stateful QuantizedKvSdpa
//!    (KV internal), driven with the facade's persistent `spawn_state()` (feed ONE token per step).
//!  • f16 : the existing CausalLlmModel incremental decode (external concat-grow KV), force_cpu.
//!   cargo run --release -p causal_llm --bin persistent_decode -- -t tok.json -m m.nnef.tgz -n 256
use anyhow::Result;
use clap::Parser;
use std::time::Instant;
use tokenizers::Tokenizer;
use tract::prelude::*;

tract::impl_ndarray_interop!();

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    tokenizer: String,
    #[arg(short, long)]
    model: String,
    #[arg(short, default_value = "256")]
    n: usize,
    #[arg(default_value = "The history of computing began in the nineteenth century when")]
    prompt: String,
}

fn report(tag: &str, samples: &[(f64, f64)]) {
    for (i, (ctx, ms)) in samples.iter().enumerate() {
        if i % 64 == 0 || i + 1 == samples.len() {
            println!("  ctx={ctx:>4.0}  {ms:>6.2}ms  {:>5.1} tok/s", 1e3 / ms);
        }
    }
    let half = &samples[samples.len() / 2..];
    let mean_ms = half.iter().map(|s| s.1).sum::<f64>() / half.len() as f64;
    println!("[{tag}] steady-state mean: {:.1} tok/s ({mean_ms:.2}ms/tok)\n", 1e3 / mean_ms);
}

fn argmax_last(t: &Tensor) -> Result<i64> {
    let f = t.convert_to(DatumType::F32)?;
    let v = f.as_slice::<f32>()?;
    let vocab = *t.shape()?.last().unwrap();
    let last = &v[v.len() - vocab..];
    Ok(last
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            },
        )
        .0 as i64)
}

// int4: facade load → detect_all → fuse → stateful op; persistent spawn_state, feed 1 token/step.
fn decode_int4(model: &str, ids: &[i64], n: usize) -> Result<()> {
    let nnef = tract::nnef()?.with_tract_transformers()?;
    let mut nn = nnef.load(model)?;
    nn.transform("transformers_detect_all")?;
    nn.transform("fuse_quantized_kv_sdpa")?;
    println!("[int4] fused, inputs={}", nn.input_count()?);
    let runnable = runtime_for_name("default")?.prepare(nn)?;
    let mut state = runnable.spawn_state()?;

    let pre: Tensor = ndarray::Array2::from_shape_vec((1, ids.len()), ids.to_vec())?.tract()?;
    let t0 = Instant::now();
    let out = state.run(vec![pre])?;
    println!("[int4] prefill({} tok)={:.0}ms", ids.len(), t0.elapsed().as_secs_f64() * 1e3);
    let mut next = argmax_last(&out[0])?;
    let mut ctx = ids.len();
    let mut samples = Vec::new();
    for _ in 0..n {
        let tk: Tensor = ndarray::Array2::from_shape_vec((1, 1), vec![next])?.tract()?;
        let t = Instant::now();
        let out = state.run(vec![tk])?;
        samples.push(((ctx + 1) as f64, t.elapsed().as_secs_f64() * 1e3));
        ctx += 1;
        next = argmax_last(&out[0])?;
    }
    report("int4", &samples);
    Ok(())
}

// f16 baseline: the existing external-KV incremental decode, forced to CPU.
fn decode_f16(tok_path: &str, model: &str, prompt: &str, n: usize) -> Result<()> {
    use causal_llm::{CausalLlmModel, CausalLlmModelConfig};
    let llm = CausalLlmModel::from_paths_and_conf(
        tok_path,
        model,
        CausalLlmModelConfig { force_cpu: true },
    )?;
    let mut st = llm.spawn()?;
    st.append_text(prompt)?;
    let t0 = Instant::now();
    st.generate_next_token()?; // prefill
    println!("[f16 ] prefill={:.0}ms", t0.elapsed().as_secs_f64() * 1e3);
    let mut samples = Vec::new();
    for _ in 0..n {
        let t = Instant::now();
        st.generate_next_token()?;
        samples.push((st.seq.len() as f64, t.elapsed().as_secs_f64() * 1e3));
    }
    report("f16 ", &samples);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tok = Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!(e))?;
    let ids: Vec<i64> = tok
        .encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?
        .get_ids()
        .iter()
        .map(|&x| x as i64)
        .collect();
    println!("prompt {} tokens, generating {} (CPU)\n", ids.len(), args.n);
    decode_f16(&args.tokenizer, &args.model, &args.prompt, args.n)?;
    decode_int4(&args.model, &ids, args.n)?;
    Ok(())
}
