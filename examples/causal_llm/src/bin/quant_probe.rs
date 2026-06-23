//! Probe: does `fuse_quantized_kv_sdpa` fire on a real Qwen graph, and does the int4 KV path
//! produce a sane next-token prediction? A single forward (prefill) builds the stateful op's cache
//! within the call, so this works without a persistent-state driver. Compares the int4 model's
//! argmax/top-5 next token against the unfused (f16-cache) model on the same prompt.
//!   cargo run --release -p causal_llm --bin quant_probe -- -t tok.json -m model.nnef.tgz "prompt"
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
    #[arg(default_value = "The capital of France is")]
    prompt: String,
    /// Tokens to generate (greedy)
    #[arg(short, default_value = "24")]
    n: usize,
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            },
        )
        .0
}

fn build(model_path: &str, fuse: bool) -> Result<(tract::Runnable, Vec<Tensor>, &'static str)> {
    let nnef = tract::nnef()?.with_tract_transformers()?;
    let mut nn = nnef.load(model_path)?;
    nn.transform("transformers_detect_all")?;
    if fuse {
        nn.transform("fuse_quantized_kv_sdpa")?;
    } else {
        nn.transform("unfold-kv-cache")?;
    }
    // Empty KV caches for inputs 1.. (the unfolded baseline has them; the fused int4 model has none)
    let n = nn.input_count()?;
    let mut empties = Vec::new();
    for i in 1..n {
        let f = nn.input_fact(i)?;
        let dt = f.datum_type()?;
        let mut shape = vec![];
        for ax in 0..f.rank()? {
            let v =
                f.dim(ax)?.to_int64().map(|v| v as usize).unwrap_or(if ax == 0 { 1 } else { 0 });
            shape.push(v);
        }
        let nb = shape.iter().product::<usize>() * dt.size_of();
        empties.push(Tensor::from_bytes(dt, &shape, &vec![0u8; nb])?);
    }
    let r = runtime_for_name("default")?.prepare(nn)?;
    Ok((r, empties, if fuse { "int4" } else { "f16 " }))
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
    println!("prompt = {:?}  ({} tokens)", args.prompt, ids.len());

    for fuse in [false, true] {
        let t0 = Instant::now();
        let (runnable, empties, tag) = match build(&args.model, fuse) {
            Ok(x) => x,
            Err(e) => {
                println!("[{}] BUILD FAILED: {e:#}", if fuse { "int4" } else { "f16 " });
                continue;
            }
        };
        let load_ms = t0.elapsed().as_secs_f64() * 1e3;
        let n_inputs = 1 + empties.len();

        // Greedy generation. Stateless re-feed of the growing sequence each step (the int4 op
        // rebuilds its cache within each forward) — O(T²) so not a decode-tok/s number, but it
        // shows real multi-token coherence and lets int4 be compared to f16 token-for-token.
        let mut seq = ids.clone();
        let t1 = Instant::now();
        for _ in 0..args.n {
            let mut inputs: Vec<Tensor> =
                vec![ndarray::Array2::from_shape_vec((1, seq.len()), seq.clone())?.tract()?];
            inputs.extend(empties.iter().cloned());
            let out = runnable.run(inputs)?;
            let logits = out[0].convert_to(DatumType::F32)?;
            let next = argmax(logits.as_slice::<f32>()?) as i64;
            seq.push(next);
        }
        let gen_ms = t1.elapsed().as_secs_f64() * 1e3;
        let gen_ids: Vec<u32> = seq[ids.len()..].iter().map(|&x| x as u32).collect();
        let text = tok.decode(&gen_ids, true).unwrap_or_default();
        println!(
            "[{tag}] load={load_ms:.0}ms inputs={n_inputs} gen({}tok)={:.0}ms\n      → {text:?}",
            args.n, gen_ms
        );
    }
    println!(
        "\n(int4 firing = inputs=1, KV internal/stateful; quality = does int4 text match f16)"
    );
    Ok(())
}
