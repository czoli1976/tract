//! Bonsai Image (ternary Flux2 Klein 4B) with tract, from the NNEF archives of
//! tract-nnef/bonsai-image-ternary-4B: the Qwen3 encoder, the ternary DiT and the Flux2 VAE.

use std::path::{Path, PathBuf};

use anyhow::*;
use rand::SeedableRng;
use tract::prelude::*;

tract::impl_ndarray_interop!();

pub const TEXT_LEN: usize = 64;
pub const TEXT_DIM: usize = 7680;
pub const LATENT_DIM: usize = 128;

/// Runtimes the demo can name. Only `wgpu` runs the ternary transformer today.
pub const RUNTIMES: &[&str] = &["wgpu", "metal", "cuda"];

/// What `Generator::generate` reports as it goes.
pub enum Progress {
    Loading(&'static str),
    Encoding,
    Step { step: usize, steps: usize },
    Decoding,
}

pub struct Generator {
    runtime: Runtime,
    assets: PathBuf,
    tokenizer: tokenizers::Tokenizer,
}

impl Generator {
    /// `assets` holds the three `*.nnef.tar` archives and `tokenizer/tokenizer.json`.
    pub fn new(runtime: &str, assets: &Path) -> Result<Generator> {
        Ok(Generator {
            runtime: tract::runtime_for_name(runtime)?,
            assets: assets.to_path_buf(),
            tokenizer: tokenizers::Tokenizer::from_file(assets.join("tokenizer/tokenizer.json"))
                .map_err(|e| anyhow!("{e}"))?,
        })
    }

    pub fn runtime_name(&self) -> Result<String> {
        self.runtime.name()
    }

    /// One image for `prompt`, `size` pixels square and a multiple of 16.
    pub fn generate(
        &self,
        prompt: &str,
        size: usize,
        steps: usize,
        seed: u64,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<image::RgbImage> {
        let embeddings = self.embed(prompt, progress)?;
        let latent = self.denoise(&embeddings, size, steps, seed, progress)?;
        self.decode(&latent, progress)
    }

    /// The text encoder's `(1, TEXT_LEN, TEXT_DIM)` embeddings of the chat-templated prompt.
    pub fn embed(&self, prompt: &str, progress: &mut dyn FnMut(Progress)) -> Result<Tensor> {
        let [ids, mask] = tokenize(&self.tokenizer, prompt)?;
        progress(Progress::Loading("text encoder"));
        let text_encoder = self.load("text_encoder_gptq.nnef.tar")?;
        progress(Progress::Encoding);
        let embeddings = text_encoder.run([ids.tract()?, mask.tract()?])?[0].clone();
        Ok(embeddings)
    }

    /// The `(1, 128, size/16, size/16)` latent after `steps` flow-matching steps from `seed`.
    pub fn denoise(
        &self,
        embeddings: &Tensor,
        size: usize,
        steps: usize,
        seed: u64,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<ndarray::Array4<f32>> {
        let side = size / 16;
        let tokens = side * side;
        let sigmas = schedule(steps, tokens);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut x = normal(&mut rng, tokens * LATENT_DIM);
        let [image_ids, text_ids] = position_ids(side)?;

        progress(Progress::Loading("transformer"));
        let transformer = self.load("transformer.nnef.tar")?;
        for step in 0..steps {
            progress(Progress::Step { step, steps });
            let (sigma, next) = (sigmas[step], sigmas[step + 1]);
            let hidden = ndarray::ArrayD::from_shape_vec(vec![1, tokens, LATENT_DIM], x.clone())?;
            let t = ndarray::arr1(&[sigma]);
            let out = transformer.run([
                hidden.tract()?,
                embeddings.clone(),
                t.tract()?,
                image_ids.clone().tract()?,
                text_ids.clone().tract()?,
            ])?;
            for (x, v) in x.iter_mut().zip(out[0].as_slice::<f32>()?) {
                *x += (next - sigma) * v;
            }
        }
        let hidden = ndarray::ArrayD::from_shape_vec(vec![1, tokens, LATENT_DIM], x)?;
        tokens_to_latent(&hidden.tract()?)
    }

    pub fn decode(
        &self,
        latent: &ndarray::Array4<f32>,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<image::RgbImage> {
        progress(Progress::Loading("VAE decoder"));
        let vae_decoder = self.load("vae_decoder.nnef.tar")?;
        progress(Progress::Decoding);
        let image = vae_decoder.run([latent.clone().tract()?])?;
        let view = image[0].ndarray4::<f32>()?;
        let (_, channels, h, w) = view.dim();
        ensure!(channels == 3, "the VAE gave {channels} channels, not 3");
        let mut pixels = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    let val = (view[[0, ch, y, x]].clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0;
                    pixels[(y * w + x) * 3 + ch] = val as u8;
                }
            }
        }
        image::RgbImage::from_raw(w as u32, h as u32, pixels).context("image size")
    }

    fn load(&self, archive: &str) -> Result<Runnable> {
        let mut nnef = tract::nnef()?;
        nnef.enable_tract_transformers()?;
        let mut model = nnef.load_read(&mut std::fs::File::open(self.assets.join(archive))?)?;
        model.transform("transformers_detect_all")?;
        self.runtime.prepare(model)
    }
}

/// `(1, N, C)` row-major tokens over a square grid to `(1, C, side, side)`.
pub fn tokens_to_latent(tokens: &Tensor) -> Result<ndarray::Array4<f32>> {
    let view = tokens.ndarray3::<f32>()?;
    let (batch, n, c) = view.dim();
    ensure!(batch == 1, "expected one image, got batch {batch}");
    let side = (n as f64).sqrt() as usize;
    ensure!(side * side == n, "{n} tokens is not a square grid");
    let channels_first = view.index_axis(ndarray::Axis(0), 0).t().as_standard_layout().into_owned();
    Ok(channels_first.into_shape_with_order((1, c, side, side))?)
}

/// `(input_ids, attention_mask)` for the chat-templated prompt, padded to `TEXT_LEN`.
pub fn tokenize(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
) -> Result<[ndarray::Array2<i64>; 2]> {
    let text = format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let enc = tokenizer.encode(text, false).map_err(|e| anyhow!("{e}"))?;
    let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
    ids.truncate(TEXT_LEN);
    let mut mask = vec![1i64; ids.len()];
    let pad = tokenizer.token_to_id("<|endoftext|>").context("no pad token")? as i64;
    ids.resize(TEXT_LEN, pad);
    mask.resize(TEXT_LEN, 0);
    Ok([
        ndarray::Array2::from_shape_vec((1, TEXT_LEN), ids)?,
        ndarray::Array2::from_shape_vec((1, TEXT_LEN), mask)?,
    ])
}

/// FLUX.2 Klein schedule: linear sigmas under an exponential time shift, whose mu follows the
/// image token count. Klein takes the sigma itself as its timestep.
pub fn schedule(steps: usize, tokens: usize) -> Vec<f32> {
    let tokens = tokens as f64;
    let (m200, m10) = (0.00016927 * tokens + 0.45666666, 8.73809524e-05 * tokens + 1.89833333);
    let a = (m200 - m10) / 190.0;
    let mu = (a * steps as f64 + m200 - 200.0 * a).exp();
    let mut sigmas: Vec<f32> = (0..steps)
        .map(|i| 1.0 - i as f64 * (1.0 - 1.0 / steps as f64) / (steps - 1).max(1) as f64)
        .map(|lin| (mu / (mu + (1.0 / lin - 1.0))) as f32)
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// Flux2 four-axis position IDs for a square image grid and for the text: `(N, 4)` and
/// `(TEXT_LEN, 4)`.
pub fn position_ids(side: usize) -> Result<[ndarray::Array2<f32>; 2]> {
    let image: Vec<f32> = (0..side)
        .flat_map(|h| (0..side).flat_map(move |w| [0.0, h as f32, w as f32, 0.0]))
        .collect();
    let text: Vec<f32> = (0..TEXT_LEN).flat_map(|l| [0.0, 0.0, 0.0, l as f32]).collect();
    Ok([
        ndarray::Array2::from_shape_vec((side * side, 4), image)?,
        ndarray::Array2::from_shape_vec((TEXT_LEN, 4), text)?,
    ])
}

fn normal(rng: &mut rand::rngs::StdRng, len: usize) -> Vec<f32> {
    use rand_distr::{Distribution, StandardNormal};
    (0..len).map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, rng)).collect()
}
