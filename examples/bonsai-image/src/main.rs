//! Prompt to PNG with Bonsai Image on tract. See `lib.rs`.

use std::path::PathBuf;

use anyhow::*;
use bonsai_image::{Generator, Progress};

#[derive(clap::Parser)]
struct Args {
    /// Text prompt
    #[arg(short, long, default_value = "a photo of a bonsai tree")]
    prompt: String,
    /// Denoising steps
    #[arg(short, long, default_value_t = 4)]
    steps: usize,
    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Image side, a multiple of 16
    #[arg(long, default_value_t = 256)]
    size: usize,
    /// Runtime: wgpu, metal or cuda
    #[arg(long, default_value = "wgpu")]
    runtime: String,
    /// Directory of the NNEF archives and tokenizer/tokenizer.json
    #[arg(long, default_value = "assets")]
    assets: PathBuf,
    /// Output file
    #[arg(short, long, default_value = "output.png")]
    output: PathBuf,
}

fn main() -> Result<()> {
    use clap::Parser as _;
    let args = Args::parse();

    let generator = Generator::new(&args.runtime, &args.assets)?;
    eprintln!("runtime: {}", generator.runtime_name()?);
    let image =
        generator.generate(&args.prompt, args.size, args.steps, args.seed, &mut |p| match p {
            Progress::Loading(what) => eprintln!("Loading {what}..."),
            Progress::Encoding => eprintln!("Encoding prompt..."),
            Progress::Step { step, steps } => eprintln!("Denoising step {}/{steps}", step + 1),
            Progress::Decoding => eprintln!("Decoding image..."),
        })?;
    image.save(&args.output)?;
    eprintln!("Saved {}", args.output.display());
    Ok(())
}
