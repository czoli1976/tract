# Bonsai Image with tract

Text to image with PrismML's Ternary Bonsai Image 4B, a Qwen3 text encoder, a ternary Flux.2
Klein transformer and its VAE, on tract's `wgpu` runtime.

The weights stay packed on the GPU: `Q2_0_T` for the transformer's ternary matmuls, `Q4_0` for
the encoder's. The NNEF archives are the exports in
[tract-nnef/bonsai-image-ternary-4B](https://huggingface.co/tract-nnef/bonsai-image-ternary-4B),
which take the image size as a symbolic dimension, so one archive serves every size. The encoder
is its GPTQ-calibrated one.

## Native

```sh
huggingface-cli download tract-nnef/bonsai-image-ternary-4B transformer.nnef.tar text_encoder_gptq.nnef.tar vae_decoder.nnef.tar --local-dir assets
huggingface-cli download prism-ml/bonsai-image-ternary-4B-unpacked tokenizer/tokenizer.json --local-dir assets
cargo run --release -p bonsai-image -- -p "a photo of a bonsai tree"
```

`--size` picks the square image side, 256 by default, a multiple of 16. The archives were
exported for 64 text tokens, which the demo uses throughout.

## In the browser

See [web](web).
