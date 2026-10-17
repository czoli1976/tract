#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# ONNX LSTM (T=3, batch=8, H=16) lowered to Scan, both state initializers sharing
# one zero constant. Reference outputs come from onnxruntime.
$TRACT_RUN --nnef-tract-core . run --input-from-bundle io.npz --assert-output-bundle io.npz
$TRACT_RUN --nnef-tract-core . -O run --input-from-bundle io.npz --assert-output-bundle io.npz
