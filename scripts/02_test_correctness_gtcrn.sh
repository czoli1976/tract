#!/usr/bin/env bash
# Bit-exact verification on GTCRN: every output tensor element must match exactly.
#
# Inputs (env): BASELINE_BIN, PATCHED_BIN — from 00_apply_patch.sh
# Model:        models/gtcrn.onnx — from models/get_gtcrn.sh

set -euo pipefail
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/tract-k1-repro}"
BASELINE_BIN="${BASELINE_BIN:-$WORK_DIR/tract-baseline/target/release/tract}"
PATCHED_BIN="${PATCHED_BIN:-$WORK_DIR/tract-patched/target/release/tract}"

MODEL="$KIT_DIR/models/gtcrn.onnx"
[[ -f "$MODEL" ]] || { echo "Missing $MODEL — run models/get_gtcrn.sh first"; exit 1; }

# Generate fixed-seed inputs
INPUTS_NPZ=/tmp/gtcrn_inputs.npz
python3 -c "
import numpy as np
np.random.seed(42)
np.savez('$INPUTS_NPZ',
    mix=np.random.randn(1, 257, 1, 2).astype(np.float32),
    conv_cache=np.random.randn(2, 1, 16, 16, 33).astype(np.float32),
    tra_cache=np.random.randn(2, 3, 1, 1, 16).astype(np.float32),
    inter_cache=np.random.randn(2, 1, 33, 16).astype(np.float32),
)
"

ARGS="--onnx-ignore-output-shapes --onnx-ignore-value-info $MODEL \
  -i mix:1,257,1,2,f32 \
  -i conv_cache:2,1,16,16,33,f32 \
  -i tra_cache:2,3,1,1,16,f32 \
  -i inter_cache:2,1,33,16,f32"

echo ">> Running baseline"
$BASELINE_BIN $ARGS run --input-from-npz $INPUTS_NPZ --save-outputs-npz /tmp/gtcrn_baseline.npz | tail -1

echo ">> Running patched"
$PATCHED_BIN $ARGS run --input-from-npz $INPUTS_NPZ --save-outputs-npz /tmp/gtcrn_patched.npz | tail -1

echo
echo ">> Comparing every output tensor element-wise"
python3 "$KIT_DIR/scripts/lib_compare.py" /tmp/gtcrn_baseline.npz /tmp/gtcrn_patched.npz
