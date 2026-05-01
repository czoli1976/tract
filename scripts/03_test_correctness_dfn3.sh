#!/usr/bin/env bash
# Bit-exact verification on DFN3's three ONNX models (enc, erb_dec, df_dec).
#
# Inputs (env):
#   BASELINE_BIN, PATCHED_BIN — from 00_apply_patch.sh
#   DFN3_DIR — directory containing enc.onnx, erb_dec.onnx, df_dec.onnx
#              (extract DeepFilterNet3_onnx.tar.gz from the DFN releases)

set -euo pipefail
WORK_DIR="${WORK_DIR:-/tmp/tract-k1-repro}"
BASELINE_BIN="${BASELINE_BIN:-$WORK_DIR/tract-baseline/target/release/tract}"
PATCHED_BIN="${PATCHED_BIN:-$WORK_DIR/tract-patched/target/release/tract}"
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

[[ -n "${DFN3_DIR:-}" ]] || { echo "Set DFN3_DIR to a dir containing enc.onnx, erb_dec.onnx, df_dec.onnx"; exit 1; }
[[ -f "$DFN3_DIR/enc.onnx" ]] || { echo "Missing $DFN3_DIR/enc.onnx"; exit 1; }

# Generate fixed-seed inputs (T=100 frames at 10ms hop = 1s of audio)
python3 -c "
import numpy as np
np.random.seed(42)
np.savez('/tmp/dfn3_enc_in.npz',
    feat_erb=np.random.randn(1, 1, 100, 32).astype(np.float32),
    feat_spec=np.random.randn(1, 2, 100, 96).astype(np.float32))
np.savez('/tmp/dfn3_erb_dec_in.npz',
    e0=np.random.randn(1, 64, 100, 32).astype(np.float32),
    e1=np.random.randn(1, 64, 100, 16).astype(np.float32),
    e2=np.random.randn(1, 64, 100, 8).astype(np.float32),
    e3=np.random.randn(1, 64, 100, 8).astype(np.float32),
    emb=np.random.randn(1, 100, 512).astype(np.float32))
np.savez('/tmp/dfn3_df_dec_in.npz',
    emb=np.random.randn(1, 100, 512).astype(np.float32),
    c0=np.random.randn(1, 64, 100, 96).astype(np.float32))
"

run() {
  local name=$1; local model=$2; local inputs=$3; shift 3
  echo "=== $name ==="
  $BASELINE_BIN --onnx-ignore-output-shapes --onnx-ignore-value-info $model "$@" run \
    --input-from-npz $inputs --save-outputs-npz /tmp/${name}_baseline.npz | tail -1
  $PATCHED_BIN  --onnx-ignore-output-shapes --onnx-ignore-value-info $model "$@" run \
    --input-from-npz $inputs --save-outputs-npz /tmp/${name}_patched.npz | tail -1
  python3 "$KIT_DIR/scripts/lib_compare.py" /tmp/${name}_baseline.npz /tmp/${name}_patched.npz
}

run dfn3_enc "$DFN3_DIR/enc.onnx" /tmp/dfn3_enc_in.npz \
  -i feat_erb:1,1,100,32,f32 -i feat_spec:1,2,100,96,f32

run dfn3_erb_dec "$DFN3_DIR/erb_dec.onnx" /tmp/dfn3_erb_dec_in.npz \
  -i e0:1,64,100,32,f32 -i e1:1,64,100,16,f32 -i e2:1,64,100,8,f32 \
  -i e3:1,64,100,8,f32 -i emb:1,100,512,f32

run dfn3_df_dec "$DFN3_DIR/df_dec.onnx" /tmp/dfn3_df_dec_in.npz \
  -i emb:1,100,512,f32 -i c0:1,64,100,96,f32
