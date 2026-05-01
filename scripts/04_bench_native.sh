#!/usr/bin/env bash
# Interleaved native bench with stats analysis.
#
# Usage:  bash 04_bench_native.sh <model_label>
#         where <model_label> is "gtcrn" or "dfn3"
#
# Inputs (env): BASELINE_BIN, PATCHED_BIN
#               DFN3_DIR (only if model=dfn3)
#               RUNS (default 5), MAX_TIME_MS (default 10000), WARMUP_MS (default 1000)

set -euo pipefail
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/tract-k1-repro}"
BASELINE_BIN="${BASELINE_BIN:-$WORK_DIR/tract-baseline/target/release/tract}"
PATCHED_BIN="${PATCHED_BIN:-$WORK_DIR/tract-patched/target/release/tract}"

RUNS="${RUNS:-5}"
MAX_TIME_MS="${MAX_TIME_MS:-10000}"
WARMUP_MS="${WARMUP_MS:-1000}"

MODEL_LABEL="${1:-}"
[[ -n "$MODEL_LABEL" ]] || { echo "Usage: $0 <model_label>  (gtcrn|dfn3-enc|dfn3-erb_dec|dfn3-df_dec)"; exit 1; }

case "$MODEL_LABEL" in
  gtcrn)
    MODEL="$KIT_DIR/models/gtcrn.onnx"
    INPUT_ARGS="-i mix:1,257,1,2,f32 -i conv_cache:2,1,16,16,33,f32 -i tra_cache:2,3,1,1,16,f32 -i inter_cache:2,1,33,16,f32"
    ;;
  dfn3-enc)
    MODEL="${DFN3_DIR:?set DFN3_DIR}/enc.onnx"
    INPUT_ARGS="-i feat_erb:1,1,100,32,f32 -i feat_spec:1,2,100,96,f32"
    ;;
  dfn3-erb_dec)
    MODEL="${DFN3_DIR:?set DFN3_DIR}/erb_dec.onnx"
    INPUT_ARGS="-i e0:1,64,100,32,f32 -i e1:1,64,100,16,f32 -i e2:1,64,100,8,f32 -i e3:1,64,100,8,f32 -i emb:1,100,512,f32"
    ;;
  dfn3-df_dec)
    MODEL="${DFN3_DIR:?set DFN3_DIR}/df_dec.onnx"
    INPUT_ARGS="-i emb:1,100,512,f32 -i c0:1,64,100,96,f32"
    ;;
  *) echo "Unknown model_label: $MODEL_LABEL"; exit 1;;
esac

[[ -f "$MODEL" ]] || { echo "Missing $MODEL"; exit 1; }

declare -a B P
echo "== Interleaved bench: $MODEL_LABEL ($RUNS pairs, ${MAX_TIME_MS}ms each, ${WARMUP_MS}ms warmup) =="
for r in $(seq 1 $RUNS); do
  b_line=$($BASELINE_BIN --onnx-ignore-output-shapes --onnx-ignore-value-info "$MODEL" $INPUT_ARGS \
    -O bench --allow-random-input --max-time $MAX_TIME_MS --warmup-time $WARMUP_MS 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' | grep "ms/i" | tail -1)
  p_line=$($PATCHED_BIN  --onnx-ignore-output-shapes --onnx-ignore-value-info "$MODEL" $INPUT_ARGS \
    -O bench --allow-random-input --max-time $MAX_TIME_MS --warmup-time $WARMUP_MS 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' | grep "ms/i" | tail -1)
  b_ms=$(echo "$b_line" | grep -oE '[0-9]+\.[0-9]+ ms/i' | grep -oE '^[0-9]+\.[0-9]+')
  p_ms=$(echo "$p_line" | grep -oE '[0-9]+\.[0-9]+ ms/i' | grep -oE '^[0-9]+\.[0-9]+')
  echo "  baseline $r: $b_line"
  echo "  patched  $r: $p_line"
  B+=("$b_ms")
  P+=("$p_ms")
done

echo
python3 "$KIT_DIR/scripts/lib_stats.py" "${B[@]}" -- "${P[@]}"
