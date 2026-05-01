#!/usr/bin/env bash
# Show the einsum→Mul rewrite via tract's optimized-graph dump.
#
# Usage:  bash 05_diff_optimized_graph.sh <model.onnx> [-i input_specs...]
#
# Empty diff → fix doesn't fire on this model (confirms it's a no-op there).
# Non-empty diff → look for "OptMatMul → Mul" pairs to see where the rewrite landed.

set -euo pipefail
WORK_DIR="${WORK_DIR:-/tmp/tract-k1-repro}"
BASELINE_BIN="${BASELINE_BIN:-$WORK_DIR/tract-baseline/target/release/tract}"
PATCHED_BIN="${PATCHED_BIN:-$WORK_DIR/tract-patched/target/release/tract}"

MODEL="${1:?usage: $0 model.onnx [-i input_specs...]}"; shift

dump() {
  local bin="$1"; shift
  "$bin" --onnx-ignore-output-shapes --onnx-ignore-value-info "$MODEL" "$@" -O dump 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g'
}

dump "$BASELINE_BIN" "$@" > /tmp/k1_b_dump.txt
dump "$PATCHED_BIN"  "$@" > /tmp/k1_p_dump.txt

DIFF_LINES=$( { diff /tmp/k1_b_dump.txt /tmp/k1_p_dump.txt || true; } | wc -l | tr -d ' ')
echo "=== Optimized graph diff: $DIFF_LINES lines ==="
if [[ "$DIFF_LINES" -eq 0 ]]; then
  echo "  → fix does NOT fire on this model (graphs identical)"
  exit 0
fi
echo "  → fix fires; showing OptMatMul→Mul rewrite sites:"
{ diff /tmp/k1_b_dump.txt /tmp/k1_p_dump.txt || true; } | grep -E "^[<>] .*((OptMatMul|Mul).*einsum|einsum.*pack_a)" | head -20
echo
echo "Full diff at /tmp/k1_b_dump.txt vs /tmp/k1_p_dump.txt"
