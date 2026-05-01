#!/usr/bin/env bash
# Run the einsum + matmul + quant test suites on the patched tract checkout.
#
# Expected: 235 passed, 0 failed in tract-core --lib (231 pre-existing + 4 new K=1).

set -euo pipefail
WORK_DIR="${WORK_DIR:-/tmp/tract-k1-repro}"
PATCHED_DIR="$WORK_DIR/tract-patched"

cd "$PATCHED_DIR"

echo ">> Running einsum unit tests (should include 4 new K=1 cases)"
cargo test --release -p tract-core --lib ops::einsum 2>&1 | tail -15

echo
echo ">> Running matmul + quant tests (should not regress)"
cargo test --release -p tract-core --lib ops::matmul ops::quant 2>&1 | tail -15

echo
echo ">> Full tract-core lib test suite"
cargo test --release -p tract-core --lib 2>&1 | tail -3
