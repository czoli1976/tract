#!/usr/bin/env bash
# Set up baseline + patched tract checkouts and build both CLI binaries.
#
# Inputs (env):
#   TRACT_BASELINE   path to your existing tract checkout (or fresh clone)
#   WORK_DIR         where to put the patched copy (default: /tmp/tract-k1-repro)
#
# Outputs:
#   $WORK_DIR/tract-baseline/target/release/tract   (untouched main)
#   $WORK_DIR/tract-patched/target/release/tract    (with K=1 fix applied)

set -euo pipefail
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/tract-k1-repro}"
BASE_REF="${BASE_REF:-b36f34e92}"   # sonos/tract main as of 2026-04-30; rebase if drifted

if [[ -z "${TRACT_BASELINE:-}" ]]; then
  echo "Set TRACT_BASELINE to a tract checkout to clone from."
  echo "Or run:  TRACT_BASELINE=https://github.com/sonos/tract bash $0"
  exit 1
fi

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone baseline if not already there
if [[ ! -d tract-baseline ]]; then
  if [[ -d "$TRACT_BASELINE" ]]; then
    git clone "$TRACT_BASELINE" tract-baseline
  else
    git clone "$TRACT_BASELINE" tract-baseline
  fi
  (cd tract-baseline && git checkout "$BASE_REF")
fi

# Copy and apply patch (default: main; for v0.22.1 set PATCH=k1-fix-on-tract-0.22.1.patch)
PATCH="${PATCH:-k1-fix-on-tract-main.patch}"
if [[ ! -d tract-patched ]]; then
  cp -R tract-baseline tract-patched
  (cd tract-patched && git am "$KIT_DIR/$PATCH")
fi

# Build both
echo ">> Building baseline (this may take ~10 min on first build)"
(cd tract-baseline && cargo build --release -p tract-cli --bin tract 2>&1 | tail -3)
echo ">> Building patched"
(cd tract-patched && cargo build --release -p tract-cli --bin tract 2>&1 | tail -3)

ls -la tract-baseline/target/release/tract tract-patched/target/release/tract
echo
echo "Done. Set BASELINE_BIN and PATCHED_BIN for the other scripts:"
echo "  export BASELINE_BIN=$WORK_DIR/tract-baseline/target/release/tract"
echo "  export PATCHED_BIN=$WORK_DIR/tract-patched/target/release/tract"
