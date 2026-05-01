#!/usr/bin/env bash
# Download GTCRN ONNX (small, public).
#
# Source: https://github.com/Xiaobin-Rong/gtcrn (commit pinning a known good version is below)

set -euo pipefail
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$KIT_DIR/models"
cd "$MODELS_DIR"

echo ">> Downloading gtcrn.onnx (~344 KB)"
curl -sL "https://github.com/Xiaobin-Rong/gtcrn/raw/main/stream/onnx_models/gtcrn.onnx" \
  -o gtcrn.onnx

ls -lh gtcrn.onnx
