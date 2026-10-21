#!/usr/bin/env bash
set -euo pipefail
# Use your active Python environment; all additional arguments reach the CLI.
TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$TRAINING_DIR/main_pretrain.py" \
  --config-file "$TRAINING_DIR/configs/bfno_500k_20_7_128.yaml" "$@"
