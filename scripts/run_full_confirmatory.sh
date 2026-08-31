#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src"
ACL_PYTHON_BIN="${ACL_PYTHON_BIN:-python3}"

"$ACL_PYTHON_BIN" src/run_confirmatory_ablation.py \
  --config config/ACL_confirmatory_aux.yaml \
  --resume
