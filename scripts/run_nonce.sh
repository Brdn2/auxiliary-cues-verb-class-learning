#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src"
ACL_PYTHON_BIN="${ACL_PYTHON_BIN:-python3}"

"$ACL_PYTHON_BIN" src/run_nonce.py --config configs/nonce.yaml
"$ACL_PYTHON_BIN" src/summarize_nonce.py --results-dir results/nonce
