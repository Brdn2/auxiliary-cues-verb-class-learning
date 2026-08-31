#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACL_PYTHON_BIN="${ACL_PYTHON_BIN:-python3}"
ACL_TECTONIC_BIN="${ACL_TECTONIC_BIN:-tectonic}"
export PYTHONPATH="$PROJECT_ROOT/src"

cd "$PROJECT_ROOT"
"$ACL_PYTHON_BIN" src/summarize_confirmatory.py
"$ACL_PYTHON_BIN" src/summarize_nonce.py
"$ACL_PYTHON_BIN" src/update_paper_from_confirmatory.py
"$ACL_PYTHON_BIN" src/summarize_compute_budget.py
"$ACL_PYTHON_BIN" src/make_paper_figures.py
"$ACL_PYTHON_BIN" -m pytest -q

if command -v "$ACL_TECTONIC_BIN" >/dev/null 2>&1; then
  mkdir -p build/paper output/pdf
  cd paper
  "$ACL_TECTONIC_BIN" -X compile --outdir "$PROJECT_ROOT/build/paper" --outfmt pdf --untrusted main.tex
  cd "$PROJECT_ROOT"
  cp build/paper/main.pdf output/pdf/auxiliary_cues_novel_verb_acl_draft.pdf
  echo "Final paper: $PROJECT_ROOT/output/pdf/auxiliary_cues_novel_verb_acl_draft.pdf"
else
  echo "Statistics, paper tables, figures, and tests are complete. Set ACL_TECTONIC_BIN or compile paper/main.tex with the bundled LaTeX workflow."
fi

"$ACL_PYTHON_BIN" src/audit_release_readiness.py --require-e7 --strict
"$ACL_PYTHON_BIN" src/build_public_artifact.py --require-e7 --force
