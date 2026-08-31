from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy
import pandas
import scipy
import sklearn
import spacy
import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]


def classify(path: Path) -> str:
    text = str(path)
    if "ACL_confirmatory_aux_v2" in text:
        return "E7_confirmatory"
    if "Expanded_learning_curve" in text:
        return "E0_E1_expanded"
    if "Confirmatory_ablation_v1" in text:
        return "development_confirmatory"
    if "Pilot_v4" in text:
        return "development_pilots"
    return "other"


def collect(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/training_summary.json")):
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            summary = json.loads(data)
            rows.append({
                "phase": classify(path),
                "condition": summary.get("condition", "unknown"),
                "seed": summary.get("seed", ""),
                "debug": bool(summary.get("debug", False)),
                "device": summary.get("device", "unknown"),
                "epochs": len(summary.get("history", [])),
                "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
                "summary_sha256": digest,
                "source_path": str(path),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deduplicated lower-bound compute inventory.")
    parser.add_argument("--legacy-root", default=str(ROOT.parent / "MVP_Pilot实验"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/ACL_compute_budget"))
    args = parser.parse_args()
    roots = [ROOT, Path(args.legacy_root)]
    rows = collect(roots)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "training_run_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total_hours = sum(float(row["elapsed_seconds"]) for row in rows) / 3600
    full_hours = sum(float(row["elapsed_seconds"]) for row in rows if not row["debug"]) / 3600
    nonce_models = len(list((ROOT / "models/ACL_nonce_cross_template_v2").glob("*/*/model.safetensors")))
    report = {
        "scope": "Deduplicated logged training summaries from the paper project and legacy development project.",
        "is_lower_bound": True,
        "known_omissions": [
            f"Wall-clock timing was not logged for {nonce_models} nonce-adaptation checkpoints.",
            "Evaluation, preprocessing, corpus annotation, bootstrap, and failed/interrupted runs are not timed.",
        ],
        "unique_logged_runs": len(rows),
        "unique_non_debug_runs": sum(not row["debug"] for row in rows),
        "logged_training_hours_all": total_hours,
        "logged_training_hours_non_debug": full_hours,
        "runs_by_phase": dict(Counter(row["phase"] for row in rows)),
        "runs_by_device": dict(Counter(row["device"] for row in rows)),
        "hardware": {"machine": "MacBook Air", "chip": "Apple M3", "cpu_cores": 8, "memory_gb": 16},
        "model_parameters": {"total": 946208, "trainable": 946208},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "spacy": spacy.__version__,
            "spacy_model_en_core_web_sm": "3.4.1",
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "compute_budget_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status = (
        f"Logged training summaries currently establish a lower bound of {total_hours:.1f} device-hours "
        f"across {len(rows)} deduplicated runs (including development and debug runs). "
        "This excludes untimed nonce adaptation, evaluation, preprocessing, and interrupted runs."
    )
    (ROOT / "paper/tables/compute_budget_status.tex").write_text(status + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
