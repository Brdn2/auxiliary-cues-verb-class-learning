#!/usr/bin/env python3
"""Verify the release-safe file set and recompute headline aggregate values."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = {"data", "models", "logs", "tmp", "archive", "audits", "literature"}
FORBIDDEN_RESULT_PATTERNS = {
    "mvp_per_example*.csv",
    "nonce_per_example.csv",
    "baseline_per_example.csv",
    "item_cue_features.csv",
    "aux_context_sample*.csv",
}
REQUIRED = {
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "docs/DATA_STATEMENT.md",
    "output/pdf/paper.pdf",
    "paper/main.tex",
    "results/ACL_confirmatory_aux_v2/CONFIRMATORY_DECISION.json",
}


def rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(observed: float, expected: float, tolerance: float = 5e-4) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"expected {expected}, observed {observed}")


def main() -> None:
    missing = sorted(relative for relative in REQUIRED if not (ROOT / relative).is_file())
    if missing:
        raise AssertionError(f"required release files missing: {missing}")

    leaked_roots = sorted(name for name in FORBIDDEN_ROOTS if (ROOT / name).exists())
    if leaked_roots:
        raise AssertionError(f"restricted roots present in public artifact: {leaked_roots}")

    leaked_results = sorted(
        str(path.relative_to(ROOT))
        for pattern in FORBIDDEN_RESULT_PATTERNS
        for path in (ROOT / "results").glob(f"**/{pattern}")
    )
    if leaked_results:
        raise AssertionError(f"restricted result files present: {leaked_results}")

    curve = rows("results/Expanded_learning_curve_summary/original_learning_curve.csv")
    by_scale: dict[str, list[float]] = {}
    for row in curve:
        by_scale.setdefault(row["scale"], []).append(float(row["verb_class_accuracy"]))
    close(statistics.fmean(by_scale["0720k"]), 0.5366666667)
    close(statistics.fmean(by_scale["full"]), 0.7216666667)

    baselines = {
        row["baseline"]: float(row["accuracy"])
        for row in rows("results/ACL_static_analysis_v1/baseline_summary.csv")
        if row["seed"] == "deterministic"
    }
    close(baselines["aux_only_nb"], 0.305)
    close(baselines["pos_frame_count"], 0.445)
    close(baselines["lexical_cooccurrence_nb"], 0.520)

    confirmatory = {
        (row["contrast_label"], row["metric"]): row
        for row in rows(
            "results/ACL_confirmatory_aux_v2/confirmatory_hierarchical_bootstrap.csv"
        )
    }
    total = confirmatory[("total_AUX_effect", "class_preference_score")]
    close(float(total["delta_comparison_minus_reference"]), -0.1443538974)
    close(float(total["ci95_low"]), -0.2071668166)
    close(float(total["ci95_high"]), -0.0847858851)

    decision = json.loads(
        (ROOT / "results/ACL_confirmatory_aux_v2/CONFIRMATORY_DECISION.json").read_text(
            encoding="utf-8"
        )
    )
    if decision.get("specificity_criterion_satisfied") is not False:
        raise AssertionError("confirmatory decision must reject the AUX-specific claim")

    nonce = {
        (row["comparison"], row["metric"]): row
        for row in rows("results/ACL_nonce_cross_template_v2/hierarchical_bootstrap.csv")
    }
    shuffle_accuracy = nonce[("aux_shuffled_exposure", "verb_class_accuracy")]
    shuffle_cps = nonce[("aux_shuffled_exposure", "class_preference_score")]
    close(float(shuffle_accuracy["delta_comparison_minus_reference"]), -0.0326666667)
    close(float(shuffle_cps["delta_comparison_minus_reference"]), -0.0956817987)

    print(
        "public artifact verification passed: "
        f"{len(REQUIRED)} required files and headline aggregates reproduced"
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, OSError, ValueError) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
