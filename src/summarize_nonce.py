from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from acl_statistics import hierarchical_paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "zero_shot",
    "intact_exposure",
    "aux_ablated_exposure",
    "aux_shuffled_exposure",
    "matched_function_ablated_exposure",
}
SEEDS = {2026, 2027, 2028}
N_ITEMS = 500
METRICS = ("verb_class_accuracy", "class_preference_score")
COMPARISONS = (
    ("aux_ablated_exposure", "diagnostic_interventions_6"),
    ("aux_shuffled_exposure", "diagnostic_interventions_6"),
    ("matched_function_ablated_exposure", "diagnostic_interventions_6"),
    ("zero_shot", "adaptation_check_2"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def validate_nonce_matrix(rows: list[dict[str, str]]) -> None:
    if {row["condition"] for row in rows} != CONDITIONS:
        raise RuntimeError("Nonce condition set is incomplete or unexpected.")
    reference_ids: set[int] | None = None
    seen: set[tuple[str, int, int]] = set()
    for condition in sorted(CONDITIONS):
        observed_seeds = {int(row["seed"]) for row in rows if row["condition"] == condition}
        if observed_seeds != SEEDS:
            raise RuntimeError(f"Incomplete nonce seeds for {condition}: {sorted(observed_seeds)}")
        for seed in sorted(SEEDS):
            subset = [row for row in rows if row["condition"] == condition and int(row["seed"]) == seed]
            ids = {int(row["example_id"]) for row in subset}
            if len(subset) != N_ITEMS or len(ids) != N_ITEMS:
                raise RuntimeError(
                    f"Incomplete or duplicated nonce cell {condition}/seed={seed}: "
                    f"rows={len(subset)}, unique_items={len(ids)}"
                )
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise RuntimeError(f"Nonce evaluation-item mismatch for {condition}/seed={seed}")
            for item_id in ids:
                key = (condition, seed, item_id)
                if key in seen:
                    raise RuntimeError(f"Duplicate nonce evaluation key: {key}")
                seen.add(key)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in input order."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def add_holm_columns(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["multiplicity_family"])].append(index)
    for family, indices in grouped.items():
        raw = [float(rows[index]["bootstrap_p_two_sided"]) for index in indices]
        samples = [int(rows[index]["bootstrap_samples"]) for index in indices]
        if len(set(samples)) != 1:
            raise RuntimeError(f"Inconsistent bootstrap sample counts in {family}")
        adjusted = holm_adjust(raw)
        upper = holm_adjust([max(value, 1 / samples[0]) for value in raw])
        for index, value, bound in zip(indices, adjusted, upper):
            rows[index]["holm_adjusted_p"] = value
            rows[index]["holm_adjusted_p_upper_bound"] = bound
            rows[index]["multiplicity_family_size"] = len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the diagnostic nonce experiment.")
    parser.add_argument("--results-dir", default=str(ROOT / "results/ACL_nonce_cross_template_v2"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=51001)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    rows = read_csv(results_dir / "nonce_per_example.csv")
    validate_nonce_matrix(rows)

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], int(row["seed"]))].append(row)
    metrics_rows = [
        {
            "condition": condition,
            "seed": seed,
            "n_examples": len(items),
            **{
                metric: statistics.mean(float(item[metric]) for item in items)
                for metric in METRICS
            },
        }
        for (condition, seed), items in sorted(grouped.items())
    ]

    bootstrap_rows: list[dict[str, Any]] = []
    for comparison_index, (comparison, family) in enumerate(COMPARISONS):
        for metric_index, metric in enumerate(METRICS):
            result = hierarchical_paired_bootstrap(
                rows,
                "intact_exposure",
                comparison,
                metric,
                args.samples,
                args.seed + comparison_index * len(METRICS) + metric_index,
            )
            result["multiplicity_family"] = family
            bootstrap_rows.append(result)
    add_holm_columns(bootstrap_rows)

    seed_rows: list[dict[str, Any]] = []
    for comparison, family in COMPARISONS:
        for seed in sorted(SEEDS):
            for metric in METRICS:
                reference = next(
                    row for row in metrics_rows if row["condition"] == "intact_exposure" and row["seed"] == seed
                )
                compared = next(
                    row for row in metrics_rows if row["condition"] == comparison and row["seed"] == seed
                )
                seed_rows.append(
                    {
                        "reference": "intact_exposure",
                        "comparison": comparison,
                        "multiplicity_family": family,
                        "seed": seed,
                        "metric": metric,
                        "delta_comparison_minus_reference": float(compared[metric]) - float(reference[metric]),
                    }
                )

    # All outputs are written only after the full matrix and all statistics pass.
    write_csv(results_dir / "metrics_summary.csv", metrics_rows)
    write_csv(results_dir / "hierarchical_bootstrap.csv", bootstrap_rows)
    write_csv(results_dir / "nonce_seed_deltas.csv", seed_rows)
    (results_dir / "hierarchical_bootstrap.json").write_text(
        json.dumps(bootstrap_rows, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "complete",
        "design_label": "diagnostic",
        "rows": len(rows),
        "conditions": sorted(CONDITIONS),
        "seeds": sorted(SEEDS),
        "items_per_cell": N_ITEMS,
        "multiplicity_families": {
            "diagnostic_interventions_6": "Three intervention contrasts versus intact exposure, across two metrics.",
            "adaptation_check_2": "Zero-shot versus intact exposure, across two metrics.",
        },
    }
    (results_dir / "NONCE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
