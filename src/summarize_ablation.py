from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from statistical_analysis import hierarchical_paired_bootstrap
from update_paper import validate_complete


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONDITIONS = {"original", "AUX_target_ablation", "AUX_matched_content", "AUX_matched_function", "AUX_identity_shuffle"}
EXPECTED_SEEDS = {2026, 2027, 2028, 2029, 2030}
EXPECTED_ITEMS = 600


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def summarize_training(model_dir: Path, expected_seeds: set[int], results_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for condition in sorted(EXPECTED_CONDITIONS):
        for seed in sorted(expected_seeds):
            path = model_dir / condition / str(seed) / "training_summary.json"
            if not path.exists():
                raise RuntimeError(f"Missing full training summary: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("debug") or int(summary["seed"]) != seed or summary["condition"] != condition:
                raise RuntimeError(f"Invalid full training summary: {path}")
            history = summary["history"]
            if len(history) != 5 or [int(epoch["epoch"]) for epoch in history] != [1, 2, 3, 4, 5]:
                raise RuntimeError(f"Incomplete five-epoch history: {path}")
            rows.append({
                "condition": condition,
                "seed": seed,
                "final_train_loss": float(history[-1]["train_loss"]),
                "final_dev_loss": float(history[-1]["dev_loss"]),
                "best_dev_loss": min(float(epoch["dev_loss"]) for epoch in history),
                "best_dev_epoch": min(history, key=lambda epoch: float(epoch["dev_loss"]))["epoch"],
                "elapsed_seconds": float(summary["elapsed_seconds"]),
                "device": summary["device"],
                "max_length": summary["max_length"],
            })
    write_csv(results_dir / "training_diagnostics.csv", rows)
    aggregate: dict[str, Any] = {}
    for condition in sorted(EXPECTED_CONDITIONS):
        condition_rows = [row for row in rows if row["condition"] == condition]
        aggregate[condition] = {}
        for metric in ("final_train_loss", "final_dev_loss", "best_dev_loss", "elapsed_seconds"):
            values = [float(row[metric]) for row in condition_rows]
            aggregate[condition][metric] = {
                "mean": statistics.mean(values),
                "seed_sd": statistics.stdev(values),
            }
    (results_dir / "training_diagnostics_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results/natural_corpus"))
    parser.add_argument("--model-dir", default=str(ROOT / "models/natural_corpus"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=81001)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    rows = read_csv(results_dir / "mvp_per_example.csv")
    # This must precede every write below: no partial, duplicated, or
    # evaluation-item-misaligned matrix may produce paper-facing statistics.
    validate_complete(rows, EXPECTED_SEEDS, EXPECTED_ITEMS)
    expected_seeds = EXPECTED_SEEDS
    summarize_training(Path(args.model_dir), expected_seeds, results_dir)
    comparisons = [
        ("original", "AUX_target_ablation", "total_AUX_effect"),
        ("original", "AUX_matched_content", "content_damage_effect"),
        ("original", "AUX_matched_function", "function_damage_effect"),
        ("original", "AUX_identity_shuffle", "AUX_identity_effect"),
        ("AUX_matched_content", "AUX_target_ablation", "AUX_specific_vs_content"),
        ("AUX_matched_function", "AUX_target_ablation", "AUX_specific_vs_function"),
    ]
    bootstrap_rows: list[dict[str, Any]] = []
    for index, (reference, comparison, label) in enumerate(comparisons):
        for metric in ("class_preference_score", "verb_class_accuracy"):
            result = hierarchical_paired_bootstrap(rows, reference, comparison, metric, args.samples, args.seed + index * 2)
            result["contrast_label"] = label
            bootstrap_rows.append(result)
    write_csv(results_dir / "confirmatory_hierarchical_bootstrap.csv", bootstrap_rows)
    (results_dir / "confirmatory_hierarchical_bootstrap.json").write_text(json.dumps(bootstrap_rows, indent=2) + "\n", encoding="utf-8")

    means: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in ("class_preference_score", "verb_class_accuracy"):
            means[(row["condition"], int(row["seed"]), metric)].append(float(row[metric]))
    seed_rows: list[dict[str, Any]] = []
    for reference, comparison, label in comparisons:
        seeds = sorted({seed for condition, seed, _ in means if condition == reference})
        for seed in seeds:
            for metric in ("class_preference_score", "verb_class_accuracy"):
                ref = sum(means[(reference, seed, metric)]) / len(means[(reference, seed, metric)])
                comp = sum(means[(comparison, seed, metric)]) / len(means[(comparison, seed, metric)])
                seed_rows.append({
                    "contrast_label": label, "reference": reference, "comparison": comparison, "seed": seed,
                    "metric": metric, "reference_mean": ref, "comparison_mean": comp,
                    "delta_comparison_minus_reference": comp - ref,
                })
    write_csv(results_dir / "confirmatory_seed_deltas.csv", seed_rows)

    primary = [row for row in bootstrap_rows if row["metric"] == "class_preference_score"]
    lines = ["# Confirmatory AUX summary", "", "Primary metric: class preference score. Deltas are comparison minus reference.", ""]
    for row in primary:
        p = float(row["bootstrap_p_two_sided"])
        p_text = f"{p:.4f}" if p > 0 else f"<{1 / int(row['bootstrap_samples']):.4f}"
        lines.append(
            f"- {row['contrast_label']}: Δ={float(row['delta_comparison_minus_reference']):.4f}, "
            f"95% CI [{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}], p={p_text}."
        )
    lines.extend(["", "The AUX-specific claim requires both `AUX_specific_vs_content` and `AUX_specific_vs_function` to be negative with confidence intervals excluding zero."])
    (results_dir / "CONFIRMATORY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
