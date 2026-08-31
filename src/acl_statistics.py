from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def hierarchical_paired_bootstrap(
    rows: list[dict[str, str]], reference: str, comparison: str, metric: str, samples: int, seed: int,
) -> dict[str, Any]:
    lookup = {(row["condition"], int(row["seed"]), int(row["example_id"])): float(row[metric]) for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows if row["condition"] == reference})
    paired: dict[int, list[float]] = {}
    for model_seed in seeds:
        ids = sorted({int(row["example_id"]) for row in rows if row["condition"] == reference and int(row["seed"]) == model_seed})
        paired[model_seed] = [lookup[(comparison, model_seed, item)] - lookup[(reference, model_seed, item)] for item in ids]
    observed = float(np.mean([value for values in paired.values() for value in values]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for draw in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for model_seed in sampled_seeds:
            values = np.asarray(paired[int(model_seed)])
            seed_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        draws[draw] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    p_two_sided = 2 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))
    return {
        "reference": reference, "comparison": comparison, "metric": metric, "n_seeds": len(seeds),
        "n_items_per_seed": len(next(iter(paired.values()))), "delta_comparison_minus_reference": observed,
        "ci95_low": float(low), "ci95_high": float(high), "bootstrap_p_two_sided": min(p_two_sided, 1.0),
        "bootstrap_samples": samples, "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "results/Expanded_learning_curve_full/mvp_per_example.csv"))
    parser.add_argument("--output", default=str(ROOT / "results/ACL_statistics/legacy_hierarchical_bootstrap.csv"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--reference", default="original")
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    conditions = sorted({row["condition"] for row in rows if row["condition"] != args.reference})
    metrics = ["verb_class_accuracy", "class_preference_score"]
    results = [
        hierarchical_paired_bootstrap(rows, args.reference, condition, metric, args.samples, args.seed + index)
        for index, (condition, metric) in enumerate((c, m) for c in conditions for m in metrics)
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output.with_suffix(".json")).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
