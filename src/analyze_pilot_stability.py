from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "Pilot_v4_1_roberta"
PRIMARY_COMPARISON_SUFFIX = "minus_matched_content_masked"
PRIMARY_METRICS = [
    "target_verb_logprob",
    "target_verb_logprob_sum",
    "verb_class_accuracy",
    "class_preference_score",
    "top5_surface_accuracy",
    "top5_lemma_accuracy",
]


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sign(value: float, epsilon: float = 1e-12) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def direction_label(signs: list[int]) -> str:
    nonzero = [s for s in signs if s != 0]
    if not nonzero:
        return "all_zero"
    if all(s > 0 for s in nonzero):
        return "positive_consistent"
    if all(s < 0 for s in nonzero):
        return "negative_consistent"
    return "mixed"


def stability_label(mean_delta: float, sd_delta: float, signs: list[int]) -> str:
    label = direction_label(signs)
    if label == "mixed":
        return "unstable_direction"
    if label == "all_zero":
        return "near_zero"
    if abs(mean_delta) < 1e-9:
        return "near_zero"
    ratio = abs(mean_delta) / max(sd_delta, 1e-12)
    if ratio >= 2.0:
        return "directionally_stable"
    if ratio >= 1.0:
        return "directionally_consistent_but_small"
    return "directionally_consistent_low_margin"


def summarize_paired_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row["comparison"].endswith(PRIMARY_COMPARISON_SUFFIX):
            continue
        grouped[(row["category"], row["metric"], row["comparison"])].append(row)

    out: list[dict[str, Any]] = []
    for (category, metric, comparison), items in sorted(grouped.items()):
        values = [float(item["mean_delta"]) for item in sorted(items, key=lambda x: str(x["seed"]))]
        signs = [sign(v) for v in values]
        mean_delta = mean(values)
        sd_delta = pstdev(values) if len(values) > 1 else 0.0
        positive = sum(1 for s in signs if s > 0)
        negative = sum(1 for s in signs if s < 0)
        zero = sum(1 for s in signs if s == 0)
        out.append(
            {
                "category": category,
                "metric": metric,
                "comparison": comparison,
                "n_seeds": len(values),
                "mean_delta": mean_delta,
                "sd_delta": sd_delta,
                "min_delta": min(values),
                "max_delta": max(values),
                "positive_seeds": positive,
                "negative_seeds": negative,
                "zero_seeds": zero,
                "direction": direction_label(signs),
                "stability": stability_label(mean_delta, sd_delta, signs),
                "seed_values": " ".join(f"{v:.6f}" for v in values),
            }
        )
    return out


def summarize_training(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_seed_condition = {(row["seed"], row["condition"]): row for row in rows}
    categories = sorted({condition.split("_", 1)[0] for _, condition in by_seed_condition if condition.endswith("_masked_p100")})
    seeds = sorted({row["seed"] for row in rows})
    out: list[dict[str, Any]] = []
    for category in categories:
        fn = f"{category}_masked_p100"
        content = f"{category}_matched_content_masked_p100"
        strict = f"{category}_strict_content_control_p100"
        for right in ["full_input", content, strict]:
            train_values: list[float] = []
            dev_values: list[float] = []
            for seed in seeds:
                left_row = by_seed_condition.get((seed, fn))
                right_row = by_seed_condition.get((seed, right))
                if not left_row or not right_row:
                    continue
                train_values.append(float(left_row["final_train_loss"]) - float(right_row["final_train_loss"]))
                dev_values.append(float(left_row["final_dev_loss"]) - float(right_row["final_dev_loss"]))
            if not dev_values:
                continue
            out.append(
                {
                    "category": category,
                    "comparison": f"{fn}_minus_{right}",
                    "n_seeds": len(dev_values),
                    "mean_train_loss_delta": mean(train_values),
                    "mean_dev_loss_delta": mean(dev_values),
                    "sd_dev_loss_delta": pstdev(dev_values) if len(dev_values) > 1 else 0.0,
                    "dev_loss_direction": direction_label([sign(v) for v in dev_values]),
                    "dev_loss_seed_values": " ".join(f"{v:.6f}" for v in dev_values),
                }
            )
    return out


def summarize_metric_families(stability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family = {
        "target_verb_logprob": "target_probability",
        "target_verb_logprob_sum": "target_probability_sum",
        "class_preference_score": "class_preference",
        "verb_class_accuracy": "class_accuracy",
        "top5_surface_accuracy": "topk_surface",
        "top5_lemma_accuracy": "topk_lemma",
    }
    rows = [row for row in stability_rows if row["metric"] in PRIMARY_METRICS]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[family.get(row["metric"], row["metric"])].append(row)
    out: list[dict[str, Any]] = []
    for metric_family, items in sorted(grouped.items()):
        stable = sum(1 for row in items if str(row["stability"]).startswith("directionally"))
        mixed = sum(1 for row in items if row["stability"] == "unstable_direction")
        positive = sum(1 for row in items if row["direction"] == "positive_consistent")
        negative = sum(1 for row in items if row["direction"] == "negative_consistent")
        out.append(
            {
                "metric_family": metric_family,
                "n_category_metrics": len(items),
                "directionally_consistent_count": stable,
                "mixed_direction_count": mixed,
                "positive_consistent_count": positive,
                "negative_consistent_count": negative,
            }
        )
    return out


def format_row_lookup(rows: list[dict[str, Any]], metric: str) -> dict[str, dict[str, Any]]:
    return {row["category"]: row for row in rows if row["metric"] == metric and row["comparison"].endswith(PRIMARY_COMPARISON_SUFFIX)}


def write_report(
    results_dir: Path,
    stability_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> None:
    target = format_row_lookup(stability_rows, "target_verb_logprob")
    cps = format_row_lookup(stability_rows, "class_preference_score")
    top5 = format_row_lookup(stability_rows, "top5_lemma_accuracy")
    lines = [
        "# Pilot_v4_1_roberta Stability Diagnostics",
        "",
        "## Scope",
        "",
        "This report is a Pilot diagnostic, not a theoretical causal analysis. It summarizes seed-direction stability for function-masked minus matched-content-masked MVP deltas and checks whether training-loss differences tell the same story as MVP scoring.",
        "",
        "## Primary Delta Stability",
        "",
        "| Category | Target logprob mean | Target direction | Class pref. mean | Class pref. direction | Top5 lemma mean | Top5 lemma direction |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for category in sorted(target):
        target_row = target[category]
        cps_row = cps.get(category, {})
        top5_row = top5.get(category, {})
        lines.append(
            f"| {category} | {target_row['mean_delta']:.4f} | {target_row['stability']} | "
            f"{float(cps_row.get('mean_delta', math.nan)):.4f} | {cps_row.get('stability', 'missing')} | "
            f"{float(top5_row.get('mean_delta', math.nan)):.4f} | {top5_row.get('stability', 'missing')} |"
        )

    lines.extend(
        [
            "",
            "## Metric-Family Summary",
            "",
            "| Metric family | Category-metric cells | Directionally consistent | Mixed direction | Positive consistent | Negative consistent |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in family_rows:
        lines.append(
            f"| {row['metric_family']} | {row['n_category_metrics']} | {row['directionally_consistent_count']} | "
            f"{row['mixed_direction_count']} | {row['positive_consistent_count']} | {row['negative_consistent_count']} |"
        )

    lines.extend(
        [
            "",
            "## Training-Loss Diagnostics",
            "",
            "| Category | Comparison | Mean dev-loss delta | Direction | Seed values |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in training_rows:
        if not row["comparison"].endswith("matched_content_masked_p100"):
            continue
        lines.append(
            f"| {row['category']} | {row['comparison']} | {row['mean_dev_loss_delta']:.4f} | {row['dev_loss_direction']} | {row['dev_loss_seed_values']} |"
        )

    lines.extend(
        [
            "",
            "## Main-Experiment Implication",
            "",
            "- The v4.1 effects are weak and often low-margin across seeds; no category should be treated as final theory evidence.",
            "- Target logprob and top-k metrics can point in different directions, so the main experiment should pre-register a primary metric and report the others as diagnostics.",
            "- `target_verb_logprob` and `target_verb_logprob_sum` should both remain in the output until subword effects are fully characterized.",
            "- Training-loss differences do not by themselves validate cue effects; PRON in particular shows that content-control distribution changes can dominate the training objective.",
            "- At least 5 seeds are advisable for the main experiment because 3-seed Pilot deltas still show mixed or low-margin direction for several category/metric cells.",
        ]
    )
    (results_dir / "pilot_stability_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    paired = csv_read(results_dir / "paired_delta_summary.csv")
    training = csv_read(results_dir / "training_condition_summary.csv")
    stability_rows = summarize_paired_deltas(paired)
    training_rows = summarize_training(training)
    family_rows = summarize_metric_families(stability_rows)

    csv_write(results_dir / "metric_stability_summary.csv", stability_rows)
    csv_write(results_dir / "training_loss_delta_summary.csv", training_rows)
    csv_write(results_dir / "metric_family_stability_summary.csv", family_rows)
    write_report(results_dir, stability_rows, training_rows, family_rows)
    print(f"Stability diagnostics written to {results_dir}")


if __name__ == "__main__":
    main()
