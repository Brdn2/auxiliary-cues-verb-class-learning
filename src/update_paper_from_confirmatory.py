from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [
    "original",
    "AUX_target_ablation",
    "AUX_matched_content",
    "AUX_matched_function",
    "AUX_identity_shuffle",
]
LABELS = {
    "original": "the Original condition",
    "AUX_target_ablation": "AUX-identity ablation",
    "AUX_matched_content": "the matched-content control",
    "AUX_matched_function": "the matched-function control",
    "AUX_identity_shuffle": "the AUX-identity-shuffle condition",
}
TABLE_LABELS = {
    "original": "Original",
    "AUX_target_ablation": "AUX-identity ablation",
    "AUX_matched_content": "Matched-content ablation",
    "AUX_matched_function": "Matched-function ablation",
    "AUX_identity_shuffle": "AUX-identity shuffle",
}
PRIMARY_SPECIFICITY_CONTRASTS = ("AUX_specific_vs_content", "AUX_specific_vs_function")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_complete(rows: list[dict[str, str]], expected_seeds: set[int], expected_items: int) -> None:
    observed_conditions = {row["condition"] for row in rows}
    if observed_conditions != set(CONDITIONS):
        raise RuntimeError(
            f"Incomplete condition set: expected={CONDITIONS}, observed={sorted(observed_conditions)}"
        )

    reference_ids: set[int] | None = None
    seen: set[tuple[str, int, int]] = set()
    for condition in CONDITIONS:
        observed_seeds = {int(row["seed"]) for row in rows if row["condition"] == condition}
        if observed_seeds != expected_seeds:
            raise RuntimeError(
                f"Incomplete seeds for {condition}: expected={sorted(expected_seeds)}, "
                f"observed={sorted(observed_seeds)}"
            )
        for seed in sorted(expected_seeds):
            subset = [row for row in rows if row["condition"] == condition and int(row["seed"]) == seed]
            item_ids = {int(row["example_id"]) for row in subset}
            if len(subset) != expected_items or len(item_ids) != expected_items:
                raise RuntimeError(
                    f"Incomplete or duplicated evaluation cell {condition}/seed={seed}: "
                    f"rows={len(subset)}, unique_items={len(item_ids)}, expected={expected_items}"
                )
            if reference_ids is None:
                reference_ids = item_ids
            elif item_ids != reference_ids:
                raise RuntimeError(f"Evaluation-item mismatch for {condition}/seed={seed}")
            for item_id in item_ids:
                key = (condition, seed, item_id)
                if key in seen:
                    raise RuntimeError(f"Duplicate evaluation key: {key}")
                seen.add(key)


def seed_condition_means(rows: list[dict[str, str]]) -> dict[str, dict[str, list[float]]]:
    cells: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in ("verb_class_accuracy", "class_preference_score"):
            cells[(row["condition"], int(row["seed"]), metric)].append(float(row[metric]))
    output: dict[str, dict[str, list[float]]] = {}
    for condition in CONDITIONS:
        output[condition] = {}
        for metric in ("verb_class_accuracy", "class_preference_score"):
            output[condition][metric] = [
                statistics.mean(values)
                for (cell_condition, _seed, cell_metric), values in sorted(cells.items())
                if cell_condition == condition and cell_metric == metric
            ]
    return output


def contrast_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["contrast_label"], row["metric"]): row for row in rows}


def seed_direction_counts(rows: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        if float(row["delta_comparison_minus_reference"]) < 0:
            counts[(row["contrast_label"], row["metric"])] += 1
    return counts


def p_text(row: dict[str, str]) -> str:
    p = float(row["bootstrap_p_two_sided"])
    samples = int(row["bootstrap_samples"])
    formatted = f"{1 / samples:.4f}".lstrip("0") if p == 0 else f"{p:.4f}".lstrip("0")
    return f"$p<{formatted}$" if p == 0 else f"$p={formatted}$"


def effect_text(row: dict[str, str], digits: int = 3) -> str:
    delta = float(row["delta_comparison_minus_reference"])
    low = float(row["ci95_low"])
    high = float(row["ci95_high"])
    return f"$\\Delta={delta:.{digits}f}$, 95\\% CI [{low:.{digits}f}, {high:.{digits}f}], {p_text(row)}"


def specificity_decision(
    contrasts: dict[tuple[str, str], dict[str, str]], direction_counts: dict[tuple[str, str], int]
) -> bool:
    for label in PRIMARY_SPECIFICITY_CONTRASTS:
        row = contrasts[(label, "class_preference_score")]
        if float(row["ci95_high"]) >= 0 or direction_counts[(label, "class_preference_score")] < 4:
            return False
    return True


def render_table(means: dict[str, dict[str, list[float]]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Training condition & Accuracy (\%) & \cps{} \\",
        r"\midrule",
    ]
    for condition in CONDITIONS:
        accuracy = means[condition]["verb_class_accuracy"]
        cps = means[condition]["class_preference_score"]
        lines.append(
            f"{TABLE_LABELS[condition]} & {100 * statistics.mean(accuracy):.1f} "
            f"({100 * statistics.stdev(accuracy):.1f}) & "
            f"${statistics.mean(cps):.3f}$ ({statistics.stdev(cps):.3f}) \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Natural-corpus results over five initialization seeds and the same 600 class-balanced items per seed. Parentheses give seed-level standard deviations.}",
            r"\label{tab:confirmatory}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_training_diagnostics(summary: dict[str, Any]) -> str:
    short_labels = {
        "original": "Original",
        "AUX_target_ablation": "AUX-identity ablation",
        "AUX_matched_content": "Matched-content ablation",
        "AUX_matched_function": "Matched-function ablation",
        "AUX_identity_shuffle": "AUX-identity shuffle",
    }
    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Condition & Train & Dev & Hours \\",
        r"\midrule",
    ]
    for condition in CONDITIONS:
        row = summary[condition]
        train = row["final_train_loss"]
        dev = row["final_dev_loss"]
        elapsed = row["elapsed_seconds"]
        lines.append(
            f"{short_labels[condition]} & {train['mean']:.3f} ({train['seed_sd']:.3f}) & "
            f"{dev['mean']:.3f} ({dev['seed_sd']:.3f}) & "
            f"{elapsed['mean'] / 3600:.2f} ({elapsed['seed_sd'] / 3600:.2f}) \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Confirmatory-ablation optimization diagnostics (mean with seed-level standard deviation). Dev loss is always measured on the unchanged Original-condition split.}",
            r"\label{tab:training-diagnostics}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_results(
    means: dict[str, dict[str, list[float]]],
    contrasts: dict[tuple[str, str], dict[str, str]],
    direction_counts: dict[tuple[str, str], int],
) -> tuple[str, str, dict[str, Any]]:
    total = contrasts[("total_AUX_effect", "class_preference_score")]
    content = contrasts[("AUX_specific_vs_content", "class_preference_score")]
    function = contrasts[("AUX_specific_vs_function", "class_preference_score")]
    identity = contrasts[("AUX_identity_effect", "class_preference_score")]
    specific = specificity_decision(contrasts, direction_counts)
    total_significant = float(total["ci95_high"]) < 0

    if specific:
        decision = "The preregistered criterion for an AUX-specific contribution is therefore satisfied."
        abstract = (
            f"In the five-seed natural-corpus experiment, AUX ablation lowers CPS by "
            f"{abs(float(total['delta_comparison_minus_reference'])):.3f} relative to Original and satisfies "
            "the preregistered AUX-specificity criterion."
        )
    elif total_significant:
        decision = (
            "The target ablation shows an overall AUX contribution, but the preregistered criterion "
            "for AUX specificity against both controls is not satisfied."
        )
        abstract = (
            f"In the five-seed natural-corpus experiment, AUX-identity ablation lowers CPS by "
            f"{abs(float(total['delta_comparison_minus_reference'])):.3f} relative to the Original condition "
            f"(95\\% CI [{float(total['ci95_low']):.3f}, {float(total['ci95_high']):.3f}]), but does not satisfy "
            "the preregistered specificity criterion against either matched-damage control."
        )
    else:
        decision = (
            "The preregistered natural-corpus experiment does not establish a reliable overall AUX effect "
            "or AUX specificity."
        )
        abstract = "The five-seed natural-corpus experiment does not establish a reliable auxiliary-specific contribution under our preregistered criterion."

    original_cps = statistics.mean(means["original"]["class_preference_score"])
    target_cps = statistics.mean(means["AUX_target_ablation"]["class_preference_score"])
    results = (
        f"Across five seeds, the Original condition obtains a mean \\cps{{}} of {original_cps:.3f}, compared with "
        f"{target_cps:.3f} after AUX-identity ablation ({effect_text(total)}). "
        f"Relative to the matched-content control, the ablation difference is {effect_text(content)} "
        f"and is negative in {direction_counts[('AUX_specific_vs_content', 'class_preference_score')]}/5 seeds. "
        f"Relative to the matched-function control, the difference is {effect_text(function)} "
        f"and is negative in {direction_counts[('AUX_specific_vs_function', 'class_preference_score')]}/5 seeds. "
        f"The AUX-identity-shuffle condition also yields a lower \\cps{{}} than the Original condition ({effect_text(identity)}).\n"
    )
    decision_record = {
        "specificity_criterion_satisfied": specific,
        "overall_aux_effect_ci_below_zero": total_significant,
        "required_specificity_contrasts": list(PRIMARY_SPECIFICITY_CONTRASTS),
        "negative_seed_counts": {
            label: direction_counts[(label, "class_preference_score")]
            for label in PRIMARY_SPECIFICITY_CONTRASTS
        },
        "interpretation": decision,
    }
    return results, abstract + "\n", decision_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate and insert complete E7 results into the ACL manuscript.")
    parser.add_argument("--results-dir", default=str(ROOT / "results/ACL_confirmatory_aux_v2"))
    parser.add_argument("--paper-dir", default=str(ROOT / "paper"))
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--expected-items", type=int, default=600)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    paper_dir = Path(args.paper_dir)
    expected_seeds = {int(seed) for seed in args.seeds.split(",")}
    rows = read_csv(results_dir / "mvp_per_example.csv")
    validate_complete(rows, expected_seeds, args.expected_items)

    bootstrap_rows = read_csv(results_dir / "confirmatory_hierarchical_bootstrap.csv")
    seed_delta_rows = read_csv(results_dir / "confirmatory_seed_deltas.csv")
    contrasts = contrast_lookup(bootstrap_rows)
    direction_counts = seed_direction_counts(seed_delta_rows)
    required = {
        (label, metric)
        for label in (
            "total_AUX_effect",
            "content_damage_effect",
            "function_damage_effect",
            "AUX_identity_effect",
            "AUX_specific_vs_content",
            "AUX_specific_vs_function",
        )
        for metric in ("class_preference_score", "verb_class_accuracy")
    }
    bootstrap_keys = [(row["contrast_label"], row["metric"]) for row in bootstrap_rows]
    if len(bootstrap_keys) != len(set(bootstrap_keys)) or set(bootstrap_keys) != required:
        raise RuntimeError(
            "Invalid bootstrap contrast matrix: "
            f"missing={sorted(required - set(bootstrap_keys))}, "
            f"unexpected={sorted(set(bootstrap_keys) - required)}, duplicates={len(bootstrap_keys) - len(set(bootstrap_keys))}"
        )
    expected_seed_delta_keys = {
        (label, metric, seed)
        for label, metric in required
        for seed in expected_seeds
    }
    seed_delta_keys = [
        (row["contrast_label"], row["metric"], int(row["seed"]))
        for row in seed_delta_rows
    ]
    if len(seed_delta_keys) != len(set(seed_delta_keys)) or set(seed_delta_keys) != expected_seed_delta_keys:
        raise RuntimeError(
            "Invalid seed-delta matrix: "
            f"missing={len(expected_seed_delta_keys - set(seed_delta_keys))}, "
            f"unexpected={len(set(seed_delta_keys) - expected_seed_delta_keys)}, "
            f"duplicates={len(seed_delta_keys) - len(set(seed_delta_keys))}"
        )

    means = seed_condition_means(rows)
    results_text, abstract_text, decision = render_results(means, contrasts, direction_counts)
    training_summary = json.loads(
        (results_dir / "training_diagnostics_summary.json").read_text(encoding="utf-8")
    )
    table_dir = paper_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "confirmatory.tex").write_text(render_table(means), encoding="utf-8")
    (table_dir / "confirmatory_results_text.tex").write_text(results_text, encoding="utf-8")
    (table_dir / "confirmatory_abstract_status.tex").write_text(abstract_text, encoding="utf-8")
    (table_dir / "training_diagnostics.tex").write_text(
        render_training_diagnostics(training_summary), encoding="utf-8"
    )
    (results_dir / "CONFIRMATORY_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
