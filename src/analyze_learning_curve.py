from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCALES = ["0720k", "2000k", "full"]
CHANCE = 0.20


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for scale in SCALES:
        results = PROJECT / "results" / "learning_curve" / scale
        metrics_path = results / "metrics_summary.csv"
        per_item_path = results / "mvp_per_example.csv"
        if not metrics_path.exists() or not per_item_path.exists():
            missing.append(scale)
            continue
        metrics = [row for row in read_csv(metrics_path) if row["condition"] == "original"]
        per_item = [row for row in read_csv(per_item_path) if row["condition"] == "original"]
        for metric_row in metrics:
            seed = int(metric_row["seed"])
            predicted = Counter(row["predicted_class"] for row in per_item if int(row["seed"]) == seed)
            total = sum(predicted.values())
            dominant_class, dominant_count = predicted.most_common(1)[0]
            rows.append({
                "scale": scale,
                "seed": seed,
                "verb_class_accuracy": float(metric_row["verb_class_accuracy"]),
                "class_preference_score": float(metric_row["class_preference_score"]),
                "top5_lemma_accuracy": float(metric_row["top5_lemma_accuracy"]),
                "dominant_predicted_class": dominant_class,
                "dominant_prediction_share": dominant_count / max(total, 1),
            })

    output_dir = PROJECT / "results" / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    gates = {}
    for scale in SCALES:
        scale_rows = [row for row in rows if row["scale"] == scale]
        if not scale_rows:
            continue
        mean_accuracy = sum(float(row["verb_class_accuracy"]) for row in scale_rows) / len(scale_rows)
        positive_preferences = sum(float(row["class_preference_score"]) > 0 for row in scale_rows)
        acceptable_seeds = sum(float(row["verb_class_accuracy"]) >= 0.23 for row in scale_rows)
        max_dominance = max(float(row["dominant_prediction_share"]) for row in scale_rows)
        passed = mean_accuracy >= 0.25 and acceptable_seeds >= 2 and positive_preferences == len(scale_rows) and max_dominance <= 0.60
        gates[scale] = {
            "mean_accuracy": mean_accuracy,
            "seeds_accuracy_at_least_0.23": acceptable_seeds,
            "positive_class_preference_seeds": positive_preferences,
            "max_dominant_prediction_share": max_dominance,
            "evaluation_validity_gate": "pass" if passed else "fail",
        }
    recommendation = "run_aux_followup" if gates.get("full", {}).get("evaluation_validity_gate") == "pass" else "run_babyberta_original_only_before_any_ablation"
    summary = {"chance_accuracy": CHANCE, "missing_scales": missing, "gates": gates, "recommendation": recommendation}
    (output_dir / "learning_curve_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
