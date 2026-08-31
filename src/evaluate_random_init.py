from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from pilot_utils import masked_mean, set_seed
from run_pilot_v4_roberta import (
    build_verb_maps,
    candidate_metrics,
    candidate_surfaces_for_evaluation,
    load_merged_config,
    load_tokenizer,
    make_model,
    score_candidates_for_example,
    select_mvp_examples,
)


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config/ACL_confirmatory_aux.yaml"))
    parser.add_argument("--seeds", default="61001,61002,61003,61004,61005")
    parser.add_argument("--output-dir", default=str(ROOT / "results/ACL_random_init"))
    args = parser.parse_args()
    config = load_merged_config(Path(args.config))
    tokenizer = load_tokenizer(config)
    surface_map, _ = build_verb_maps(config)
    candidates, _ = candidate_surfaces_for_evaluation(config, tokenizer)
    candidate_ids = {surface: tokenizer.encode(surface, add_special_tokens=False) for surface in candidates}
    examples, _ = select_mvp_examples(config, int(config["evaluation"]["max_examples"]), set(candidates))
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    rows: list[dict] = []
    for seed in seeds:
        set_seed(seed)
        model = make_model(config, tokenizer, 128).to(device)
        model.eval()
        for example_id, example in enumerate(examples):
            scores, _, _ = score_candidates_for_example(model, tokenizer, example, candidate_ids, 128, device)
            pred_surface, pred_class, accuracy, cps = candidate_metrics(scores, surface_map, example["target_class"])
            rows.append({
                "condition": "random_init", "seed": seed, "example_id": example_id, "line_id": example["line_id"],
                "target_class": example["target_class"], "target_surface": example["target_surface"],
                "predicted_surface": pred_surface, "predicted_class": pred_class,
                "verb_class_accuracy": accuracy, "class_preference_score": cps,
            })
        print(f"random-init seed={seed} complete", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "mvp_per_example.csv", rows)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    summary = [
        {
            "condition": "random_init", "seed": seed, "n_examples": len(items),
            "verb_class_accuracy": masked_mean([float(row["verb_class_accuracy"]) for row in items]),
            "class_preference_score": masked_mean([float(row["class_preference_score"]) for row in items]),
        }
        for seed, items in sorted(grouped.items())
    ]
    write_csv(output_dir / "metrics_summary.csv", summary)
    (output_dir / "run_summary.json").write_text(json.dumps({"device": str(device), "seeds": seeds, "examples": len(examples)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
