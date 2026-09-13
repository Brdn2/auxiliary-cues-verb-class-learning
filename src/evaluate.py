from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import RobertaForMaskedLM

from data_utils import build_verb_maps
from experiment_utils import load_json, project_path
from train import (
    candidate_metrics,
    candidate_surfaces_for_evaluation,
    csv_write,
    load_merged_config,
    load_tokenizer,
    paired_delta,
    score_candidates_for_example,
    select_mvp_examples,
    selected_conditions,
    summarize_metrics,
    tokenization_audit,
    write_decision_notes,
    write_error_analysis,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "Pilot_v4_1_roberta_babylm2026_eval250.yaml"


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_existing_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {(row["condition"], row["seed"], int(row["example_id"])) for row in csv.DictReader(f)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_progress(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_selected_evaluation(
    rows: list[dict[str, str]], conditions: list[str], seeds: list[int], n_examples: int
) -> None:
    selected = [
        row
        for row in rows
        if row.get("condition") in set(conditions) and int(row.get("seed", -1)) in set(seeds)
    ]
    keys = [(row["condition"], int(row["seed"]), int(row["example_id"])) for row in selected]
    expected = {
        (condition, seed, example_id)
        for condition in conditions
        for seed in seeds
        for example_id in range(n_examples)
    }
    observed = set(keys)
    if len(keys) != len(observed):
        raise RuntimeError("Duplicate rows remain in the selected evaluation matrix.")
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        unexpected = sorted(observed - expected)[:10]
        raise RuntimeError(
            "Incomplete selected evaluation matrix: "
            f"rows={len(keys)}, expected={len(expected)}, missing_sample={missing}, "
            f"unexpected_sample={unexpected}"
        )


def max_length_from_training_summaries(config: dict[str, Any], conditions: list[str], seeds: list[int]) -> int:
    lengths: list[int] = []
    for condition in conditions:
        for seed in seeds:
            summary_path = project_path(config["paths"]["model_dir"]) / condition / str(seed) / "training_summary.json"
            if summary_path.exists():
                summary = load_json(summary_path)
                if "max_length" in summary:
                    lengths.append(int(summary["max_length"]))
    if not lengths:
        raise FileNotFoundError("No training summaries with max_length found; train models before evaluation.")
    return max(lengths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual MVP evaluation with progress/resume support.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-dir", help="Override results directory.")
    parser.add_argument("--audits-dir", help="Override audits directory.")
    parser.add_argument("--max-examples", type=int, help="Override evaluation.max_examples.")
    parser.add_argument("--max-candidates-per-class", type=int, help="Override evaluation.max_candidates_per_class.")
    parser.add_argument("--conditions", help="Comma-separated condition list. Default: all selected conditions.")
    parser.add_argument("--seeds", help="Comma-separated seed list. Default: config training.seeds.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in mvp_per_example.csv.")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    config = load_merged_config(Path(args.config))
    if args.results_dir:
        config["paths"]["results_dir"] = args.results_dir
    if args.audits_dir:
        config["paths"]["audits_dir"] = args.audits_dir
    if args.max_examples is not None:
        config["evaluation"]["max_examples"] = int(args.max_examples)
    if args.max_candidates_per_class is not None:
        config["evaluation"]["max_candidates_per_class"] = int(args.max_candidates_per_class)

    results_dir = project_path(config["paths"]["results_dir"])
    audits_dir = project_path(config["paths"]["audits_dir"])
    mvp_path = results_dir / "mvp_per_example.csv"
    progress_path = results_dir / "progress.json"
    training_path = results_dir / "training_condition_summary.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)
    if mvp_path.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to append a fresh evaluation to existing file {mvp_path}. "
            "Use --resume or choose a new --results-dir."
        )

    manifest = load_json(project_path(config["paths"]["conditions_dir"]) / "condition_manifest.json")
    conditions = parse_csv_arg(args.conditions) or selected_conditions(manifest, debug=False)
    seeds = [int(seed) for seed in (parse_csv_arg(args.seeds) or config["training"]["seeds"])]

    tokenizer = load_tokenizer(config)
    tokenization_audit(config, tokenizer)
    surface_map, _ = build_verb_maps(config)
    candidate_surfaces, candidate_audit_rows = candidate_surfaces_for_evaluation(config, tokenizer)
    candidate_token_ids = {surface: tokenizer.encode(surface, add_special_tokens=False) for surface in candidate_surfaces}
    examples, example_audit_rows = select_mvp_examples(config, int(config["evaluation"]["max_examples"]), set(candidate_surfaces))
    csv_write(audits_dir / "candidate_set_audit.csv", candidate_audit_rows)
    csv_write(audits_dir / "evaluation_item_audit.csv", example_audit_rows)

    max_length = max_length_from_training_summaries(config, conditions, seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    total = len(conditions) * len(seeds) * len(examples)
    expected_keys = {
        (condition, str(seed), example_id)
        for condition in conditions
        for seed in seeds
        for example_id in range(len(examples))
    }
    existing = (read_existing_keys(mvp_path) & expected_keys) if args.resume else set()
    completed = len(existing)
    started = time.time()
    train_rows: list[dict[str, Any]] = []

    for condition in conditions:
        for seed in seeds:
            model_dir = project_path(config["paths"]["model_dir"]) / condition / str(seed)
            model = RobertaForMaskedLM.from_pretrained(model_dir).to(device)
            model.eval()
            summary = load_json(model_dir / "training_summary.json")
            final = summary["history"][-1]
            train_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "debug": False,
                    "epochs": len(summary["history"]),
                    "final_train_loss": final["train_loss"],
                    "final_dev_loss": final["dev_loss"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "device": summary["device"],
                }
            )
            for example_id, example in enumerate(examples):
                key = (condition, str(seed), example_id)
                if key in existing:
                    continue
                scores, sum_scores, piece_counts = score_candidates_for_example(model, tokenizer, example, candidate_token_ids, max_length, device)
                pred_surface, pred_class, acc, cps = candidate_metrics(scores, surface_map, example["target_class"])
                ranked_surfaces = [s for s, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
                top_surfaces = ranked_surfaces[:5]
                top_lemmas = []
                for surface in top_surfaces:
                    lemma = surface_map[surface]["lemma"]
                    if lemma not in top_lemmas:
                        top_lemmas.append(lemma)
                row = {
                    "condition": condition,
                    "seed": seed,
                    "example_id": example_id,
                    "line_id": example["line_id"],
                    "target_surface": example["target_surface"],
                    "target_lemma": example["target_lemma"],
                    "target_class": example["target_class"],
                    "target_verb_logprob": scores.get(example["target_surface"], float("nan")),
                    "target_verb_logprob_sum": sum_scores.get(example["target_surface"], float("nan")),
                    "target_verb_piece_count": piece_counts.get(example["target_surface"], ""),
                    "predicted_surface": pred_surface,
                    "predicted_class": pred_class,
                    "verb_class_accuracy": acc,
                    "class_preference_score": cps,
                    "top1_surface_accuracy": 1.0 if example["target_surface"] in ranked_surfaces[:1] else 0.0,
                    "top3_surface_accuracy": 1.0 if example["target_surface"] in ranked_surfaces[:3] else 0.0,
                    "top5_surface_accuracy": 1.0 if example["target_surface"] in ranked_surfaces[:5] else 0.0,
                    "top1_lemma_accuracy": 1.0 if any(surface_map[s]["lemma"] == example["target_lemma"] for s in ranked_surfaces[:1]) else 0.0,
                    "top3_lemma_accuracy": 1.0 if any(surface_map[s]["lemma"] == example["target_lemma"] for s in ranked_surfaces[:3]) else 0.0,
                    "top5_lemma_accuracy": 1.0 if any(surface_map[s]["lemma"] == example["target_lemma"] for s in ranked_surfaces[:5]) else 0.0,
                    "top5_surfaces": " ".join(top_surfaces),
                    "top5_lemmas": " ".join(top_lemmas),
                    "surface_topk_accuracy": 1.0 if example["target_surface"] in ranked_surfaces[:5] else 0.0,
                }
                append_csv(mvp_path, row)
                completed += 1
                if completed % max(int(args.progress_every), 1) == 0 or completed == total:
                    elapsed = time.time() - started
                    rate = completed / max(elapsed, 1e-9)
                    remaining = max(total - completed, 0) / max(rate, 1e-9)
                    previous_progress = load_json(progress_path) if progress_path.exists() else {}
                    progress = {
                        **{key: previous_progress[key] for key in ["total_conditions", "total_models", "completed_models", "pending_models", "failures"] if key in previous_progress},
                        "status": "running",
                        "config": str(args.config),
                        "results_dir": str(results_dir),
                        "device": str(device),
                        "conditions": conditions,
                        "seeds": seeds,
                        "n_examples": len(examples),
                        "n_candidates": len(candidate_surfaces),
                        "completed": completed,
                        "total": total,
                        "percent": round(completed / max(total, 1) * 100, 3),
                        "current_condition": condition,
                        "current_seed": seed,
                        "current_example_id": example_id,
                        "elapsed_seconds": round(elapsed, 1),
                        "estimated_remaining_seconds": round(remaining, 1),
                        "updated_at": time.time(),
                    }
                    write_progress(progress_path, progress)
                    print(f"[{progress['percent']:6.2f}%] {condition} seed={seed} example={example_id} remaining~{progress['estimated_remaining_seconds']}s", flush=True)
            del model

    rows = read_csv_rows(mvp_path)
    validate_selected_evaluation(rows, conditions, seeds, len(examples))
    csv_write(training_path, train_rows)
    summarize_metrics(rows, results_dir / "metrics_summary.csv")
    paired_delta(rows, config, results_dir / "paired_delta_summary.csv")
    write_error_analysis(rows, config, results_dir / "error_analysis.md")
    write_decision_notes(config, rows, train_rows, debug=False)
    progress = load_json(progress_path) if progress_path.exists() else {}
    progress.update({"status": "complete", "completed": total, "total": total, "percent": 100.0, "finished_at": time.time()})
    write_progress(progress_path, progress)
    print(f"Manual MVP evaluation complete. Results: {results_dir}")


if __name__ == "__main__":
    main()
