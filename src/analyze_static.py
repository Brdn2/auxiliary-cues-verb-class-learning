from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import pointbiserialr, spearmanr

from train import load_merged_config, select_mvp_examples


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_AUX = {
    "am", "are", "be", "been", "being", "can", "ca", "could", "did", "do", "does",
    "had", "has", "have", "is", "may", "might", "must", "need", "ought", "oughta",
    "shall", "sha", "should", "was", "were", "will", "wo", "would",
    "'d", "'ll", "'m", "'re", "'s", "'ve", "mighta",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_aux(token: str) -> str:
    if token in CANONICAL_AUX:
        return "canonical"
    cleaned = token.strip("⌈⌉⌊⌋∆.,!?;:'\"-_")
    if cleaned in CANONICAL_AUX or any(cleaned.endswith(aux) for aux in CANONICAL_AUX if len(aux) >= 3):
        return "transcription_or_tokenization_artifact"
    return "likely_mistag"


def aux_audit(records: list[dict[str, Any]], sample_size: int, seed: int, output_dir: Path) -> dict[str, Any]:
    inventory: Counter[str] = Counter()
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record_idx, record in enumerate(records):
        for token_idx, (token, tag) in enumerate(zip(record["tokens"], record["pos"])):
            if tag != "AUX":
                continue
            inventory[token] += 1
            if len(occurrences[token]) < 30:
                occurrences[token].append({
                    "record_idx": record_idx, "line_id": record["line_id"], "token_idx": token_idx,
                    "token": token, "left_context": " ".join(record["tokens"][max(0, token_idx - 5):token_idx]),
                    "right_context": " ".join(record["tokens"][token_idx + 1:token_idx + 6]), "raw": record["raw"],
                })
    inventory_rows = [
        {"token": token, "count": count, "proportion": count / sum(inventory.values()), "audit_class": classify_aux(token)}
        for token, count in inventory.most_common()
    ]
    write_csv(output_dir / "aux_inventory.csv", inventory_rows)

    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    suspicious = [row for row in inventory_rows if row["audit_class"] != "canonical"]
    for row in suspicious:
        sample.extend({**item, "automatic_class": row["audit_class"], "review_label": "", "review_notes": ""} for item in occurrences[row["token"]])
    canonical_pool = [item for token in inventory if classify_aux(token) == "canonical" for item in occurrences[token]]
    rng.shuffle(canonical_pool)
    remaining = max(0, sample_size - len(sample))
    sample.extend({**item, "automatic_class": "canonical", "review_label": "", "review_notes": ""} for item in canonical_pool[:remaining])
    sample = sample[:sample_size]
    for sample_id, item in enumerate(sample, 1):
        item["sample_id"] = sample_id
    blinded = [{key: value for key, value in item.items() if key != "automatic_class"} for item in sample]
    key_rows = [{"sample_id": item["sample_id"], "automatic_class": item["automatic_class"]} for item in sample]
    write_csv(output_dir / "aux_context_sample_for_blind_review.csv", blinded)
    write_csv(output_dir / "aux_context_sample_automatic_key.csv", key_rows)

    class_counts = Counter()
    for token, count in inventory.items():
        class_counts[classify_aux(token)] += count
    report = {
        "aux_tokens": sum(inventory.values()), "aux_types": len(inventory), "class_event_counts": dict(class_counts),
        "canonical_event_rate": class_counts["canonical"] / sum(inventory.values()),
        "suspicious_event_rate": (sum(inventory.values()) - class_counts["canonical"]) / sum(inventory.values()),
        "sample_size": len(sample),
        "scope": "Lexicon-based corpus audit; the exported blind-review sheet still requires independent human annotation.",
    }
    (output_dir / "aux_audit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# AUX annotation audit\n\n"
        "This audit separates exact lexical plausibility from human contextual validity. "
        "`aux_inventory.csv` is exhaustive. `aux_context_sample_for_blind_review.csv` is a deterministic, "
        "context-rich sample whose `review_label` and `review_notes` columns must be completed independently. "
        "The automatic class must be hidden from annotators before review.\n",
        encoding="utf-8",
    )
    return report


def verb_maps(config: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    for cls, lemmas in config["verb_lexicon"].items():
        for forms in lemmas.values():
            for form in forms:
                mapping[form] = cls
    return mapping, sorted(config["verb_lexicon"])


def frame(record: dict[str, Any], index: int, radius: int = 2) -> str:
    left = record["pos"][max(0, index - radius):index]
    right = record["pos"][index + 1:index + radius + 1]
    return " ".join([*(['<BOS>'] * (radius - len(left))), *left, "<V>", *right, *(['<EOS>'] * (radius - len(right)))])


def log_nb_score(features: list[str], cls: str, feature_counts: dict[str, Counter[str]], totals: Counter[str], vocab_size: int, alpha: float) -> float:
    denominator = totals[cls] + alpha * max(vocab_size, 1)
    return sum(math.log((feature_counts[cls][feature] + alpha) / denominator) for feature in features)


def argmax(scores: dict[str, float]) -> str:
    return max(sorted(scores), key=lambda key: scores[key])


def build_training_statistics(records: list[dict[str, Any]], train_indices: list[int], surface_map: dict[str, str], window: int) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    lexical_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lexical_totals: Counter[str] = Counter()
    aux_counts: dict[str, Counter[str]] = defaultdict(Counter)
    aux_totals: Counter[str] = Counter()
    frame_counts: dict[str, Counter[str]] = defaultdict(Counter)
    surface_class: dict[str, Counter[str]] = defaultdict(Counter)
    for record_idx in train_indices:
        record = records[record_idx]
        for index, (token, tag) in enumerate(zip(record["tokens"], record["pos"])):
            if tag != "VERB" or token not in surface_map:
                continue
            cls = surface_map[token]
            class_counts[cls] += 1
            surface_class[token][cls] += 1
            context = record["tokens"][max(0, index - window):index] + record["tokens"][index + 1:index + window + 1]
            lexical_counts[cls].update(context)
            lexical_totals[cls] += len(context)
            aux = [t for t, p in zip(record["tokens"], record["pos"]) if p == "AUX"]
            aux_counts[cls].update(aux)
            aux_totals[cls] += len(aux)
            frame_counts[frame(record, index)][cls] += 1
    return {
        "class_counts": class_counts, "lexical_counts": lexical_counts, "lexical_totals": lexical_totals,
        "aux_counts": aux_counts, "aux_totals": aux_totals, "frame_counts": frame_counts, "surface_class": surface_class,
    }


def evaluate_baselines(
    records: list[dict[str, Any]], examples: list[dict[str, Any]], stats: dict[str, Any], classes: list[str],
    window: int, alpha: float, random_seeds: list[int], output_dir: Path,
) -> list[dict[str, Any]]:
    lexical_vocab = {token for counts in stats["lexical_counts"].values() for token in counts}
    aux_vocab = {token for counts in stats["aux_counts"].values() for token in counts}
    total_classes = sum(stats["class_counts"].values())
    rows: list[dict[str, Any]] = []
    baseline_names = ["majority_frequency", "pos_frame_count", "lexical_cooccurrence_nb", "aux_only_nb", "lexical_identity_ceiling"]
    for example_id, example in enumerate(examples):
        target = example["target_class"]
        record = {"tokens": example["tokens"], "pos": example["pos"]}
        index = int(example["target_idx"])
        context = record["tokens"][max(0, index - window):index] + record["tokens"][index + 1:index + window + 1]
        aux_context = [t for t, p in zip(record["tokens"], record["pos"]) if p == "AUX"]
        prior = {cls: math.log((stats["class_counts"][cls] + alpha) / (total_classes + alpha * len(classes))) for cls in classes}
        predictions = {
            "majority_frequency": argmax(prior),
            "pos_frame_count": argmax({cls: prior[cls] + math.log(stats["frame_counts"][frame(record, index)][cls] + alpha) for cls in classes}),
            "lexical_cooccurrence_nb": argmax({cls: prior[cls] + log_nb_score(context, cls, stats["lexical_counts"], stats["lexical_totals"], len(lexical_vocab), alpha) for cls in classes}),
            "aux_only_nb": argmax({cls: prior[cls] + log_nb_score(aux_context, cls, stats["aux_counts"], stats["aux_totals"], len(aux_vocab), alpha) for cls in classes}),
            "lexical_identity_ceiling": argmax(dict(stats["surface_class"].get(example["target_surface"], stats["class_counts"]))),
        }
        for name in baseline_names:
            rows.append({
                "baseline": name, "seed": "deterministic", "example_id": example_id, "line_id": example["line_id"],
                "target_surface": example["target_surface"], "target_class": target,
                "predicted_class": predictions[name], "accuracy": float(predictions[name] == target),
                "aux_cue_count": len(aux_context), "frame": frame(record, index),
            })
        for seed in random_seeds:
            prediction = random.Random(seed + example_id).choice(classes)
            rows.append({
                "baseline": "uniform_random", "seed": seed, "example_id": example_id, "line_id": example["line_id"],
                "target_surface": example["target_surface"], "target_class": target,
                "predicted_class": prediction, "accuracy": float(prediction == target),
                "aux_cue_count": len(aux_context), "frame": frame(record, index),
            })
    write_csv(output_dir / "baseline_per_example.csv", rows)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["baseline"]), str(row["seed"]))].append(float(row["accuracy"]))
    summary = [
        {"baseline": name, "seed": seed, "n_examples": len(values), "accuracy": sum(values) / len(values)}
        for (name, seed), values in sorted(grouped.items())
    ]
    write_csv(output_dir / "baselines.csv", summary)
    return rows


def cue_analysis(
    examples: list[dict[str, Any]], stats: dict[str, Any], classes: list[str], legacy_predictions: Path, output_dir: Path,
) -> None:
    total_joint = sum(stats["aux_totals"].values())
    aux_marginal = Counter()
    for cls in classes:
        aux_marginal.update(stats["aux_counts"][cls])
    cue_rows: list[dict[str, Any]] = []
    for cue, cue_total in aux_marginal.most_common():
        probs = [(stats["aux_counts"][cls][cue] + 0.5) / (cue_total + 0.5 * len(classes)) for cls in classes]
        entropy = -sum(p * math.log2(p) for p in probs)
        best_index = int(np.argmax(probs))
        mi_contribution = 0.0
        for cls in classes:
            joint = stats["aux_counts"][cls][cue]
            if joint:
                p_joint = joint / total_joint
                p_cue = cue_total / total_joint
                p_cls = stats["aux_totals"][cls] / total_joint
                mi_contribution += p_joint * math.log2(p_joint / (p_cue * p_cls))
        cue_rows.append({
            "aux_cue": cue, "count": cue_total, "reliability": max(probs), "entropy_bits": entropy,
            "most_associated_class": classes[best_index], "mi_contribution_bits": mi_contribution,
            **{f"p_{cls}_given_cue": probs[i] for i, cls in enumerate(classes)},
        })
    write_csv(output_dir / "aux_cue_statistics.csv", cue_rows)

    prediction_rows: list[dict[str, str]] = []
    with legacy_predictions.open(encoding="utf-8", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    predictions = {(int(row["example_id"]), int(row["seed"])): row for row in prediction_rows if row["condition"] == "original"}
    item_rows: list[dict[str, Any]] = []
    for example_id, example in enumerate(examples):
        aux = [token for token, tag in zip(example["tokens"], example["pos"]) if tag == "AUX"]
        target = example["target_class"]
        class_scores = {
            cls: sum(math.log((stats["aux_counts"][cls][cue] + 0.5) / (stats["aux_totals"][cls] + 0.5 * max(len(aux_marginal), 1))) for cue in aux)
            for cls in classes
        }
        target_score = class_scores[target]
        foil_score = sum(score for cls, score in class_scores.items() if cls != target) / (len(classes) - 1)
        for seed in sorted({key[1] for key in predictions}):
            model_row = predictions.get((example_id, seed))
            if model_row is None:
                continue
            item_rows.append({
                "example_id": example_id, "seed": seed, "line_id": example["line_id"], "target_class": target,
                "aux_cue_count": len(aux), "aux_cues": " ".join(aux), "aux_log_odds": target_score - foil_score,
                "model_accuracy": float(model_row["verb_class_accuracy"]),
                "model_class_preference_score": float(model_row["class_preference_score"]),
            })
    write_csv(output_dir / "item_cue_features.csv", item_rows)
    correlation_rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in item_rows}):
        subset = [row for row in item_rows if int(row["seed"]) == seed]
        x = [float(row["aux_log_odds"]) for row in subset]
        y_cps = [float(row["model_class_preference_score"]) for row in subset]
        y_acc = [float(row["model_accuracy"]) for row in subset]
        rho, rho_p = spearmanr(x, y_cps)
        point, point_p = pointbiserialr(y_acc, x)
        correlation_rows.append({
            "seed": seed, "n": len(subset), "spearman_aux_logodds_vs_cps": rho, "spearman_p": rho_p,
            "pointbiserial_aux_logodds_vs_accuracy": point, "pointbiserial_p": point_p,
        })
    write_csv(output_dir / "cue_correlations.csv", correlation_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/static_analysis.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = load_merged_config(ROOT / "configs/learning_curve.yaml")
    records = read_jsonl(ROOT / cfg["paths"]["records"])
    splits = json.loads((ROOT / cfg["paths"]["splits"]).read_text(encoding="utf-8"))
    audit_dir = ROOT / cfg["paths"]["audit_dir"]
    output_dir = ROOT / cfg["paths"]["output_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = aux_audit(records, int(cfg["audit"]["sample_size"]), int(cfg["seed"]), audit_dir)

    selected_surfaces: set[str] = set()
    with (ROOT / "audits/legacy_expanded_full/candidate_set_audit.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            selected_surfaces.update(row["selected_surface_list"].split())
    examples, _ = select_mvp_examples(model_cfg, int(model_cfg["evaluation"]["max_examples"]), selected_surfaces)
    surface_map, classes = verb_maps(model_cfg)
    stats = build_training_statistics(records, list(map(int, splits["train"])), surface_map, int(cfg["baselines"]["context_window"]))
    rows = evaluate_baselines(
        records, examples, stats, classes, int(cfg["baselines"]["context_window"]), float(cfg["baselines"]["alpha"]),
        list(map(int, cfg["baselines"]["random_seeds"])), output_dir,
    )
    cue_analysis(examples, stats, classes, ROOT / cfg["paths"]["legacy_predictions"], output_dir)
    summary = {
        "aux_audit": audit, "evaluation_examples": len(examples), "baseline_rows": len(rows),
        "classes": classes, "no_target_token_in_context_baselines": True,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
