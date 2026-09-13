from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForMaskedLM

from experiment_utils import masked_mean, set_seed
from train import (
    candidate_metrics,
    load_merged_config,
    load_tokenizer,
    score_candidates_for_example,
)


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_surface_map(config: dict[str, Any]) -> dict[str, str]:
    return {
        surface: cls
        for cls, lemmas in config["verb_lexicon"].items()
        for forms in lemmas.values()
        for surface in forms
    }


def frame_signature(record: dict[str, Any], index: int, radius: int = 2) -> str:
    left = record["pos"][max(0, index - radius):index]
    right = record["pos"][index + 1:index + radius + 1]
    return " ".join([*(['<BOS>'] * (radius - len(left))), *left, "<V>", *right, *(['<EOS>'] * (radius - len(right)))])


def collect_occurrences(
    records: list[dict[str, Any]], indices: list[int], surface_map: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record_idx in indices:
        record = records[record_idx]
        eligible = [
            index for index, (token, tag) in enumerate(zip(record["tokens"], record["pos"]))
            if tag == "VERB" and token in surface_map
        ]
        if len(eligible) != 1:
            continue
        index = eligible[0]
        cls = surface_map[record["tokens"][index]]
        output[cls].append({
            "record_idx": record_idx, "line_id": record["line_id"], "tokens": list(record["tokens"]),
            "pos": list(record["pos"]), "target_idx": index, "source_surface": record["tokens"][index],
            "target_class": cls, "frame": frame_signature(record, index),
        })
    return output


def sample_protocol(
    train: dict[str, list[dict[str, Any]]], test: dict[str, list[dict[str, Any]]], cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(int(cfg["seed"]))
    exposures: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    n_exposure = int(cfg["sampling"]["exposure_items_per_class"])
    n_eval = int(cfg["sampling"]["evaluation_items_per_class"])
    n_frames = int(cfg["sampling"]["exposure_frame_types_per_class"])
    classes = sorted(cfg["nonce_by_class"])
    aux_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    aux_totals: Counter[str] = Counter()
    for cls in classes:
        for item in train[cls]:
            for token, tag in zip(item["tokens"], item["pos"]):
                if tag == "AUX":
                    aux_class_counts[token][cls] += 1
                    aux_totals[token] += 1

    def diagnosticity(item: dict[str, Any]) -> float:
        cues = [token for token, tag in zip(item["tokens"], item["pos"]) if tag == "AUX"]
        if not cues:
            return float("-inf")
        cls = item["target_class"]
        values = []
        for cue in cues:
            probs = {label: (aux_class_counts[cue][label] + 0.5) / (aux_totals[cue] + 0.5 * len(classes)) for label in classes}
            values.append(math.log(probs[cls]) - sum(math.log(probs[label]) for label in classes if label != cls) / (len(classes) - 1))
        return sum(values) / len(values)

    for cls, nonce in cfg["nonce_by_class"].items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in train[cls]:
            if cfg["sampling"].get("require_aux_in_every_exposure") and "AUX" not in item["pos"]:
                continue
            item = {**item, "aux_diagnosticity": diagnosticity(item)}
            grouped[item["frame"]].append(item)
        frames = [frame for frame, items in grouped.items() if len(items) >= 3]
        frames.sort(key=lambda frame: sum(item["aux_diagnosticity"] for item in grouped[frame]) / len(grouped[frame]), reverse=True)
        if not cfg["sampling"].get("select_high_diagnosticity_aux_contexts"):
            rng.shuffle(frames)
        exposure_frames = set(frames[:n_frames])
        pool = [item for frame in exposure_frames for item in grouped[frame]]
        if cfg["sampling"].get("select_high_diagnosticity_aux_contexts"):
            pool.sort(key=lambda item: item["aux_diagnosticity"], reverse=True)
        else:
            rng.shuffle(pool)
        selected_exposure = pool[:n_exposure]
        eval_pool = [item for item in test[cls] if item["frame"] not in exposure_frames]
        rng.shuffle(eval_pool)
        selected_eval = eval_pool[:n_eval]
        if len(selected_exposure) < n_exposure or len(selected_eval) < n_eval:
            raise RuntimeError(f"Insufficient cross-template items for {cls}: exposure={len(selected_exposure)}, eval={len(selected_eval)}")
        for item in selected_exposure:
            copied = dict(item)
            copied["nonce"] = nonce
            copied["tokens"] = list(item["tokens"])
            copied["tokens"][item["target_idx"]] = nonce
            exposures.append(copied)
        for item in selected_eval:
            copied = dict(item)
            copied["nonce"] = nonce
            copied["target_surface"] = nonce
            copied["target_lemma"] = nonce
            copied["tokens"] = list(item["tokens"])
            copied["tokens"][item["target_idx"]] = nonce
            evaluations.append(copied)
        audit[cls] = {
            "nonce": nonce, "exposure_items": len(selected_exposure), "evaluation_items": len(selected_eval),
            "exposure_frame_types": sorted(exposure_frames),
            "evaluation_frame_types": len({item["frame"] for item in selected_eval}),
            "exact_frame_overlap": len(exposure_frames & {item["frame"] for item in selected_eval}),
            "mean_exposure_aux_diagnosticity": sum(item["aux_diagnosticity"] for item in selected_exposure) / len(selected_exposure),
            "all_exposures_contain_aux": all("AUX" in item["pos"] for item in selected_exposure),
        }
    rng.shuffle(exposures)
    rng.shuffle(evaluations)
    return exposures, evaluations, audit


def apply_exposure_condition(item: dict[str, Any], condition: str, ablation_token: str) -> dict[str, Any]:
    output = dict(item)
    output["tokens"] = list(item["tokens"])
    aux_positions = [index for index, tag in enumerate(item["pos"]) if tag == "AUX"]
    if condition == "aux_ablated_exposure":
        for index in aux_positions:
            output["tokens"][index] = ablation_token
    elif condition == "matched_function_ablated_exposure":
        allowed = {"PRON", "DET", "ADP", "PART", "CCONJ", "SCONJ"}
        candidates = [index for index, tag in enumerate(item["pos"]) if tag in allowed and index != item["target_idx"]]
        used: set[int] = set()
        for aux_index in aux_positions:
            available = [index for index in candidates if index not in used]
            if not available:
                break
            chosen = min(available, key=lambda index: abs(index - aux_index))
            used.add(chosen)
            output["tokens"][chosen] = ablation_token
    return output


def condition_exposures(items: list[dict[str, Any]], condition: str, ablation_token: str, seed: int) -> list[dict[str, Any]]:
    output = [apply_exposure_condition(item, condition, ablation_token) for item in items]
    if condition != "aux_shuffled_exposure":
        return output
    positions: list[tuple[int, int]] = []
    tokens: list[str] = []
    for item_index, item in enumerate(items):
        for token_index, tag in enumerate(item["pos"]):
            if tag == "AUX":
                positions.append((item_index, token_index))
                tokens.append(item["tokens"][token_index])
    rng = random.Random(seed)
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    for (item_index, token_index), token in zip(positions, shuffled):
        output[item_index]["tokens"][token_index] = token
    return output


class TargetMaskDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for item in items:
            target_ids = tokenizer.encode(item["nonce"], add_special_tokens=False)
            tokens = list(item["tokens"])
            tokens[int(item["target_idx"])] = " ".join([tokenizer.mask_token] * len(target_ids))
            encoded = tokenizer(" ".join(tokens), max_length=max_length, truncation=True)
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention = torch.tensor(encoded["attention_mask"], dtype=torch.long)
            labels = torch.full_like(input_ids, -100)
            positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(positions) < len(target_ids):
                continue
            for position, token_id in zip(positions[:len(target_ids)], target_ids):
                labels[position] = token_id
            self.rows.append((input_ids, attention, labels))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def collate(batch, pad_id: int):
    ids, attention, labels = zip(*batch)
    return {
        "input_ids": pad_sequence(ids, batch_first=True, padding_value=pad_id),
        "attention_mask": pad_sequence(attention, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
    }


def adapt_model(model, tokenizer, items: list[dict[str, Any]], cfg: dict[str, Any], device: torch.device) -> list[float]:
    dataset = TargetMaskDataset(items, tokenizer, int(cfg["adaptation"]["max_length"]))
    loader = DataLoader(
        dataset, batch_size=int(cfg["adaptation"]["batch_size"]), shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["adaptation"]["learning_rate"]),
        weight_decay=float(cfg["adaptation"]["weight_decay"]),
    )
    history: list[float] = []
    model.train()
    for _ in range(int(cfg["adaptation"]["epochs"])):
        losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(masked_mean(losses))
    return history


def evaluate_model(model, tokenizer, evaluations: list[dict[str, Any]], nonce_by_class: dict[str, str], device, seed: int, condition: str, max_length: int) -> list[dict[str, Any]]:
    custom_map = {nonce: {"class": cls, "lemma": nonce} for cls, nonce in nonce_by_class.items()}
    candidate_ids = {nonce: tokenizer.encode(nonce, add_special_tokens=False) for nonce in nonce_by_class.values()}
    rows: list[dict[str, Any]] = []
    model.eval()
    for example_id, example in enumerate(evaluations):
        scores, _, pieces = score_candidates_for_example(model, tokenizer, example, candidate_ids, max_length, device)
        pred_surface, pred_class, accuracy, cps = candidate_metrics(scores, custom_map, example["target_class"])
        rows.append({
            "condition": condition, "seed": seed, "example_id": example_id, "line_id": example["line_id"],
            "source_surface": example["source_surface"], "target_nonce": example["nonce"], "target_class": example["target_class"],
            "frame": example["frame"], "predicted_nonce": pred_surface, "predicted_class": pred_class,
            "verb_class_accuracy": accuracy, "class_preference_score": cps,
            "target_piece_count": pieces[example["nonce"]],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/nonce.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = load_merged_config(ROOT / cfg["model_config"])
    source = yaml.safe_load((ROOT / model_cfg["source_config"]).read_text(encoding="utf-8"))
    base_source = yaml.safe_load((ROOT / source["base_source_config"]).read_text(encoding="utf-8"))
    source = {**base_source, **source, "paths": {**base_source["paths"], **source["paths"]}}
    records = read_jsonl(ROOT / source["paths"]["processed_dir"] / "utterances.jsonl")
    splits = json.loads((ROOT / source["paths"]["splits_dir"] / "splits.json").read_text(encoding="utf-8"))
    surface_map = build_surface_map(source)
    train = collect_occurrences(records, list(map(int, splits["train"])), surface_map)
    test = collect_occurrences(records, list(map(int, splits["test"])), surface_map)
    exposures, evaluations, protocol = sample_protocol(train, test, cfg)
    tokenizer = load_tokenizer(model_cfg)
    protocol["nonce_piece_counts"] = {nonce: len(tokenizer.encode(nonce, add_special_tokens=False)) for nonce in cfg["nonce_by_class"].values()}
    audit_dir = ROOT / cfg["audit_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(audit_dir / "exposure_items.csv", exposures)
    write_csv(audit_dir / "evaluation_items.csv", evaluations)

    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for seed in map(int, cfg["model_seeds"]):
        base_path = ROOT / cfg["base_model_dir"] / str(seed)
        base = RobertaForMaskedLM.from_pretrained(base_path).to(device)
        rows.extend(evaluate_model(base, tokenizer, evaluations, cfg["nonce_by_class"], device, seed, "zero_shot", int(cfg["adaptation"]["max_length"])))
        del base
        if device.type == "mps":
            torch.mps.empty_cache()
        for condition in cfg["adaptation"]["conditions"]:
            set_seed(seed)
            model = RobertaForMaskedLM.from_pretrained(base_path).to(device)
            conditioned = condition_exposures(exposures, condition, model_cfg["special_tokens"]["ablation"], seed + 17001)
            history = adapt_model(model, tokenizer, conditioned, cfg, device)
            rows.extend(evaluate_model(model, tokenizer, evaluations, cfg["nonce_by_class"], device, seed, condition, int(cfg["adaptation"]["max_length"])))
            model_dir = ROOT / cfg["adapted_model_dir"] / condition / str(seed)
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            training_rows.append({"condition": condition, "seed": seed, "epochs": len(history), "initial_loss": history[0], "final_loss": history[-1], "device": str(device)})
            print(f"nonce condition={condition} seed={seed} final_loss={history[-1]:.4f}", flush=True)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
    output_dir = ROOT / cfg["output_dir"]
    write_csv(output_dir / "nonce_per_example.csv", rows)
    write_csv(output_dir / "adaptation_summary.csv", training_rows)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], int(row["seed"]))].append(row)
    metrics = [
        {"condition": condition, "seed": seed, "n_examples": len(items),
         "verb_class_accuracy": masked_mean([float(item["verb_class_accuracy"]) for item in items]),
         "class_preference_score": masked_mean([float(item["class_preference_score"]) for item in items])}
        for (condition, seed), items in sorted(grouped.items())
    ]
    write_csv(output_dir / "metrics_summary.csv", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
