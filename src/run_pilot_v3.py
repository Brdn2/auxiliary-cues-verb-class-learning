from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_utils import (
    build_verb_maps,
    build_vocab,
    load_prepared,
    load_split_records,
    parse_corpus,
    save_prepared,
    split_records,
    vocab_from_data,
)
from models import TransformerMLM
from pilot_utils import (
    ensure_dirs,
    get_device,
    load_config,
    load_json,
    masked_mean,
    project_path,
    save_json,
    set_seed,
    write_jsonl,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "Pilot_v3.yaml"
METRICS = [
    "target_verb_logprob",
    "verb_class_accuracy",
    "class_preference_score",
    "surface_topk_accuracy",
    "top5_lemma_accuracy",
]


def stable_int(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def prepare_base_data(config: dict[str, Any], force: bool = False) -> None:
    processed_dir = project_path(config["paths"]["processed_dir"])
    splits_dir = project_path(config["paths"]["splits_dir"])
    if not force and (processed_dir / "utterances.jsonl").exists() and (splits_dir / "splits.json").exists():
        return
    records = parse_corpus(config)
    splits = split_records(records, config)
    train_records = [records[i] for i in splits["train"]]
    vocab_data = build_vocab(train_records, config)
    save_prepared(records, splits, vocab_data, config)
    save_json(
        processed_dir / "corpus_summary.json",
        {
            "version": config["version"],
            "utterances": len(records),
            "split_sizes": {split: len(indices) for split, indices in splits.items()},
            "vocab_size": len(vocab_data["id_to_token"]),
            "pos_counts": dict(Counter(pos for record in records for pos in record["pos"])),
        },
    )


def record_split_lookup(splits: dict[str, list[int]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for split, indices in splits.items():
        for idx in indices:
            lookup[int(idx)] = split
    return lookup


def nearest_verb_distance(pos: list[str], token_index: int) -> int | str:
    distances = [abs(i - token_index) for i, upos in enumerate(pos) if upos == "VERB"]
    return min(distances) if distances else ""


def rel_position_bin(token_index: int, length: int) -> str:
    rel = token_index / max(length - 1, 1)
    if rel < 0.25:
        return "q1"
    if rel < 0.50:
        return "q2"
    if rel < 0.75:
        return "q3"
    return "q4"


def length_bin(length: int) -> str:
    if length <= 5:
        return "<=5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    return ">20"


def frequency_bin(token: str, counts: dict[str, int]) -> str:
    count = int(counts.get(token, 0))
    if count <= 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    if count <= 100:
        return "21-100"
    return ">100"


def event_row(
    records: list[dict[str, Any]],
    record_idx: int,
    token_index: int,
    split: str,
    counts: dict[str, int],
    category: str,
    event_type: str,
) -> dict[str, Any]:
    record = records[record_idx]
    token = record["tokens"][token_index]
    pos = record["pos"][token_index]
    length = len(record["tokens"])
    return {
        "event_type": event_type,
        "category": category,
        "split": split,
        "record_idx": record_idx,
        "line_id": record["line_id"],
        "token_index": token_index,
        "original_token": token,
        "original_upos": pos,
        "token_frequency": counts.get(token, 0),
        "token_frequency_bin": frequency_bin(token, counts),
        "utterance_length": length,
        "utterance_length_bin": length_bin(length),
        "relative_position": round(token_index / max(length - 1, 1), 4),
        "relative_position_bin": rel_position_bin(token_index, length),
        "nearest_verb_distance": nearest_verb_distance(record["pos"], token_index),
        "source_distribution": "UD-English-CHILDES-parental",
    }


def function_indices(record: dict[str, Any], category: str) -> list[int]:
    return [i for i, upos in enumerate(record["pos"]) if upos == category]


def content_indices(record: dict[str, Any], config: dict[str, Any]) -> list[int]:
    allowed = set(config["pilot_v3"]["content_control_pos"])
    return [i for i, upos in enumerate(record["pos"]) if upos in allowed]


def choose_content_match(
    function_event: dict[str, Any],
    pools: dict[tuple[str, str, str], list[dict[str, Any]]],
    split_pools: dict[str, list[dict[str, Any]]],
    used: set[tuple[int, int]],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    key = (
        function_event["split"],
        function_event["utterance_length_bin"],
        function_event["relative_position_bin"],
    )
    candidate_lists = [
        pools.get(key, []),
        split_pools.get(function_event["split"], []),
    ]
    for status, candidates in [("matched", candidate_lists[0]), ("fallback_matched", candidate_lists[1])]:
        available = [
            event for event in candidates if (int(event["record_idx"]), int(event["token_index"])) not in used
        ]
        if available:
            idx = stable_int(config["seed"], function_event["category"], function_event["line_id"], function_event["token_index"], status) % len(available)
            return available[idx], status
    return None, "unmatched"


def make_condition_records(
    records: list[dict[str, Any]],
    replacements: dict[int, dict[int, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        new_record = dict(record)
        tokens = list(record["tokens"])
        for token_index, replacement in replacements.get(idx, {}).items():
            tokens[int(token_index)] = replacement
        new_record["tokens"] = tokens
        output.append(new_record)
    return output


def generate_conditions(config: dict[str, Any], force: bool = False) -> None:
    conditions_dir = project_path(config["paths"]["conditions_dir"])
    manifest_path = conditions_dir / "condition_manifest.json"
    if not force and manifest_path.exists():
        return
    prepare_base_data(config, force=force)
    records, splits, vocab = load_prepared(config)
    split_lookup = record_split_lookup(splits)
    train_counts = load_json(project_path(config["paths"]["processed_dir"]) / "vocab.json")["counts"]
    fw_mask = config["special_tokens"]["fw_mask"]
    content_mask = config["special_tokens"]["content_mask"]

    content_events: list[dict[str, Any]] = []
    function_events_by_category: dict[str, list[dict[str, Any]]] = {}
    for record_idx, record in enumerate(records):
        split = split_lookup[record_idx]
        for token_index in content_indices(record, config):
            content_events.append(event_row(records, record_idx, token_index, split, train_counts, "CONTENT", "content_candidate"))
        for category in config["pilot_v3"]["categories"]:
            for token_index in function_indices(record, category):
                function_events_by_category.setdefault(category, []).append(
                    event_row(records, record_idx, token_index, split, train_counts, category, "function")
                )

    pools: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    split_pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in content_events:
        key = (event["split"], event["utterance_length_bin"], event["relative_position_bin"])
        pools[key].append(event)
        split_pools[event["split"]].append(event)

    audit_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"version": config["version"], "conditions": {}}
    full_records = records
    condition_files: dict[str, Path] = {}
    full_dir = conditions_dir / "full_input"
    full_dir.mkdir(parents=True, exist_ok=True)
    condition_files["full_input"] = full_dir / "utterances.jsonl"
    write_jsonl(condition_files["full_input"], full_records)
    manifest["conditions"]["full_input"] = {
        "type": "baseline",
        "file": str(condition_files["full_input"].relative_to(project_path("."))),
        "utterances": len(full_records),
        "mask_events": 0,
    }

    for category, events in function_events_by_category.items():
        replacements: dict[int, dict[int, str]] = defaultdict(dict)
        for event in events:
            replacements[int(event["record_idx"])][int(event["token_index"])] = fw_mask
            audit_rows.append({**event, "condition": f"{category}_masked_p100", "replacement_token": fw_mask, "match_status": "function_masked"})
        condition_name = f"{category}_masked_p100"
        condition_dir = conditions_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)
        condition_files[condition_name] = condition_dir / "utterances.jsonl"
        write_jsonl(condition_files[condition_name], make_condition_records(records, replacements))
        manifest["conditions"][condition_name] = {
            "type": "function_masked",
            "category": category,
            "file": str(condition_files[condition_name].relative_to(project_path("."))),
            "utterances": len(records),
            "mask_events": len(events),
        }

        used: set[tuple[int, int]] = set()
        control_replacements: dict[int, dict[int, str]] = defaultdict(dict)
        matched = 0
        for event in events:
            match, status = choose_content_match(event, pools, split_pools, used, config)
            if match is not None:
                used.add((int(match["record_idx"]), int(match["token_index"])))
                control_replacements[int(match["record_idx"])][int(match["token_index"])] = content_mask
                matched += 1
                audit_rows.append(
                    {
                        **match,
                        "condition": f"{category}_matched_content_masked_p100",
                        "category": category,
                        "replacement_token": content_mask,
                        "match_status": status,
                        "matched_function_line_id": event["line_id"],
                        "matched_function_token_index": event["token_index"],
                        "matched_function_token": event["original_token"],
                        "matched_function_upos": event["original_upos"],
                    }
                )
            else:
                audit_rows.append(
                    {
                        **event,
                        "condition": f"{category}_matched_content_masked_p100",
                        "replacement_token": "",
                        "match_status": status,
                    }
                )
        control_name = f"{category}_matched_content_masked_p100"
        control_dir = conditions_dir / control_name
        control_dir.mkdir(parents=True, exist_ok=True)
        condition_files[control_name] = control_dir / "utterances.jsonl"
        write_jsonl(condition_files[control_name], make_condition_records(records, control_replacements))
        manifest["conditions"][control_name] = {
            "type": "matched_content_masked",
            "category": category,
            "file": str(condition_files[control_name].relative_to(project_path("."))),
            "utterances": len(records),
            "mask_events": matched,
            "unmatched_events": len(events) - matched,
        }

    audits_dir = project_path(config["paths"]["audits_dir"])
    write_csv(audits_dir / "mask_matching_audit.csv", audit_rows)
    tokenization_rows = []
    for name in [config["special_tokens"]["fw_mask"], config["special_tokens"]["content_mask"], config["special_tokens"]["mask_verb"], config["special_tokens"]["mask"]]:
        tokenization_rows.append(
            {
                "token": name,
                "token_id": vocab.token_to_id.get(name, ""),
                "is_single_token": name in vocab.token_to_id,
                "decoded_text": name if name in vocab.token_to_id else "",
            }
        )
    write_csv(audits_dir / "tokenization_audit.csv", tokenization_rows)
    save_json(manifest_path, manifest)


class MLMDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], vocab) -> None:
        self.records = records
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "input_ids": [self.vocab.bos_id] + self.vocab.encode(record["tokens"]) + [self.vocab.eos_id],
            "pos": ["SPECIAL"] + record["pos"] + ["SPECIAL"],
        }


def collate_mlm(batch: list[dict[str, Any]], vocab, mlm_probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), vocab.pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        ids = list(item["input_ids"])
        maskable = [i for i, tag in enumerate(item["pos"]) if tag != "SPECIAL"]
        chosen = [i for i in maskable if random.random() < mlm_probability]
        if not chosen and maskable:
            chosen = [random.choice(maskable)]
        for i in chosen:
            labels[row, i] = ids[i]
            ids[i] = vocab.mask_verb_id if item["pos"][i] == "VERB" else vocab.mask_id
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return input_ids, labels


def run_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device, grad_clip: float, max_batches: int | None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total = 0.0
    batches = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return total / max(batches, 1)


def load_condition_records(config: dict[str, Any], condition: str, split: str) -> list[dict[str, Any]]:
    manifest = load_json(project_path(config["paths"]["conditions_dir"]) / "condition_manifest.json")
    all_records = []
    path = project_path(manifest["conditions"][condition]["file"])
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))
    _, splits, _ = load_prepared(config)
    return load_split_records(all_records, splits, split)


def train_condition(config: dict[str, Any], condition: str, seed: int, debug: bool = False) -> Path:
    set_seed(seed)
    records, _, vocab = load_prepared(config)
    train_records = load_condition_records(config, condition, "train")
    dev_records = load_condition_records(config, condition, "dev")
    transformer_cfg = dict(config["transformer"])
    max_batches = None
    if debug:
        transformer_cfg["epochs"] = 1
        transformer_cfg["batch_size"] = 16
        max_batches = 3
        train_records = train_records[:256]
        dev_records = dev_records[:128]

    train_loader = DataLoader(
        MLMDataset(train_records, vocab),
        batch_size=int(transformer_cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda batch: collate_mlm(batch, vocab, float(transformer_cfg["mlm_probability"])),
    )
    dev_loader = DataLoader(
        MLMDataset(dev_records, vocab),
        batch_size=int(transformer_cfg["batch_size"]),
        shuffle=False,
        collate_fn=lambda batch: collate_mlm(batch, vocab, float(transformer_cfg["mlm_probability"])),
    )
    device = get_device()
    model = TransformerMLM(
        vocab_size=len(vocab.id_to_token),
        pad_id=vocab.pad_id,
        max_length=int(config["data"]["max_utterance_tokens"]) + 2,
        embedding_dim=int(transformer_cfg["embedding_dim"]),
        nhead=int(transformer_cfg["nhead"]),
        num_layers=int(transformer_cfg["num_layers"]),
        dim_feedforward=int(transformer_cfg["dim_feedforward"]),
        dropout=float(transformer_cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(transformer_cfg["learning_rate"]))
    history = []
    started = time.time()
    for epoch in range(1, int(transformer_cfg["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, float(transformer_cfg["grad_clip"]), max_batches)
        dev_loss = run_epoch(model, dev_loader, None, device, float(transformer_cfg["grad_clip"]), max_batches)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})
        print(f"{condition} seed={seed} epoch {epoch}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}", flush=True)
    model_dir = project_path(config["paths"]["transformer_model_dir"]) / condition / str(seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = model_dir / ("checkpoint_debug.pt" if debug else "checkpoint.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": transformer_cfg,
            "vocab": {"token_to_id": vocab.token_to_id, "id_to_token": vocab.id_to_token},
            "history": history,
            "condition": condition,
            "seed": seed,
            "debug": debug,
            "elapsed_seconds": round(time.time() - started, 2),
            "max_length": int(config["data"]["max_utterance_tokens"]) + 2,
        },
        ckpt,
    )
    return ckpt


def load_transformer(config: dict[str, Any], condition: str, seed: int, debug: bool, vocab, device: torch.device):
    ckpt = project_path(config["paths"]["transformer_model_dir"]) / condition / str(seed) / ("checkpoint_debug.pt" if debug else "checkpoint.pt")
    checkpoint = torch.load(ckpt, map_location=device)
    cfg = checkpoint["model_config"]
    model = TransformerMLM(
        vocab_size=len(vocab.id_to_token),
        pad_id=vocab.pad_id,
        max_length=int(checkpoint["max_length"]),
        embedding_dim=int(cfg["embedding_dim"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def select_mvp_examples(test_records: list[dict[str, Any]], surface_map: dict[str, dict[str, str]], config: dict[str, Any], max_examples: int | None) -> list[dict[str, Any]]:
    rng = random.Random(int(config["seed"]))
    examples = []
    for record in test_records:
        eligible = [i for i, (token, upos) in enumerate(zip(record["tokens"], record["pos"])) if upos == "VERB" and token in surface_map]
        if not eligible:
            continue
        target_idx = rng.choice(eligible)
        target = record["tokens"][target_idx]
        examples.append(
            {
                "line_id": record["line_id"],
                "tokens": record["tokens"],
                "pos": record["pos"],
                "target_idx": target_idx,
                "target_surface": target,
                "target_lemma": surface_map[target]["lemma"],
                "target_class": surface_map[target]["class"],
            }
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def score_candidates(model, example: dict[str, Any], candidate_surfaces: list[str], vocab, device: torch.device) -> tuple[dict[str, float], torch.Tensor]:
    tokens = list(example["tokens"])
    tokens[int(example["target_idx"])] = vocab.id_to_token[vocab.mask_verb_id]
    ids = [vocab.bos_id] + vocab.encode(tokens) + [vocab.eos_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask_seq_idx = int(example["target_idx"]) + 1
    with torch.no_grad():
        logits = model(input_ids)[0, mask_seq_idx]
        log_probs = F.log_softmax(logits, dim=-1)
    return {surface: float(log_probs[vocab.token_to_id[surface]].detach().cpu()) for surface in candidate_surfaces}, logits.detach().cpu()


def candidate_metrics(scores: dict[str, float], surface_map: dict[str, dict[str, str]], target_class: str) -> tuple[str, str, float, float]:
    pred_surface = max(scores.items(), key=lambda item: item[1])[0]
    pred_class = surface_map[pred_surface]["class"]
    class_scores: dict[str, list[float]] = defaultdict(list)
    for surface, score in scores.items():
        class_scores[surface_map[surface]["class"]].append(score)
    target_score = masked_mean(class_scores[target_class])
    foil_scores = [masked_mean(values) for cls, values in class_scores.items() if cls != target_class]
    return pred_surface, pred_class, 1.0 if pred_class == target_class else 0.0, target_score - masked_mean(foil_scores)


def top5_lemma_accuracy(logits: torch.Tensor, surface_map: dict[str, dict[str, str]], vocab, target_lemma: str) -> tuple[float, list[str]]:
    log_probs = F.log_softmax(logits, dim=-1)
    scored = []
    for surface, meta in surface_map.items():
        if surface in vocab.token_to_id:
            scored.append((surface, meta["lemma"], float(log_probs[vocab.token_to_id[surface]])))
    top = sorted(scored, key=lambda item: item[2], reverse=True)[:5]
    lemmas = []
    for _, lemma, _ in top:
        if lemma not in lemmas:
            lemmas.append(lemma)
    return (1.0 if target_lemma in lemmas else 0.0), lemmas


def surface_topk_accuracy(scores: dict[str, float], target_surface: str, k: int) -> float:
    top = [surface for surface, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]]
    return 1.0 if target_surface in top else 0.0


def evaluate(config: dict[str, Any], debug: bool = False) -> None:
    records, splits, vocab = load_prepared(config)
    surface_map, class_to_surfaces = build_verb_maps(config, vocab)
    candidate_surfaces = sorted({surface for values in class_to_surfaces.values() for surface in values if surface in surface_map and surface in vocab.token_to_id})
    max_examples = 50 if debug else config["evaluation"]["max_examples"]
    examples = select_mvp_examples(load_split_records(records, splits, "test"), surface_map, config, max_examples)
    conditions = config["pilot_v3"]["primary_conditions"]
    seeds = config["evaluation"]["debug_seeds"] if debug else config["evaluation"]["seeds"]
    device = get_device()
    rows = []
    summary_rows = []
    for condition in conditions:
        for seed in seeds:
            model, checkpoint = load_transformer(config, condition, int(seed), debug, vocab, device)
            for example_id, example in enumerate(examples):
                scores, logits = score_candidates(model, example, candidate_surfaces, vocab, device)
                pred_surface, pred_class, class_acc, cps = candidate_metrics(scores, surface_map, example["target_class"])
                top5_acc, top5_lemmas = top5_lemma_accuracy(logits, surface_map, vocab, example["target_lemma"])
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "example_id": example_id,
                        "line_id": example["line_id"],
                        "target_surface": example["target_surface"],
                        "target_lemma": example["target_lemma"],
                        "target_class": example["target_class"],
                        "target_verb_logprob": scores.get(example["target_surface"], float("nan")),
                        "predicted_surface": pred_surface,
                        "predicted_class": pred_class,
                        "verb_class_accuracy": class_acc,
                        "class_preference_score": cps,
                        "surface_topk_accuracy": surface_topk_accuracy(scores, example["target_surface"], int(config["evaluation"]["topk_surface"])),
                        "top5_lemma_accuracy": top5_acc,
                        "top5_lemmas": " ".join(top5_lemmas),
                    }
                )
            final = checkpoint["history"][-1]
            summary_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "epochs": len(checkpoint["history"]),
                    "final_train_loss": final["train_loss"],
                    "final_dev_loss": final["dev_loss"],
                    "elapsed_seconds": checkpoint["elapsed_seconds"],
                    "debug": debug,
                }
            )
    results_dir = project_path(config["paths"]["results_dir"])
    suffix = "_debug" if debug else ""
    write_csv(results_dir / f"mvp_per_example{suffix}.csv", rows)
    write_csv(results_dir / f"training_condition_summary{suffix}.csv", summary_rows)
    aggregate_metrics(rows, results_dir / f"metrics_summary{suffix}.csv")
    paired_delta(rows, config, results_dir / f"paired_delta_summary{suffix}.csv")
    write_error_analysis(rows, config, results_dir / f"error_analysis{suffix}.md")


def aggregate_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["seed"]))].append(row)
    output = []
    for (condition, seed), items in sorted(grouped.items()):
        output.append({"condition": condition, "seed": seed, "n_examples": len(items), **{metric: masked_mean([float(item[metric]) for item in items]) for metric in METRICS}})
    write_csv(path, output)


def paired_delta(rows: list[dict[str, Any]], config: dict[str, Any], path: Path) -> None:
    by_key = {(str(row["seed"]), int(row["example_id"]), str(row["condition"])): row for row in rows}
    output = []
    for seed in sorted({str(row["seed"]) for row in rows}):
        example_ids = sorted({int(row["example_id"]) for row in rows if str(row["seed"]) == seed})
        for category in config["pilot_v3"]["categories"]:
            fn = f"{category}_masked_p100"
            ctrl = f"{category}_matched_content_masked_p100"
            for metric in METRICS:
                function_minus_full = []
                control_minus_full = []
                adjusted = []
                for example_id in example_ids:
                    full = by_key.get((seed, example_id, "full_input"))
                    frow = by_key.get((seed, example_id, fn))
                    crow = by_key.get((seed, example_id, ctrl))
                    if not full or not frow or not crow:
                        continue
                    f_delta = float(frow[metric]) - float(full[metric])
                    c_delta = float(crow[metric]) - float(full[metric])
                    function_minus_full.append(f_delta)
                    control_minus_full.append(c_delta)
                    adjusted.append(f_delta - c_delta)
                for name, values in [
                    (f"{fn}_minus_full_input", function_minus_full),
                    (f"{ctrl}_minus_full_input", control_minus_full),
                    (f"{category}_matched_control_adjusted_delta", adjusted),
                ]:
                    if values:
                        output.append({"seed": seed, "category": category, "metric": metric, "comparison": name, "n_pairs": len(values), "mean_delta": masked_mean(values)})
    write_csv(path, output)


def write_error_analysis(rows: list[dict[str, Any]], config: dict[str, Any], path: Path) -> None:
    by_key = {(str(row["seed"]), int(row["example_id"]), str(row["condition"])): row for row in rows}
    lines = ["# Pilot_v3 Error Analysis", ""]
    for category in config["pilot_v3"]["categories"]:
        fn = f"{category}_masked_p100"
        ctrl = f"{category}_matched_content_masked_p100"
        lines.append(f"## {category}")
        examples = []
        for key, frow in by_key.items():
            seed, example_id, condition = key
            if condition != fn:
                continue
            crow = by_key.get((seed, example_id, ctrl))
            full = by_key.get((seed, example_id, "full_input"))
            if not crow or not full:
                continue
            if float(frow["verb_class_accuracy"]) == 0.0 and float(crow["verb_class_accuracy"]) == 1.0:
                examples.append((frow, crow, full))
        lines.append(f"- function_masked failed while matched_content succeeded: {len(examples)} cases.")
        for frow, crow, full in examples[:10]:
            lines.append(
                f"- seed={frow['seed']} example={frow['example_id']} target={frow['target_surface']}/{frow['target_class']} "
                f"full_pred={full['predicted_surface']}/{full['predicted_class']} "
                f"function_pred={frow['predicted_surface']}/{frow['predicted_class']} "
                f"content_pred={crow['predicted_surface']}/{crow['predicted_class']}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--conditions", nargs="*", default=None)
    args = parser.parse_args()
    config = load_config(CONFIG_PATH)
    ensure_dirs(config)
    prepare_base_data(config, force=args.force_prepare)
    generate_conditions(config, force=args.force_prepare)
    conditions = args.conditions or list(config["pilot_v3"]["primary_conditions"])
    seeds = list(config["evaluation"]["debug_seeds"] if args.debug else config["evaluation"]["seeds"])
    if not args.skip_train:
        for condition in conditions:
            for seed in seeds:
                train_condition(config, condition, int(seed), debug=args.debug)
    evaluate(config, debug=args.debug)
    print("Pilot_v3 complete.")


if __name__ == "__main__":
    main()
