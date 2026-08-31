from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast

from data_utils import build_verb_maps, build_vocab, load_prepared, load_split_records, parse_corpus, save_prepared, split_records
from pilot_utils import ensure_dirs, load_config, load_json, masked_mean, project_path, save_json, set_seed, write_jsonl


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "Pilot_v4_roberta.yaml"
METRICS = [
    "target_verb_logprob",
    "target_verb_logprob_sum",
    "verb_class_accuracy",
    "class_preference_score",
    "top1_surface_accuracy",
    "top3_surface_accuracy",
    "top5_surface_accuracy",
    "top1_lemma_accuracy",
    "top3_lemma_accuracy",
    "top5_lemma_accuracy",
]


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def stable_int(*parts: Any) -> int:
    return int(hashlib.md5("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12], 16)


def load_merged_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("base_experiment_config"):
        base = load_config(project_path(config["base_experiment_config"]))
        merged = dict(base)
        for key, value in config.items():
            if key in {"paths", "training", "roberta", "evaluation", "tokenizer", "ablation_design"} and isinstance(value, dict):
                merged[key] = {**base.get(key, {}), **value}
            else:
                merged[key] = value
        config = merged
    source = source_config(config)
    for key in ("split", "data", "verb_lexicon"):
        if key not in config:
            config[key] = source[key]
    return config


def source_config(config: dict[str, Any]) -> dict[str, Any]:
    source = load_config(project_path(config["source_config"]))
    if source.get("base_source_config"):
        base = load_config(project_path(source["base_source_config"]))
        merged = dict(base)
        for key, value in source.items():
            if key == "paths" and isinstance(value, dict):
                merged["paths"] = {**base.get("paths", {}), **value}
            else:
                merged[key] = value
        source = merged
    return source


def load_source_records(config: dict[str, Any]):
    return load_prepared(source_config(config))


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_path(".."), text=True).strip()
    except Exception:
        return "unknown"


def prepare_base_data(config: dict[str, Any]) -> None:
    processed = project_path(config["paths"]["processed_dir"]) / "utterances.jsonl"
    splits_path = project_path(config["paths"]["splits_dir"]) / "splits.json"
    if processed.exists() and splits_path.exists():
        return
    records = parse_corpus(config)
    splits = split_records(records, config)
    train_records = load_split_records(records, splits, "train")
    save_prepared(records, splits, build_vocab(train_records, config), config)


def split_lookup(splits: dict[str, list[int]]) -> dict[int, str]:
    return {int(idx): split for split, indices in splits.items() for idx in indices}


def bin_label(value: float | int | str, boundaries: list[float | int], prefix: str = "<=") -> str:
    if value == "":
        return "none"
    number = float(value)
    for boundary in boundaries:
        if number <= float(boundary):
            return f"{prefix}{boundary}"
    return f">{boundaries[-1]}"


def relative_position(token_index: int, length: int) -> float:
    return round(token_index / max(length - 1, 1), 4)


def nearest_verb_distance(pos: list[str], token_index: int) -> int | str:
    distances = [abs(i - token_index) for i, upos in enumerate(pos) if upos == "VERB"]
    return min(distances) if distances else ""


def first_target_class(record: dict[str, Any], surface_map: dict[str, dict[str, str]]) -> str:
    classes = [surface_map[t]["class"] for t, upos in zip(record["tokens"], record["pos"]) if upos == "VERB" and t in surface_map]
    unique = sorted(set(classes))
    if not unique:
        return "none"
    if len(unique) > 1:
        return "multiple"
    return unique[0]


def token_frequency_bin(token: str, counts: dict[str, int], config: dict[str, Any]) -> str:
    return bin_label(int(counts.get(token, 0)), config["bins"]["frequency"])


def category_indices(record: dict[str, Any], category: str, config: dict[str, Any]) -> list[int]:
    if category == "COMP":
        allow = set(config["comp_allowlist"])
        return [i for i, (t, p) in enumerate(zip(record["tokens"], record["pos"])) if p == "SCONJ" and t in allow]
    return [i for i, p in enumerate(record["pos"]) if p == category]


def content_indices(record: dict[str, Any], config: dict[str, Any]) -> list[int]:
    allowed = set(config["content_control_pos"])
    excluded = set(config["excluded_content_pos"])
    return [i for i, p in enumerate(record["pos"]) if p in allowed and p not in excluded]


def make_event(
    records: list[dict[str, Any]],
    record_idx: int,
    token_index: int,
    split: str,
    counts: dict[str, int],
    category: str,
    event_type: str,
    surface_map: dict[str, dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    record = records[record_idx]
    token = record["tokens"][token_index]
    dist = nearest_verb_distance(record["pos"], token_index)
    rel = relative_position(token_index, len(record["tokens"]))
    return {
        "category": category,
        "event_type": event_type,
        "split": split,
        "record_idx": record_idx,
        "line_id": record["line_id"],
        "token_index": token_index,
        "token": token,
        "upos": record["pos"][token_index],
        "utterance_length": len(record["tokens"]),
        "utterance_length_bin": bin_label(len(record["tokens"]), config["bins"]["utterance_length"]),
        "relative_position": rel,
        "relative_position_bin": bin_label(rel, config["bins"]["relative_position"]),
        "nearest_verb_distance": dist,
        "nearest_verb_distance_bin": bin_label(dist, config["bins"]["nearest_verb_distance"]),
        "token_frequency": counts.get(token, 0),
        "token_frequency_bin": token_frequency_bin(token, counts, config),
        "target_verb_class_if_applicable": first_target_class(record, surface_map),
        "source_distribution": config.get("source_distribution", "UD-English-CHILDES-parental"),
    }


def event_match_key(event: dict[str, Any], relax: str | None = None) -> tuple[str, ...]:
    fields = ["split", "utterance_length_bin", "relative_position_bin", "nearest_verb_distance_bin", "token_frequency_bin", "target_verb_class_if_applicable"]
    if relax == "relaxed_frequency":
        fields.remove("token_frequency_bin")
    elif relax == "relaxed_distance":
        fields.remove("nearest_verb_distance_bin")
    elif relax == "relaxed_position":
        fields.remove("relative_position_bin")
    elif relax == "split_only_fallback":
        fields = ["split"]
    return tuple(str(event[field]) for field in fields)


def choose_match(
    function_event: dict[str, Any],
    pools: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]],
    reuse: Counter[tuple[int, int]],
    config: dict[str, Any],
    cursors: dict[tuple[str, tuple[str, ...]], int] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    levels = [None, "relaxed_frequency", "relaxed_distance", "relaxed_position", "split_only_fallback"]
    names = ["exact_matched", "relaxed_frequency", "relaxed_distance", "relaxed_position", "split_only_fallback"]
    allow_reuse = bool(config.get("matching", {}).get("allow_content_reuse_within_category", True))
    for relax, name in zip(levels, names):
        candidates = pools[name].get(event_match_key(function_event, relax), [])
        if not allow_reuse:
            if not candidates:
                continue
            pool_key = (name, event_match_key(function_event, relax))
            if cursors is None:
                start = stable_int(config["seed"], function_event["category"], function_event["line_id"], function_event["token_index"], name) % len(candidates)
            else:
                cursor = cursors.get(pool_key, 0)
                if cursor >= len(candidates):
                    continue
                start = cursor
            for offset in range(len(candidates)):
                chosen_index = (start + offset) % len(candidates)
                chosen = candidates[chosen_index]
                key = (int(chosen["record_idx"]), int(chosen["token_index"]))
                if key in reuse:
                    continue
                reuse[key] = 1
                if cursors is not None:
                    cursors[pool_key] = chosen_index + 1
                return chosen, name
            if cursors is not None:
                cursors[pool_key] = start + len(candidates)
            continue
        if candidates:
            min_reuse = min(reuse[(int(c["record_idx"]), int(c["token_index"]))] for c in candidates)
            least_used = [c for c in candidates if reuse[(int(c["record_idx"]), int(c["token_index"]))] == min_reuse]
            idx = stable_int(config["seed"], function_event["category"], function_event["line_id"], function_event["token_index"], name) % len(least_used)
            chosen = least_used[idx]
            reuse[(int(chosen["record_idx"]), int(chosen["token_index"]))] += 1
            return chosen, name
    return None, "unmatched"


def apply_replacements(records: list[dict[str, Any]], replacements: dict[int, dict[int, str]]) -> list[dict[str, Any]]:
    output = []
    for idx, record in enumerate(records):
        updated = dict(record)
        tokens = list(record["tokens"])
        for token_index, replacement in replacements.get(idx, {}).items():
            tokens[int(token_index)] = replacement
        updated["tokens"] = tokens
        output.append(updated)
    return output


def create_ablation_conditions_and_audits(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Create train-only target/random/identity conditions for the confirmatory design."""
    ensure_dirs(config)
    conditions_dir = project_path(config["paths"]["conditions_dir"])
    audits_dir = project_path(config["paths"]["audits_dir"])
    manifest_path = conditions_dir / "condition_manifest.json"
    if manifest_path.exists() and not force:
        return load_json(manifest_path)

    records, splits, _ = load_source_records(config)
    lookup = split_lookup(splits)
    categories = list(config["categories"])
    ablation_token = config["special_tokens"]["ablation"]
    random_seeds = [int(x) for x in config["ablation_design"]["random_control_corpus_seeds"]]
    identity_seed = int(config["ablation_design"]["identity_shuffle_seed"])
    write_position_audit = bool(config["ablation_design"].get("write_position_audit", True))
    write_overlap_audit = bool(config["ablation_design"].get("write_overlap_audit", True))
    train_indices = set(int(x) for x in splits["train"])

    def split_hashes_for(source_records: list[dict[str, Any]]) -> dict[str, str]:
        digests = {split: hashlib.sha256() for split in splits}
        # Condition files are streamed in original record order. Hash the
        # baseline in that same order; split index lists are shuffled and would
        # otherwise produce false mismatches despite identical record content.
        for index, record in enumerate(source_records):
            payload = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
            digests[lookup[index]].update(payload + b"\n")
        return {split: digest.hexdigest() for split, digest in digests.items()}

    original_path = conditions_dir / "original" / "utterances.jsonl"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(original_path, records)
    split_hashes = split_hashes_for(records)
    manifest: dict[str, Any] = {
        "version": config["version"],
        "design": "target_random_identity_ablation_v1",
        "git_commit": git_head(),
        "raw_corpus": config["paths"]["raw_corpus"],
        "raw_corpus_hash": sha256_file(project_path(config["paths"]["raw_corpus"])),
        "split_hash": sha256_json(splits),
        "original_split_hashes": split_hashes,
        "config_hash": sha256_json(config),
        "train_only_manipulation": True,
        "train_categories": categories,
        "conditions": {
            "original": {
                "type": "original",
                "file": str(original_path.relative_to(project_path("."))),
                "replacement_count": 0,
                "file_hash": sha256_file(original_path),
                "split_hashes": split_hashes,
            }
        },
        "category_audit": {},
    }
    audit_rows: list[dict[str, Any]] = []
    random_position_sets: dict[str, dict[str, set[tuple[int, int]]]] = defaultdict(dict)

    def save_condition(name: str, typ: str, category: str, replacements: dict[int, dict[int, str]], **extra: Any) -> None:
        path = conditions_dir / name / "utterances.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        split_digests = {split: hashlib.sha256() for split in splits}
        with path.open("w", encoding="utf-8") as handle:
            for record_idx, record in enumerate(records):
                updated = record
                if record_idx in replacements:
                    updated = dict(record)
                    tokens = list(record["tokens"])
                    for token_index, replacement in replacements[record_idx].items():
                        tokens[int(token_index)] = replacement
                    updated["tokens"] = tokens
                handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
                split_digests[lookup[record_idx]].update(json.dumps(updated, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n")
        condition_split_hashes = {split: digest.hexdigest() for split, digest in split_digests.items()}
        if condition_split_hashes["dev"] != split_hashes["dev"] or condition_split_hashes["test"] != split_hashes["test"]:
            raise AssertionError(f"{name}: dev/test changed in a train-only design")
        manifest["conditions"][name] = {
            "type": typ,
            "category": category,
            "file": str(path.relative_to(project_path("."))),
            "replacement_token": ablation_token if typ != "identity_shuffle" else None,
            "replacement_count": sum(len(v) for v in replacements.values()),
            "file_hash": sha256_file(path),
            "split_hashes": condition_split_hashes,
            **extra,
        }

    for category in categories:
        target_positions: list[tuple[int, int]] = []
        for record_idx in sorted(train_indices):
            record = records[record_idx]
            target_positions.extend((record_idx, token_idx) for token_idx in category_indices(record, category, config))
        target_set = set(target_positions)
        target_replacements: dict[int, dict[int, str]] = defaultdict(dict)
        for record_idx, token_idx in target_positions:
            target_replacements[record_idx][token_idx] = ablation_token
            if write_position_audit:
                audit_rows.append({"condition": f"{category}_target_ablation", "condition_type": "target_ablation", "category": category, "corpus_seed": "", "record_idx": record_idx, "line_id": records[record_idx]["line_id"], "token_index": token_idx, "original_token": records[record_idx]["tokens"][token_idx], "replacement_token": ablation_token, "upos": records[record_idx]["pos"][token_idx], "split": "train"})
        save_condition(f"{category}_target_ablation", "target_ablation", category, target_replacements, eligible_count=len(target_positions))

        random_pool = [
            (record_idx, token_idx)
            for record_idx in sorted(train_indices)
            for token_idx, upos in enumerate(records[record_idx]["pos"])
            if (record_idx, token_idx) not in target_set and upos != "PUNCT"
        ]
        if len(random_pool) < len(target_positions):
            raise ValueError(f"{category}: random pool smaller than target event count")
        for random_index, corpus_seed in enumerate(random_seeds, 1):
            name = f"{category}_random_ablation_r{random_index:02d}"
            chosen = random.Random(corpus_seed + stable_int(config["version"], category)).sample(random_pool, len(target_positions))
            if write_overlap_audit:
                random_position_sets[category][name] = set(chosen)
            replacements: dict[int, dict[int, str]] = defaultdict(dict)
            pos_counts: Counter[str] = Counter()
            for record_idx, token_idx in chosen:
                replacements[record_idx][token_idx] = ablation_token
                pos_counts[records[record_idx]["pos"][token_idx]] += 1
                if write_position_audit:
                    audit_rows.append({"condition": name, "condition_type": "random_ablation", "category": category, "corpus_seed": corpus_seed, "record_idx": record_idx, "line_id": records[record_idx]["line_id"], "token_index": token_idx, "original_token": records[record_idx]["tokens"][token_idx], "replacement_token": ablation_token, "upos": records[record_idx]["pos"][token_idx], "split": "train"})
            save_condition(name, "random_ablation", category, replacements, corpus_seed=corpus_seed, target_count=len(target_positions), selected_count=len(chosen), selected_unique_count=len(set(chosen)), selected_pos_counts=dict(pos_counts))

        identity_positions = list(target_positions)
        original_tokens = [records[r]["tokens"][i] for r, i in identity_positions]
        rng = random.Random(identity_seed + stable_int(config["version"], category))
        # Sort equal forms into blocks, randomize within blocks, then rotate by the
        # largest block. This preserves the exact multiset and attains a complete
        # derangement whenever no single form occupies more than half the events.
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, token in enumerate(original_tokens):
            grouped_indices[token].append(index)
        for indices in grouped_indices.values():
            rng.shuffle(indices)
        token_order = list(grouped_indices)
        rng.shuffle(token_order)
        sorted_indices = [index for token in token_order for index in grouped_indices[token]]
        sorted_tokens = [original_tokens[index] for index in sorted_indices]
        shift = max(Counter(original_tokens).values(), default=0)
        rotated_tokens = sorted_tokens[shift:] + sorted_tokens[:shift]
        best = list(original_tokens)
        for index, replacement in zip(sorted_indices, rotated_tokens):
            best[index] = replacement
        best_changed = sum(a != b for a, b in zip(original_tokens, best))
        if Counter(original_tokens) != Counter(best):
            raise AssertionError(f"{category}: identity shuffle changed token frequencies")
        identity_replacements: dict[int, dict[int, str]] = defaultdict(dict)
        for (record_idx, token_idx), replacement in zip(identity_positions, best):
            identity_replacements[record_idx][token_idx] = replacement
            if write_position_audit:
                audit_rows.append({"condition": f"{category}_identity_shuffle", "condition_type": "identity_shuffle", "category": category, "corpus_seed": identity_seed, "record_idx": record_idx, "line_id": records[record_idx]["line_id"], "token_index": token_idx, "original_token": records[record_idx]["tokens"][token_idx], "replacement_token": replacement, "upos": records[record_idx]["pos"][token_idx], "split": "train"})
        save_condition(f"{category}_identity_shuffle", "identity_shuffle", category, identity_replacements, corpus_seed=identity_seed, event_count=len(identity_positions), changed_count=best_changed, unchanged_count=len(identity_positions) - best_changed, original_token_counts=dict(Counter(original_tokens)), shuffled_token_counts=dict(Counter(best)))
        manifest["category_audit"][category] = {"target_train_events": len(target_positions), "random_pool_size": len(random_pool), "identity_changed": best_changed, "identity_unchanged": len(identity_positions) - best_changed}

    overlap_rows = []
    for category, named_sets in random_position_sets.items():
        names = sorted(named_sets)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                overlap = len(named_sets[left] & named_sets[right])
                overlap_rows.append({"category": category, "left_condition": left, "right_condition": right, "overlap_count": overlap, "overlap_rate": overlap / max(len(named_sets[left]), 1)})
    if write_position_audit:
        csv_write(audits_dir / "ablation_position_audit.csv", audit_rows)
    if write_overlap_audit:
        csv_write(audits_dir / "random_control_overlap_audit.csv", overlap_rows)
    save_json(manifest_path, manifest)
    return manifest


def create_conditions_and_audits(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    if config.get("experiment_design") == "acl_aux_confirmatory_v2":
        from acl_conditions import build_acl_conditions

        return build_acl_conditions(config, force)
    if config.get("experiment_design") == "target_random_identity_ablation_v1":
        return create_ablation_conditions_and_audits(config, force)
    ensure_dirs(config)
    conditions_dir = project_path(config["paths"]["conditions_dir"])
    audits_dir = project_path(config["paths"]["audits_dir"])
    manifest_path = conditions_dir / "condition_manifest.json"
    if manifest_path.exists() and not force:
        return load_json(manifest_path)

    records, splits, _ = load_source_records(config)
    surface_map, _ = build_verb_maps(config)
    counts = load_json(project_path(config["paths"]["processed_dir"]) / "vocab.json")["counts"]
    lookup = split_lookup(splits)

    function_rows: list[dict[str, Any]] = []
    content_events: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record_idx, record in enumerate(records):
        split = lookup[record_idx]
        for idx in content_indices(record, config):
            content_events.append(make_event(records, record_idx, idx, split, counts, "CONTENT", "content", surface_map, config))
        for category in config["categories"]:
            for idx in category_indices(record, category, config):
                event = make_event(records, record_idx, idx, split, counts, category, "function", surface_map, config)
                function_rows.append(event)
                by_category[category].append(event)

    csv_write(audits_dir / "function_event_audit.csv", function_rows)

    pools: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]] = {name: defaultdict(list) for name in ["exact_matched", "relaxed_frequency", "relaxed_distance", "relaxed_position", "split_only_fallback"]}
    for event in content_events:
        for relax, name in [(None, "exact_matched"), ("relaxed_frequency", "relaxed_frequency"), ("relaxed_distance", "relaxed_distance"), ("relaxed_position", "relaxed_position"), ("split_only_fallback", "split_only_fallback")]:
            pools[name][event_match_key(event, relax)].append(event)

    full_path = conditions_dir / "full_input" / "utterances.jsonl"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(full_path, records)

    manifest: dict[str, Any] = {
        "version": config["version"],
        "git_commit": git_head(),
        "raw_corpus": config["paths"]["raw_corpus"],
        "raw_corpus_hash": sha256_file(project_path(config["paths"]["raw_corpus"])),
        "split_hash": sha256_json(splits),
        "verb_lexicon_hash": sha256_json(config["verb_lexicon"]),
        "config_hash": sha256_json(config),
        "conditions": {
            "full_input": {"type": "baseline", "file": str(full_path.relative_to(project_path("."))), "mask_count": 0, "fallback_count": 0, "unmatched_count": 0, "reuse_max": 0}
        },
        "category_audit": {},
        "train_categories": [],
    }

    mask_rows: list[dict[str, Any]] = []
    matched_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in config["categories"]:
        function_events = by_category[category]
        train_count = sum(1 for e in function_events if e["split"] == "train")
        manifest["category_audit"][category] = {"eligible_events": len(function_events), "train_events": train_count}

        function_replacements: dict[int, dict[int, str]] = defaultdict(dict)
        for event in function_events:
            function_replacements[int(event["record_idx"])][int(event["token_index"])] = config["special_tokens"]["fw_mask"]
            mask_rows.append({**event, "condition": f"{category}_masked_p100", "replacement_token": config["special_tokens"]["fw_mask"], "match_status": "function_masked", "reuse_count": 0})

        reuse: Counter[tuple[int, int]] = Counter()
        cursors: dict[tuple[str, tuple[str, ...]], int] = {}
        content_replacements: dict[int, dict[int, str]] = defaultdict(dict)
        strict_replacements: dict[int, dict[int, str]] = defaultdict(dict)
        matched_events: list[dict[str, Any]] = []
        fallback_count = 0
        unmatched_count = 0
        status_counts: Counter[str] = Counter()
        for event in function_events:
            match, status = choose_match(event, pools, reuse, config, cursors)
            status_counts[status] += 1
            if status != "exact_matched":
                fallback_count += 1
            if match is None:
                unmatched_count += 1
                mask_rows.append({**event, "condition": f"{category}_matched_content_masked_p100", "replacement_token": "", "match_status": status, "reuse_count": ""})
                continue
            content_replacements[int(match["record_idx"])][int(match["token_index"])] = config["special_tokens"]["content_mask"]
            strict_replacements[int(match["record_idx"])][int(match["token_index"])] = config["special_tokens"]["fw_mask"]
            content_row = {
                **match,
                "category": category,
                "condition": f"{category}_matched_content_masked_p100",
                "replacement_token": config["special_tokens"]["content_mask"],
                "match_status": status,
                "matched_function_line_id": event["line_id"],
                "matched_function_token_index": event["token_index"],
                "matched_function_token": event["token"],
                "matched_function_upos": event["upos"],
                "reuse_count": reuse[(int(match["record_idx"]), int(match["token_index"]))],
            }
            matched_events.append(content_row)
            mask_rows.append(content_row)
        matched_by_category[category] = matched_events

        for condition, replacements, token_name, typ in [
            (f"{category}_masked_p100", function_replacements, config["special_tokens"]["fw_mask"], "function_masked"),
            (f"{category}_matched_content_masked_p100", content_replacements, config["special_tokens"]["content_mask"], "matched_content_masked"),
            (f"{category}_strict_content_control_p100", strict_replacements, config["special_tokens"]["fw_mask"], "strict_content_control"),
        ]:
            path = conditions_dir / condition / "utterances.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(path, apply_replacements(records, replacements))
            manifest["conditions"][condition] = {
                "type": typ,
                "category": category,
                "replacement_token": token_name,
                "file": str(path.relative_to(project_path("."))),
                "mask_count": sum(len(v) for v in replacements.values()),
                "matched_event_count": len(matched_events) if typ != "function_masked" else len(function_events),
                "eligible_function_event_count": len(function_events),
                "fallback_count": fallback_count if typ != "function_masked" else 0,
                "unmatched_count": unmatched_count if typ != "function_masked" else 0,
                "match_status_counts": dict(status_counts) if typ != "function_masked" else {},
                "reuse_max": max(reuse.values()) if reuse else 0,
            }

    csv_write(audits_dir / "mask_matching_audit.csv", mask_rows)
    balance_rows, category_status = build_balance_audit(config, by_category, matched_by_category)
    csv_write(audits_dir / "balance_audit_summary.csv", balance_rows)
    for category, status in category_status.items():
        manifest["category_audit"][category].update(status)
    if bool(config.get("training", {}).get("train_all_categories", False)):
        train_categories = list(config["categories"])
    else:
        train_categories = [
            c for c in config["categories"]
            if c in config["default_train_categories"] and manifest["category_audit"][c]["decision"] in {"go", "caution"}
        ]
        if manifest["category_audit"].get("COMP", {}).get("decision") in {"go", "caution"}:
            train_categories.append("COMP")
    manifest["train_categories"] = list(dict.fromkeys(train_categories))
    save_json(manifest_path, manifest)
    return manifest


def build_balance_audit(config: dict[str, Any], function_by_category: dict[str, list[dict[str, Any]]], content_by_category: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    dimensions = ["token_frequency_bin", "nearest_verb_distance_bin", "utterance_length_bin", "relative_position_bin", "target_verb_class_if_applicable", "split", "source_distribution"]
    rows = []
    status_by_category: dict[str, dict[str, Any]] = {}
    for category in config["categories"]:
        category_max = 0.0
        fn = function_by_category.get(category, [])
        ct = content_by_category.get(category, [])
        unmatched = max(len(fn) - len(ct), 0)
        for dim in dimensions:
            f_counts = Counter(str(e[dim]) for e in fn)
            c_counts = Counter(str(e[dim]) for e in ct)
            bins = sorted(set(f_counts) | set(c_counts))
            dim_max = 0.0
            for value in bins:
                f = f_counts[value]
                c = c_counts[value]
                f_prop = f / max(len(fn), 1)
                c_prop = c / max(len(ct), 1)
                normalized = abs(f_prop - c_prop)
                dim_max = max(dim_max, normalized)
                rows.append(
                    {
                        "category": category,
                        "dimension": dim,
                        "bin": value,
                        "function_count": f,
                        "content_count": c,
                        "absolute_difference": abs(f - c),
                        "normalized_difference": normalized,
                        "max_bin_imbalance": dim_max,
                    }
                )
            category_max = max(category_max, dim_max)
        if unmatched > 0:
            decision = "no_go"
        elif len(fn) == 0 or sum(1 for e in fn if e["split"] == "train") < int(config["audit"]["min_train_events_for_training"]):
            decision = "no_go"
        elif category_max <= float(config["audit"]["go_max_normalized_imbalance"]):
            decision = "go"
        elif category_max <= float(config["audit"]["caution_max_normalized_imbalance"]):
            decision = "caution"
        else:
            decision = "no_go"
        for row in rows:
            if row["category"] == category:
                row["category_max_imbalance"] = category_max
                row["status"] = decision
                row["unmatched_count"] = unmatched
        status_by_category[category] = {"max_balance_imbalance": category_max, "unmatched_count": unmatched, "decision": decision}
    return rows, status_by_category


def train_tokenizer(config: dict[str, Any], force: bool = False) -> RobertaTokenizerFast:
    tokenizer_dir = project_path(config["paths"]["tokenizer_dir"])
    vocab_file = tokenizer_dir / "vocab.json"
    merges_file = tokenizer_dir / "merges.txt"
    if not force and vocab_file.exists() and merges_file.exists():
        return load_tokenizer(config)
    records, splits, _ = load_source_records(config)
    train_path = tokenizer_dir / "full_input_train.txt"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as f:
        for record in load_split_records(records, splits, "train"):
            f.write(" ".join(record["tokens"]) + "\n")
    special = config["special_tokens"]
    special_tokens = [special["bos"], special["pad"], special["eos"], special["unk"], special["mask"], special["fw_mask"], special["content_mask"]]
    if special.get("ablation"):
        special_tokens.append(special["ablation"])
    special_tokens = list(dict.fromkeys(special_tokens))
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[str(train_path)], vocab_size=int(config["tokenizer"]["vocab_size"]), min_frequency=int(config["tokenizer"]["min_frequency"]), special_tokens=special_tokens)
    tokenizer.save_model(str(tokenizer_dir))
    return load_tokenizer(config)


def load_tokenizer(config: dict[str, Any]) -> RobertaTokenizerFast:
    special = config["special_tokens"]
    tokenizer_dir = project_path(config["paths"]["tokenizer_dir"])
    tokenizer = RobertaTokenizerFast(
        vocab_file=str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt"),
        bos_token=special["bos"],
        eos_token=special["eos"],
        unk_token=special["unk"],
        pad_token=special["pad"],
        mask_token=special["mask"],
        add_prefix_space=True,
    )
    additional = [special["fw_mask"], special["content_mask"]]
    if special.get("ablation"):
        additional.append(special["ablation"])
    tokenizer.add_special_tokens({"additional_special_tokens": list(dict.fromkeys(additional))})
    return tokenizer


def tokenization_audit(config: dict[str, Any], tokenizer: RobertaTokenizerFast) -> None:
    rows = []
    samples = [
        ["look", config["special_tokens"]["fw_mask"], "at", "that"],
        ["you", config["special_tokens"]["content_mask"], "it"],
        ["i", "want", config["special_tokens"]["mask"], "."],
    ]
    audited_tokens = [config["special_tokens"]["fw_mask"], config["special_tokens"]["content_mask"], config["special_tokens"]["mask"]]
    if config["special_tokens"].get("ablation"):
        audited_tokens.append(config["special_tokens"]["ablation"])
    audited_tokens = list(dict.fromkeys(audited_tokens))
    for token in audited_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        rows.append({"sample_text": token, "placeholder": token, "placeholder_count": 1, "encoded_ids": " ".join(map(str, encoded)), "decoded_text": tokenizer.decode(encoded), "is_single_token": len(encoded) == 1})
    for sample in samples:
        text = " ".join(sample)
        encoded = tokenizer.encode(text, add_special_tokens=True)
        for token in audited_tokens:
            rows.append({"sample_text": text, "placeholder": token, "placeholder_count": sample.count(token), "encoded_ids": " ".join(map(str, encoded)), "decoded_text": tokenizer.decode(encoded), "is_single_token": len(tokenizer.encode(token, add_special_tokens=False)) == 1})
    csv_write(project_path(config["paths"]["audits_dir"]) / "tokenization_audit.csv", rows)


class ConditionDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: RobertaTokenizerFast, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = " ".join(self.records[idx]["tokens"])
        encoded = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors=None)
        return {"input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long)}


def collate_mlm(batch: list[dict[str, torch.Tensor]], tokenizer: RobertaTokenizerFast, mlm_probability: float) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    special_ids = set(tokenizer.all_special_ids)
    for row, item in enumerate(batch):
        ids = item["input_ids"].clone()
        length = len(ids)
        input_ids[row, :length] = ids
        attention_mask[row, :length] = 1
        candidates = [i for i, token_id in enumerate(ids.tolist()) if token_id not in special_ids]
        chosen = [i for i in candidates if random.random() < mlm_probability]
        if not chosen and candidates:
            chosen = [random.choice(candidates)]
        for i in chosen:
            labels[row, i] = input_ids[row, i]
            input_ids[row, i] = tokenizer.mask_token_id
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def condition_records(config: dict[str, Any], condition: str, split: str) -> list[dict[str, Any]]:
    manifest = load_json(project_path(config["paths"]["conditions_dir"]) / "condition_manifest.json")
    if manifest.get("train_only_manipulation") and split != "train":
        condition = "original"
    path = project_path(manifest["conditions"][condition]["file"])
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    _, splits, _ = load_source_records(config)
    return load_split_records(rows, splits, split)


def max_encoded_length(config: dict[str, Any], tokenizer: RobertaTokenizerFast) -> int:
    records, _, _ = load_source_records(config)
    max_len = 0
    for record in records:
        max_len = max(max_len, len(tokenizer.encode(" ".join(record["tokens"]), add_special_tokens=True)))
    return max(int(config["roberta"]["max_position_embeddings_min"]), max_len + 4)


def make_model(config: dict[str, Any], tokenizer: RobertaTokenizerFast, max_position_embeddings: int) -> RobertaForMaskedLM:
    rcfg = config["roberta"]
    model_config = RobertaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=max_position_embeddings,
        hidden_size=int(rcfg["hidden_size"]),
        num_hidden_layers=int(rcfg["num_hidden_layers"]),
        num_attention_heads=int(rcfg["num_attention_heads"]),
        intermediate_size=int(rcfg["intermediate_size"]),
        hidden_dropout_prob=float(rcfg["hidden_dropout_prob"]),
        attention_probs_dropout_prob=float(rcfg["attention_probs_dropout_prob"]),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return RobertaForMaskedLM(model_config)


def run_epoch(model: RobertaForMaskedLM, loader: DataLoader, optimizer, device: torch.device, grad_clip: float, max_batches: int | None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            outputs = model(**batch)
            loss = outputs.loss
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1
        if max_batches and batches >= max_batches:
            break
    return total / max(batches, 1)


def train_condition(config: dict[str, Any], condition: str, seed: int, tokenizer: RobertaTokenizerFast, debug: bool, max_length: int) -> Path:
    set_seed(seed)
    train_cfg = config["training"]
    batch_size = int(train_cfg["debug_batch_size"] if debug else train_cfg["batch_size"])
    max_train_value = train_cfg["debug_max_train_batches"] if debug else train_cfg.get("max_train_batches")
    max_eval_value = train_cfg["debug_max_eval_batches"] if debug else train_cfg.get("max_eval_batches")
    max_train = int(max_train_value) if max_train_value is not None else None
    max_eval = int(max_eval_value) if max_eval_value is not None else None
    train_records = condition_records(config, condition, "train")
    dev_records = condition_records(config, condition, "dev")
    if debug:
        train_records = train_records[:256]
        dev_records = dev_records[:128]
    train_loader = DataLoader(ConditionDataset(train_records, tokenizer, max_length), batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_mlm(b, tokenizer, float(train_cfg["mlm_probability"])))
    dev_loader = DataLoader(ConditionDataset(dev_records, tokenizer, max_length), batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_mlm(b, tokenizer, float(train_cfg["mlm_probability"])))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model = make_model(config, tokenizer, max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    started = time.time()
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, float(train_cfg["grad_clip"]), max_train)
        dev_loss = run_epoch(model, dev_loader, None, device, float(train_cfg["grad_clip"]), max_eval)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})
        print(f"{condition} seed={seed} epoch={epoch}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}", flush=True)
        if bool(train_cfg.get("save_each_epoch", False)):
            epoch_dir = project_path(config["paths"]["model_dir"]) / condition / str(seed) / f"epoch_{epoch:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
    model_dir = project_path(config["paths"]["model_dir"]) / condition / str(seed)
    if debug:
        model_dir = model_dir / "debug"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    save_json(
        model_dir / "training_summary.json",
        {"condition": condition, "seed": seed, "debug": debug, "history": history, "elapsed_seconds": round(time.time() - started, 2), "max_length": max_length, "device": str(device)},
    )
    return model_dir


def selected_conditions(manifest: dict[str, Any], debug: bool) -> list[str]:
    if manifest.get("design") == "acl_aux_confirmatory_v2":
        return [
            "original",
            "AUX_target_ablation",
            "AUX_matched_content",
            "AUX_matched_function",
            "AUX_identity_shuffle",
        ]
    if manifest.get("design") == "target_random_identity_ablation_v1":
        conditions = ["original"]
        for category in manifest["train_categories"]:
            conditions.append(f"{category}_target_ablation")
            conditions.extend(sorted(name for name, info in manifest["conditions"].items() if info.get("category") == category and info.get("type") == "random_ablation"))
            conditions.append(f"{category}_identity_shuffle")
        return conditions
    if debug:
        category = manifest["train_categories"][0] if manifest["train_categories"] else "AUX"
        return ["full_input", f"{category}_masked_p100", f"{category}_matched_content_masked_p100", f"{category}_strict_content_control_p100"]
    conditions = ["full_input"]
    for category in manifest["train_categories"]:
        conditions.extend([f"{category}_masked_p100", f"{category}_matched_content_masked_p100", f"{category}_strict_content_control_p100"])
    return conditions


def candidate_surfaces_for_evaluation(config: dict[str, Any], tokenizer: RobertaTokenizerFast) -> tuple[list[str], list[dict[str, Any]]]:
    surface_map, class_to_surfaces = build_verb_maps(config)
    rng = random.Random(int(config["seed"]) + int(config["evaluation"].get("candidate_seed_offset", 20000)))
    encoded_by_class: dict[str, list[str]] = {}
    for verb_class, surfaces in class_to_surfaces.items():
        encoded_by_class[verb_class] = sorted({s for s in surfaces if tokenizer.encode(s, add_special_tokens=False)})

    class_balanced = bool(config["evaluation"].get("class_balanced_candidates", False))
    max_per_class = config["evaluation"].get("max_candidates_per_class")
    if class_balanced:
        positive_counts = [len(v) for v in encoded_by_class.values() if v]
        target_n = min(positive_counts) if positive_counts else 0
        if max_per_class is not None:
            target_n = min(target_n, int(max_per_class))
    else:
        target_n = None

    candidates: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    for verb_class, surfaces in sorted(encoded_by_class.items()):
        selected = list(surfaces)
        if target_n is not None and len(selected) > target_n:
            selected = sorted(rng.sample(selected, target_n))
        candidates.extend(selected)
        audit_rows.append(
            {
                "verb_class": verb_class,
                "available_surfaces": len(surfaces),
                "selected_surfaces": len(selected),
                "class_balanced_candidates": class_balanced,
                "max_candidates_per_class": "" if max_per_class is None else max_per_class,
                "selected_surface_list": " ".join(selected),
            }
        )
    return sorted(set(candidates)), audit_rows


def select_mvp_examples(config: dict[str, Any], max_examples: int | None, allowed_surfaces: set[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, splits, _ = load_source_records(config)
    surface_map, _ = build_verb_maps(config)
    rng = random.Random(int(config["seed"]))
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_split_records(records, splits, "test"):
        eligible = [
            i
            for i, (t, p) in enumerate(zip(record["tokens"], record["pos"]))
            if p == "VERB" and t in surface_map and (allowed_surfaces is None or t in allowed_surfaces)
        ]
        if not eligible:
            continue
        idx = rng.choice(eligible)
        target = record["tokens"][idx]
        example = {"line_id": record["line_id"], "tokens": record["tokens"], "pos": record["pos"], "target_idx": idx, "target_surface": target, "target_lemma": surface_map[target]["lemma"], "target_class": surface_map[target]["class"]}
        by_class[surface_map[target]["class"]].append(example)

    balance_examples = bool(config["evaluation"].get("balance_examples_by_class", False))
    if balance_examples:
        positive_counts = [len(v) for v in by_class.values() if v]
        per_class = min(positive_counts) if positive_counts else 0
        if max_examples:
            per_class = min(per_class, max(1, max_examples // max(len(positive_counts), 1)))
        examples = []
        for verb_class in sorted(by_class):
            candidates = list(by_class[verb_class])
            rng.shuffle(candidates)
            examples.extend(candidates[:per_class])
        rng.shuffle(examples)
    else:
        examples = []
        for verb_class in sorted(by_class):
            examples.extend(by_class[verb_class])
        examples.sort(key=lambda x: int(x["line_id"]))
        if max_examples:
            examples = examples[:max_examples]

    audit_rows = []
    selected_counts = Counter(str(e["target_class"]) for e in examples)
    available_counts = {verb_class: len(items) for verb_class, items in by_class.items()}
    for verb_class in sorted(set(available_counts) | set(selected_counts)):
        audit_rows.append(
            {
                "verb_class": verb_class,
                "available_examples": available_counts.get(verb_class, 0),
                "selected_examples": selected_counts.get(verb_class, 0),
                "balance_examples_by_class": balance_examples,
                "allowed_surface_filter": allowed_surfaces is not None,
            }
        )
    return examples, audit_rows


def score_candidates_for_example(
    model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizerFast,
    example: dict[str, Any],
    candidate_token_ids: dict[str, list[int]],
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    grouped: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for candidate, ids in candidate_token_ids.items():
        if ids:
            grouped[len(ids)].append((candidate, ids))
    scores: dict[str, float] = {}
    sum_scores: dict[str, float] = {}
    piece_counts: dict[str, int] = {}
    for piece_count, candidates in grouped.items():
        tokens = list(example["tokens"])
        tokens[int(example["target_idx"])] = " ".join([tokenizer.mask_token] * piece_count)
        encoded = tokenizer(" ".join(tokens), max_length=max_length, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        mask_positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) < piece_count:
            for candidate, _ in candidates:
                scores[candidate] = float("-inf")
                sum_scores[candidate] = float("-inf")
                piece_counts[candidate] = piece_count
            continue
        target_positions = mask_positions[:piece_count]
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            log_probs = F.log_softmax(logits, dim=-1)
        for candidate, ids in candidates:
            total_score = float(sum(float(log_probs[pos, cid].detach().cpu()) for pos, cid in zip(target_positions, ids)))
            sum_scores[candidate] = total_score
            scores[candidate] = total_score / max(len(ids), 1)
            piece_counts[candidate] = len(ids)
    return scores, sum_scores, piece_counts


def candidate_metrics(scores: dict[str, float], surface_map: dict[str, dict[str, str]], target_class: str) -> tuple[str, str, float, float]:
    pred_surface = max(scores.items(), key=lambda item: item[1])[0]
    pred_class = surface_map[pred_surface]["class"]
    grouped: dict[str, list[float]] = defaultdict(list)
    for surface, score in scores.items():
        grouped[surface_map[surface]["class"]].append(score)
    target_score = masked_mean(grouped[target_class])
    foil = [masked_mean(v) for cls, v in grouped.items() if cls != target_class]
    return pred_surface, pred_class, 1.0 if pred_class == target_class else 0.0, target_score - masked_mean(foil)


def evaluate(config: dict[str, Any], conditions: list[str], debug: bool, tokenizer: RobertaTokenizerFast, max_length: int) -> None:
    surface_map, class_to_surfaces = build_verb_maps(config)
    candidate_surfaces, candidate_audit_rows = candidate_surfaces_for_evaluation(config, tokenizer)
    candidate_token_ids = {surface: tokenizer.encode(surface, add_special_tokens=False) for surface in candidate_surfaces}
    max_examples = int(config["evaluation"]["debug_max_examples"] if debug else config["evaluation"]["max_examples"])
    examples, example_audit_rows = select_mvp_examples(config, max_examples, set(candidate_surfaces))
    audit_dir = project_path(config["paths"]["audits_dir"])
    suffix = "_debug" if debug else ""
    csv_write(audit_dir / f"candidate_set_audit{suffix}.csv", candidate_audit_rows)
    csv_write(audit_dir / f"evaluation_item_audit{suffix}.csv", example_audit_rows)
    seeds = config["training"]["debug_seeds"] if debug else config["training"]["seeds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    rows = []
    train_rows = []
    for condition in conditions:
        for seed in seeds:
            model_dir = project_path(config["paths"]["model_dir"]) / condition / str(seed)
            if debug:
                model_dir = model_dir / "debug"
            model = RobertaForMaskedLM.from_pretrained(model_dir).to(device)
            model.eval()
            summary = load_json(model_dir / "training_summary.json")
            final = summary["history"][-1]
            train_rows.append({"condition": condition, "seed": seed, "debug": debug, "epochs": len(summary["history"]), "final_train_loss": final["train_loss"], "final_dev_loss": final["dev_loss"], "elapsed_seconds": summary["elapsed_seconds"], "device": summary["device"]})
            for example_id, example in enumerate(examples):
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
                }
                row["surface_topk_accuracy"] = row["top5_surface_accuracy"]
                rows.append(row)
    suffix = "_debug" if debug else ""
    results = project_path(config["paths"]["results_dir"])
    csv_write(results / f"mvp_per_example{suffix}.csv", rows)
    csv_write(results / f"training_condition_summary{suffix}.csv", train_rows)
    summarize_metrics(rows, results / f"metrics_summary{suffix}.csv")
    paired_delta(rows, config, results / f"paired_delta_summary{suffix}.csv")
    write_error_analysis(rows, config, results / f"error_analysis{suffix}.md")
    write_decision_notes(config, rows, train_rows, debug)


def summarize_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["seed"]))].append(row)
    out = []
    for (condition, seed), items in sorted(grouped.items()):
        out.append({"condition": condition, "seed": seed, "n_examples": len(items), **{metric: masked_mean([float(x[metric]) for x in items]) for metric in METRICS}})
    csv_write(path, out)


def paired_delta(rows: list[dict[str, Any]], config: dict[str, Any], path: Path) -> None:
    if config.get("experiment_design") == "target_random_identity_ablation_v1":
        return ablation_paired_delta(rows, config, path)
    by_key = {(str(r["seed"]), int(r["example_id"]), str(r["condition"])): r for r in rows}
    categories = sorted({c.split("_", 1)[0] for c in {str(r["condition"]) for r in rows} if c.endswith("_masked_p100")})
    out = []
    for category in categories:
        fn = f"{category}_masked_p100"
        content = f"{category}_matched_content_masked_p100"
        strict = f"{category}_strict_content_control_p100"
        for seed in sorted({str(r["seed"]) for r in rows}):
            example_ids = sorted({int(r["example_id"]) for r in rows if str(r["seed"]) == seed})
            comparisons = [
                (f"{fn}_minus_full_input", fn, "full_input"),
                (f"{content}_minus_full_input", content, "full_input"),
                (f"{strict}_minus_full_input", strict, "full_input"),
                (f"{fn}_minus_matched_content_masked", fn, content),
                (f"{fn}_minus_strict_content_control", fn, strict),
            ]
            for metric in METRICS:
                for name, left, right in comparisons:
                    values = []
                    for eid in example_ids:
                        l = by_key.get((seed, eid, left))
                        r = by_key.get((seed, eid, right))
                        if l and r:
                            values.append(float(l[metric]) - float(r[metric]))
                    if values:
                        out.append({"category": category, "seed": seed, "metric": metric, "comparison": name, "n_pairs": len(values), "mean_delta": masked_mean(values)})
    csv_write(path, out)


def ablation_paired_delta(rows: list[dict[str, Any]], config: dict[str, Any], path: Path) -> None:
    """Write item-paired contrasts plus a across-random-corpus summary."""
    by_key = {(str(r["seed"]), int(r["example_id"]), str(r["condition"])): r for r in rows}
    seeds = sorted({str(r["seed"]) for r in rows})
    conditions = sorted({str(r["condition"]) for r in rows})
    out: list[dict[str, Any]] = []
    for category in config["categories"]:
        target = f"{category}_target_ablation"
        identity = f"{category}_identity_shuffle"
        random_conditions = [c for c in conditions if c.startswith(f"{category}_random_ablation_r")]
        for seed in seeds:
            example_ids = sorted({int(r["example_id"]) for r in rows if str(r["seed"]) == seed})
            for metric in METRICS:
                comparison_specs = [
                    ("original_minus_target", "original", target),
                    ("original_minus_identity_shuffle", "original", identity),
                    ("identity_shuffle_minus_target", identity, target),
                ] + [(f"target_minus_{random_condition.rsplit('_', 1)[-1]}", target, random_condition) for random_condition in random_conditions]
                random_deltas: list[float] = []
                for name, left, right in comparison_specs:
                    values = []
                    for example_id in example_ids:
                        left_row = by_key.get((seed, example_id, left))
                        right_row = by_key.get((seed, example_id, right))
                        if left_row and right_row:
                            values.append(float(left_row[metric]) - float(right_row[metric]))
                    if values:
                        mean_delta = masked_mean(values)
                        out.append({"category": category, "seed": seed, "metric": metric, "comparison": name, "n_pairs": len(values), "mean_delta": mean_delta, "random_control_count": ""})
                        if name.startswith("target_minus_r"):
                            random_deltas.append(mean_delta)
                if random_deltas:
                    mean_random = masked_mean(random_deltas)
                    variance = masked_mean([(x - mean_random) ** 2 for x in random_deltas])
                    out.append({"category": category, "seed": seed, "metric": metric, "comparison": "target_minus_random_mean", "n_pairs": len(example_ids), "mean_delta": mean_random, "random_control_count": len(random_deltas), "random_delta_std": math.sqrt(variance), "random_delta_min": min(random_deltas), "random_delta_max": max(random_deltas), "target_worse_than_random_count": sum(delta < 0 for delta in random_deltas)})
    csv_write(path, out)


def write_error_analysis(rows: list[dict[str, Any]], config: dict[str, Any], path: Path) -> None:
    by_key = {(str(r["seed"]), int(r["example_id"]), str(r["condition"])): r for r in rows}
    categories = sorted({c.split("_", 1)[0] for c in {str(r["condition"]) for r in rows} if c.endswith("_masked_p100")})
    lines = ["# Pilot_v4 RoBERTa Error Analysis", ""]
    for category in categories:
        fn = f"{category}_masked_p100"
        content = f"{category}_matched_content_masked_p100"
        strict = f"{category}_strict_content_control_p100"
        lines.extend([f"## {category}", ""])
        class_counts = Counter()
        failed_content_success = []
        failed_strict_success = []
        both_failed = []
        for key, frow in by_key.items():
            seed, eid, condition = key
            if condition != fn:
                continue
            crow = by_key.get((seed, eid, content))
            srow = by_key.get((seed, eid, strict))
            if not crow or not srow:
                continue
            if float(frow["verb_class_accuracy"]) == 0:
                class_counts[str(frow["target_class"])] += 1
                if float(crow["verb_class_accuracy"]) == 1:
                    failed_content_success.append((frow, crow))
                if float(srow["verb_class_accuracy"]) == 1:
                    failed_strict_success.append((frow, srow))
                if float(crow["verb_class_accuracy"]) == 0 and float(srow["verb_class_accuracy"]) == 0:
                    both_failed.append((frow, crow, srow))
        lines.append(f"- function failed / matched-content succeeded: {len(failed_content_success)}")
        lines.append(f"- function failed / strict-control succeeded: {len(failed_strict_success)}")
        lines.append(f"- function/content/strict all failed: {len(both_failed)}")
        lines.append(f"- failed target-class distribution: {dict(class_counts)}")
        if category == "PRON":
            mental = [x for x in failed_content_success if x[0]["target_class"] == "mental_communication"]
            lines.append(f"- PRON mental/communication function-failed-content-succeeded cases: {len(mental)}")
        for item in failed_content_success[:8]:
            frow, crow = item
            lines.append(f"- seed={frow['seed']} example={frow['example_id']} target={frow['target_surface']}/{frow['target_class']} function={frow['predicted_surface']}/{frow['predicted_class']} content={crow['predicted_surface']}/{crow['predicted_class']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision_notes(config: dict[str, Any], rows: list[dict[str, Any]], train_rows: list[dict[str, Any]], debug: bool) -> None:
    manifest = load_json(project_path(config["paths"]["conditions_dir"]) / "condition_manifest.json")
    suffix = "_debug" if debug else ""
    if manifest.get("design") == "target_random_identity_ablation_v1":
        lines = [
            f"# {config['version']} Decision Notes{suffix}", "", "## Completion", "",
            f"- Conditions: {len(manifest['conditions'])} (Original + Target/Random/Identity for AUX, DET, COMP).",
            f"- Model seeds: {config['training']['debug_seeds'] if debug else config['training']['seeds']}.",
            "- Manipulations apply to train only; dev/test are byte-equivalent to Original at the record-content level.",
            "- Primary metric: class_preference_score; all other MVP metrics are diagnostic.",
            "- PRON is excluded from the confirmatory experiment.",
            "- Noun evaluation is out of scope for this run.", "", "## Interpretation", "",
            "- Original − Target tests whether the functional-word category is useful.",
            "- Target − Random tests whether the loss exceeds equal-count general input damage.",
            "- Original − Identity shuffle tests whether stable form–context mappings matter.",
        ]
        path = project_path(config["paths"]["results_dir"]) / f"pilot_decision_notes{suffix}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    if manifest.get("design") == "acl_aux_confirmatory_v2":
        lines = [
            f"# {config['version']} Decision Notes{suffix}", "", "## Design", "",
            "- Confirmatory conditions: Original, AUX target ablation, sentence-damage-matched content ablation, token-covariate-matched function-word ablation, and AUX identity shuffle.",
            "- Every manipulation is train-only; dev and test hashes must equal Original.",
            "- The two matched controls are complementary and are not described as interchangeable random controls.",
            "- Primary metric: class preference score; verb-class accuracy is the secondary discrete metric.",
            f"- Model seeds: {config['training']['debug_seeds'] if debug else config['training']['seeds']}.",
            "", "## Interpretation guardrails", "",
            "- Target versus Original estimates total usefulness of AUX information plus any AUX-specific distribution shift.",
            "- Target versus matched controls is the specificity comparison.",
            "- Identity shuffle tests whether stable AUX-form/context mappings matter while preserving AUX unigram counts.",
            "- A claim of AUX-specific causality requires the target effect to exceed both matched controls across seeds.",
        ]
        path = project_path(config["paths"]["results_dir"]) / f"pilot_decision_notes{suffix}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    allow_reuse = bool(config.get("matching", {}).get("allow_content_reuse_within_category", True))
    lines = [f"# {config['version']} RoBERTa Decision Notes{suffix}", "", "## Completion", "", f"- Completed RoBERTa-style run: {not debug}.", f"- Model: hidden={config['roberta']['hidden_size']}, layers={config['roberta']['num_hidden_layers']}, heads={config['roberta']['num_attention_heads']}, intermediate={config['roberta']['intermediate_size']}.", f"- Training budget: epochs={config['training']['epochs']}, max_train_batches={config['training']['max_train_batches']}, max_eval_batches={config['training']['max_eval_batches']}, seeds={config['training']['seeds']}.", f"- Content-control reuse within category: {allow_reuse}.", "- MVP scoring masks only the target verb position at evaluation time; AUX/PRON/DET/COMP tokens remain present in evaluation contexts.", "- `target_verb_logprob` is subword-length-normalized; `target_verb_logprob_sum` is retained for comparison with earlier aggregation.", "- Candidate sets and evaluation items are audited for class balance in `candidate_set_audit*.csv` and `evaluation_item_audit*.csv`.", ""]
    lines.extend(["## Category Decisions", ""])
    for category, info in manifest["category_audit"].items():
        cond = manifest["conditions"][f"{category}_matched_content_masked_p100"]
        lines.append(f"- {category}: {info['decision']} | events={info['eligible_events']} train={info['train_events']} matched_mask_count={cond['mask_count']} max_balance_imbalance={info['max_balance_imbalance']:.4f} unmatched={info['unmatched_count']} reuse_max={cond['reuse_max']}")
    lines.extend(["", "## Interpretation", "", "- This run is a small RoBERTa-style Pilot from scratch, not a reuse of the Pilot_v3 word-level Transformer results.", "- Results remain Pilot evidence; do not treat them as final causal claims.", "- If content-control reuse is disabled, event-level mask counts are stricter but matching balance can degrade; no-go balance categories should be interpreted as pipeline diagnostics, not clean causal comparisons.", "- Content-control ease should be checked through final_dev_loss differences in training_condition_summary.csv.", ""])
    path = project_path(config["paths"]["results_dir"]) / f"pilot_decision_notes{suffix}.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = load_merged_config(Path(args.config))
    ensure_dirs(config)
    prepare_base_data(source_config(config))
    manifest = create_conditions_and_audits(config, force=args.force)
    tokenizer = train_tokenizer(config, force=args.force)
    tokenization_audit(config, tokenizer)
    if args.prepare_only:
        print(f"{config['version']} prepared.")
        return
    max_length = max_encoded_length(config, tokenizer)
    conditions = selected_conditions(manifest, args.debug)
    seeds = config["training"]["debug_seeds"] if args.debug else config["training"]["seeds"]
    if not args.skip_train:
        for condition in conditions:
            for seed in seeds:
                train_condition(config, condition, int(seed), tokenizer, args.debug, max_length)
    evaluate(config, conditions, args.debug, tokenizer, max_length)
    print(f"{config['version']} complete.")


if __name__ == "__main__":
    main()
