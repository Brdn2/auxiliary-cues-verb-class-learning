from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _source_config(config: dict[str, Any]) -> dict[str, Any]:
    source = _load_yaml(ROOT / config["source_config"])
    if source.get("base_source_config"):
        base = _load_yaml(ROOT / source["base_source_config"])
        merged = dict(base)
        merged.update({k: v for k, v in source.items() if k != "paths"})
        merged["paths"] = {**base.get("paths", {}), **source.get("paths", {})}
        source = merged
    return source


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_noise(seed: int, *parts: Any) -> float:
    value = "::".join(map(str, (seed, *parts))).encode("utf-8")
    return int(hashlib.md5(value).hexdigest()[:12], 16) / float(16**12)


def _stable_int(*parts: Any) -> int:
    return int(hashlib.md5("::".join(map(str, parts)).encode("utf-8")).hexdigest()[:12], 16)


def _nearest_verb(pos: list[str], index: int) -> int:
    distances = [abs(index - j) for j, tag in enumerate(pos) if tag == "VERB"]
    return min(distances) if distances else 99


def _relative(index: int, length: int) -> float:
    return index / max(length - 1, 1)


def _event(records: list[dict[str, Any]], record_idx: int, token_idx: int, counts: Counter[str]) -> dict[str, Any]:
    record = records[record_idx]
    token = record["tokens"][token_idx]
    return {
        "record_idx": record_idx,
        "line_id": record["line_id"],
        "token_idx": token_idx,
        "token": token,
        "upos": record["pos"][token_idx],
        "length": len(record["tokens"]),
        "relative_position": _relative(token_idx, len(record["tokens"])),
        "verb_distance": _nearest_verb(record["pos"], token_idx),
        "log_frequency": math.log1p(counts[token]),
    }


def _cost(target: dict[str, Any], control: dict[str, Any], cfg: dict[str, Any], seed: int) -> float:
    weights = cfg["ablation_design"]["matching"]
    return (
        float(weights["relative_position_weight"]) * abs(target["relative_position"] - control["relative_position"])
        + float(weights["verb_distance_weight"]) * abs(target["verb_distance"] - control["verb_distance"]) / 10.0
        + float(weights["frequency_weight"]) * abs(target["log_frequency"] - control["log_frequency"]) / 10.0
        + 1e-6 * _stable_noise(seed, target["record_idx"], target["token_idx"], control["token_idx"])
    )


def _smd(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return float("nan")
    pooled = math.sqrt((statistics.pvariance(left) + statistics.pvariance(right)) / 2.0)
    return 0.0 if pooled == 0 else (statistics.mean(left) - statistics.mean(right)) / pooled


def _write_condition(
    records: list[dict[str, Any]],
    splits: dict[str, list[int]],
    output_path: Path,
    replacements: dict[int, dict[int, str]],
) -> tuple[dict[str, str], int]:
    lookup = {int(index): split for split, indices in splits.items() for index in indices}
    digests = {split: hashlib.sha256() for split in splits}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record_idx, record in enumerate(records):
            updated = record
            if replacements.get(record_idx):
                updated = dict(record)
                updated["tokens"] = list(record["tokens"])
                for token_idx, token in replacements[record_idx].items():
                    updated["tokens"][token_idx] = token
                    changed += 1
            line = json.dumps(updated, ensure_ascii=False)
            handle.write(line + "\n")
            digests[lookup[record_idx]].update(json.dumps(updated, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n")
    return {split: digest.hexdigest() for split, digest in digests.items()}, changed


def _select_controls(
    records: list[dict[str, Any]],
    train_indices: set[int],
    targets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    targets_by_record: dict[int, list[dict[str, Any]]] = defaultdict(list)
    candidates_by_record: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in targets:
        targets_by_record[event["record_idx"]].append(event)
    for event in candidates:
        candidates_by_record[event["record_idx"]].append(event)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int]] = set()
    pending: list[dict[str, Any]] = []
    status = Counter()
    if bool(config["ablation_design"]["matching"].get("same_utterance_first", True)):
        for record_idx in sorted(targets_by_record):
            available = list(candidates_by_record.get(record_idx, []))
            for target in sorted(targets_by_record[record_idx], key=lambda x: x["token_idx"]):
                if not available:
                    pending.append(target)
                    continue
                best = min(available, key=lambda item: _cost(target, item, config, seed))
                available.remove(best)
                chosen = dict(best)
                chosen["matched_target"] = target
                chosen["match_status"] = "same_utterance"
                selected.append(chosen)
                selected_keys.add((chosen["record_idx"], chosen["token_idx"]))
                status["same_utterance"] += 1
    else:
        pending = list(targets)

    remaining = [event for event in candidates if (event["record_idx"], event["token_idx"]) not in selected_keys]
    exact_pools: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    freq_rel_pools: dict[tuple[int, int], list[int]] = defaultdict(list)
    freq_pools: dict[int, list[int]] = defaultdict(list)
    global_pool = list(range(len(remaining)))
    for index, event in enumerate(remaining):
        freq_bin = int(event["log_frequency"] * 2)
        rel_bin = min(int(event["relative_position"] * 10), 9)
        distance_bin = min(int(event["verb_distance"]), 5)
        exact_pools[(freq_bin, rel_bin, distance_bin)].append(index)
        freq_rel_pools[(freq_bin, rel_bin)].append(index)
        freq_pools[freq_bin].append(index)
    rng = random.Random(seed)
    for pools in (exact_pools, freq_rel_pools, freq_pools):
        for pool in pools.values():
            rng.shuffle(pool)
    rng.shuffle(global_pool)
    used_global: set[int] = set()

    def pop_unused(pool: list[int]) -> int | None:
        while pool and pool[-1] in used_global:
            pool.pop()
        return pool.pop() if pool else None

    for target in pending:
        target_freq = int(target["log_frequency"] * 2)
        target_rel = min(int(target["relative_position"] * 10), 9)
        target_distance = min(int(target["verb_distance"]), 5)
        best_idx = pop_unused(exact_pools[(target_freq, target_rel, target_distance)])
        match_status = "global_exact_stratum"
        if best_idx is None:
            best_idx = pop_unused(freq_rel_pools[(target_freq, target_rel)])
            match_status = "global_frequency_position"
        if best_idx is None:
            best_idx = pop_unused(freq_pools[target_freq])
            match_status = "global_frequency"
        if best_idx is None:
            best_idx = pop_unused(global_pool)
            match_status = "global_unstratified"
        if best_idx is None:
            raise RuntimeError("Insufficient unique content-control positions")
        used_global.add(best_idx)
        chosen = dict(remaining[best_idx])
        chosen["matched_target"] = target
        chosen["match_status"] = match_status
        selected.append(chosen)
        status[match_status] += 1
    if len(selected) != len(targets):
        raise AssertionError("Control and target event counts differ")
    if len({(x["record_idx"], x["token_idx"]) for x in selected}) != len(selected):
        raise AssertionError("A matched-control position was reused")
    return selected, status


def build_acl_conditions(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    conditions_dir = ROOT / config["paths"]["conditions_dir"]
    audits_dir = ROOT / config["paths"]["audits_dir"]
    manifest_path = conditions_dir / "condition_manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)

    source = _source_config(config)
    records = _read_jsonl(ROOT / source["paths"]["processed_dir"] / "utterances.jsonl")
    splits = json.loads((ROOT / source["paths"]["splits_dir"] / "splits.json").read_text(encoding="utf-8"))
    train_indices = set(map(int, splits["train"]))
    counts = Counter(token for index in train_indices for token in records[index]["tokens"])
    target_surfaces = {
        surface
        for lemmas in source["verb_lexicon"].values()
        for forms in lemmas.values()
        for surface in forms
    }
    targets = [
        _event(records, record_idx, token_idx, counts)
        for record_idx in sorted(train_indices)
        for token_idx, tag in enumerate(records[record_idx]["pos"])
        if tag == "AUX"
    ]
    control_specs = list(config["ablation_design"]["matched_controls"])
    ablation = config["special_tokens"]["ablation"]
    manifest: dict[str, Any] = {
        "version": config["version"],
        "design": "acl_aux_confirmatory_v2",
        "train_only_manipulation": True,
        "train_categories": ["AUX"],
        "target_train_events": len(targets),
        "eligible_control_events": {},
        "conditions": {},
    }

    original_path = conditions_dir / "original" / "utterances.jsonl"
    original_hashes, _ = _write_condition(records, splits, original_path, {})
    manifest["conditions"]["original"] = {
        "type": "original", "file": str(original_path.relative_to(ROOT)),
        "replacement_count": 0, "split_hashes": original_hashes, "file_hash": _sha256(original_path),
    }
    target_replacements: dict[int, dict[int, str]] = defaultdict(dict)
    for event in targets:
        target_replacements[event["record_idx"]][event["token_idx"]] = ablation
    target_path = conditions_dir / "AUX_target_ablation" / "utterances.jsonl"
    target_hashes, target_changed = _write_condition(records, splits, target_path, target_replacements)
    manifest["conditions"]["AUX_target_ablation"] = {
        "type": "target_ablation", "category": "AUX", "file": str(target_path.relative_to(ROOT)),
        "replacement_token": ablation, "replacement_count": target_changed,
        "split_hashes": target_hashes, "file_hash": _sha256(target_path),
    }

    audit_path = audits_dir / "matched_pairs.csv.gz"
    audit_fields = [
        "condition", "match_status", "target_record_idx", "target_line_id", "target_token_idx", "target_token",
        "control_record_idx", "control_line_id", "control_token_idx", "control_token", "control_upos",
        "target_length", "control_length", "target_relative_position", "control_relative_position",
        "target_verb_distance", "control_verb_distance", "target_log_frequency", "control_log_frequency",
    ]
    pair_summaries: list[dict[str, Any]] = []
    with gzip.open(audit_path, "wt", encoding="utf-8", newline="") as audit_handle:
        writer = csv.DictWriter(audit_handle, fieldnames=audit_fields)
        writer.writeheader()
        for spec in control_specs:
            name = str(spec["name"])
            seed = int(spec["seed"])
            allowed_pos = set(spec["allowed_pos"])
            candidates = [
                _event(records, record_idx, token_idx, counts)
                for record_idx in sorted(train_indices)
                for token_idx, (token, tag) in enumerate(zip(records[record_idx]["tokens"], records[record_idx]["pos"]))
                if tag in allowed_pos and not (bool(spec.get("exclude_target_verbs")) and token in target_surfaces)
            ]
            manifest["eligible_control_events"][name] = len(candidates)
            control_config = deepcopy(config)
            control_config["ablation_design"]["matching"].update(spec.get("matching_overrides", {}))
            selected, status = _select_controls(records, train_indices, targets, candidates, control_config, seed)
            replacements: dict[int, dict[int, str]] = defaultdict(dict)
            for control in selected:
                target = control["matched_target"]
                replacements[control["record_idx"]][control["token_idx"]] = ablation
                writer.writerow({
                    "condition": name, "match_status": control["match_status"],
                    "target_record_idx": target["record_idx"], "target_line_id": target["line_id"],
                    "target_token_idx": target["token_idx"], "target_token": target["token"],
                    "control_record_idx": control["record_idx"], "control_line_id": control["line_id"],
                    "control_token_idx": control["token_idx"], "control_token": control["token"], "control_upos": control["upos"],
                    "target_length": target["length"], "control_length": control["length"],
                    "target_relative_position": target["relative_position"], "control_relative_position": control["relative_position"],
                    "target_verb_distance": target["verb_distance"], "control_verb_distance": control["verb_distance"],
                    "target_log_frequency": target["log_frequency"], "control_log_frequency": control["log_frequency"],
                })
            path = conditions_dir / name / "utterances.jsonl"
            split_hashes, changed = _write_condition(records, splits, path, replacements)
            if split_hashes["dev"] != original_hashes["dev"] or split_hashes["test"] != original_hashes["test"]:
                raise AssertionError(f"{name} modified dev/test data")
            target_affected = len({x["record_idx"] for x in targets})
            control_affected = len({x["record_idx"] for x in selected})
            summary = {
                "condition": name,
                "same_utterance_rate": status["same_utterance"] / len(targets),
                "global_fallback_rate": (len(targets) - status["same_utterance"]) / len(targets),
                "match_status_counts": dict(status),
                "target_affected_utterances": target_affected,
                "control_affected_utterances": control_affected,
                "affected_utterance_ratio": control_affected / target_affected,
                "length_smd": _smd([x["length"] for x in targets], [x["length"] for x in selected]),
                "relative_position_smd": _smd([x["relative_position"] for x in targets], [x["relative_position"] for x in selected]),
                "verb_distance_smd": _smd([x["verb_distance"] for x in targets], [x["verb_distance"] for x in selected]),
                "log_frequency_smd": _smd([x["log_frequency"] for x in targets], [x["log_frequency"] for x in selected]),
                "control_pos_counts": dict(Counter(x["upos"] for x in selected)),
            }
            pair_summaries.append(summary)
            manifest["conditions"][name] = {
                "type": "matched_ablation", "category": "AUX", "corpus_seed": seed,
                "allowed_control_pos": sorted(allowed_pos),
                "file": str(path.relative_to(ROOT)), "replacement_token": ablation,
                "replacement_count": changed, "split_hashes": split_hashes, "file_hash": _sha256(path), **summary,
            }

    identity_seed = int(config["ablation_design"]["identity_shuffle_seed"])
    original_tokens = [event["token"] for event in targets]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(original_tokens):
        grouped[token].append(index)
    identity_salt = config["ablation_design"].get("identity_shuffle_salt", config["version"])
    rng = random.Random(identity_seed + _stable_int(identity_salt, "AUX"))
    for indices in grouped.values():
        rng.shuffle(indices)
    token_order = list(grouped)
    rng.shuffle(token_order)
    ordered_indices = [index for token in token_order for index in grouped[token]]
    ordered_tokens = [original_tokens[index] for index in ordered_indices]
    shift = max(Counter(original_tokens).values())
    rotated = ordered_tokens[shift:] + ordered_tokens[:shift]
    replacements_list = list(original_tokens)
    for index, token in zip(ordered_indices, rotated):
        replacements_list[index] = token
    if Counter(replacements_list) != Counter(original_tokens):
        raise AssertionError("Identity shuffle did not preserve the AUX unigram distribution")
    identity_replacements: dict[int, dict[int, str]] = defaultdict(dict)
    for event, token in zip(targets, replacements_list):
        identity_replacements[event["record_idx"]][event["token_idx"]] = token
    identity_path = conditions_dir / "AUX_identity_shuffle" / "utterances.jsonl"
    identity_hashes, identity_events = _write_condition(records, splits, identity_path, identity_replacements)
    changed = sum(a != b for a, b in zip(original_tokens, replacements_list))
    manifest["conditions"]["AUX_identity_shuffle"] = {
        "type": "identity_shuffle", "category": "AUX", "corpus_seed": identity_seed,
        "file": str(identity_path.relative_to(ROOT)), "replacement_count": identity_events,
        "changed_count": changed, "unchanged_count": len(targets) - changed,
        "split_hashes": identity_hashes, "file_hash": _sha256(identity_path),
    }
    manifest["matching_summary"] = pair_summaries
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (audits_dir / "matching_summary.json").write_text(json.dumps(pair_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config/ACL_confirmatory_aux.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = _load_yaml(Path(args.config))
    manifest = build_acl_conditions(config, args.force)
    print(json.dumps({"conditions": list(manifest["conditions"]), "target_train_events": manifest["target_train_events"]}, indent=2))


if __name__ == "__main__":
    main()
