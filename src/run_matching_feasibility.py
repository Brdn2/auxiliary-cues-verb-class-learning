from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data_utils import build_verb_maps, load_prepared
from pilot_utils import load_config, load_json, project_path
from run_pilot_v4_roberta import (
    bin_label,
    category_indices,
    content_indices,
    event_match_key,
    make_event,
    prepare_base_data,
    split_lookup,
    stable_int,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "Pilot_v5_matching_feasibility.yaml"
RELAXATION_ORDER = [
    (None, "exact_matched"),
    ("relaxed_frequency", "relaxed_frequency"),
    ("relaxed_distance", "relaxed_distance"),
    ("relaxed_position", "relaxed_position"),
    ("split_only_fallback", "split_only_fallback"),
]
BALANCE_DIMS = [
    "token_frequency_bin",
    "nearest_verb_distance_bin",
    "utterance_length_bin",
    "relative_position_bin",
    "target_verb_class_if_applicable",
    "split",
    "source_distribution",
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


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_merged_config(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source = load_config(project_path(config["source_config"]))
    for key in ("split", "data", "verb_lexicon"):
        if key not in config:
            config[key] = source[key]
    return config


def load_source_records(config: dict[str, Any]):
    source = load_config(project_path(config["source_config"]))
    return load_prepared(source)


def decision(max_imbalance: float, unmatched_count: int, train_events: int, config: dict[str, Any]) -> str:
    if unmatched_count > 0:
        return "no_go"
    if train_events < int(config["audit"]["min_train_events_for_training"]):
        return "no_go"
    if max_imbalance <= float(config["audit"]["go_max_normalized_imbalance"]):
        return "go"
    if max_imbalance <= float(config["audit"]["caution_max_normalized_imbalance"]):
        return "caution"
    return "no_go"


def max_balance_imbalance(function_events: list[dict[str, Any]], content_events: list[dict[str, Any]]) -> float:
    if not function_events or not content_events:
        return 1.0
    category_max = 0.0
    for dim in BALANCE_DIMS:
        f_counts = Counter(str(e[dim]) for e in function_events)
        c_counts = Counter(str(e[dim]) for e in content_events)
        bins = set(f_counts) | set(c_counts)
        dim_max = 0.0
        for value in bins:
            f_prop = f_counts[value] / max(len(function_events), 1)
            c_prop = c_counts[value] / max(len(content_events), 1)
            dim_max = max(dim_max, abs(f_prop - c_prop))
        category_max = max(category_max, dim_max)
    return category_max


def build_events(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    records, splits, _ = load_source_records(config)
    surface_map, _ = build_verb_maps(config)
    counts = load_json(project_path(config["paths"]["processed_dir"]) / "vocab.json")["counts"]
    lookup = split_lookup(splits)
    function_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    content_events: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records):
        split = lookup[record_idx]
        for idx in content_indices(record, config):
            content_events.append(make_event(records, record_idx, idx, split, counts, "CONTENT", "content", surface_map, config))
        for category in config["categories"]:
            for idx in category_indices(record, category, config):
                function_by_category[category].append(make_event(records, record_idx, idx, split, counts, category, "function", surface_map, config))
    return function_by_category, content_events


def build_pools(content_events: list[dict[str, Any]]) -> dict[str, dict[tuple[str, ...], list[dict[str, Any]]]]:
    pools: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]] = {name: defaultdict(list) for _, name in RELAXATION_ORDER}
    for event in content_events:
        for relax, name in RELAXATION_ORDER:
            pools[name][event_match_key(event, relax)].append(event)
    return pools


def parse_reuse_cap(raw: Any) -> int | None:
    if str(raw).lower() == "unlimited":
        return None
    return int(raw)


def allowed_relaxations(max_level: str) -> list[tuple[str | None, str]]:
    names = [name for _, name in RELAXATION_ORDER]
    if max_level not in names:
        raise ValueError(f"Unknown relaxation level: {max_level}")
    return RELAXATION_ORDER[: names.index(max_level) + 1]


def choose_feasibility_match(
    function_event: dict[str, Any],
    pools: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]],
    reuse: Counter[tuple[int, int]],
    config: dict[str, Any],
    reuse_cap: int | None,
    max_level: str,
    cursors: dict[tuple[str, tuple[str, ...]], int] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    for relax, name in allowed_relaxations(max_level):
        candidates = pools[name].get(event_match_key(function_event, relax), [])
        if not candidates:
            continue
        pool_key = (name, event_match_key(function_event, relax))
        if cursors is not None:
            cursor = cursors.get(pool_key, 0)
            if reuse_cap == 1 and cursor >= len(candidates):
                continue
            start = cursor % len(candidates)
        else:
            start = stable_int(config["seed"], function_event["category"], function_event["line_id"], function_event["token_index"], name, reuse_cap) % len(candidates)
        for offset in range(len(candidates)):
            candidate_index = (start + offset) % len(candidates)
            candidate = candidates[candidate_index]
            key = (int(candidate["record_idx"]), int(candidate["token_index"]))
            count = reuse[key]
            if reuse_cap is not None and count >= reuse_cap:
                continue
            if cursors is not None:
                cursors[pool_key] = candidate_index + 1
            reuse[key] += 1
            return candidate, name
        if cursors is not None:
            cursors[pool_key] = start + len(candidates)
    return None, "unmatched"


def simulate_strategy(
    config: dict[str, Any],
    function_by_category: dict[str, list[dict[str, Any]]],
    pools: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]],
    reuse_cap: int | None,
    max_level: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    strategy = f"reuse_{'unlimited' if reuse_cap is None else reuse_cap}_through_{max_level}"
    for category in config["categories"]:
        function_events = function_by_category[category]
        train_events = sum(1 for event in function_events if event["split"] == "train")
        reuse: Counter[tuple[int, int]] = Counter()
        cursors: dict[tuple[str, tuple[str, ...]], int] = {}
        status_counts: Counter[str] = Counter()
        matched_events: list[dict[str, Any]] = []
        unique_content_events: dict[tuple[int, int], dict[str, Any]] = {}
        for event in function_events:
            match, status = choose_feasibility_match(event, pools, reuse, config, reuse_cap, max_level, cursors)
            status_counts[status] += 1
            if match is None:
                continue
            key = (int(match["record_idx"]), int(match["token_index"]))
            matched = dict(match)
            matched["category"] = category
            matched_events.append(matched)
            unique_content_events.setdefault(key, matched)
            detail_rows.append(
                {
                    "strategy": strategy,
                    "reuse_cap": "unlimited" if reuse_cap is None else reuse_cap,
                    "max_relaxation": max_level,
                    "category": category,
                    "function_line_id": event["line_id"],
                    "function_token_index": event["token_index"],
                    "function_token": event["token"],
                    "matched_line_id": match["line_id"],
                    "matched_token_index": match["token_index"],
                    "matched_token": match["token"],
                    "match_status": status,
                    "reuse_count_after_match": reuse[key],
                }
            )
        unmatched = status_counts["unmatched"]
        fallback = len(matched_events) - status_counts["exact_matched"]
        event_imbalance = max_balance_imbalance(function_events, matched_events)
        unique_events = list(unique_content_events.values())
        unique_imbalance = max_balance_imbalance(function_events, unique_events)
        max_reuse = max(reuse.values()) if reuse else 0
        rows.append(
            {
                "strategy": strategy,
                "reuse_cap": "unlimited" if reuse_cap is None else reuse_cap,
                "max_relaxation": max_level,
                "category": category,
                "eligible_events": len(function_events),
                "train_events": train_events,
                "matched_event_count": len(matched_events),
                "unique_content_positions": len(unique_events),
                "event_parity": round(len(matched_events) / max(len(function_events), 1), 6),
                "unique_mask_parity": round(len(unique_events) / max(len(function_events), 1), 6),
                "unmatched_count": unmatched,
                "unmatched_rate": round(unmatched / max(len(function_events), 1), 6),
                "fallback_count": fallback,
                "fallback_rate": round(fallback / max(len(function_events), 1), 6),
                "exact_rate": round(status_counts["exact_matched"] / max(len(function_events), 1), 6),
                "reuse_max": max_reuse,
                "event_weighted_max_imbalance": round(event_imbalance, 6),
                "unique_mask_max_imbalance": round(unique_imbalance, 6),
                "event_weighted_decision": decision(event_imbalance, unmatched, train_events, config),
                "unique_mask_decision": decision(unique_imbalance, unmatched, train_events, config),
                "exact_matched": status_counts["exact_matched"],
                "relaxed_frequency": status_counts["relaxed_frequency"],
                "relaxed_distance": status_counts["relaxed_distance"],
                "relaxed_position": status_counts["relaxed_position"],
                "split_only_fallback": status_counts["split_only_fallback"],
            }
        )
    return rows, detail_rows


def write_report(config: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    strict_rows = [r for r in rows if str(r["reuse_cap"]) == "1" and r["max_relaxation"] == "split_only_fallback"]
    best_unique: dict[str, dict[str, Any]] = {}
    best_event: dict[str, dict[str, Any]] = {}
    for row in rows:
        category = str(row["category"])
        if float(row["event_parity"]) == 1.0:
            current = best_event.get(category)
            if current is None or float(row["event_weighted_max_imbalance"]) < float(current["event_weighted_max_imbalance"]):
                best_event[category] = row
        if float(row["unique_mask_parity"]) == 1.0:
            current = best_unique.get(category)
            if current is None or float(row["unique_mask_max_imbalance"]) < float(current["unique_mask_max_imbalance"]):
                best_unique[category] = row

    lines = [
        f"# {config['version']} Matching Feasibility Report",
        "",
        "## Scope",
        "",
        "This report does not train a model. It tests whether matched content controls can satisfy event-level parity, unique mask-count parity, and balance before the main experiment.",
        "",
        "## Strict No-Reuse Strategy",
        "",
        "| Category | Event parity | Unique mask parity | Unique imbalance | Decision | Fallback rate |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in sorted(strict_rows, key=lambda x: str(x["category"])):
        lines.append(
            f"| {row['category']} | {row['event_parity']:.3f} | {row['unique_mask_parity']:.3f} | {row['unique_mask_max_imbalance']:.4f} | {row['unique_mask_decision']} | {row['fallback_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Best Unique-Parity Strategy",
            "",
            "| Category | Strategy | Unique imbalance | Decision | Fallback rate | Reuse max |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for category in sorted(config["categories"]):
        row = best_unique.get(category)
        if row is None:
            lines.append(f"| {category} | none |  | no_go |  |  |")
        else:
            lines.append(
                f"| {category} | {row['strategy']} | {row['unique_mask_max_imbalance']:.4f} | {row['unique_mask_decision']} | {row['fallback_rate']:.3f} | {row['reuse_max']} |"
            )

    lines.extend(
        [
            "",
            "## Best Event-Weighted Strategy",
            "",
            "| Category | Strategy | Event-weighted imbalance | Unique mask parity | Decision | Reuse max |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for category in sorted(config["categories"]):
        row = best_event.get(category)
        if row is None:
            lines.append(f"| {category} | none |  |  | no_go |  |")
        else:
            lines.append(
                f"| {category} | {row['strategy']} | {row['event_weighted_max_imbalance']:.4f} | {row['unique_mask_parity']:.3f} | {row['event_weighted_decision']} | {row['reuse_max']} |"
            )

    split_fallback_rows = [
        row
        for row in rows
        if row["max_relaxation"] == "split_only_fallback" and str(row["reuse_cap"]) in {"1", "2", "3", "5", "unlimited"}
    ]
    lines.extend(
        [
            "",
            "## Reuse-Cap Tradeoff Under Full Relaxation",
            "",
            "| Reuse cap | Category | Event decision | Event imbalance | Unique decision | Unique parity | Unique imbalance | Reuse max |",
            "|---:|---|---|---:|---|---:|---:|---:|",
        ]
    )
    cap_order = {"1": 1, "2": 2, "3": 3, "5": 5, "unlimited": 99}
    for row in sorted(split_fallback_rows, key=lambda x: (cap_order[str(x["reuse_cap"])], str(x["category"]))):
        lines.append(
            f"| {row['reuse_cap']} | {row['category']} | {row['event_weighted_decision']} | {row['event_weighted_max_imbalance']:.4f} | {row['unique_mask_decision']} | {row['unique_mask_parity']:.3f} | {row['unique_mask_max_imbalance']:.4f} | {row['reuse_max']} |"
        )

    lines.extend(
        [
            "",
            "## Main-Experiment Implication",
            "",
            "If strict unique parity is required, categories marked `go` have clean matched controls under the current corpus and annotation pipeline; `caution`/`no_go` categories still need either more corpus support, a different matching policy, or explicit qualification.",
            "",
            "If limited reuse is allowed, reuse_cap=5 gives event-weighted go/caution decisions for all four categories under full relaxation, but it still masks far fewer unique content positions than functional-word events, especially PRON. That design must be pre-specified as a reuse-weighted control rather than described as strict unique event parity.",
            "",
            "Unlimited reuse gives the best event-weighted balance for AUX/PRON/DET, but it relies on heavy reuse in high-frequency categories. It is useful as a diagnostic upper bound, not as the cleanest main-experiment control.",
            "",
            "The strongest next step is to train/evaluate only after the chosen matching policy is explicitly fixed and the annotation risk for this corpus is documented.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--write-details", action="store_true")
    args = parser.parse_args()

    config = load_merged_config(Path(args.config))
    prepare_base_data(load_config(project_path(config["source_config"])))
    results_dir = project_path(config["paths"]["results_dir"])
    audits_dir = project_path(config["paths"]["audits_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)

    function_by_category, content_events = build_events(config)
    pools = build_pools(content_events)
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for raw_cap in config["matching_feasibility"]["reuse_caps"]:
        reuse_cap = parse_reuse_cap(raw_cap)
        for max_level in config["matching_feasibility"]["max_relaxation_levels"]:
            strategy_rows, strategy_details = simulate_strategy(config, function_by_category, pools, reuse_cap, str(max_level))
            rows.extend(strategy_rows)
            if args.write_details:
                detail_rows.extend(strategy_details)

    csv_write(results_dir / "matching_feasibility_summary.csv", rows)
    if args.write_details:
        csv_write(audits_dir / "matching_feasibility_details.csv", detail_rows)
    save_json(
        results_dir / "matching_feasibility_manifest.json",
        {
            "version": config["version"],
            "source_config": config["source_config"],
            "categories": config["categories"],
            "reuse_caps": config["matching_feasibility"]["reuse_caps"],
            "max_relaxation_levels": config["matching_feasibility"]["max_relaxation_levels"],
            "n_content_events": len(content_events),
            "function_event_counts": {category: len(function_by_category[category]) for category in config["categories"]},
        },
    )
    write_report(config, rows, results_dir / "matching_feasibility_report.md")
    print(f"{config['version']} matching feasibility complete.")


if __name__ == "__main__":
    main()
