from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pilot_utils import load_json, project_path
from run_pilot_v4_roberta import (
    create_conditions_and_audits,
    evaluate,
    load_merged_config,
    max_encoded_length,
    prepare_base_data,
    selected_conditions,
    source_config,
    tokenization_audit,
    train_condition,
    train_tokenizer,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "Confirmatory_ablation_v1_babylm2026.yaml"


def write_progress(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def model_complete(config: dict[str, Any], condition: str, seed: int, debug: bool = False) -> bool:
    model_dir = project_path(config["paths"]["model_dir"]) / condition / str(seed)
    if debug:
        model_dir = model_dir / "debug"
    required = [
        model_dir / "training_summary.json",
        model_dir / "config.json",
        model_dir / "model.safetensors",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def training_progress(config: dict[str, Any], conditions: list[str], seeds: list[int], current: str = "", failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    completed = [{"condition": c, "seed": s} for c in conditions for s in seeds if model_complete(config, c, s)]
    total = len(conditions) * len(seeds)
    return {
        "status": "training" if len(completed) < total else "trained",
        "total_conditions": len(conditions),
        "total_models": total,
        "completed_models": len(completed),
        "pending_models": total - len(completed),
        "completed": completed,
        "current": current,
        "failures": failures or [],
        "updated_at": time.time(),
    }


def parse_csv(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def run_resumable_evaluation(config_path: Path, resume: bool, conditions: list[str] | None = None, seeds: list[int] | None = None) -> None:
    command = [sys.executable, str(Path(__file__).with_name("run_mvp_eval_with_progress.py")), "--config", str(config_path)]
    if resume:
        command.append("--resume")
    if conditions:
        command.extend(["--conditions", ",".join(conditions)])
    if seeds:
        command.extend(["--seeds", ",".join(str(seed) for seed in seeds)])
    subprocess.run(command, check=True)
    config = load_merged_config(config_path)
    expected_conditions = {"original", "AUX_target_ablation", "AUX_matched_content", "AUX_matched_function", "AUX_identity_shuffle"}
    expected_seeds = {int(seed) for seed in config["training"]["seeds"]}
    covers_full_run = set(conditions or expected_conditions) == expected_conditions and {int(seed) for seed in (seeds or expected_seeds)} == expected_seeds
    if config.get("experiment_design") == "acl_aux_confirmatory_v2" and covers_full_run:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("summarize_confirmatory.py")), "--results-dir", str(project_path(config["paths"]["results_dir"]))],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirmatory target/random/identity ablation experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conditions", help="Comma-separated condition subset. Defaults to all generated conditions.")
    parser.add_argument("--seeds", help="Comma-separated model seed subset. Defaults to config seeds.")
    args = parser.parse_args()
    if args.train_only and args.eval_only:
        parser.error("--train-only and --eval-only are mutually exclusive")

    config_path = Path(args.config).resolve()
    config = load_merged_config(config_path)
    prepare_base_data(source_config(config))
    manifest = create_conditions_and_audits(config, force=args.force_prepare)
    tokenizer = train_tokenizer(config, force=args.force_prepare)
    tokenization_audit(config, tokenizer)
    all_conditions = selected_conditions(manifest, debug=args.smoke)
    conditions = parse_csv(args.conditions) or all_conditions
    unknown = sorted(set(conditions) - set(all_conditions))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    configured_seeds = [int(seed) for seed in (config["training"]["debug_seeds"] if args.smoke else config["training"]["seeds"])]
    seeds = [int(seed) for seed in (parse_csv(args.seeds) or configured_seeds)]
    if args.prepare_only:
        print(f"Prepared {len(conditions)} conditions: {config['paths']['conditions_dir']}")
        return

    if args.eval_only:
        run_resumable_evaluation(config_path, resume=args.resume, conditions=conditions, seeds=seeds)
        return

    max_length = max_encoded_length(config, tokenizer)
    if args.smoke:
        for condition in conditions:
            if not model_complete(config, condition, seeds[0], debug=True):
                train_condition(config, condition, seeds[0], tokenizer, True, max_length)
        evaluate(config, conditions, True, tokenizer, max_length)
        print(f"Smoke test complete for {len(conditions)} conditions.")
        return

    progress_path = project_path(config["paths"]["results_dir"]) / "progress.json"
    failures: list[dict[str, Any]] = []
    write_progress(progress_path, training_progress(config, conditions, seeds))
    for condition in conditions:
        for seed in seeds:
            if model_complete(config, condition, seed):
                continue
            current = f"{condition}/seed={seed}"
            write_progress(progress_path, training_progress(config, conditions, seeds, current, failures))
            try:
                train_condition(config, condition, seed, tokenizer, False, max_length)
            except Exception as exc:
                failures.append({"condition": condition, "seed": seed, "error": repr(exc)})
                write_progress(progress_path, training_progress(config, conditions, seeds, current, failures))
                raise
    write_progress(progress_path, training_progress(config, conditions, seeds, failures=failures))
    if not args.train_only:
        run_resumable_evaluation(config_path, resume=True, conditions=conditions, seeds=seeds)


if __name__ == "__main__":
    main()
