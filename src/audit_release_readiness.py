from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from build_public_artifact import load_manifest, resolve_manifest
from summarize_nonce import validate_nonce_matrix
from update_paper_from_confirmatory import validate_complete


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = {2026, 2027, 2028, 2029, 2030}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check(name: str, passed: bool, evidence: str, required: bool = True) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "required": required, "evidence": evidence}


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def validate_matrix(
    rows: list[dict[str, str]], conditions: set[str], seeds: set[int], expected_items: int
) -> None:
    if {row["condition"] for row in rows} != conditions:
        raise RuntimeError("condition set mismatch")
    reference_ids: set[int] | None = None
    keys: list[tuple[str, int, int]] = []
    for condition in sorted(conditions):
        for seed in sorted(seeds):
            subset = [row for row in rows if row["condition"] == condition and int(row["seed"]) == seed]
            ids = {int(row["example_id"]) for row in subset}
            if len(subset) != expected_items or len(ids) != expected_items:
                raise RuntimeError(
                    f"invalid cell {condition}/seed={seed}: rows={len(subset)}, ids={len(ids)}"
                )
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise RuntimeError(f"evaluation-item mismatch for {condition}/seed={seed}")
            keys.extend((condition, seed, item_id) for item_id in ids)
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate evaluation keys")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit evidence required to call the ACL artifact ready.")
    parser.add_argument("--require-e7", action="store_true")
    parser.add_argument("--require-human-audit", action="store_true")
    parser.add_argument("--require-source-license", action="store_true")
    parser.add_argument(
        "--require-submission",
        action="store_true",
        help="Require every machine-checkable pre-submission gate, including E7, human audit, license, and checklist closure.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.require_submission:
        args.require_e7 = True
        args.require_human_audit = True
        args.require_source_license = True
    checks: list[dict[str, Any]] = []

    hash_expectations = {
        ROOT / "paper/acl.sty": "19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de",
        ROOT / "paper/acl_natbib.bst": "6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e",
        ROOT / "data/expanded_childes/childes_train2_cds_full_cleaned.txt": "1360cb99960028a3d9013a8a82ed028722611fbd94d6e1a6f58e4e81def7f749",
        ROOT / "data/expanded_childes/pos/childes_train2_cds_full_spacy_pos.txt": "c9abf03b5252565c2fa18213a8a2e09c0ef9b909d5fdd14cdfb533ec03bd5eaf",
        ROOT / "data/splits_expanded_full/splits.json": "05909372bd0245c90ff07bfe88f11a1cc5120c9b4fbe07e8a52d61026fd223be",
    }
    for path, expected in hash_expectations.items():
        observed = sha256(path) if path.exists() else "missing"
        checks.append(check(f"hash:{path.relative_to(ROOT)}", observed == expected, f"observed={observed}"))

    required_docs = [
        "README.md",
        "docs/DATA_STATEMENT.md",
        "docs/MODEL_CARD.md",
        "docs/EXPERIMENT_REGISTRY.md",
        "docs/REUSED_MODEL_PROVENANCE.md",
        "docs/ACL_STYLE_PROVENANCE.md",
        "docs/PUBLIC_ARTIFACT_POLICY.md",
        "config/public_artifact_manifest.json",
        "src/build_public_artifact.py",
        "paper/ARR_RESPONSIBLE_NLP_CHECKLIST_DRAFT.md",
        "paper/main.tex",
        "requirements.txt",
    ]
    for relative in required_docs:
        path = ROOT / relative
        checks.append(check(f"artifact:{relative}", path.exists() and path.stat().st_size > 0, display_path(path)))

    try:
        public_files = resolve_manifest(load_manifest(), require_e7=False)
        public_ok, public_evidence = True, f"allowlisted_files={len(public_files)}"
    except Exception as error:
        public_ok, public_evidence = False, repr(error)
    checks.append(check("artifact:public_allowlist_preflight", public_ok, public_evidence))

    # E0: every registered learning-curve scale must pass its frozen validity gate.
    learning_gate_path = ROOT / "results/Expanded_learning_curve_summary/learning_curve_gate.json"
    try:
        learning_gate = json.loads(learning_gate_path.read_text(encoding="utf-8"))
        e0_ok = not learning_gate.get("missing_scales") and set(learning_gate.get("gates", {})) == {"0720k", "2000k", "full"}
        e0_ok = e0_ok and all(
            item.get("evaluation_validity_gate") == "pass" for item in learning_gate["gates"].values()
        )
        e0_evidence = json.dumps(learning_gate)
    except Exception as error:
        e0_ok, e0_evidence = False, repr(error)
    checks.append(check("E0:learning_curve_valid", e0_ok, e0_evidence))

    # E1: exploratory full-scale intervention matrix, exactly five conditions x three seeds x 600 items.
    legacy_path = ROOT / "results/Expanded_learning_curve_full/mvp_per_example.csv"
    try:
        legacy_rows = read_csv(legacy_path)
        validate_matrix(
            legacy_rows,
            {"original", "AUX_target_ablation", "AUX_random_ablation_r01", "AUX_random_ablation_r02", "AUX_identity_shuffle"},
            {2026, 2027, 2028},
            600,
        )
        e1_ok, e1_evidence = True, f"rows={len(legacy_rows)}"
    except Exception as error:
        e1_ok, e1_evidence = False, repr(error)
    checks.append(check("E1:legacy_ablation_matrix", e1_ok, e1_evidence))
    legacy_stats = ROOT / "results/ACL_statistics/legacy_hierarchical_bootstrap.csv"
    checks.append(check("E1:hierarchical_statistics", legacy_stats.exists() and legacy_stats.stat().st_size > 0, display_path(legacy_stats)))

    # E2--E5: automatic annotation audit, leakage-controlled baselines/mechanism, and random initialization.
    aux_summary_path = ROOT / "audits/ACL_static_analysis_v1/aux_audit_summary.json"
    blind_sheet_path = ROOT / "audits/ACL_static_analysis_v1/aux_context_sample_for_blind_review.csv"
    try:
        aux_summary = json.loads(aux_summary_path.read_text(encoding="utf-8"))
        blind_rows = read_csv(blind_sheet_path)
        e2_ok = aux_summary.get("aux_tokens") == 935364 and aux_summary.get("sample_size") == 500 and len(blind_rows) == 500
        e2_evidence = f"aux_tokens={aux_summary.get('aux_tokens')}; blind_rows={len(blind_rows)}"
    except Exception as error:
        e2_ok, e2_evidence = False, repr(error)
    checks.append(check("E2:automatic_AUX_audit", e2_ok, e2_evidence))

    static_summary_path = ROOT / "results/ACL_static_analysis_v1/run_summary.json"
    correlations_path = ROOT / "results/ACL_static_analysis_v1/cue_predictive_correlations.csv"
    try:
        static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
        baseline_rows = read_csv(ROOT / "results/ACL_static_analysis_v1/baseline_per_example.csv")
        correlations = read_csv(correlations_path)
        expected_baselines = {
            "majority_frequency", "aux_only_nb", "pos_frame_count", "lexical_cooccurrence_nb",
            "lexical_identity_ceiling", "uniform_random",
        }
        e3_ok = static_summary.get("baseline_rows") == len(baseline_rows) == 9000
        e3_ok = e3_ok and len({row["example_id"] for row in baseline_rows}) == 600
        e3_ok = e3_ok and {row["baseline"] for row in baseline_rows} == expected_baselines
        e3_ok = e3_ok and static_summary.get("evaluation_examples") == 600
        e3_ok = e3_ok and static_summary.get("no_target_token_in_context_baselines") is True
        e5_ok = len(correlations) == 3 and {int(row["seed"]) for row in correlations} == {2026, 2027, 2028}
        static_evidence = json.dumps(static_summary)
    except Exception as error:
        e3_ok = e5_ok = False
        static_evidence = repr(error)
    checks.append(check("E3:leakage_controlled_baselines", e3_ok, static_evidence))
    checks.append(check("E5:cue_mechanism_correlations", e5_ok, f"{static_evidence}; correlations={display_path(correlations_path)}"))

    random_path = ROOT / "results/ACL_random_init/mvp_per_example.csv"
    try:
        random_rows = read_csv(random_path)
        validate_matrix(random_rows, {"random_init"}, {61001, 61002, 61003, 61004, 61005}, 600)
        e4_ok, e4_evidence = True, f"rows={len(random_rows)}"
    except Exception as error:
        e4_ok, e4_evidence = False, repr(error)
    checks.append(check("E4:random_initialization_matrix", e4_ok, e4_evidence))

    nonce_dir = ROOT / "results/ACL_nonce_cross_template_v2"
    try:
        nonce_rows = read_csv(nonce_dir / "nonce_per_example.csv")
        validate_nonce_matrix(nonce_rows)
        nonce_summary = json.loads((nonce_dir / "NONCE_SUMMARY.json").read_text(encoding="utf-8"))
        nonce_bootstrap = read_csv(nonce_dir / "hierarchical_bootstrap.csv")
        protocol = json.loads((ROOT / "audits/ACL_nonce_cross_template_v2/protocol.json").read_text(encoding="utf-8"))
        protocol_classes = [value for value in protocol.values() if isinstance(value, dict) and "nonce" in value]
        e6_ok = nonce_summary.get("status") == "complete" and nonce_summary.get("design_label") == "diagnostic"
        e6_ok = e6_ok and len(nonce_bootstrap) == 8 and all(row.get("holm_adjusted_p_upper_bound") for row in nonce_bootstrap)
        e6_ok = e6_ok and len(protocol_classes) == 5 and all(
            item.get("exact_frame_overlap") == 0 and item.get("all_exposures_contain_aux") is True
            for item in protocol_classes
        )
        e6_evidence = f"rows={len(nonce_rows)}; bootstrap_rows={len(nonce_bootstrap)}; protocol_classes={len(protocol_classes)}"
    except Exception as error:
        e6_ok, e6_evidence = False, repr(error)
    checks.append(check("E6:nonce_matrix_statistics_protocol", e6_ok, e6_evidence))

    progress_path = ROOT / "results/ACL_confirmatory_aux_v2/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    e7_trained = progress.get("completed_models") == 25 and not progress.get("failures")
    checks.append(check("E7:25_checkpoints", e7_trained, json.dumps(progress), required=args.require_e7))

    full_rows_path = ROOT / "results/ACL_confirmatory_aux_v2/mvp_per_example.csv"
    matrix_ok = False
    matrix_evidence = "missing"
    if full_rows_path.exists():
        try:
            rows = read_csv(full_rows_path)
            validate_complete(rows, EXPECTED_SEEDS, 600)
            matrix_ok = True
            matrix_evidence = f"rows={len(rows)}"
        except Exception as error:
            matrix_evidence = repr(error)
    checks.append(check("E7:aligned_evaluation_matrix", matrix_ok, matrix_evidence, required=args.require_e7))

    statistical_outputs = [
        "confirmatory_hierarchical_bootstrap.csv",
        "confirmatory_hierarchical_bootstrap.json",
        "confirmatory_seed_deltas.csv",
        "training_diagnostics.csv",
        "training_diagnostics_summary.json",
        "CONFIRMATORY_SUMMARY.md",
        "CONFIRMATORY_DECISION.json",
    ]
    for filename in statistical_outputs:
        path = ROOT / "results/ACL_confirmatory_aux_v2" / filename
        checks.append(check(f"E7:output:{filename}", path.exists() and path.stat().st_size > 0, display_path(path), required=args.require_e7))

    try:
        bootstrap_rows = read_csv(ROOT / "results/ACL_confirmatory_aux_v2/confirmatory_hierarchical_bootstrap.csv")
        seed_delta_rows = read_csv(ROOT / "results/ACL_confirmatory_aux_v2/confirmatory_seed_deltas.csv")
        diagnostics_rows = read_csv(ROOT / "results/ACL_confirmatory_aux_v2/training_diagnostics.csv")
        expected_metrics = {"class_preference_score", "verb_class_accuracy"}
        expected_contrasts = {
            "total_AUX_effect", "content_damage_effect", "function_damage_effect", "AUX_identity_effect",
            "AUX_specific_vs_content", "AUX_specific_vs_function",
        }
        bootstrap_keys = {(row["contrast_label"], row["metric"]) for row in bootstrap_rows}
        seed_delta_keys = {(row["contrast_label"], row["metric"], int(row["seed"])) for row in seed_delta_rows}
        expected_bootstrap = {(contrast, metric) for contrast in expected_contrasts for metric in expected_metrics}
        expected_deltas = {(contrast, metric, seed) for contrast, metric in expected_bootstrap for seed in EXPECTED_SEEDS}
        e7_stats_ok = len(bootstrap_rows) == len(expected_bootstrap) and bootstrap_keys == expected_bootstrap
        e7_stats_ok = e7_stats_ok and len(seed_delta_rows) == len(expected_deltas) and seed_delta_keys == expected_deltas
        e7_stats_ok = e7_stats_ok and len(diagnostics_rows) == 25
        e7_stats_ok = e7_stats_ok and {
            (row["condition"], int(row["seed"])) for row in diagnostics_rows
        } == {(condition, seed) for condition in {"original", "AUX_target_ablation", "AUX_matched_content", "AUX_matched_function", "AUX_identity_shuffle"} for seed in EXPECTED_SEEDS}
        decision = json.loads((ROOT / "results/ACL_confirmatory_aux_v2/CONFIRMATORY_DECISION.json").read_text(encoding="utf-8"))
        e7_stats_ok = e7_stats_ok and decision.get("specificity_criterion_satisfied") is False
        e7_stats_ok = e7_stats_ok and decision.get("overall_aux_effect_ci_below_zero") is True
        e7_stats_evidence = f"bootstrap={len(bootstrap_rows)}; seed_deltas={len(seed_delta_rows)}; diagnostics={len(diagnostics_rows)}"
    except Exception as error:
        e7_stats_ok, e7_stats_evidence = False, repr(error)
    checks.append(check("E7:statistical_matrices_and_decision", e7_stats_ok, e7_stats_evidence, required=args.require_e7))

    import_provenance = ROOT / "audits/ACL_confirmatory_aux_v2/MBP_E7_SIX_IMPORT_PROVENANCE.json"
    try:
        imported = json.loads(import_provenance.read_text(encoding="utf-8"))
        import_ok = imported.get("status") == "imported_with_author_accepted_runtime_heterogeneity"
        import_ok = import_ok and len(imported.get("imported_cells", [])) == 6
        import_ok = import_ok and imported.get("validation", {}).get("manifest_model_and_summary_sha256_verified") is True
        import_ok = import_ok and imported.get("overlap_handling", {}).get("selected_source") == "MBP archive"
        import_evidence = f"imported_cells={len(imported.get('imported_cells', []))}; status={imported.get('status')}"
    except Exception as error:
        import_ok, import_evidence = False, repr(error)
    checks.append(check("E7:import_provenance", import_ok, import_evidence, required=args.require_e7))

    main_text = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "paper/tables/confirmatory.tex",
            ROOT / "paper/tables/confirmatory_results_text.tex",
            ROOT / "paper/tables/confirmatory_abstract_status.tex",
        )
    )
    placeholders_absent = "running" not in generated_text and "Pending full run" not in generated_text
    checks.append(check("paper:no_E7_placeholders", placeholders_absent, generated_text[:500], required=args.require_e7))
    checks.append(check("paper:official_ACL_review_style", "\\usepackage[review]{acl}" in main_text, "paper/main.tex"))

    pdf = ROOT / "output/pdf/auxiliary_cues_novel_verb_acl_draft.pdf"
    checks.append(check("paper:compiled_pdf", pdf.exists() and pdf.stat().st_size > 50000, display_path(pdf)))
    paper_sources = [
        path for pattern in ("**/*.tex", "**/*.bib", "**/*.sty", "**/*.bst", "figures/*.pdf")
        for path in (ROOT / "paper").glob(pattern)
    ]
    latest_source_mtime = max(path.stat().st_mtime for path in paper_sources)
    pdf_fresh = pdf.exists() and pdf.stat().st_mtime >= latest_source_mtime
    checks.append(
        check(
            "paper:compiled_pdf_fresh",
            pdf_fresh,
            f"pdf_mtime={pdf.stat().st_mtime if pdf.exists() else 'missing'}; latest_source_mtime={latest_source_mtime}",
        )
    )

    human_report = ROOT / "audits/ACL_static_analysis_v1/human_annotation/results/human_aux_audit_report.json"
    human_complete = False
    human_evidence = "missing"
    if human_report.exists():
        report = json.loads(human_report.read_text(encoding="utf-8"))
        human_complete = report.get("status") == "complete"
        human_evidence = report.get("status", "unknown")
    checks.append(check("human_AUX_audit", human_complete, human_evidence, required=args.require_human_audit))

    license_record = ROOT / "docs/SOURCE_LICENSE_VERIFICATION.json"
    license_verified = False
    license_evidence = "missing"
    if license_record.exists():
        try:
            record = json.loads(license_record.read_text(encoding="utf-8"))
            required_fields = ("status", "source_artifact", "terms_or_license", "verified_by", "verified_at")
            license_verified = record.get("status") == "verified" and all(record.get(field) for field in required_fields)
            license_evidence = json.dumps(record, ensure_ascii=False)
        except Exception as error:
            license_evidence = repr(error)
    checks.append(
        check(
            "source_license_verified",
            license_verified,
            license_evidence,
            required=args.require_source_license,
        )
    )

    checklist = ROOT / "paper/ARR_RESPONSIBLE_NLP_CHECKLIST_DRAFT.md"
    checklist_text = checklist.read_text(encoding="utf-8").lower()
    open_markers = ("| partial |", "| unresolved |", "| pending ", "| awaiting authors |")
    checklist_closed = not any(marker in checklist_text for marker in open_markers)
    checks.append(
        check(
            "responsible_NLP_checklist_closed",
            checklist_closed,
            "open status markers remain" if not checklist_closed else "no open status markers",
            required=args.require_submission,
        )
    )

    required_failures = [item for item in checks if item["required"] and item["status"] != "pass"]
    submission_gate_names = {
        "E7:25_checkpoints",
        "E7:aligned_evaluation_matrix",
        "paper:no_E7_placeholders",
        "human_AUX_audit",
        "source_license_verified",
        "responsible_NLP_checklist_closed",
    }
    submission_checks = [
        item
        for item in checks
        if item["required"] or item["name"] in submission_gate_names or item["name"].startswith("E7:output:")
    ]
    submission_failures = [item for item in submission_checks if item["status"] != "pass"]
    report = {
        "status": "ready_for_selected_gate" if not required_failures else "not_ready_for_selected_gate",
        "submission_ready": not submission_failures,
        "required_failures": [item["name"] for item in required_failures],
        "submission_blockers": [item["name"] for item in submission_failures],
        "checks": checks,
    }
    output = ROOT / "results/release_readiness.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and required_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
