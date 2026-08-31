from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acl_static_analysis import classify_aux
from acl_statistics import hierarchical_paired_bootstrap
from build_public_artifact import (
    FORBIDDEN_NAME_FRAGMENTS,
    FORBIDDEN_ROOTS,
    load_manifest,
    normalize_relative,
    resolve_manifest,
)
from score_aux_human_audit import agreement
from run_mvp_eval_with_progress import validate_selected_evaluation
from summarize_nonce import holm_adjust, validate_nonce_matrix
from update_paper_from_confirmatory import specificity_decision, validate_complete


def test_aux_audit_separates_canonical_and_noise():
    assert classify_aux("could") == "canonical"
    assert classify_aux("⌈could") == "transcription_or_tokenization_artifact"
    assert classify_aux("tummyache") == "likely_mistag"


def test_hierarchical_bootstrap_preserves_paired_direction():
    rows = []
    for seed in (1, 2, 3):
        for item in range(20):
            rows.append({"condition": "original", "seed": str(seed), "example_id": str(item), "score": "1.0"})
            rows.append({"condition": "ablated", "seed": str(seed), "example_id": str(item), "score": "0.0"})
    result = hierarchical_paired_bootstrap(rows, "original", "ablated", "score", 500, 7)
    assert result["delta_comparison_minus_reference"] == -1.0
    assert result["ci95_high"] < 0


def test_paper_gate_rejects_partial_evaluation():
    rows = [
        {"condition": condition, "seed": "1", "example_id": "0"}
        for condition in (
            "original",
            "AUX_target_ablation",
            "AUX_matched_content",
            "AUX_matched_function",
            "AUX_identity_shuffle",
        )
    ]
    try:
        validate_complete(rows, {1, 2}, 1)
    except RuntimeError as error:
        assert "Incomplete seeds" in str(error)
    else:
        raise AssertionError("Partial evaluation must not pass the paper gate")


def test_eval_gate_rejects_duplicate_rows():
    rows = [
        {"condition": "original", "seed": "1", "example_id": "0"},
        {"condition": "original", "seed": "1", "example_id": "0"},
    ]
    try:
        validate_selected_evaluation(rows, ["original"], [1], 1)
    except RuntimeError as error:
        assert "Duplicate rows" in str(error)
    else:
        raise AssertionError("Duplicate evaluation rows must not pass the evaluation gate")


def test_holm_adjustment_is_monotone_and_mapped_to_input_order():
    adjusted = holm_adjust([0.2, 0.001, 0.03])
    assert adjusted == [0.2, 0.003, 0.06]


def test_nonce_gate_rejects_partial_matrix():
    try:
        validate_nonce_matrix([{"condition": "intact_exposure", "seed": "2026", "example_id": "0"}])
    except RuntimeError as error:
        assert "condition set" in str(error)
    else:
        raise AssertionError("Partial nonce output must not pass the summary gate")


def test_specificity_decision_requires_both_controls_and_four_seeds():
    contrasts = {
        ("AUX_specific_vs_content", "class_preference_score"): {"ci95_high": "-0.01"},
        ("AUX_specific_vs_function", "class_preference_score"): {"ci95_high": "-0.02"},
    }
    counts = {
        ("AUX_specific_vs_content", "class_preference_score"): 5,
        ("AUX_specific_vs_function", "class_preference_score"): 4,
    }
    assert specificity_decision(contrasts, counts)
    counts[("AUX_specific_vs_function", "class_preference_score")] = 3
    assert not specificity_decision(contrasts, counts)


def test_human_audit_agreement_reports_multiclass_and_binary_metrics():
    result = agreement(
        ["valid_aux", "mistag", "transcription_or_tokenization_artifact", "uncertain"],
        ["valid_aux", "mistag", "mistag", "uncertain"],
    )
    assert result["raw_exact_agreement"] == 0.75
    assert result["definite_binary_n"] == 3
    assert result["definite_binary_raw_agreement"] == 1.0


def test_public_artifact_allowlist_excludes_sensitive_classes():
    files = resolve_manifest(load_manifest(), require_e7=False)
    assert files
    for path in files:
        assert path.parts[0] not in FORBIDDEN_ROOTS
        assert not any(fragment in path.name.lower() for fragment in FORBIDDEN_NAME_FRAGMENTS)


def test_public_artifact_rejects_parent_traversal():
    try:
        normalize_relative("../restricted.txt")
    except RuntimeError as error:
        assert "project-relative" in str(error)
    else:
        raise AssertionError("Parent traversal must not pass the artifact allowlist")
