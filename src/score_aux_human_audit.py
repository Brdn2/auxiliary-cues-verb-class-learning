from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"valid_aux", "mistag", "transcription_or_tokenization_artifact", "uncertain"}
AUTO_TO_HUMAN = {
    "canonical": "valid_aux",
    "likely_mistag": "mistag",
    "transcription_or_tokenization_artifact": "transcription_or_tokenization_artifact",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def index_completed(rows: list[dict[str, str]], name: str, expected_ids: set[str]) -> dict[str, dict[str, str]]:
    if len(rows) != len(expected_ids):
        raise RuntimeError(f"{name}: expected {len(expected_ids)} rows, found {len(rows)}")
    indexed = {row["sample_id"]: row for row in rows}
    if set(indexed) != expected_ids or len(indexed) != len(rows):
        raise RuntimeError(f"{name}: sample IDs are missing or duplicated")
    invalid = Counter(row["review_label"].strip() for row in rows if row["review_label"].strip() not in LABELS)
    if invalid:
        raise RuntimeError(f"{name}: blank or invalid labels: {dict(invalid)}")
    return indexed


def agreement(labels_a: list[str], labels_b: list[str]) -> dict[str, Any]:
    exact = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    definite = [(a, b) for a, b in zip(labels_a, labels_b) if "uncertain" not in (a, b)]
    binary_a = ["valid" if a == "valid_aux" else "invalid" for a, _ in definite]
    binary_b = ["valid" if b == "valid_aux" else "invalid" for _, b in definite]
    return {
        "n_items": len(labels_a),
        "raw_exact_agreement": exact,
        "multiclass_cohen_kappa": float(cohen_kappa_score(labels_a, labels_b)),
        "definite_binary_n": len(definite),
        "definite_binary_raw_agreement": (
            sum(a == b for a, b in zip(binary_a, binary_b)) / len(definite) if definite else None
        ),
        "definite_binary_cohen_kappa": (
            float(cohen_kappa_score(binary_a, binary_b)) if definite else None
        ),
        "annotator_a_label_counts": dict(Counter(labels_a)),
        "annotator_b_label_counts": dict(Counter(labels_b)),
    }


def adjudication_accuracy(
    adjudicated: dict[str, dict[str, str]], key: dict[str, str], inventory: list[dict[str, str]]
) -> dict[str, Any]:
    labels = {sample_id: row["review_label"].strip() for sample_id, row in adjudicated.items()}
    exact = [labels[item] == AUTO_TO_HUMAN[auto] for item, auto in key.items()]
    by_auto: dict[str, list[bool]] = {}
    for item, auto in key.items():
        by_auto.setdefault(auto, []).append(labels[item] == AUTO_TO_HUMAN[auto])

    token_counts = {row["token"]: int(row["count"]) for row in inventory}
    strata = {"rare_1_99": [], "medium_100_9999": [], "frequent_10000_plus": []}
    for sample_id, row in adjudicated.items():
        count = token_counts[row["token"]]
        stratum = "rare_1_99" if count < 100 else "medium_100_9999" if count < 10000 else "frequent_10000_plus"
        strata[stratum].append(labels[sample_id] == AUTO_TO_HUMAN[key[sample_id]])
    return {
        "adjudicated_automatic_exact_accuracy": sum(exact) / len(exact),
        "accuracy_by_automatic_class": {
            name: {"n": len(values), "accuracy": sum(values) / len(values)}
            for name, values in by_auto.items()
        },
        "accuracy_by_token_frequency_stratum": {
            name: {"n": len(values), "accuracy": sum(values) / len(values) if values else None}
            for name, values in strata.items()
        },
        "adjudicated_label_counts": dict(Counter(labels.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and score the two-person blinded AUX audit.")
    audit_dir = ROOT / "audits/ACL_static_analysis_v1"
    parser.add_argument("--annotator-a", default=str(audit_dir / "human_annotation/annotator_A.csv"))
    parser.add_argument("--annotator-b", default=str(audit_dir / "human_annotation/annotator_B.csv"))
    parser.add_argument("--adjudicated", default=str(audit_dir / "human_annotation/adjudicated.csv"))
    parser.add_argument("--output-dir", default=str(audit_dir / "human_annotation/results"))
    args = parser.parse_args()

    key_rows = read_csv(audit_dir / "aux_context_sample_automatic_key.csv")
    key = {row["sample_id"]: row["automatic_class"] for row in key_rows}
    expected_ids = set(key)
    rows_a = read_csv(Path(args.annotator_a))
    rows_b = read_csv(Path(args.annotator_b))
    indexed_a = index_completed(rows_a, "annotator A", expected_ids)
    indexed_b = index_completed(rows_b, "annotator B", expected_ids)
    ordered_ids = sorted(expected_ids, key=int)
    labels_a = [indexed_a[item]["review_label"].strip() for item in ordered_ids]
    labels_b = [indexed_b[item]["review_label"].strip() for item in ordered_ids]
    report: dict[str, Any] = {"independent_agreement": agreement(labels_a, labels_b)}

    disagreements: list[dict[str, Any]] = []
    for item in ordered_ids:
        if indexed_a[item]["review_label"].strip() != indexed_b[item]["review_label"].strip():
            row = dict(indexed_a[item])
            row["annotator_a_label"] = indexed_a[item]["review_label"].strip()
            row["annotator_b_label"] = indexed_b[item]["review_label"].strip()
            row["review_label"] = ""
            row["review_notes"] = ""
            disagreements.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "disagreements_for_adjudication.csv",
        disagreements,
        list(rows_a[0]) + ["annotator_a_label", "annotator_b_label"],
    )

    adjudicated_path = Path(args.adjudicated)
    if adjudicated_path.exists():
        adjudicated_rows = read_csv(adjudicated_path)
        indexed_adjudicated = index_completed(adjudicated_rows, "adjudicated", expected_ids)
        report["adjudicated_audit"] = adjudication_accuracy(
            indexed_adjudicated, key, read_csv(audit_dir / "aux_inventory.csv")
        )
        report["status"] = "complete"
    else:
        report["status"] = "awaiting_adjudication"
        report["adjudication_instructions"] = (
            "Resolve all disagreements without consulting the automatic key, then create a complete "
            "500-row adjudicated.csv by copying agreed labels and adjudicated disagreement labels."
        )
    (output_dir / "human_aux_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
