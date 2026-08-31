from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from run_pilot_v4_roberta import METRICS, ablation_paired_delta, selected_conditions  # noqa: E402
from run_mvp_eval_with_progress import read_existing_keys  # noqa: E402


class ConfirmatoryAblationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = PROJECT / "data/conditions/Confirmatory_ablation_v1_babylm2026/condition_manifest.json"
        if not cls.manifest_path.exists():
            raise unittest.SkipTest("Run --prepare-only before artifact tests")
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_condition_inventory_excludes_pron(self) -> None:
        conditions = selected_conditions(self.manifest, debug=False)
        self.assertEqual(len(conditions), 22)
        self.assertNotIn("PRON", self.manifest["train_categories"])
        self.assertFalse(any(name.startswith("PRON_") for name in conditions))

    def test_target_random_counts_and_unique_positions(self) -> None:
        audit_path = PROJECT / "audits/Confirmatory_ablation_v1_babylm2026/ablation_position_audit.csv"
        with audit_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for category in self.manifest["train_categories"]:
            target = self.manifest["conditions"][f"{category}_target_ablation"]["replacement_count"]
            for index in range(1, 6):
                name = f"{category}_random_ablation_r{index:02d}"
                info = self.manifest["conditions"][name]
                self.assertEqual(info["replacement_count"], target)
                selected = [(row["record_idx"], row["token_index"]) for row in rows if row["condition"] == name]
                self.assertEqual(len(selected), len(set(selected)))

    def test_identity_shuffle_frequency_and_split_hashes(self) -> None:
        dev_hashes = set()
        test_hashes = set()
        for info in self.manifest["conditions"].values():
            dev_hashes.add(info["split_hashes"]["dev"])
            test_hashes.add(info["split_hashes"]["test"])
        self.assertEqual(len(dev_hashes), 1)
        self.assertEqual(len(test_hashes), 1)
        for category in self.manifest["train_categories"]:
            info = self.manifest["conditions"][f"{category}_identity_shuffle"]
            self.assertEqual(Counter(info["original_token_counts"]), Counter(info["shuffled_token_counts"]))
            self.assertEqual(info["event_count"], info["changed_count"] + info["unchanged_count"])


class ConfirmatoryComparisonTests(unittest.TestCase):
    def test_comparison_names_and_random_summary(self) -> None:
        conditions = ["original", "AUX_target_ablation", "AUX_identity_shuffle"] + [f"AUX_random_ablation_r{i:02d}" for i in range(1, 6)]
        rows = []
        values = {condition: float(index) for index, condition in enumerate(conditions)}
        for condition in conditions:
            for example_id in (0, 1):
                row = {"condition": condition, "seed": 1, "example_id": example_id}
                row.update({metric: values[condition] for metric in METRICS})
                rows.append(row)
        config = {"experiment_design": "target_random_identity_ablation_v1", "categories": ["AUX"]}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "paired.csv"
            ablation_paired_delta(rows, config, output)
            with output.open(encoding="utf-8", newline="") as handle:
                result = list(csv.DictReader(handle))
        names = {row["comparison"] for row in result}
        self.assertTrue({"original_minus_target", "original_minus_identity_shuffle", "identity_shuffle_minus_target", "target_minus_random_mean"}.issubset(names))
        self.assertTrue(all(f"target_minus_r{i:02d}" in names for i in range(1, 6)))

    def test_resume_keys_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mvp.csv"
            path.write_text("condition,seed,example_id\noriginal,2026,0\noriginal,2026,0\noriginal,2026,1\n", encoding="utf-8")
            self.assertEqual(read_existing_keys(path), {("original", "2026", 0), ("original", "2026", 1)})


if __name__ == "__main__":
    unittest.main()
