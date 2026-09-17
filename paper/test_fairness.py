"""Small integrity tests for replacement evidence, without training models."""

import copy
import unittest

from build import PANEL_SEEDS
from update_fairness import correction_key, replace


class FairnessEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"recipe": "process-reward", "framework": framework, "seed": seed,
             "examples_per_second": 64 * rate, "steps_per_second": rate}
            for framework, rate in (("reference", 1), ("representax", 2))
            for seed in PANEL_SEEDS
        ]
        aggregate = {"recipe": "process-reward", "frameworks": {
            "reference": {}, "representax": {}}}
        self.evidence = {"panels": {"gpu-rtx4090": {
            "runs": copy.deepcopy(self.records), "aggregates": [aggregate]}}, "sources": {}}

    def test_complete_pair_is_replaced_and_old_evidence_preserved(self):
        before = copy.deepcopy(self.evidence)
        result = replace(self.evidence, "gpu", "process-reward", self.records, {})
        self.assertEqual(self.evidence, before)
        key = correction_key("gpu", "process-reward")
        self.assertEqual(result["corrections"][key]["superseded_runs"],
                         before["panels"]["gpu-rtx4090"]["runs"])
        self.assertEqual(result["panels"]["gpu-rtx4090"]["aggregates"][0]
                         ["representax_to_reference_ratio"], 2)
        with self.assertRaisesRegex(ValueError, "already promoted"):
            replace(result, "gpu", "process-reward", self.records, {})

    def test_missing_or_duplicate_seed_fails_closed(self):
        for records in (self.records[:-1], self.records[:-1] + [self.records[0]]):
            with self.assertRaisesRegex(ValueError, "Incomplete paired"):
                replace(self.evidence, "gpu", "process-reward", records, {})

    def test_different_recipe_cannot_be_substituted(self):
        records = copy.deepcopy(self.records)
        records[0]["recipe"] = "outcome-reward"
        with self.assertRaisesRegex(ValueError, "Wrong replacement recipe"):
            replace(self.evidence, "gpu", "process-reward", records, {})


if __name__ == "__main__":
    unittest.main()
