"""Focused checks for paper statistics and the selected evidence inventory."""

import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

from build import HERE, describe, freeze, paired_rates, warm_rows


class StatisticsTests(unittest.TestCase):
    def test_excludes_first_use_and_retains_last_twenty_complete_intervals(self):
        rows = [
            {
                "event": "training_step",
                "iteration": i,
                "metrics": {"perf/step_seconds": 1.0},
            }
            for i in range(25)
        ]
        rows[0]["metrics"]["perf/compilation_and_first_step_seconds"] = 100.0
        rows[-1]["metrics"]["perf/excluded_from_steady_state"] = True
        rows.append({"event": "evaluation", "metrics": {"perf/step_seconds": 99}})
        selected = warm_rows(rows)
        self.assertEqual([r["iteration"] for r in selected], list(range(4, 24)))

    def test_pairing_uses_seed_not_record_order(self):
        panel = {
            "runs": [
                {
                    "recipe": "test",
                    "framework": "reference",
                    "seed": 42,
                    "examples_per_second": 20,
                },
                {
                    "recipe": "test",
                    "framework": "representax",
                    "seed": 7,
                    "examples_per_second": 30,
                },
                {
                    "recipe": "test",
                    "framework": "reference",
                    "seed": 7,
                    "examples_per_second": 10,
                },
                {
                    "recipe": "test",
                    "framework": "representax",
                    "seed": 42,
                    "examples_per_second": 40,
                },
            ]
        }
        self.assertEqual(paired_rates(panel, "test"), [3.0, 2.0])
        panel["runs"].pop()
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            paired_rates(panel, "test")

    def test_sample_sd_and_nonfinite_rejection(self):
        self.assertEqual(describe([1, 2, 3])["sd"], 1.0)
        with self.assertRaises(ValueError):
            describe([1, math.nan])

    def test_freeze_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "evidence.json"
            target.write_text("retained")
            with self.assertRaisesRegex(ValueError, "already frozen"):
                freeze(Path(directory), target)
            self.assertEqual(target.read_text(), "retained")


class FrozenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((HERE / "evidence.json").read_text())

    def test_panel_counts_and_aggregate_reconstruction(self):
        for name, panel in self.evidence["panels"].items():
            self.assertEqual(
                len(panel["runs"]), 5 if name.endswith("torchinductor") else 130
            )
            for aggregate in panel["aggregates"]:
                for framework, values in aggregate["frameworks"].items():
                    rates = [
                        r["examples_per_second"]
                        for r in panel["runs"]
                        if r["recipe"] == aggregate["recipe"]
                        and r["framework"] == framework
                    ]
                    if not rates:
                        continue
                    self.assertEqual(len(rates), 5)
                    self.assertAlmostEqual(
                        statistics.median(rates), values["median_examples_per_second"]
                    )

    def test_scaling_reconstruction(self):
        rows = self.evidence["scaling"]["rows"]
        self.assertEqual(len(rows), 12)
        for row in rows:
            baseline = next(
                r for r in rows if r["seed"] == row["seed"] and r["gpus"] == 1
            )
            self.assertAlmostEqual(
                row["tokens_per_second"] / baseline["tokens_per_second"], row["speedup"]
            )
            self.assertAlmostEqual(row["speedup"] / row["gpus"], row["efficiency"])

    def test_negative_and_transfer_evidence_not_hidden(self):
        late = self.evidence["learning"]["Late interaction / NanoMSMARCO"]
        self.assertLess(late["final"]["mean"], late["initial"]["mean"])
        self.assertEqual(
            set(self.evidence["transfer_final_only"]),
            {"trec-dl-2019", "natural-questions"},
        )


if __name__ == "__main__":
    unittest.main()
