"""Check the scope and numerical reconstruction of the padding correction."""

import copy
import json
import statistics
import unittest

from build import HERE, PANEL_SEEDS, warm_rows
from update_process_reward import CORRECTION, replace_runs


class CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((HERE / "evidence.json").read_text())
        later = cls.evidence.get("corrections", {}).get(
            "fairness-20260916-gpu-process-reward")
        if later:
            # Test the earlier padding-only correction against its own snapshot.
            panel = cls.evidence["panels"]["gpu-rtx4090"]
            old = {(r["framework"], r["seed"]): r for r in later["superseded_runs"]}
            panel["runs"] = [old[r["framework"], r["seed"]]
                             if r["recipe"] == "process-reward" else r
                             for r in panel["runs"]]
            panel["aggregates"] = [later["superseded_aggregate"]
                                   if r["recipe"] == "process-reward" else r
                                   for r in panel["aggregates"]]

    def test_correction_reconstructs_and_changes_only_five_cells(self):
        current = self.evidence
        record = current["corrections"][CORRECTION]
        before = copy.deepcopy(current)
        del before["corrections"][CORRECTION]
        replacements = [
            row
            for row in before["panels"]["gpu-rtx4090"]["runs"]
            if (row["recipe"], row["framework"]) == ("process-reward", "reference")
        ]
        old = {row["seed"]: row for row in record["superseded_reference_runs"]}
        panel = before["panels"]["gpu-rtx4090"]
        panel["runs"] = [
            old[row["seed"]] if row in replacements else row for row in panel["runs"]
        ]
        panel["aggregates"] = [
            record["superseded_aggregate"] if row["recipe"] == "process-reward" else row
            for row in panel["aggregates"]
        ]
        reconstructed = replace_runs(before, replacements, record["provenance"])
        self.assertEqual(reconstructed, current)
        with self.assertRaisesRegex(ValueError, "already applied"):
            replace_runs(current, replacements, record["provenance"])

    def test_corrected_rates_shapes_and_resume_window(self):
        panel = self.evidence["panels"]["gpu-rtx4090"]
        records = [
            row
            for row in panel["runs"]
            if row["recipe"] == "process-reward" and row["framework"] == "reference"
        ]
        self.assertEqual({row["seed"] for row in records}, set(PANEL_SEEDS))
        for row in records:
            excluded = [
                x["iteration"]
                for x in row["metrics"]
                if x["metrics"]["perf/excluded_from_steady_state"]
            ]
            self.assertEqual(excluded, [1, 12])
            warm = warm_rows(row["metrics"])
            self.assertEqual(len(warm), 20)
            self.assertAlmostEqual(
                row["examples_per_second"],
                1280 / sum(x["metrics"]["perf/step_seconds"] for x in warm),
            )
        rate = statistics.median(row["examples_per_second"] for row in records)
        self.assertAlmostEqual(rate, 16.946012039863685)
        aggregate = next(
            row for row in panel["aggregates"] if row["recipe"] == "process-reward"
        )
        self.assertAlmostEqual(
            aggregate["representax_to_reference_ratio"], 3.3181076066264676
        )
        self.assertEqual(
            self.evidence["corrections"][CORRECTION]["provenance"]["launch"][
                "padding_length"
            ],
            256,
        )


if __name__ == "__main__":
    unittest.main()
