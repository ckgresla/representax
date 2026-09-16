"""Regression checks for the manuscript's derived numbers and qualifications."""

import json
from pathlib import Path
import statistics as stats
import tempfile
import unittest
from unittest.mock import patch

import report


HERE = Path(__file__).resolve().parent


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = report.load()
        cls.methods = json.loads((HERE / "methods.json").read_text())

    def test_tables_are_reproducible_from_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tables").mkdir()
            with patch.object(report, "HERE", root):
                report.framework_tables(self.e)
                report.learning_tables(self.e)
                report.scaling_table(self.e)
            generated = sorted((root / "tables").glob("*.org"))
            self.assertEqual(len(generated), 8)
            for file in generated:
                self.assertEqual(file.read_bytes(), (HERE / "tables" / file.name).read_bytes())
            self.assertEqual((root / "analysis.json").read_bytes(), (HERE / "analysis.json").read_bytes())

    def test_historical_methods_cover_exactly_the_measured_panel(self):
        key = lambda r: (r["platform"], r["recipe"], r["seed"], r["framework"])
        actual = [key(r) for r in self.methods["paired_runs"]]
        expected = [(p, r["recipe"], r["seed"], r["framework"]) for p, panel in self.e["panels"].items() for r in panel["runs"]]
        self.assertEqual(len(actual), 265)
        self.assertEqual(len(set(actual)), 265)
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(len(self.methods["sources"]), 693)
        for r in self.methods["paired_runs"]:
            if r["manifest"]:
                self.assertIn(r["manifest"], self.methods["data_manifests"])
            if r["environment"]:
                self.assertIn(r["environment"], self.methods["environments"])

    def test_omni_changes_use_each_arms_actual_initial_checkpoint(self):
        for strategy in report.STRATEGIES:
            for dataset in report.DATASETS:
                key = f"valid/{dataset}/cosine_ndcg@10"
                deltas = []
                for seed in report.SEEDS:
                    history = self.e["omni"]["runs"][f"{strategy}/seed-{seed}"]["evaluation_history"]
                    self.assertEqual(history[0]["iteration"], 0)
                    self.assertEqual(history[-1]["iteration"], 2000)
                    deltas.append(history[-1]["metrics"][key] - history[0]["metrics"][key])
                recorded = self.e["omni"]["groups"][strategy]["quality"][key]["delta"]
                self.assertAlmostEqual(stats.mean(deltas), recorded["mean"])
                self.assertAlmostEqual(stats.stdev(deltas), recorded["sample_standard_deviation"])

    def test_negative_results_and_scope_are_retained(self):
        text = (HERE / "paper.org").read_text().split("* Author Notes")[0]
        for phrase in ("0.7104 to 0.6925", "within 256 tokens", "batch one and 24 videos", "not a causal ablation", "finite-budget learning", "per-device negatives"):
            self.assertIn(phrase, text)
        self.assertNotIn("Draft:", text)
        self.assertNotIn("TODO", text)

    def test_framework_plot_does_not_clip_seed_observations(self):
        for platform in ("gpu-rtx4090", "tpu-v5e-16"):
            for recipe in report.LABELS:
                values = report.paired_rates(self.e["panels"][platform], recipe)
                self.assertTrue(all(.45 < x < 16 for x in values), (platform, recipe, values))

    def test_missing_startup_is_not_reported_as_zero(self):
        audit = json.loads((HERE / "analysis.json").read_text())
        self.assertEqual(audit["gpu/dense-retrieval"]["reference"]["first_use_seconds"], [None] * 5)
        self.assertIn("A dash means unrecorded, not zero", (HERE / "tables/gpu-startup.org").read_text())

    def test_saved_early_step_timing_covers_every_run(self):
        native, reference = [], []
        for panel in self.e["panels"].values():
            for run in panel["runs"]:
                diagnostic = report.startup_diagnostics(run)
                if run["framework"] == "reference":
                    reference.append(diagnostic)
                    self.assertGreater(diagnostic["reference_step_1_seconds"], 0)
                    self.assertGreater(diagnostic["reference_step_2_seconds"], 0)
                    self.assertIsNone(diagnostic["native_first_use_total_seconds"])
                else:
                    native.append(diagnostic)
                    self.assertGreater(diagnostic["initial_setup_seconds"], 0)
                    self.assertGreater(diagnostic["native_first_use_total_seconds"], 0)
                    self.assertIsNone(diagnostic["reference_step_1_seconds"])
        self.assertEqual(len(native), 130)
        self.assertEqual(len(reference), 135)

    def test_missing_steps_are_not_replaced_with_later_intervals(self):
        run = {
            "framework": "reference", "seed": 7,
            "resolved_directory": "test", "metrics_sha256": "test",
            "metrics": [{"event": "training_step", "iteration": 3,
                         "metrics": {"perf/step_seconds": 1.0}}],
        }
        diagnostic = report.startup_diagnostics(run)
        self.assertIsNone(diagnostic["reference_step_1_seconds"])
        self.assertIsNone(diagnostic["reference_step_2_seconds"])

    def test_native_events_and_reference_steps_stay_separate(self):
        runs = self.e["panels"]["gpu-rtx4090"]["runs"]
        run = next(r for r in runs if r["recipe"] == "late-interaction"
                   and r["framework"] == "representax" and r["seed"] == 7)
        diagnostic = report.startup_diagnostics(run)
        events = diagnostic["native_first_use_events"]
        self.assertEqual(len(events), 7)
        self.assertAlmostEqual(diagnostic["native_first_use_total_seconds"],
                               sum(row["seconds"] for row in events))
        self.assertGreater(diagnostic["native_first_use_total_seconds"],
                           events[0]["seconds"])
        self.assertEqual(report.startup_cells([], [diagnostic]), ["---"] * 4)


if __name__ == "__main__":
    unittest.main()
