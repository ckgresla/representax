"""Provenance checks for the small, training-free design-diagnostics capture."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import collect_design


class DesignCaptureTests(unittest.TestCase):
    def test_capture_projects_fields_and_preserves_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run"
            source.mkdir()
            raw = json.dumps({"seed": 17, "steps": 30, "batch_size": 128,
                              "steady_state_examples_per_second": 42.5,
                              "private_path": "/private/not-for-export"}).encode()
            (source / "report.json").write_bytes(raw)
            with patch.object(collect_design, "SOURCES", {"test": "run"}):
                result = collect_design.collect(root)
            row = result["rows"]["test"]
            self.assertEqual(row["source"], "run/report.json")
            self.assertEqual(row["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(row["report"]["steady_state_examples_per_second"], 42.5)
            self.assertIsNone(row["report"]["compilation_and_first_use_seconds"])
            self.assertNotIn("private_path", row["report"])

    def test_wrong_measurement_contract_is_rejected(self):
        for report in ({"seed": 42, "steps": 30}, {"seed": 17, "steps": 100}):
            with self.subTest(report=report), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "report.json").write_text(json.dumps(report))
                with patch.object(collect_design, "SOURCES", {"test": "."}):
                    with self.assertRaisesRegex(ValueError, "Unexpected diagnostic contract"):
                        collect_design.collect(root)

    def test_cli_refuses_to_replace_frozen_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("existing evidence\n")
            result = subprocess.run(
                [sys.executable, collect_design.__file__, directory, "--output", str(output)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Evidence already exists", result.stderr)
            self.assertEqual(output.read_text(), "existing evidence\n")


if __name__ == "__main__":
    unittest.main()
