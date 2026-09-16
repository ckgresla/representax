"""Promote the verified, matched-padding GPU reward rerun into paper evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics as stats
from pathlib import Path

from build import HERE, PANEL_SEEDS, require, warm_rows

CORRECTION = "process-reward-gpu-padding256"
ROOT = Path(
    "/raid/representax-paper/10-cross-accelerator-framework-comparison/"
    "gpu-rtx4090-process-reward-padding256"
)


def replace_runs(evidence, replacements, provenance):
    """Replace only the five reference cells, retaining their original records."""
    corrected = copy.deepcopy(evidence)
    require(len(replacements) == 5, "Expected five corrected runs")
    require({row["seed"] for row in replacements} == set(PANEL_SEEDS), "Seed mismatch")
    for row in replacements:
        require(
            (row["recipe"], row["framework"]) == ("process-reward", "reference"),
            "Correction must only replace process-reward references",
        )
    require(
        CORRECTION not in corrected.get("corrections", {}), "Correction already applied"
    )
    panel = corrected["panels"]["gpu-rtx4090"]
    previous = [
        row
        for row in panel["runs"]
        if (row["recipe"], row["framework"]) == ("process-reward", "reference")
    ]
    require(len(previous) == 5, "Expected five original references")
    by_seed = {row["seed"]: row for row in replacements}
    panel["runs"] = [
        by_seed[row["seed"]] if row in previous else row for row in panel["runs"]
    ]
    aggregate = next(
        row for row in panel["aggregates"] if row["recipe"] == "process-reward"
    )
    old_aggregate = copy.deepcopy(aggregate)
    reference = aggregate["frameworks"]["reference"]
    for unit, field in (
        ("examples", "examples_per_second"),
        ("steps", "steps_per_second"),
    ):
        values = [row[field] for row in replacements]
        for name, function in (
            ("median", stats.median),
            ("mean", stats.mean),
            ("stdev", stats.stdev),
        ):
            reference[f"{name}_{unit}_per_second"] = function(values)
    aggregate["representax_to_reference_ratio"] = (
        aggregate["frameworks"]["representax"]["median_examples_per_second"]
        / reference["median_examples_per_second"]
    )
    corrected.setdefault("corrections", {})[CORRECTION] = {
        "reason": "Original GPU reference padded to 2048; native used 256. "
        "Rerun reference with padding 256, retaining science and data. "
        "Exclude first-use steps 1 and 12 around checkpoint/resume.",
        "superseded_reference_runs": previous,
        "superseded_aggregate": old_aggregate,
        "provenance": provenance,
    }
    return corrected


def collect(evidence, root=ROOT):
    sources = {}

    def read(path, *, expected=None, jsonl=False):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(
            expected is None or expected == "sha256:" + digest, f"Hash mismatch: {path}"
        )
        sources[str(path)] = {"sha256": digest, "bytes": len(raw)}
        return (
            [json.loads(line) for line in raw.splitlines() if line.strip()]
            if jsonl
            else json.loads(raw)
        )

    launch = read(root / "launch.json")
    require(
        (
            launch["padding_length"],
            launch["global_batch"],
            launch["micro_batch"],
            launch["total_steps"],
            launch["excluded_steps"],
        )
        == (256, 64, 2, 22, [1, 12]),
        "Unexpected correction configuration",
    )
    for relative, expected in launch["source_hashes"].items():
        path = root / "source" / relative
        raw = path.read_bytes()
        require(
            "sha256:" + hashlib.sha256(raw).hexdigest() == expected,
            f"Source mismatch: {path}",
        )
    assets = Path("/raid/representax-paper-assets")
    for name, expected in launch["data_hashes"].items():
        read(
            assets / "process-data" / name,
            expected=expected,
            jsonl=name.endswith("jsonl"),
        )
    records = []
    for seed in PANEL_SEEDS:
        directory = root / f"seed-{seed}" / "reference"
        run = read(directory / "run.json")
        require(
            run["status"] == "complete" and run["exit_code"] == 0,
            f"Incomplete seed {seed}",
        )
        summary = read(directory / "summary.json", expected=run["summary_sha256"])
        metrics = read(
            directory / "metrics.jsonl", expected=run["metrics_sha256"], jsonl=True
        )
        inputs = read(directory / "run/input-audit.json")
        require(
            summary["observed_training_shapes"] == [[2, 256]], "Wrong execution shape"
        )
        require(
            summary["steps"] == 22 and summary["excluded_steps"] == [1, 12],
            "Wrong schedule",
        )
        require(
            summary["batch_size"] == 64 and summary["micro_batch_size"] == 2,
            "Wrong batch",
        )
        require(
            summary["resumed"] and summary["reload_probe_exact"], "Lifecycle failed"
        )
        native = next(
            row
            for row in evidence["panels"]["gpu-rtx4090"]["runs"]
            if (row["recipe"], row["framework"], row["seed"])
            == ("process-reward", "representax", seed)
        )
        native_metrics = [
            row["metrics"]
            for row in native["metrics"]
            if "perf/tokens" in row.get("metrics", {})
        ]
        counts = inputs["train_token_counts"]
        require(len(native_metrics) == 22, "Native update count differs")
        for step, row in enumerate(native_metrics):
            require(row["perf/token_capacity"] == 64 * 256, "Native shape differs")
            require(
                row["perf/tokens"] == sum(counts[step * 64 : (step + 1) * 64]),
                "Native batch order differs",
            )
        measured = warm_rows(metrics)
        require(
            len(metrics) == 22 and len(measured) == 20, "Wrong measured interval count"
        )
        durations = [row["metrics"]["perf/step_seconds"] for row in measured]
        require(all(math.isfinite(x) and x > 0 for x in durations), "Invalid duration")
        rate = sum(row["metrics"]["perf/examples"] for row in measured) / sum(durations)
        require(
            math.isclose(
                rate, summary["steady_state"]["examples_per_second"], rel_tol=1e-12
            ),
            "Rate mismatch",
        )
        records.append(
            {
                "recipe": "process-reward",
                "framework": "reference",
                "seed": seed,
                "measured_updates": 20,
                "total_step_seconds": sum(durations),
                "steps_per_second": 20 / sum(durations),
                "examples_per_second": rate,
                "median_step_seconds": stats.median(durations),
                "first_loss": summary["losses"][0],
                "final_loss": summary["losses"][-1],
                "summary_sha256": run["summary_sha256"],
                "metrics_sha256": run["metrics_sha256"],
                "source_commit": launch["git"]["commit"],
                "result_directory": str(directory),
                "resolved_directory": str(directory),
                "metrics": metrics,
                "run": run,
                "contract": None,
                "source": launch["git"],
                "data_manifest_path": str(assets / "process-data/manifest.json"),
                "data_manifest_sha256": launch["data_hashes"]["manifest.json"],
            }
        )
    result = replace_runs(
        evidence, records, {"launch": launch, "source_directory": str(root)}
    )
    result["sources"].update(sources)
    return result


if __name__ == "__main__":
    path = HERE / "evidence.json"
    result = collect(json.loads(path.read_text()))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(
        "Promoted five corrected references; "
        "original measurements retained in corrections."
    )
