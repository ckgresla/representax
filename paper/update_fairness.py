"""Promote complete corrected five-seed pairs, preserving superseded evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

from build import HERE, PANEL_SEEDS, require, warm_rows

ROOT = Path("/raid/representax-paper/10-cross-accelerator-framework-comparison")
RECIPES = ("late-interaction", "outcome-reward", "process-reward", "audio-text", "video-text")
PLATFORMS = {"gpu": "gpu-rtx4090", "tpu": "tpu-v5e-16"}
BATCHES = {
    "gpu": dict(zip(RECIPES, (512, 128, 64, 32, 2))),
    "tpu": dict(zip(RECIPES, (512, 128, 64, 48, 112))),
}


def correction_key(platform, recipe):
    return f"fairness-20260916-{platform}-{recipe}"


def training_losses(training, summary):
    saved = [row for row in summary.get("training_metrics", ())
             if "step" in row and "loss" in row]
    by_step = {row["step"]: row["loss"] for row in saved}
    require(len(by_step) == len(saved), "Duplicate steps in summary loss history")
    losses = [row["metrics"].get("train/loss", by_step.get(row["iteration"]))
              for row in training]
    require(all(value is not None and math.isfinite(value) for value in losses),
            "Missing or nonfinite training loss")
    return losses


def collect(platform, recipe, root=ROOT):
    records, sources = [], {}

    def read(path, expected=None, jsonl=False):
        raw = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        require(expected is None or expected == digest, f"Hash mismatch: {path}")
        sources[str(path)] = {"sha256": digest.removeprefix("sha256:"), "bytes": len(raw)}
        return ([json.loads(line) for line in raw.splitlines() if line.strip()]
                if jsonl else json.loads(raw))

    base = root / f"{platform}-fairness-20260916"
    for seed in PANEL_SEEDS:
        for framework in ("representax", "reference"):
            scope = "global" if recipe in {"audio-text", "late-interaction"} else "local"
            variant = (f"representax-{scope}" if framework == "representax" and
                       recipe in {"audio-text", "video-text", "late-interaction"} else framework)
            directory = base / f"seed-{seed}" / recipe / variant
            run = read(directory / "run.json")
            require((run["platform"], run["recipe"], run["framework"], run["seed"],
                     run["steps"], run["status"]) ==
                    (platform, recipe, framework, seed, 22, "completed"),
                    f"Run identity/status mismatch: {directory}")
            require(run["source"]["working_tree_clean"], f"Unfrozen source: {directory}")
            summary = read(directory / "summary.json", run["summary_sha256"])
            metrics = read(directory / "metrics.jsonl", run["metrics_sha256"], jsonl=True)
            metrics_path = directory / "metrics.jsonl"
            if framework == "representax" and not metrics:
                # The TPU-derived launcher omitted the flat GPU native path.
                # Preserve its empty canonical file and cite the actual log.
                metrics_path = directory / "run/metrics.jsonl"
                metrics = read(metrics_path, jsonl=True)
            metrics_hash = "sha256:" + sources[str(metrics_path)]["sha256"]
            read(directory / "data-manifest.json", run["data_manifest_sha256"])
            read(directory / "environment.json")
            batch = summary.get("global_batch_size", summary.get("batch_size"))
            require(batch == BATCHES[platform][recipe], f"Wrong batch: {directory}")
            if platform == "tpu":
                read(directory / "collection.json")
                if framework == "reference":
                    require(summary.get("final_parameter_sha256") or
                            summary.get("final_adapter_sha256"),
                            f"Missing final replica verification: {directory}")
            training = [row for row in metrics if row.get("event") == "training_step"]
            require([row["iteration"] for row in training] == list(range(1, 23)),
                    f"Incorrect update coverage: {directory}")
            losses = training_losses(training, summary)
            # GPU reference timers include save work in the next interval.
            # Exclude the same update in both frameworks; keep raw logs intact.
            excluded = {1, 2, 12} if platform == "gpu" else set()
            measured = [row for row in warm_rows(metrics) if row["iteration"] not in excluded]
            require(len(measured) >= 15, f"Insufficient warm updates: {directory}")
            durations = [row["metrics"]["perf/step_seconds"] for row in measured]
            require(all(math.isfinite(t) and t > 0 for t in durations),
                    f"Invalid timing: {directory}")
            require(all(row["metrics"]["perf/examples"] == batch for row in measured),
                    f"Measured batch mismatch: {directory}")
            seconds = sum(durations)
            records.append({
                "recipe": recipe, "framework": framework, "seed": seed,
                "measured_updates": len(measured), "total_step_seconds": seconds,
                "measured_iterations": [row["iteration"] for row in measured],
                "steps_per_second": len(measured) / seconds,
                "examples_per_second": batch * len(measured) / seconds,
                "median_step_seconds": stats.median(durations),
                "first_loss": losses[0], "final_loss": losses[-1],
                "loss_history": losses,
                "summary_sha256": run["summary_sha256"], "metrics_sha256": metrics_hash,
                "metrics_path": str(metrics_path),
                "source_commit": run["source"]["commit"], "source": run["source"],
                "result_directory": str(directory), "resolved_directory": str(directory),
                "metrics": metrics, "run": run, "contract": summary.get("contract"),
                "analysis_excluded_iterations": sorted(excluded),
                "data_manifest_path": str(directory / "data-manifest.json"),
                "data_manifest_sha256": run["data_manifest_sha256"],
            })
    require(len({row["data_manifest_sha256"] for row in records}) == 1,
            "Paired data manifests differ")
    for seed in PANEL_SEEDS:
        paired = [row["measured_iterations"] for row in records if row["seed"] == seed]
        require(paired[0] == paired[1], f"Paired warm intervals differ for seed {seed}")
    return records, sources


def replace(evidence, platform, recipe, records, sources):
    key = correction_key(platform, recipe)
    require(key not in evidence.get("corrections", {}), "Correction already promoted")
    expected = {(framework, seed) for framework in ("reference", "representax") for seed in PANEL_SEEDS}
    actual = [(row["framework"], row["seed"]) for row in records]
    require(len(actual) == len(expected) and set(actual) == expected, "Incomplete paired seed panel")
    require(all(row["recipe"] == recipe for row in records), "Wrong replacement recipe")
    updated = copy.deepcopy(evidence)
    panel = updated["panels"][PLATFORMS[platform]]
    old_runs = [row for row in panel["runs"] if row["recipe"] == recipe]
    require(len(old_runs) == 10, "Expected ten superseded records")
    old_aggregate = next(row for row in panel["aggregates"] if row["recipe"] == recipe)
    aggregate = copy.deepcopy(old_aggregate)
    for framework in ("reference", "representax"):
        group = [row for row in records if row["framework"] == framework]
        for unit in ("examples", "steps"):
            values = [row[f"{unit}_per_second"] for row in group]
            for name, function in (("median", stats.median), ("mean", stats.mean), ("stdev", stats.stdev)):
                aggregate["frameworks"][framework][f"{name}_{unit}_per_second"] = function(values)
    aggregate["representax_to_reference_ratio"] = (
        aggregate["frameworks"]["representax"]["median_examples_per_second"] /
        aggregate["frameworks"]["reference"]["median_examples_per_second"])
    replacements = {(row["framework"], row["seed"]): row for row in records}
    panel["runs"] = [replacements[row["framework"], row["seed"]] if row["recipe"] == recipe else row
                     for row in panel["runs"]]
    panel["aggregates"] = [aggregate if row["recipe"] == recipe else row for row in panel["aggregates"]]
    updated.setdefault("corrections", {})[key] = {
        "reason": "Corrected paired protocol; see paper/fairness-audit.md and recorded source commits.",
        "superseded_runs": old_runs, "superseded_aggregate": old_aggregate,
        "platform": platform, "recipe": recipe,
    }
    updated["sources"].update(sources)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    records, sources = collect(args.platform, args.recipe)
    path = HERE / "evidence.json"
    updated = replace(json.loads(path.read_text()), args.platform, args.recipe, records, sources)
    if not args.check_only:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(updated, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    print(f"Validated {len(records)} runs: {args.platform} {args.recipe}")


if __name__ == "__main__":
    main()
