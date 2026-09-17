"""Run and collect the approved TPU replacement panel before its hard deadline."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


PROJECT = "project-3fed1b4b-1d3e-4ddf-bf8"
ZONE = "us-central1-a"
NODE = "representax-video-v5e16-20260917"
DEADLINE = 1789663500  # 2026-09-17 16:45:00 UTC; additional $14 authorization.
OUTPUT = Path("/raid/representax-paper/10-cross-accelerator-framework-comparison/tpu-fairness-20260916")
RECIPES = ("process-reward", "outcome-reward", "late-interaction", "audio-text", "video-text")
SEEDS = (7, 42, 773, 1234, 2026)
ENV = {**os.environ, "CLOUDSDK_CONFIG": "/home/ckg/.config/gcloud-representax-tpu"}


def cloud(action, *arguments, capture=False):
    command = ["gcloud", "alpha", "compute", "tpus", "tpu-vm", action, *arguments,
               f"--project={PROJECT}", f"--zone={ZONE}", "--quiet"]
    return subprocess.run(command, env=ENV, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def ssh(command, *, capture=False):
    return cloud("ssh", f"ckg@{NODE}", "--worker=all", "--batch-size=4",
                 f"--command={command}", capture=capture)


def identity(recipe, framework, seed):
    scope = "global" if recipe in {"late-interaction", "audio-text"} else "local"
    variant = f"representax-{scope}" if framework == "representax" and recipe in {
        "late-interaction", "audio-text", "video-text"} else framework
    relative = Path(f"seed-{seed}") / recipe / variant
    phase = "paired-sync" if recipe == "process-reward" and framework == "reference" and seed == 7 else "paired"
    if recipe == "outcome-reward" and framework == "reference" and seed == 7:
        phase = "paired-parameter-gradients"
    if recipe == "late-interaction" and framework == "reference":
        phase = "paired-pinned-collectives"
    return relative, phase


def validate(directory):
    run = json.loads((directory / "run.json").read_text())
    if run["status"] != "completed" or run["steps"] != 22:
        raise RuntimeError(f"Incomplete cell: {directory}")
    if not run["source"]["working_tree_clean"]:
        raise RuntimeError(f"Dirty source: {directory}")
    for name, extension in (("metrics", "jsonl"), ("summary", "json")):
        digest = "sha256:" + hashlib.sha256((directory / f"{name}.{extension}").read_bytes()).hexdigest()
        if digest != run[f"{name}_sha256"]:
            raise RuntimeError(f"Invalid {name} hash: {directory}")
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("losses") and not all(math.isfinite(float(x)) for x in summary["losses"]):
        raise RuntimeError(f"Nonfinite loss: {directory}")
    if run["framework"] == "reference" and not (
        summary.get("final_parameter_sha256") or summary.get("final_adapter_sha256")
    ):
        raise RuntimeError(f"Missing final replica check: {directory}")
    return run


def collect(recipe, framework, seed):
    relative, phase = identity(recipe, framework, seed)
    archive_name = f"{recipe}-{framework}-{seed}.tgz"
    remote_member = Path(phase) / relative
    ssh(f"tar -czf ~/{archive_name} -C ~/representax-fairness-results {shlex.quote(str(remote_member))}")
    checksums = ssh(f"sha256sum ~/{archive_name}", capture=True)
    expected = Counter(re.findall(r"^([0-9a-f]{64})\s", checksums, re.MULTILINE))
    observed = Counter()
    candidates = []
    for worker in range(4):
        raw = OUTPUT / "workers" / f"worker-{worker}"
        raw.mkdir(parents=True, exist_ok=True)
        archive = raw / archive_name
        cloud("scp", f"ckg@{NODE}:~/{archive_name}", str(archive), f"--worker={worker}")
        observed[hashlib.sha256(archive.read_bytes()).hexdigest()] += 1
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(raw)], check=True)
        cell = raw / remote_member
        if (cell / "summary.json").exists():
            validate(cell)
            candidates.append(cell)
    if expected != observed or sum(expected.values()) != 4:
        raise RuntimeError("Worker archive checksums differ")
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one rank-zero result; found {candidates}")
    destination = OUTPUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidates[0], destination)
    validate(destination)
    (destination / "collection.json").write_text(json.dumps({
        "remote_phase": phase, "source_worker_directory": str(candidates[0]),
        "allocation": {"project": PROJECT, "zone": ZONE, "node": NODE},
        "archive_sha256": sorted(observed.elements()), "collected_at_unix": time.time(),
    }, indent=2) + "\n")
    print(f"collected {recipe} {framework} seed={seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--framework", choices=("reference", "representax"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    for recipe in (args.recipe,) if args.recipe else RECIPES:
        for seed in (args.seed,) if args.seed else SEEDS:
            for framework in (args.framework,) if args.framework else ("reference", "representax"):
                relative, phase = identity(recipe, framework, seed)
                if (OUTPUT / relative / "run.json").exists():
                    validate(OUTPUT / relative)
                    continue
                if time.time() >= DEADLINE - 300:
                    print("Stopping before the approved TPU shutdown deadline", flush=True)
                    return
                if not args.collect_only:
                    print(f"starting {recipe} {framework} seed={seed}", flush=True)
                    ssh(f"bash ~/fairness_tpu_rerun.sh {recipe} {framework} {seed} 22 {phase}")
                collect(recipe, framework, seed)


if __name__ == "__main__":
    main()
