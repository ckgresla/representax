"""Resume the approved corrected GPU panel using the frozen experiment runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


CHECKOUT = Path("/raid/representax-paper/checkouts/fairness-tpu-3491fb0")
REVISION = "23a61b0221d6d05651df4eac08671062e8572aaf"
RECOVERY_CHECKOUT = Path("/raid/representax-paper/checkouts/fairness-gpu-recovery-20260916")
RECOVERY_REVISION = "026e80f182a2f618dc4850e6c79132b1293e6575"
CAPACITY_CHECKOUT = Path("/raid/representax-paper/checkouts/fairness-capacity-20260916")
CAPACITY_REVISION = "cdcb1047a4fb269fc994b110ff532ae1e38b3b72"
ENVIRONMENTS = Path("/raid/representax-paper/checkouts/gpu-sweep-fbe44b1/experiments")
OUTPUT = Path("/raid/representax-paper/10-cross-accelerator-framework-comparison/gpu-fairness-20260916")
ASSETS = Path("/raid/representax-paper-assets")
SEEDS = (7, 42, 773, 1234, 2026)
RECIPES = {0: ("late-interaction",), 1: ("outcome-reward",),
           2: ("process-reward", "video-text"), 3: ("audio-text",)}


def completed(directory):
    path = directory / "run.json"
    if not path.exists():
        return False
    deadline = time.monotonic() + 2400
    while True:
        run = json.loads(path.read_text())
        if run["status"] != "running":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Existing run did not finish: {directory}")
        time.sleep(15)
    if run["status"] != "completed":
        raise RuntimeError(f"Inspect failed run before retrying: {directory}")
    if run["source"]["commit"] not in {REVISION, RECOVERY_REVISION, CAPACITY_REVISION} or not run["source"]["working_tree_clean"]:
        raise RuntimeError(f"Unexpected source: {directory}")
    for name in ("metrics", "summary"):
        artifact = directory / f"{name}.{'jsonl' if name == 'metrics' else 'json'}"
        actual = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != run[f"{name}_sha256"]:
            raise RuntimeError(f"Artifact hash mismatch: {artifact}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=tuple(RECIPES), required=True)
    gpu = parser.parse_args().gpu
    for recipe in RECIPES[gpu]:
        checkout, expected_revision = CHECKOUT, REVISION
        if recipe == "outcome-reward":
            checkout, expected_revision = RECOVERY_CHECKOUT, RECOVERY_REVISION
        elif recipe in {"late-interaction", "audio-text", "video-text"}:
            checkout, expected_revision = CAPACITY_CHECKOUT, CAPACITY_REVISION
        scope = "global" if recipe in {"late-interaction", "audio-text"} else "local"
        for seed in SEEDS:
            for framework in ("reference", "representax"):
                variant = (f"representax-{scope}" if framework == "representax"
                           and recipe in {"late-interaction", "audio-text", "video-text"} else framework)
                root = OUTPUT / f"seed-{seed}"
                destination = root / recipe / variant
                if completed(destination):
                    print(f"verified {recipe} {framework} seed={seed}", flush=True)
                    continue
                revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
                dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True)
                if revision != expected_revision or dirty:
                    raise RuntimeError("Frozen source checkout changed")
                python = ENVIRONMENTS / (".venv-late-interaction" if framework == "reference"
                                          and recipe == "late-interaction" else ".venv") / "bin/python"
                env = {**os.environ, "PYTHONPATH": f"{checkout}/src:{checkout}",
                       "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
                       "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                       "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                       "JAX_COMPILATION_CACHE_DIR": str(OUTPUT.parent / "gpu-rtx4090/caches/jax" / recipe)}
                command = [str(python), "experiments/10-cross-accelerator-framework-comparison/run.py",
                           "suite", "--platform", "gpu", "--gpu", str(gpu), "--recipe", recipe,
                           "--framework", framework, "--seed", str(seed), "--steps", "22",
                           "--negative-scope", scope, "--asset-root", str(ASSETS), "--output", str(root)]
                print(f"starting {recipe} {framework} seed={seed} GPU={gpu}", flush=True)
                subprocess.run(command, cwd=checkout, env=env, check=True)
                completed(destination)
                print(f"completed {recipe} {framework} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
