"""Add the two complete pretrained transfer baselines without changing finals."""

import hashlib
import json
import math
from pathlib import Path

from build import HERE, SEEDS, require

ROOT = Path("/raid/representax-paper/11-dense-retrieval-convergence")


def collect(root=ROOT):
    sources, reports = {}, {}

    def verify(path, expected=None):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        require(expected is None or expected == "sha256:" + digest,
                f"Baseline artifact hash mismatch: {path}")
        sources[str(path)] = {"sha256": digest, "bytes": path.stat().st_size}

    def read(path):
        verify(path)
        return json.loads(path.read_text())

    directory = root / "initial-transfer"
    provenance = read(directory / "gpu-placement-restart/launch.json")
    verify(directory / "gpu-placement-restart/run.py.snapshot", provenance["source_sha256"])
    verify(directory / "gpu-placement-restart/source.patch",
           provenance["git"]["working_tree_patch_sha256"])
    for name in ("trec-dl-2019", "natural-questions"):
        initial = read(directory / f"{name}.json")
        require(initial["checkpoint_stage"] == "initial" and initial["dataset"] == name,
                f"Not an initial checkpoint evaluation: {name}")
        require(initial["seed"] == SEEDS[0], "Unexpected shared baseline seed")
        require(all(math.isfinite(value) for value in initial["metrics"].values()),
                f"Nonfinite evaluation metric: {name}")
        verify(Path(initial["data_manifest"]), initial["data_manifest_sha256"])
        verify(root / f"runs/seed-{SEEDS[0]}/run/run.json", initial["source_run_config_sha256"])
        for filename, digest in initial["model_files_sha256"].items():
            verify(Path(initial["artifact"]) / filename, digest)
        for seed in SEEDS:
            final = read(root / f"runs/seed-{seed}/transfer/{name}.json")
            for field in ("dataset", "data_manifest_sha256", "evaluation_batch_size",
                          "queries", "encoded_examples", "batches"):
                require(initial[field] == final[field], f"Evaluation contract differs: {name}/{field}")
        reports[name] = initial
    return reports, sources, provenance


def main():
    path = HERE / "evidence.json"
    evidence = json.loads(path.read_text())
    require("transfer_initial" not in evidence, "Transfer baselines already recorded")
    reports, sources, provenance = collect()
    evidence["transfer_initial"] = reports
    evidence["sources"].update(sources)
    evidence["transfer_initial_provenance"] = provenance
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print("Recorded both full-corpus initial baselines; final scores unchanged")


if __name__ == "__main__":
    main()
