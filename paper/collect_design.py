"""Capture selected existing Experiment 09 diagnostics; never launch training."""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCES = {
    "native_coarse": "real-trajectory-30-step/seed-17/custom-vjp",
    "native_rematerialized": "real-trajectory-30-step/seed-17/rematerialized",
    "native_fine": "padding-ablation-30-step/seed-17/rx-fine-buckets",
    "eager_native": "real-trajectory-30-step/seed-17/st-eager",
    "eager_fixed": "padding-ablation-30-step/seed-17/st-eager-fixed",
    "inductor_native": "real-trajectory-30-step/seed-17/st-inductor",
    "inductor_fixed": "padding-ablation-30-step/seed-17/st-inductor-fixed",
    "native_optimized": "optimization/compile-reduced-20260905-01",
    "native_cached": "optimization/compile-reduced-20260905-01/cache-hit-check",
}
FIELDS = (
    "framework", "seed", "steps", "completed_iterations", "world_size",
    "batch_size", "model_id", "revision", "parameter_count", "maximum_length",
    "mixed_precision", "cache_chunk_size", "grad_cache_implementation",
    "sequence_length_buckets", "steady_state_examples", "steady_state_seconds",
    "steady_state_step_count", "steady_state_examples_per_second",
    "compiled_signature_count", "compilation_and_first_use_seconds",
    "final_train_loss",
)


def collect(root):
    rows = {}
    for name, relative in SOURCES.items():
        path = root / relative / "report.json"
        raw = path.read_bytes()
        report = json.loads(raw)
        if report["seed"] != 17 or report["steps"] != 30:
            raise ValueError(f"Unexpected diagnostic contract: {name}")
        rows[name] = {
            "source": str(Path(relative) / "report.json"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "report": {field: report.get(field) for field in FIELDS},
        }
    return {
        "schema": "representax-design-diagnostics-v1",
        "experiment": "09-modernbert-dense-retrieval",
        "scope": "Existing single-seed systems diagnostics, not quality evidence or a replicated benchmark.",
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "design-evidence.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence already exists; choose a new output path")
    args.output.write_text(json.dumps(collect(args.root), indent=2, sort_keys=True) + "\n")
