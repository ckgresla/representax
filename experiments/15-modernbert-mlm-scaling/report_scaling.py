"""Summarize a completed seed-7 scaling screen without launching training."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def summarize(root: Path) -> dict:
    reports = [
        json.loads((root / f"{n}gpu/result.json").read_text()) for n in (1, 2, 4)
    ]
    if any(r.get("status") != "passed" for r in reports):
        raise ValueError("all three device-count runs must pass before aggregation")
    strong = len({r["configuration"]["tokens_per_update"] for r in reports}) == 1
    rows: list[dict[str, Any]] = []
    for n, report in zip((1, 2, 4), reports, strict=True):
        phase = report["phases"][0]
        observations = phase["observations"]
        warm = [r["seconds"] for r in observations[1:]]
        if len(warm) != 20 or not all(
            r["finite"] and not r["skipped"] for r in observations
        ):
            raise ValueError("expected 20 finite warm updates per run")
        mean = statistics.mean(warm)
        rows.append(
            dict(
                gpus=n,
                global_batch=phase["global_batch"],
                accumulation=phase["accumulation"],
                tokens_per_update=report["configuration"]["tokens_per_update"],
                warm_updates=len(warm),
                step_seconds=mean,
                steps_per_second=1 / mean,
                examples_per_second=phase["global_batch"] / mean,
                tokens_per_second=report["configuration"]["tokens_per_update"] / mean,
                step_time_cv=statistics.stdev(warm) / mean,
                compile_or_cache_load_seconds=phase["compile_seconds"],
                compiled_memory_gib=phase["compiled_required_bytes_per_device"] / 2**30,
                initial_loss=observations[0]["loss"],
                final_loss=observations[-1]["loss"],
                source=str(root / f"{n}gpu/result.json"),
                source_sha256=hashlib.sha256(
                    (root / f"{n}gpu/result.json").read_bytes()
                ).hexdigest(),
            )
        )
    for row in rows:
        row["speedup"] = row["tokens_per_second"] / rows[0]["tokens_per_second"]
        row["efficiency"] = row["speedup"] / row["gpus"]
    parity = None
    if strong:
        obs = [r["phases"][0]["observations"] for r in reports]
        keys = (
            "tokens_sha256",
            "corrupted_sha256",
            "labels_sha256",
            "positions_sha256",
            "supervised_tokens",
        )
        for step in zip(*obs, strict=True):
            for key in keys:
                if len({r[key] for r in step}) != 1:
                    raise ValueError(f"scientific batch mismatch: {key}")
        parity = {
            "input_and_mask_hashes_match": True,
            "max_absolute_loss_range": max(
                max(r["loss"] for r in step) - min(r["loss"] for r in step)
                for step in zip(*obs, strict=True)
            ),
        }
    result = {
        "kind": "strong" if strong else "weak",
        "seed": 7,
        "rows": rows,
        "parity": parity,
        "limitations": [
            "One seed; no cross-seed uncertainty estimate.",
            "Complete optimizer intervals include masking and placement; "
            "compilation and first update excluded.",
        ],
    }
    if not strong:
        result["limitations"].append(
            "Weak scaling changes global batches and is not a "
            "trajectory-parity comparison."
        )
    if (root / "selection.json").exists():
        result["limitations"].append(
            "One-GPU baseline selected from this microbatch sweep; "
            "independent seeded repetitions remain outstanding."
        )
    (root / "aggregate.json").write_text(json.dumps(result, indent=2))
    lines = [
        f"# {result['kind'].title()} DDP Scaling",
        "",
        "505M ModernBERT MLM; length 512; local microbatch "
        f"{reports[0]['configuration']['local_microbatch']}; BF16 compute, "
        "FP32 masters/AdamW; full activation checkpointing.",
        "",
        "| GPUs | Global batch | Accumulation | Steps/s | Tokens/s | Speedup | "
        "Efficiency | Step-time CV | Compile/load s | Compiled GiB/device |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['gpus']} | {r['global_batch']} | {r['accumulation']} | "
            f"{r['steps_per_second']:.3f} | {r['tokens_per_second']:,.0f} | "
            f"{r['speedup']:.3f}x | {r['efficiency']:.1%} | "
            f"{r['step_time_cv']:.2%} | {r['compile_or_cache_load_seconds']:.2f} | "
            f"{r['compiled_memory_gib']:.2f} |"
        )
    if parity is not None:
        lines += [
            "",
            "Inputs and masking match exactly. Maximum loss range across "
            f"topologies: {parity['max_absolute_loss_range']:.8f}.",
        ]
    lines += ["", *result["limitations"]]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run), indent=2))
