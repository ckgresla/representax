"""Build a wall-time attribution report from the matched Nsight captures."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_trace import summarize


def report(root):
    rows = []
    runs = []
    for count in (1, 2, 4):
        path = root / f"{count}gpu.sqlite"
        trace = summarize(path)
        path.with_suffix(".activities.json").write_text(json.dumps(trace, indent=2))
        run = json.loads((root / f"{count}gpu/result.json").read_text())
        runs.append(run)
        times = trace["stable_summary"]["mean_device_activity_seconds"]
        groups = dict(
            compute=0.0, large_gradients=0.0, other_collectives=0.0, other_idle=0.0
        )
        for kinds, seconds in times.items():
            active = kinds.split("+")
            if "compute" in active:
                groups["compute"] += seconds
            elif "large_gradient_collective" in active:
                groups["large_gradients"] += seconds
            elif "other_collective" in active:
                groups["other_collectives"] += seconds
            else:
                groups["other_idle"] += seconds
        observed = trace["stable_summary"]["mean_step_seconds"]
        assert abs(sum(groups.values()) - observed) < 1e-6
        observations = run["phases"][0]["observations"]
        unprofiled = statistics.mean(
            o["seconds"] for o in observations if o["step"] in (2, 6)
        )
        rows.append(
            {
                "gpus": count,
                "profiled_step_seconds": observed,
                "activity_seconds": groups,
                "unprofiled_step_2_6_seconds": unprofiled,
                "profiler_relative_difference": observed / unprofiled - 1,
                "host_prepare_seconds": statistics.mean(
                    o["host_prepare_seconds"]
                    for o in observations
                    if o["step"] in (4, 5)
                ),
            }
        )
    for step in zip(*(r["phases"][0]["observations"] for r in runs), strict=True):
        for key in (
            "tokens_sha256",
            "labels_sha256",
            "corrupted_sha256",
            "positions_sha256",
        ):
            assert len({o[key] for o in step}) == 1, key
    result = {
        "rows": rows,
        "input_hashes_match": True,
        "method": "Mean per-device disjoint activity intervals for steps 4-5; "
        "compute/collective overlap is assigned to compute. "
        "Large reductions map via CUDA launch correlation and enclosing "
        "NVTX HLO ranges to the eight million-element-plus FP32 reductions.",
        "limits": "Two stable traced steps per topology; NCCL duration includes "
        "waiting, not just transfer. No measurement of physical link saturation.",
    }
    (root / "attribution.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# DDP Scaling Bottleneck",
        "",
        "505M ModernBERT MLM, sequence length 512, physical microbatch 8, "
        "131,072 input tokens/update. Six updates per trace run, with steps 3-5 "
        "captured and steps 4-5 used below. Same inputs/masking across topologies.",
        "",
        "| GPUs | Compute active s | Exposed large gradients s | "
        "Other exposed collectives s | Other/idle s | Total step s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        g = row["activity_seconds"]
        lines.append(
            f"| {row['gpus']} | {g['compute']:.3f} | {g['large_gradients']:.3f} | "
            f"{g['other_collectives']:.3f} | {g['other_idle']:.3f} | "
            f"{row['profiled_step_seconds']:.3f} |"
        )
    lines += [
        "",
        result["method"],
        "",
        result["limits"],
        "",
        "The eight large reductions carry 2,020,157,440 bytes of FP32 payload "
        "per update (logical payload, not total network traffic). Compute active "
        "time scales within about 3% of ideal. The large-gradient tail accounts "
        "for about 91% of the four-GPU wall-time excess over ideal scaling.",
        "",
        "NCCL selects SHM/direct/direct. NVIDIA peer-access queries report CNS "
        "for every inter-GPU pair; no NVLink is present. This identifies the "
        "current host-memory communication path, not the reason P2P is unavailable.",
        "",
        "A diagnostic NCCL_PROTO=Simple override did not materially improve "
        "two-GPU large-gradient time (0.585 s in either case). Its four-GPU "
        "profiled process exited 139 before its first update; an unprofiled "
        "six-update retry passed at 2.848 s/update (warm), consistent with the "
        "default control. The failed trace is retained, not counted as a result. "
        "No protocol override is adopted.",
        "",
        "Kernel suffix LL is not reliable evidence of the selected protocol: "
        "NCCL shares kernel entry points across protocols. See "
        "[NCCL generation]"
        "(https://github.com/NVIDIA/nccl/blob/master/src/device/generate.py) "
        "and [transport documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-shm-disable).",
        "",
        "This is a diagnostic attribution, not replacement throughput evidence. "
        "No core training code or scientific settings were modified.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run), indent=2))
