"""Account for overlapping CUDA activities inside recorded optimizer intervals."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import statistics
from pathlib import Path


def partition_time(start, end, activities):
    events = [(start, "boundary", 0), (end, "boundary", 0)]
    for left, right, kind in activities:
        left, right = max(start, left), min(end, right)
        if left < right:
            events.extend(((left, kind, 1), (right, kind, -1)))
    active = collections.Counter()
    times = collections.Counter()
    previous = start
    for timestamp, kind, delta in sorted(events):
        label = "+".join(sorted(k for k, n in active.items() if n)) or "idle"
        times[label] += (timestamp - previous) / 1e9
        active[kind] += delta
        previous = timestamp
    return dict(times)


def summarize(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    kernels = [
        dict(row)
        for row in connection.execute(
            "SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.demangledName"
        )
    ]
    runtime = {
        row["correlationId"]: dict(row)
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")
    }
    thunks = collections.defaultdict(list)
    for row in connection.execute(
        "SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) AS text "
        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
        "WHERE coalesce(n.text,s.value) LIKE '%hlo_op=all-reduce%'"
    ):
        thunks[row["globalTid"]].append(dict(row))
    hlo = {
        row["instruction"].split()[0].lstrip("%"): row
        for row in json.loads(
            (path.with_suffix("") / "length-512.collectives.json").read_text()
        )
    }
    for kernel in kernels:
        kernel["activity"] = "compute"
        if "nccl" not in kernel["name"].lower():
            continue
        kernel["activity"] = "other_collective"
        launch = runtime.get(kernel["correlationId"])
        if launch is None:
            continue
        ranges = [
            r
            for r in thunks[launch["globalTid"]]
            if r["end"] is not None
            and r["start"] <= launch["start"]
            and r["end"] >= launch["end"]
        ]
        for enclosing in sorted(ranges, key=lambda r: r["end"] - r["start"]):
            match = re.search(r"hlo_op=([^,#]+)", enclosing["text"])
            if match and match[1] in hlo:
                kernel["hlo_op"] = match[1]
                operation = hlo.get(match[1])
                if operation and operation["largest_float_array_elements"] >= 1_000_000:
                    kernel["activity"] = "large_gradient_collective"
                break
    copies = [
        dict(row)
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    ]
    memsets = [
        dict(row)
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET")
    ]
    steps = list(
        connection.execute(
            "SELECT start,end,text FROM NVTX_EVENTS "
            "WHERE text LIKE 'optimizer_step_%' AND end IS NOT NULL ORDER BY start"
        )
    )
    result = {"source": str(path), "steps": []}
    for step in steps:
        start, end = step["start"], step["end"]
        row = {
            "step": int(step["text"].rsplit("_", 1)[-1]),
            "seconds": (end - start) / 1e9,
            "devices": [],
        }
        for device in sorted({k["deviceId"] for k in kernels}):
            selected = [
                k
                for k in kernels
                if k["deviceId"] == device and k["start"] < end and k["end"] > start
            ]
            transfer = [
                k
                for k in copies
                if k["deviceId"] == device and k["start"] < end and k["end"] > start
            ]
            activities = [
                (
                    k["start"],
                    k["end"],
                    k["activity"],
                )
                for k in selected
            ]
            activities += [(k["start"], k["end"], "copy") for k in transfer]
            activities += [
                (k["start"], k["end"], "memset")
                for k in memsets
                if k["deviceId"] == device and k["start"] < end and k["end"] > start
            ]
            duration = collections.Counter()
            calls = collections.Counter()
            collective_kernels = []
            for k in selected:
                duration[k["name"]] += (
                    min(end, k["end"]) - max(start, k["start"])
                ) / 1e9
                calls[k["name"]] += 1
                if "nccl" in k["name"].lower():
                    collective_kernels.append(
                        {
                            "name": k["name"],
                            "start_seconds": (k["start"] - start) / 1e9,
                            "duration_seconds": (k["end"] - k["start"]) / 1e9,
                            "grid": [k["gridX"], k["gridY"], k["gridZ"]],
                            "correlation_id": k["correlationId"],
                            "hlo_op": k.get("hlo_op"),
                            "activity": k["activity"],
                        }
                    )
            row["devices"].append(
                {
                    "device": device,
                    "exclusive_activity_seconds": partition_time(
                        start, end, activities
                    ),
                    "top_kernels": [
                        {"name": name, "summed_seconds": seconds, "calls": calls[name]}
                        for name, seconds in duration.most_common(12)
                    ],
                    "collective_kernels": collective_kernels,
                    "copy_bytes": sum(k["bytes"] for k in transfer),
                }
            )
        result["steps"].append(row)
    connection.close()
    stable = [r for r in result["steps"] if r["step"] in (4, 5)]
    if len(stable) != 2:
        raise ValueError("expected both stable profiled steps 4 and 5")
    fields = {
        k for s in stable for d in s["devices"] for k in d["exclusive_activity_seconds"]
    }
    result["stable_summary"] = {
        "mean_step_seconds": statistics.mean(s["seconds"] for s in stable),
        "mean_device_activity_seconds": {
            k: statistics.mean(
                d["exclusive_activity_seconds"].get(k, 0)
                for s in stable
                for d in s["devices"]
            )
            for k in sorted(fields)
        },
        "note": "Steps 4-5 only: excludes first traced step/startup; categories are "
        "disjoint interval unions, not sums across streams or devices. NCCL "
        "kernel time includes waiting and is not pure wire-transfer time.",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.trace.with_suffix(".activities.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["stable_summary"], indent=2))
