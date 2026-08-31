from __future__ import annotations

import json
from pathlib import Path


def _workload_names(path: Path) -> set[str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {str(workload["name"]) for workload in document["workloads"]}


def test_campaign_locks_every_workload() -> None:
    root = Path(__file__).parents[1]
    campaign = json.loads(
        (root / "benchmarks/configs/paper-campaign-v1.json").read_text(encoding="utf-8")
    )
    expected = {"dense-scaling"}
    expected.update(
        _workload_names(root / "benchmarks/configs/paper-text-reward-v1.json")
    )
    expected.update(
        _workload_names(root / "benchmarks/configs/paper-multimodal-jepa-v1.json")
    )
    workloads = {record["name"]: record for record in campaign["workloads"]}
    assert set(workloads) == expected
    assert campaign["quality_seeds"] == [7, 42, 773]
    assert campaign["timing"]["source"] == "quality-training-trajectories"
    assert campaign["timing"]["record_every_step"] is True
    assert campaign["timing"]["synchronize_for_timing"] is False
    assert campaign["timing"]["additional_fresh_process_repetitions"] == 0
    assert campaign["timing"]["scaling_world_sizes"] == [1, 2, 4]
    assert campaign["conditional_cells"]["bert-4b"]["gate"] == (
        "bert-1b-pilot-feasible"
    )
    assert all("updates" not in record for record in workloads.values())
    assert all("evaluation_every" not in record for record in workloads.values())
    assert all(len(record["frameworks"]) == 2 for record in workloads.values())
    assert workloads["dense-multilingual"]["languages"] == ["ar", "en", "hi", "ja"]
