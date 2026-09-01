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
    assert campaign["common_training"]["evaluation_progress"] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert campaign["common_training"]["lifecycle_checkpoint_progress"] == [
        0.5,
        1.0,
    ]
    assert campaign["common_training"]["final_quality_evaluator"] == "representax"
    assert all(record["training_budget"]["value"] > 0 for record in workloads.values())
    assert all(len(record["frameworks"]) == 2 for record in workloads.values())
    assert workloads["dense-multilingual"]["languages"] == ["ar", "en", "hi", "ja"]
    assert workloads["dense-realistic"]["training_budget"] == {
        "unit": "updates",
        "value": 256,
    }
    assert workloads["dense-realistic"]["evaluation_steps"] == [0, 64, 128, 192, 256]
    assert workloads["dense-realistic"]["world_size"] == 1
    assert workloads["lejepa-image-representation"]["training_budget"] == {
        "unit": "updates",
        "value": 10_000,
    }
    assert workloads["lejepa-image-representation"]["world_size"] == 2
