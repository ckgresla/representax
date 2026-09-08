"""Trace experiments 11 and 13 through worker dispatch without running training."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _experiment(directory):
    spec = importlib.util.spec_from_file_location(
        "convergence_configuration", ROOT / "experiments" / directory / "run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _JobCaptured(Exception):
    pass


def _assert_science(job, contract, seed, steps):
    assert job.training.seed == seed
    assert job.training.max_steps == steps
    assert job.training.global_batch_size == contract["global_batch_size"]
    assert job.model.parameters["revision"] == contract["model"]["revision"]
    assert job.model.parameters["parameter_dtype"] == "float32"
    assert job.model.parameters["compute_dtype"] == "bfloat16"
    assert job.loss.scale == contract["loss"]["scale"]
    assert job.loss.symmetric == contract["loss"]["symmetric"]
    optimization = contract["optimization"]
    assert job.optimization.optimizer.target == "optax.adamw"
    assert (
        job.optimization.optimizer.parameters["weight_decay"]
        == optimization["weight_decay"]
    )
    assert job.optimization.schedule.target == "optax.warmup_cosine_decay_schedule"
    assert (
        job.optimization.schedule.parameters["peak_value"]
        == optimization["learning_rate"]
    )
    assert job.optimization.schedule.parameters["decay_steps"] == steps
    assert job.optimization.max_gradient_norm == optimization["gradient_clip_norm"]
    assert job.evaluation.on_start and job.evaluation.on_end


def _dense_job(tmp_path, monkeypatch, seed=7, steps=3817):
    from benchmarks import dense_retrieval as worker
    from representax import train

    experiment = _experiment("11-dense-retrieval-convergence")
    monkeypatch.setattr(experiment, "CHECKPOINT", tmp_path)
    monkeypatch.setattr(experiment, "TRAINING_DATA", tmp_path)
    (tmp_path / f"seed-{seed}.parquet").touch()
    monkeypatch.setattr(
        worker,
        "_evaluation_data",
        lambda _: SimpleNamespace(queries={}, documents={}, relevant_documents={}),
    )
    jobs = []

    def capture(job, *args, **kwargs):
        jobs.append(job)
        raise _JobCaptured

    monkeypatch.setattr(train, "run_job", capture)
    command = experiment.worker_command(seed, 0, steps=steps)
    arguments = worker._parser().parse_args(command[3:])
    with pytest.raises(_JobCaptured):
        worker._worker(arguments)

    return experiment, jobs[0]


@pytest.mark.parametrize("seed,steps", [(7, 3817), (773, 4000)])
def test_dense_launcher_reaches_actual_worker_job(tmp_path, monkeypatch, seed, steps):
    experiment, job = _dense_job(tmp_path, monkeypatch, seed, steps)
    contract = experiment.contract(job=job)
    assert job.training.seed == seed
    assert job.training.max_steps == steps
    assert job.training.global_batch_size == contract["global_batch_size"]
    assert job.model.parameters["revision"] == contract["model"]["revision"]
    assert contract["loss"] == job.loss.model_dump(mode="json")
    assert contract["optimization"] == job.optimization.model_dump(mode="json")
    assert contract["precision"] == job.training.precision.model_dump(mode="json")
    assert job.loss.scale == 20.0 and not job.loss.symmetric
    assert job.optimization.schedule.parameters == {
        "init_value": 0.0,
        "peak_value": 2e-5,
        "warmup_steps": round(steps * 0.06),
        "decay_steps": steps,
        "end_value": 0.0,
    }
    assert (
        job.model.parameters["sequence_length_buckets"]
        == contract["sequence_length_buckets"]
    )
    cache = job.training.grad_cache
    for name, value in contract["grad_cache"].items():
        assert getattr(cache, name) == value
    assert (
        job.evaluation.batch_size
        == contract["evaluation"]["during_training_batch_size"]
    )
    assert job.checkpointing.every == steps // 2
    assert contract["checkpointing"]["iterations"] == (
        [1908, 3816, 3817] if steps == 3817 else [2000, 4000]
    )
    assert contract["checkpointing"]["keep"] == 3
    assert contract["checkpointing"]["save_final"] is True
    assert contract["export"] == "representax"
    assert job.export.enabled and job.export.huggingface is None
    assert job.data.distribution.sources[0].uri == str(
        tmp_path / f"seed-{seed}.parquet"
    )


def test_dense_pending_contract_does_not_invent_worker_settings():
    experiment = _experiment("11-dense-retrieval-convergence")
    contract = experiment.contract()

    assert "loss" not in contract
    assert "optimization" not in contract
    assert "checkpoint_progress" not in contract
    assert contract["worker_configuration"]["resolved"] is False
    assert contract["checkpointing"] == {
        "every": "optimizer_steps // 2",
        "save_final": True,
    }
    assert contract["export"] == "representax"
    assert experiment._parser().parse_args(["contract", "--seed", "7"]).seed == 7


def _record_job(experiment, config, seed):
    directory = experiment.OUTPUT / "runs" / f"seed-{seed}"
    experiment._write_json(directory / "run" / "run.json", {"config": config})
    experiment._write_json(directory / "report.json", {"seed": seed})


def test_dense_recorded_settings_are_derived_not_frozen(tmp_path, monkeypatch):
    experiment, job = _dense_job(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path)
    config = job.model_dump(mode="json")
    config["loss"].update(scale=17.0, symmetric=True)
    config["optimization"]["schedule"]["parameters"].update(
        peak_value=3e-5, warmup_steps=123
    )
    config["optimization"]["optimizer"]["parameters"]["weight_decay"] = 0.1
    config["optimization"]["max_gradient_norm"] = 2.0
    config["checkpointing"].update(every=1000, save_final=False)
    config["export"]["enabled"] = False
    _record_job(experiment, config, 7)

    contract = experiment.contract(job=experiment._worker_job(7))

    assert contract["worker_configuration"]["resolved"] is True
    assert contract["loss"] == config["loss"]
    assert contract["optimization"] == config["optimization"]
    assert contract["checkpointing"]["iterations"] == [1000, 2000, 3000]
    assert contract["export"] == "disabled"
    with pytest.raises(FileNotFoundError):
        experiment._worker_job(42)
    _record_job(experiment, config, 42)
    with pytest.raises(ValueError, match="wrong seed"):
        experiment._worker_job(42)


def test_dense_aggregate_preserves_each_workers_recorded_settings(
    tmp_path, monkeypatch
):
    experiment, job = _dense_job(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path)
    for seed in experiment.SEEDS:
        config = job.model_dump(mode="json")
        config["training"]["seed"] = seed
        config["optimization"]["schedule"]["parameters"]["warmup_steps"] = seed
        _record_job(experiment, config, seed)

    experiment.aggregate()
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert "optimization" not in summary["contract"]
    for seed in experiment.SEEDS:
        contract = summary["worker_contracts"][str(seed)]
        assert (
            contract["optimization"]["schedule"]["parameters"]["warmup_steps"] == seed
        )
        assert contract["checkpointing"]["iterations"] == [1908, 3816, 3817]


def test_dense_launch_resolves_metadata_after_worker_completes(tmp_path, monkeypatch):
    experiment, job = _dense_job(tmp_path, monkeypatch)
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path)
    monkeypatch.setattr(experiment, "_training_steps", lambda _: 3817)
    monkeypatch.setattr(experiment, "_git_state", lambda: {"commit": "test"})
    expected_command = experiment.worker_command(7, 0, steps=3817)
    launch_path = tmp_path / "runs" / "seed-7" / "launch.json"

    def worker(command, **kwargs):
        assert command == expected_command
        pending = json.loads(launch_path.read_text())
        assert pending["contract"]["worker_configuration"]["resolved"] is False
        _record_job(experiment, job.model_dump(mode="json"), 7)

    monkeypatch.setattr(experiment.subprocess, "run", worker)
    experiment.run_seed(7, 0)
    launch = json.loads(launch_path.read_text())

    assert launch["command"] == expected_command
    assert launch["git"] == {"commit": "test"}
    assert launch["contract"] == experiment.contract(job=job)


@pytest.mark.parametrize("seed", [7, 773])
def test_image_launcher_reaches_actual_worker_job(tmp_path, monkeypatch, seed):
    from experiments.preflights import image_text as worker

    experiment = _experiment("13-image-text-convergence")
    monkeypatch.setattr(experiment, "DATA", tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "training_presentations": 588800,
                "unique_image_ids": True,
                "batch_unique_captions": True,
                "relevant_documents": {"1000": [0]},
            }
        )
    )
    build_job = worker._representax_job
    jobs = []

    def capture(**kwargs):
        jobs.append(build_job(**kwargs))
        raise _JobCaptured

    monkeypatch.setattr(worker, "initialize_jax", lambda _: None)
    monkeypatch.setattr(worker, "_representax_job", capture)
    command = experiment.worker_command(seed, 0)
    arguments = worker._parser().parse_args(command[3:])
    with pytest.raises(_JobCaptured):
        worker._worker(arguments)

    job = jobs[0]
    contract = experiment.contract()
    _assert_science(job, contract, seed, experiment.STEPS)
    assert (
        job.optimization.schedule.parameters["warmup_steps"]
        == contract["optimization"]["warmup_steps"]
    )
    assert (
        job.training.grad_cache.micro_batch_size
        == contract["grad_cache_micro_batch_size"]
    )
    assert job.loss.negative_scope == contract["loss"]["negative_scope"]
    assert job.data.distribution.sources[0].uri == str(
        tmp_path / f"train-seed-{seed}.jsonl"
    )
    assert job.checkpointing.every == experiment.STEPS // 2
    assert job.export.enabled and job.export.huggingface is not None


def test_image_launcher_derives_worker_owned_settings(monkeypatch):
    from experiments.preflights import image_text as worker

    contract = replace(
        worker.frozen_contract(),
        global_batch_size=256,
        image_shape=(3, 112, 112),
        model_revision="test-revision",
    )
    monkeypatch.setattr(worker, "frozen_contract", lambda: contract)
    monkeypatch.setattr(worker, "GRAD_CACHE_MICRO_BATCH", 4)
    experiment = _experiment("13-image-text-convergence")

    assert contract.global_batch_size == experiment.GLOBAL_BATCH_SIZE
    assert experiment.STEPS == 2300
    assert experiment.WARMUP_STEPS == 138
    assert experiment.contract()["image_shape"] == [3, 112, 112]
    assert experiment.MODEL_REVISION == "test-revision"
    assert experiment.GRAD_CACHE_MICRO_BATCH == 4
