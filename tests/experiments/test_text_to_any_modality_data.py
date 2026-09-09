"""Experiment 14 caches complete pinned sources, not preflight subsets."""

import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/14-text-to-any-modality/data.py"
    )
    spec = importlib.util.spec_from_file_location("text_to_any_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("source", "repo", "revision"),
    [
        ("audio", "OpenSound/AudioCaps", "b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded"),
        ("video", "VLM2Vec/MSR-VTT", "947b139d0c591235a3f311e45a31a88461bad0d2"),
    ],
)
def test_download_complete_pinned_source(monkeypatch, tmp_path, source, repo, revision):
    module = _module()
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return tmp_path / "snapshot"

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    result = module.download(source, cache_dir=tmp_path)
    assert calls == [
        {
            "repo_id": repo,
            "repo_type": "dataset",
            "revision": revision,
            "cache_dir": tmp_path,
            "max_workers": 2,
        }
    ]
    assert result["snapshot"] == str(tmp_path / "snapshot")
    assert result["revision"] == revision


@pytest.mark.parametrize("kind", ["image", "audio", "video"])
def test_field_mapping_delegates_decoding_without_string_conversion(monkeypatch, kind):
    calls = []
    encoded, decoded = object(), object()

    def decode(value, **options):
        calls.append((value, options))
        return decoded

    monkeypatch.setattr(f"representax.data.media.decode_{kind}", decode)
    options = (
        {"seconds": 2.0}
        if kind == "audio"
        else {"frames": 2}
        if kind == "video"
        else {}
    )
    result = getattr(_module(), f"map_{kind}")(
        {"caption": "caption", kind: encoded}, **options
    )
    assert result["query"] == "caption"
    assert result["positive"][kind] is decoded
    expected = {"max_seconds": 2.0} if kind == "audio" else options
    assert calls == [(encoded, expected)]


def test_integration_job_uses_library_pipeline_and_explicit_probe_settings():
    module = importlib.import_module("experiments.14-text-to-any-modality.run")
    paths = {
        name: f"/data/{name}.jsonl" for name in ("image", "audio", "video", "text")
    }
    job, bindings = module.integration_job(paths)
    assert job.data.distribution.sampling_unit == "batch"
    assert job.data.distribution.normalized_weights == (0.25,) * 4
    assert job.data.collate.target == "representax.tasks.retrieval.RetrievalCollator"
    assert job.data.num_threads == job.data.prefetch_buffer_size == 2
    assert job.training.max_steps == 8 and job.training.global_batch_size == 2
    assert job.training.grad_cache.implementation == "custom_vjp"
    assert job.training.grad_cache.micro_batch_size == 1
    assert job.training.adapter.target_pattern == "text"
    assert job.checkpointing.every == 4
    assert bindings[f"{module.DATA_MODULE}.map_audio"].keywords == {"seconds": 2.0}
    assert bindings[f"{module.DATA_MODULE}.map_video"].keywords == {"frames": 2}


def test_integration_config_round_trip_preserves_fixed_execution():
    from representax.config import JobConfig

    module = importlib.import_module("experiments.14-text-to-any-modality.run")
    paths = {
        name: f"/data/{name}.jsonl" for name in ("image", "audio", "video", "text")
    }
    job, _ = module.integration_job(paths)
    restored = JobConfig.model_validate_json(job.model_dump_json())
    assert restored.training.grad_cache.implementation == "custom_vjp"
    assert restored.training.grad_cache.micro_batch_size == 1
    assert restored.training.global_batch_size == 2
    assert restored.loss_modifiers[0].dimensions == (32, 64, 128, 256, 512, 768)


def test_integration_cli_has_no_hyperparameter_options(monkeypatch, capsys):
    module = importlib.import_module("experiments.14-text-to-any-modality.run")
    monkeypatch.setattr("sys.argv", ["run.py", "--help"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--output" in help_text and "--resume" in help_text
    for removed in ("--execution", "--chunk-size", "--matryoshka"):
        assert removed not in help_text


def test_chunk_sweep_covers_all_modalities_with_fixed_scientific_batch():
    module = importlib.import_module("experiments.14-text-to-any-modality.chunk_sweep")
    paths = {name: f"/data/{name}.jsonl" for name in module.MODALITIES}
    assert len(module.CELLS) == 24
    for modality, chunk in module.CELLS:
        job, bindings = module.probe_job(paths, modality, chunk)
        assert (
            job.training.global_batch_size == job.training.batch.micro_batch_size == 32
        )
        assert job.training.grad_cache.micro_batch_size == chunk
        assert job.training.grad_cache.implementation == "custom_vjp"
        assert job.training.max_steps == 6
        assert job.training.adapter.target_pattern == "text"
        assert job.data.distribution.sources[0].name == modality
        assert len(job.data.distribution.sources) == 1
        assert job.checkpointing.every > job.training.max_steps
        assert not job.checkpointing.save_final and not job.export.enabled
        assert bindings[f"{module.RUN.DATA_MODULE}.map_audio"].keywords == {
            "seconds": 10.0
        }
        assert bindings[f"{module.RUN.DATA_MODULE}.map_video"].keywords == {"frames": 8}


def test_chunk_sweep_timing_keeps_input_wait_and_excludes_first_use(tmp_path):
    module = importlib.import_module("experiments.14-text-to-any-modality.chunk_sweep")
    rows = [
        {"perf/compilation_and_first_step_seconds": 20.0},
        {"perf/step_seconds": 3.0, "perf/data_wait_seconds": 1.0},
        {"perf/step_seconds": 5.0, "perf/data_wait_seconds": 2.0},
    ]
    path = tmp_path / "run/metrics.jsonl"
    path.parent.mkdir()
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "event": "training_step",
                    "metrics": {
                        **row,
                        "train/numeric_finite": True,
                        "train/skipped_update": False,
                    },
                }
            )
            for row in rows
        )
    )
    result = module.summarize(tmp_path)
    assert result["warm_examples_per_second"] == 64 / 8
    assert result["warm_observations"] == 2
    assert result["cold_compile_and_first_seconds"] == 20
    assert result["all_updates_finite"] and not result["skipped_updates"]


def test_memory_report_counts_aliased_buffers_once(tmp_path):
    module = importlib.import_module(
        "experiments.14-text-to-any-modality.memory_report"
    )
    path = tmp_path / "buffer-assignment.txt"
    path.write_text(
        "allocation 0: size 100, parameter 0, maybe-live-out:\n"
        "allocation 1: size 30, parameter 1:\n"
        "allocation 2: size 4, thread-local:\n"
        "allocation 3: size 200, preallocated-temp:\n"
        " value: <a @0> (size=150,offset=0): f32[37]\n"
        " value: <b @0> (size=150,offset=0): f32[37]\n"
    )
    report = module.buffer_memory(path)
    assert report["total_bytes"] == 330
    assert report["bytes"]["state_aliased"] == 100
    assert report["bytes"]["outputs"] == 0
    assert report["bytes"]["temporaries"] == 200
    assert report["thread_local_bytes_excluded"] == 4
    assert len(report["largest_temporary_values"]) == 2


@pytest.mark.parametrize("strategy", ["connectors", "connectors-lora", "full"])
def test_strategy_preflight_freezes_parameter_recipe_and_source_coverage(strategy):
    import numpy as np

    module = importlib.import_module(
        "experiments.14-text-to-any-modality.strategy_preflight"
    )
    paths = {
        name: f"/data/{name}.jsonl" for name in ("image", "audio", "video", "text")
    }
    job, _ = module.strategy_job(paths, strategy)
    assert job.training.global_batch_size == 32
    assert job.training.grad_cache.micro_batch_size == 2
    assert job.training.activation_rematerialization == "full"
    assert job.training.max_steps == 8 and job.checkpointing.every == 4
    sources = job.data.distribution.sources
    names = {s.name for s in sources}
    assert names == (
        {"image", "audio", "video"} if strategy == "connectors" else set(paths)
    )
    choices = np.random.default_rng(7).choice(
        len(sources), size=8, p=job.data.distribution.normalized_weights
    )
    assert set(choices) == set(range(len(sources)))
    if strategy == "full":
        assert job.training.trainable_pattern == ".*" and job.training.adapter is None
        assert not job.training.trainable_embedding_rows
    else:
        assert job.training.trainable_embedding_rows == {
            ".text.token_embedding": (128257, 128258)
        }
        assert (job.training.adapter is not None) == (strategy == "connectors-lora")
