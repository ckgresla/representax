"""Experiment 14 caches complete pinned sources, not preflight subsets."""

import importlib
import importlib.util
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
    assert job.training.grad_cache is None
    assert job.training.adapter.target_pattern == "text"
    assert job.checkpointing.every == 4
    assert bindings[f"{module.DATA_MODULE}.map_audio"].keywords == {"seconds": 2.0}
    assert bindings[f"{module.DATA_MODULE}.map_video"].keywords == {"frames": 2}


@pytest.mark.parametrize("execution", ["rematerialized", "custom_vjp"])
def test_integration_execution_options_reach_job_config(execution):
    from representax.config import JobConfig

    module = importlib.import_module("experiments.14-text-to-any-modality.run")
    paths = {
        name: f"/data/{name}.jsonl" for name in ("image", "audio", "video", "text")
    }
    job, _ = module.integration_job(
        paths, execution=execution, chunk_size=1, matryoshka=True
    )
    restored = JobConfig.model_validate_json(job.model_dump_json())
    assert restored.training.grad_cache.implementation == execution
    assert restored.training.grad_cache.micro_batch_size == 1
    assert restored.training.global_batch_size == 2
    assert restored.loss_modifiers[0].dimensions == (32, 64, 128, 256, 512, 768)


@pytest.mark.parametrize("options", [{"chunk_size": 0}, {"execution": "unknown"}])
def test_integration_rejects_invalid_execution_options(options):
    module = importlib.import_module("experiments.14-text-to-any-modality.run")
    with pytest.raises(ValueError):
        module.integration_job({}, **options)
