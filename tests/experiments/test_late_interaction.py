"""Paper late-interaction preflight contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from experiments.paper.late_interaction import (
    _build_summary,
    _encoding_timings,
    _pad_embeddings,
    _parser,
    _representax_job,
    _representax_maxsim,
    _write_flat_index,
    frozen_contract,
    representax_steady_state,
    select_training_rows,
)


def test_frozen_contract_names_one_model_dataset_and_reference() -> None:
    contract = frozen_contract()

    assert contract.model_id == "lightonai/GTE-ModernColBERT-v1"
    assert contract.model_revision == "cbbe53366e564450558f5e639dd499171f127538"
    assert contract.training_dataset["repo_id"] == (
        "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
    )
    assert contract.evaluation_dataset["subset"] == "NanoMSMARCO"
    assert contract.reference_version == "1.6.0"
    assert contract.global_batch_size == 512
    assert contract.maximum_query_length == 32
    assert contract.maximum_document_length == 256


def test_training_selection_is_ordered_and_rejects_short_sources() -> None:
    rows = (
        {"query": "", "positive": "ignored"},
        {"query": "q1", "positive": "p1"},
        {"query": "q2", "positive": "p2"},
    )

    selected = select_training_rows(rows, count=2)

    assert selected == (
        {"source_index": 1, "query": "q1", "positive": "p1"},
        {"source_index": 2, "query": "q2", "positive": "p2"},
    )
    with pytest.raises(ValueError, match="only 2 usable rows"):
        select_training_rows(rows, count=3)


def test_flat_index_is_contiguous_measured_and_hashed(tmp_path: Path) -> None:
    values = (
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 6.0]], dtype=np.float32),
    )

    report = _write_flat_index(tmp_path / "index", [11, 12], values)

    tokens = np.load(tmp_path / "index" / "tokens.npy", allow_pickle=False)
    offsets = np.load(tmp_path / "index" / "offsets.npy", allow_pickle=False)
    assert np.array_equal(tokens, np.concatenate(values))
    assert offsets.tolist() == [0, 2, 3]
    assert report["documents"] == 2
    assert report["tokens"] == 3
    assert report["embedding_dimension"] == 2
    assert report["size_bytes"] > tokens.nbytes
    assert set(report["files"]) == {
        "identifiers.json",
        "offsets.npy",
        "tokens.npy",
    }


def test_padding_preserves_tokens_and_adds_explicit_validity() -> None:
    values = (
        np.ones((2, 3), dtype=np.float32),
        np.full((1, 3), 2.0, dtype=np.float32),
    )

    padded, valid = _pad_embeddings(values, length=4)

    assert padded.shape == (2, 4, 3)
    assert valid.tolist() == [
        [True, True, False, False],
        [True, False, False, False],
    ]
    assert np.array_equal(padded[0, :2], values[0])
    assert np.array_equal(padded[1, :1], values[1])


def test_batched_representax_maxsim_matches_direct_numpy() -> None:
    queries = (
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 1.0]], dtype=np.float32),
    )
    documents = (
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        np.asarray([[-1.0, -1.0]], dtype=np.float32),
    )

    scores, timings = _representax_maxsim(queries, documents)
    expected = np.asarray(
        [
            [
                sum(max(float(query @ token) for token in document) for query in row)
                for document in documents
            ]
            for row in queries
        ],
        dtype=np.float32,
    )

    assert np.allclose(scores, expected)
    assert timings["backend"] == "jax-exact-maxsim"
    assert timings["compilation_and_first_tile_seconds"] > 0


def test_steady_state_excludes_each_compiled_step() -> None:
    rows = (
        {
            "metrics": {
                "perf/step_seconds": 9.0,
                "perf/compilation_and_first_step_seconds": 8.0,
            }
        },
        {"metrics": {"perf/step_seconds": 2.0}},
        {
            "metrics": {
                "perf/step_seconds": 7.0,
                "perf/compilation_and_first_step_seconds": 6.0,
            }
        },
        {"metrics": {"perf/step_seconds": 2.0}},
    )

    report = representax_steady_state(rows, batch_size=512)

    assert report == {
        "measured_steps": 2.0,
        "compilation_and_first_step_seconds": 14.0,
        "median_step_seconds": 2.0,
        "examples_per_second": 256.0,
    }


def test_encoding_timings_separate_compilation_and_host_phases() -> None:
    rows = (
        {
            "sequence_length": 64,
            "examples": 8,
            "preprocess_seconds": 1.0,
            "placement_seconds": 0.5,
            "encoder_seconds": 9.0,
            "host_seconds": 0.5,
        },
        {
            "sequence_length": 64,
            "examples": 8,
            "preprocess_seconds": 1.0,
            "placement_seconds": 0.5,
            "encoder_seconds": 2.0,
            "host_seconds": 0.5,
        },
        {
            "sequence_length": 128,
            "examples": 8,
            "preprocess_seconds": 1.0,
            "placement_seconds": 0.5,
            "encoder_seconds": 8.0,
            "host_seconds": 0.5,
        },
        {
            "sequence_length": 128,
            "examples": 8,
            "preprocess_seconds": 1.0,
            "placement_seconds": 0.5,
            "encoder_seconds": 4.0,
            "host_seconds": 0.5,
        },
    )

    report = _encoding_timings(rows, compiled=True)

    assert report["compiled_sequence_lengths"] == [64, 128]
    assert report["compilation_and_first_batch_seconds"] == 17.0
    assert report["warm_batches"] == 2
    assert report["warm_examples"] == 16
    assert report["warm_encoder_seconds"] == 6.0
    assert report["warm_end_to_end_seconds"] == 10.0
    assert report["warm_encoder_examples_per_second"] == pytest.approx(16 / 6)
    assert report["phase_totals_seconds"]["preprocess_seconds"] == 4.0


def test_job_uses_grad_cache_lifecycle_and_pylate_export(tmp_path: Path) -> None:
    job = _representax_job(
        checkpoint=tmp_path / "checkpoint",
        data_directory=tmp_path / "data",
        steps=4,
        seed=7,
    )

    assert job.training.global_batch_size == 512
    assert job.training.grad_cache is not None
    assert job.training.grad_cache.micro_batch_size == 8
    assert job.checkpointing.every == 2
    assert job.export.huggingface is not None
    assert "LateInteractionCheckpointAdapter" in job.export.huggingface.adapter.target
    assert job.model.parameters["query_sequence_length_buckets"] == [16, 32]
    assert job.model.parameters["document_sequence_length_buckets"] == [
        32,
        64,
        128,
        256,
    ]


def test_timing_job_has_static_shapes_and_no_large_artifacts(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"training": {"rows": 2048}}')

    job = _representax_job(
        checkpoint=tmp_path / "checkpoint",
        data_directory=data,
        steps=10,
        seed=7,
        lifecycle=False,
        static_shapes=True,
    )

    assert len(job.data.distribution.sources) == 3
    assert job.model.parameters["query_sequence_length_buckets"] == [32]
    assert job.model.parameters["document_sequence_length_buckets"] == [256]
    assert job.checkpointing is None
    assert not job.export.enabled
    assert job.export.huggingface is None
    assert not job.logging.accelerator


def test_pair_defaults_to_assigned_gpu() -> None:
    arguments = _parser().parse_args(
        [
            "pair",
            "--checkpoint",
            "/checkpoint",
            "--data-directory",
            "/data",
            "--output",
            "/output",
            "--representax-python",
            "/rx/bin/python",
            "--reference-python",
            "/ref/bin/python",
        ]
    )

    assert arguments.gpu == 1
    assert arguments.steps == 4
    assert arguments.seed == 7


def test_profiles_have_bounded_defaults() -> None:
    encoding = _parser().parse_args(
        [
            "encoding-profile",
            "--framework",
            "representax",
            "--checkpoint",
            "/checkpoint",
            "--data-directory",
            "/data",
            "--output",
            "/output.json",
        ]
    )
    training = _parser().parse_args(
        [
            "training-profile",
            "--framework",
            "pylate",
            "--checkpoint",
            "/checkpoint",
            "--data-directory",
            "/data",
            "--run-directory",
            "/run",
            "--output",
            "/output.json",
        ]
    )

    assert encoding.examples == 1536
    assert training.steps == 10


def test_existing_reports_can_be_summarized_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"rows": 2048}')
    reports = {}
    for name in ("representax", "pylate", "reload"):
        reports[name] = tmp_path / f"{name}.json"
        reports[name].write_text(f'{{"framework": "{name}"}}')
    monkeypatch.setattr(
        "experiments.paper.late_interaction.frozen_contract",
        lambda: frozen_contract(),
    )

    summary = _build_summary(
        data_directory=data,
        representax_report=reports["representax"],
        pylate_report=reports["pylate"],
        verification_report=reports["reload"],
        steps=4,
        seed=7,
        gpu=1,
    )

    assert summary["scope"] == "bounded-readiness-preflight-not-paper-result"
    assert summary["contract"]["data_manifest"] == {"rows": 2048}
    assert summary["representax"]["framework"] == "representax"
    assert summary["pylate"]["framework"] == "pylate"
