from __future__ import annotations

from experiments.paper.bert_scaling import (
    ACTIVE_UPDATE_BYTES_PER_PARAMETER,
    DEFAULT_ARTIFACT_ROOT,
    EVALUATION_DATASET_ID,
    EVALUATION_DATASET_REVISION,
    GPU_ASSIGNMENTS,
    HBM_ADMISSION_FRACTION,
    NQ_DATASET_ID,
    NQ_DATASET_REVISION,
    PRECISION_LABEL,
    PRECISION_RECIPE,
    SIZE_ORDER,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    TRAIN_DATASET_ID,
    TRAIN_DATASET_REVISION,
    EvaluationData,
    _parser,
    _timings,
    _visible_physical_gpus,
    build_job,
    four_b_feasibility,
    ladder_entry,
    load_ladder,
    select_bounded_evaluation,
    select_training_rows,
)
from scripts.validate_bert_scaling import bert_parameter_count

from representax.config import FSDPConfig
from representax.models.bert import BertConfig


def _evaluation() -> EvaluationData:
    return EvaluationData(
        queries=tuple((index, f"query {index}") for index in range(4)),
        documents=tuple((10_000 + index, f"document {index}") for index in range(16)),
        relevant_documents={
            index: frozenset((10_000 + index,)) for index in range(4)
        },
    )


def _accepted_one_b(peak: int) -> dict:
    return {
        "status": "accepted",
        "training": {
            "task": "retrieval",
            "loss": "mnr",
            "precision": PRECISION_LABEL,
            "precision_recipe": dict(PRECISION_RECIPE),
        },
        "evaluation": {
            "export_reload_max_absolute_error": 0.0,
            "exported_nq_confirmation": {"valid/nq-bounded/cosine_recall@10": 1.0},
        },
        "devices": {
            "training_jax": [
                {"peak_bytes_in_use": peak},
                {"peak_bytes_in_use": peak - 100_000_000},
            ]
        },
    }


def test_ladder_maps_every_frozen_config_to_its_exact_parameter_count() -> None:
    entries = load_ladder()

    assert tuple(entry.name for entry in entries) == SIZE_ORDER
    assert tuple(entry.expected_parameters for entry in entries) == (
        29_811_072,
        109_482_240,
        486_296_576,
        985_885_440,
        4_177_190_400,
    )
    for entry in entries:
        config = BertConfig(**entry.config_values())
        assert bert_parameter_count(config) == entry.expected_parameters
        assert config.head_dimension == 64
        assert config.intermediate_size == 4 * config.hidden_size
    assert entries[-1].admission_gate == "bert-1b-pilot-feasible"


def test_ladder_pins_dense_retrieval_sources_and_tokenizer() -> None:
    assert TRAIN_DATASET_ID == "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
    assert TRAIN_DATASET_REVISION == "0d54352548089199bde15ad7e06efe895dc80b56"
    assert EVALUATION_DATASET_ID == "sentence-transformers/NanoBEIR-en"
    assert EVALUATION_DATASET_REVISION == "beb106fbcfaa599c508c667041bf8c85fd78736b"
    assert NQ_DATASET_ID == "mteb/nq"
    assert NQ_DATASET_REVISION == "b84726e65fd226125cf7c0cbeeb5c214d49e8187"
    assert TOKENIZER_ID == "google-bert/bert-base-uncased"
    assert TOKENIZER_REVISION == "86b5e0934494bd15c9632b12f734a8a67f723594"


def test_bounded_evaluation_preserves_real_qrel_pairs_and_adds_distractors() -> None:
    queries = {f"q{index}": f"query {index}" for index in range(5)}
    documents = {f"d{index}": f"document {index}" for index in range(8)}
    qrels = (("missing", "d0"), ("q0", "d3"), ("q1", "d2"))

    result = select_bounded_evaluation(
        queries,
        documents,
        qrels,
        query_count=2,
        document_count=5,
    )

    assert result.query_source_ids == ("q0", "q1")
    assert result.document_source_ids[:2] == ("d3", "d2")
    assert result.relevant_documents == {
        0: frozenset((10_000,)),
        1: frozenset((10_001,)),
    }
    assert len(result.documents) == 5


def test_training_slice_keeps_first_distinct_queries_in_source_order() -> None:
    rows = (
        {"query": "q0", "positive": "p0"},
        {"query": "q0", "positive": "duplicate"},
        {"query": "q1", "positive": "p1"},
        {"query": "q2", "positive": "p2"},
    )

    selected = select_training_rows(rows, count=3)

    assert selected == (
        {"query": "q0", "positive": "p0"},
        {"query": "q1", "positive": "p1"},
        {"query": "q2", "positive": "p2"},
    )


def test_job_uses_mnr_evaluation_export_and_two_way_fsdp(tmp_path) -> None:
    small = build_job(
        ladder_entry("bert-110m"),
        data_directory=tmp_path,
        msmarco_evaluation=_evaluation(),
    )
    large = build_job(
        ladder_entry("bert-500m"),
        data_directory=tmp_path,
        msmarco_evaluation=_evaluation(),
    )

    assert small.task.kind == "retrieval"
    assert small.loss.kind == "mnr"
    assert small.loss.scale == 20.0
    assert not small.loss.symmetric
    assert small.training.global_batch_size == 2
    assert small.model.parameters["initialization_device"] == "default"
    assert small.model.parameters["parameter_dtype"] == "float32"
    assert small.model.parameters["compute_dtype"] == "bfloat16"
    assert small.training.precision.compute_dtype == "bfloat16"
    assert small.training.precision.parameter_dtype == "float32"
    assert small.training.mesh.device_count == 1
    assert small.evaluation is not None
    assert small.evaluation.on_start and small.evaluation.on_end
    assert small.evaluation.evaluators[0].kind == "information_retrieval"
    assert small.export.enabled
    assert small.export.selection == "final"
    assert large.model.parameters["initialization_device"] == "cpu"
    assert large.training.mesh.axis_shapes == (2,)
    assert isinstance(large.training.sharding, FSDPConfig)
    assert large.training.sharding.data_axis is None
    assert large.training.sharding.parameter_axis == "model"
    assert large.checkpointing is not None
    assert large.checkpointing.every == 2
    assert not large.checkpointing.save_final
    assert not large.checkpointing.asynchronous


def test_timing_summary_separates_compile_and_steady_updates() -> None:
    updates = (
        {
            "iteration": 1,
            "metrics": {"perf/compilation_and_first_step_seconds": 8.0},
        },
        {"iteration": 2, "metrics": {"perf/step_seconds": 0.5}},
        {
            "iteration": 3,
            "metrics": {"perf/compilation_and_first_step_seconds": 1.0},
        },
    )

    summary = _timings(updates)

    assert summary["compile_and_first_update"] == [
        {"iteration": 1, "seconds": 8.0},
        {"iteration": 3, "seconds": 1.0},
    ]
    assert summary["steady_updates"] == [{"iteration": 2, "seconds": 0.5}]
    assert summary["steady_median_seconds"] == 0.5


def test_worker_and_four_b_gate_use_the_frozen_physical_gpu_sets(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert _visible_physical_gpus() == (2, 3)
    assert GPU_ASSIGNMENTS == {
        "bert-30m": (2,),
        "bert-110m": (2,),
        "bert-500m": (2, 3),
        "bert-1b": (2, 3),
        "bert-4b": (2, 3, 4, 5),
    }


def test_four_b_passes_projection_but_remains_pending_four_gpu_canary() -> None:
    total = 24_564 * 1024**2
    result = four_b_feasibility(
        _accepted_one_b(9_350_000_000),
        device_total_bytes=(total, total, total, total),
    )

    assert result["status"] == "admitted_pending_four_gpu_canary"
    assert result["decision"] == "pending_four_gpu_canary"
    assert result["one_b_dense_retrieval_practical"]
    assert result["hbm_admission_fraction"] == HBM_ADMISSION_FRACTION
    assert result["planning"]["active_update_bytes_per_parameter"] == (
        ACTIVE_UPDATE_BYTES_PER_PARAMETER
    )
    assert result["effective_estimated_peak_bytes_per_device"] < result[
        "admission_limit_bytes_per_device"
    ]
    assert result["required_canary"] == {
        "updates": 1,
        "compiled": True,
        "devices": 4,
        "pending_gpu_indices": [4, 5],
    }


def test_four_b_rejects_projection_that_exceeds_four_gpu_reserve() -> None:
    total = 24_564 * 1024**2
    result = four_b_feasibility(
        _accepted_one_b(12_000_000_000),
        device_total_bytes=(total, total, total, total),
    )

    assert result["status"] == "not_run"
    assert result["decision"] == "skip"
    assert result["reasons"] == [
        "projected-active-update-hbm-exceeds-85-percent-reserve"
    ]


def test_four_b_rejects_a_stale_precision_recipe() -> None:
    one_b = _accepted_one_b(9_350_000_000)
    one_b["training"]["precision_recipe"]["model_parameter_dtype"] = "bfloat16"
    total = 24_564 * 1024**2

    result = four_b_feasibility(
        one_b,
        device_total_bytes=(total, total, total, total),
    )

    assert not result["one_b_dense_retrieval_practical"]
    assert result["status"] == "not_run"
    assert result["reasons"] == ["bert-1b-dense-retrieval-pilot-not-accepted"]


def test_sweep_defaults_to_artifact_root_outside_raid() -> None:
    arguments = _parser().parse_args(["sweep"])

    assert arguments.artifact_root == DEFAULT_ARTIFACT_ROOT
    assert str(arguments.artifact_root).startswith("/home/ckg/representax-artifacts/")
    assert "/raid" not in str(arguments.artifact_root)
