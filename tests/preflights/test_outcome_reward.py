"""Contracts for the paper outcome-reward preflight."""

from experiments.preflights.outcome_reward import (
    MICRO_BATCH_SIZE,
    _representax_job,
    frozen_contract,
    optimizer_token_capacities,
    preference_rows,
    reference_timing,
    steady_state,
)


def test_frozen_contract_names_qwen_ultrafeedback_and_trl() -> None:
    contract = frozen_contract()
    assert contract.model_id == "Qwen/Qwen3-0.6B"
    assert contract.model_revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert contract.dataset_id == "trl-lib/ultrafeedback_binarized"
    assert contract.dataset_revision == "47124cb5778f5d50de1c7676a412828f3ea7c555"
    assert contract.global_batch_size == 128
    assert contract.maximum_length == 1024
    assert contract.objective == "bradley-terry-log-sigmoid"
    assert contract.reference_version == "1.10.0"


def test_preference_rows_skip_sequences_that_require_truncation() -> None:
    def messages(value: str):
        return [{"role": "user", "content": value}]

    rows = (
        {
            "chosen": messages("too-long"),
            "rejected": messages("short"),
            "score_chosen": 1.0,
            "score_rejected": 0.0,
        },
        {
            "chosen": messages("chosen"),
            "rejected": messages("rejected"),
            "score_chosen": 2.0,
            "score_rejected": -1.0,
        },
    )

    def tokenize(value):
        content = value[0]["content"]
        return [1, 2, 3, 4, 5] if content == "too-long" else [1, 2]

    selected = preference_rows(rows, count=1, maximum_length=4, tokenize=tokenize)
    assert selected[0]["source_index"] == 1
    assert selected[0]["chosen_ids"] == [1, 2]
    assert selected[0]["rejected_ids"] == [1, 2]


def test_job_preserves_frozen_batch_through_gradient_accumulation(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"training_rows": 512}\n')
    job = _representax_job(
        checkpoint=tmp_path / "checkpoint",
        data_directory=data,
        steps=4,
        seed=7,
    )
    assert job.training.global_batch_size == 128
    assert job.training.batch.micro_batch_size == MICRO_BATCH_SIZE
    assert job.training.batch.gradient_accumulation_steps == 32
    assert job.task.kind == "pairwise_reward"
    assert job.loss.kind == "bradley_terry"
    assert job.training.adapter is None
    assert job.evaluation is not None
    assert job.evaluation.primary_metric == "valid/ultrafeedback/pairwise_accuracy"
    assert job.export.huggingface is not None


def test_probe_job_has_no_lifecycle_artifacts_and_fixed_pair_shape(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"training_rows": 12}\n')
    job = _representax_job(
        checkpoint=tmp_path / "checkpoint",
        data_directory=data,
        steps=3,
        seed=7,
        global_batch_size=4,
        micro_batch_size=4,
        sequence_length_buckets=(1024,),
        lifecycle=False,
    )

    assert job.training.global_batch_size == 4
    assert job.training.batch.gradient_accumulation_steps == 1
    assert job.model.parameters["sequence_length_buckets"] == [1024]
    assert job.checkpointing is None
    assert job.evaluation is None
    assert not job.export.enabled


def test_optimizer_token_capacities_groups_microbatches() -> None:
    assert optimizer_token_capacities(
        ((8, 1024), (8, 512), (8, 256), (8, 128)),
        gradient_accumulation_steps=2,
    ) == [12288, 3072]


def test_steady_state_excludes_compilation_rows() -> None:
    result = steady_state(
        (
            {
                "event": "training_step",
                "metrics": {
                    "perf/compilation_and_first_step_seconds": 8.0,
                    "perf/step_seconds": 8.0,
                },
            },
            {"event": "training_step", "metrics": {"perf/step_seconds": 2.0}},
            {"event": "training_step", "metrics": {"perf/step_seconds": 4.0}},
        ),
        batch_size=12,
    )
    assert result == {
        "measured_steps": 2.0,
        "median_step_seconds": 3.0,
        "examples_per_second": 4.0,
    }


def test_reference_timing_separates_first_eager_step() -> None:
    result = reference_timing(
        (
            {"step": 3, "seconds": 3.0},
            {"step": 1, "seconds": 5.0},
            {"step": 2, "seconds": 1.0},
        ),
        batch_size=4,
    )
    assert result["first_step_seconds"] == 5.0
    assert result["compilation_seconds"] == 0.0
    assert result["warm_steps"] == 2
    assert result["warm_median_step_seconds"] == 2.0
    assert result["warm_examples_per_second"] == 2.0
