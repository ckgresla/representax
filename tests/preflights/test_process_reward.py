"""Paper process-reward preflight contract tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from experiments.preflights.process_reward import (
    EXECUTION_SEQUENCE_LENGTH,
    MICRO_BATCH_SIZE,
    STEPS_PER_TRAJECTORY,
    _representax_job,
    frozen_contract,
    representax_steady_state,
    tokenize_trajectory,
)


class _Tokenizer:
    bos_token_id = 1

    def __call__(self, value: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [10 + ord(character) % 7 for character in value]}

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        assert value == "\n"
        return [99]


def test_frozen_process_reward_contract() -> None:
    contract = frozen_contract()
    assert contract.model_id == "Qwen/Qwen3-0.6B"
    assert contract.dataset_id == "trl-lib/math_shepherd"
    assert contract.batch_size == 64
    assert contract.maximum_length == 2048
    assert contract.objective == "stepwise-binary-cross-entropy"
    assert contract.reference_version == "1.10.0"


def test_trajectory_tokenization_supervises_each_separator() -> None:
    tokenizer = _Tokenizer()
    ids, positions = tokenize_trajectory(tokenizer, "p", ("a", "bc"))
    assert ids == [1, 10, 16, 99, 10, 11, 99]
    assert positions == [3, 6]
    assert [ids[position] for position in positions] == [99, 99]


def test_scalar_binary_logit_parameterization_is_exact() -> None:
    scores = np.asarray([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float64)
    for label in (0.0, 1.0):
        binary = np.logaddexp(0.0, scores) - label * scores
        logits = np.stack((-0.5 * scores, 0.5 * scores), axis=-1)
        log_partition = np.logaddexp(logits[:, 0], logits[:, 1])
        cross_entropy = log_partition - logits[:, int(label)]
        np.testing.assert_allclose(binary, cross_entropy, rtol=1e-14, atol=1e-14)


def test_representax_job_preserves_frozen_scientific_contract(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.jsonl", "evaluation.jsonl"):
        (data / name).write_text("{}\n")
    job = _representax_job(
        checkpoint=tmp_path / "checkpoint",
        data_directory=data,
        steps=4,
        seed=7,
    )
    assert job.training.global_batch_size == 64
    assert job.training.batch.micro_batch_size == MICRO_BATCH_SIZE
    assert job.training.batch.gradient_accumulation_steps == 32
    assert job.training.activation_rematerialization == "full"
    assert job.training.precision.compute_dtype == "bfloat16"
    assert job.evaluation is not None
    assert job.evaluation.on_start and job.evaluation.on_end
    assert job.checkpointing is not None
    assert job.checkpointing.every == 2
    assert job.export.enabled
    assert job.export.huggingface is None
    assert job.model.parameters["sequence_length_buckets"][-1] == 2048
    assert EXECUTION_SEQUENCE_LENGTH == 256
    assert STEPS_PER_TRAJECTORY == 4


def test_steady_state_excludes_compilation_rows() -> None:
    rows = (
        {
            "metrics": {
                "perf/step_seconds": 10.0,
                "perf/compilation_and_first_step_seconds": 9.0,
            }
        },
        {"metrics": {"perf/step_seconds": 2.0}},
        {"metrics": {"perf/step_seconds": 4.0}},
    )
    result = representax_steady_state(rows, batch_size=64)
    assert result["measured_steps"] == 2
    assert result["median_step_seconds"] == 3.0
    assert math.isclose(result["examples_per_second"], 128 / 6)


def test_preflight_requires_a_resumable_midpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="even integer"):
        _representax_job(
            checkpoint=tmp_path,
            data_directory=tmp_path,
            steps=3,
            seed=7,
        )
