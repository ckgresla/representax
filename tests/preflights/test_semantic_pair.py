"""Contracts for the paper semantic and pair-classification preflight."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.preflights.semantic_pair import (
    PairEvaluationCollator,
    frozen_contract,
    normalize_sts_score,
    representax_steady_state,
    sprint_preflight_rows,
)


class _Processor:
    def __call__(self, rows):
        return np.arange(len(rows), dtype=np.float32)[:, None]

    def data_contract(self):
        return {"kind": "test"}


def test_frozen_contract_uses_the_shared_mpnet_sentence_transformers_cell():
    semantic = frozen_contract("semantic-similarity")
    pair = frozen_contract("pair-classification")
    assert (
        semantic.model_id == pair.model_id == "sentence-transformers/all-mpnet-base-v2"
    )
    assert semantic.batch_size == pair.batch_size == 256
    assert semantic.maximum_length == pair.maximum_length == 128
    assert semantic.reference_version == pair.reference_version == "5.6.1"


def test_sts_scores_are_normalized_for_cosine_regression():
    assert normalize_sts_score(0) == 0
    assert normalize_sts_score(2.5) == 0.5
    assert normalize_sts_score(5) == 1
    with pytest.raises(ValueError, match="outside"):
        normalize_sts_score(5.1)


def test_sprint_subsets_are_disjoint_and_retain_both_classes():
    packed = {
        "sent1": [f"left-{index}" for index in range(5_000)],
        "sent2": [f"right-{index}" for index in range(5_000)],
        "labels": [1] * 1_000 + [0] * 4_000,
    }
    training, evaluation = sprint_preflight_rows(packed)
    assert len(training) == 2_048
    assert len(evaluation) == 2_200
    assert {row["label"] for row in training} == {0, 1}
    assert {row["label"] for row in evaluation} == {0, 1}
    assert {row["source_index"] for row in training}.isdisjoint(
        row["source_index"] for row in evaluation
    )


def test_pair_evaluation_collator_pads_only_invalid_rows():
    batch = PairEvaluationCollator(processor=_Processor(), pad_to_size=4)(
        (
            {"sentence1": "a", "sentence2": "b", "label": 1},
            {"sentence1": "c", "sentence2": "d", "label": 0},
        )
    )
    np.testing.assert_array_equal(batch.labels, [1, 0, 0, 0])
    np.testing.assert_array_equal(batch.valid, [True, True, False, False])


def test_steady_state_uses_completed_steps_after_compilation():
    rows = (
        {
            "metrics": {
                "perf/compilation_and_first_step_seconds": 3.0,
                "perf/step_seconds": 3.0,
            }
        },
        {"metrics": {"perf/step_seconds": 2.0}},
        {"metrics": {"perf/step_seconds": 4.0}},
    )
    result = representax_steady_state(rows, batch_size=12)
    assert result == {
        "measured_steps": 2.0,
        "median_step_seconds": 3.0,
        "aggregate_examples_per_second": 4.0,
    }
