"""Real Transformers preprocessing and inference parity for Qwen rerankers."""

from __future__ import annotations

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.models.qwen_reranker import load_qwen_reranker
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.performance

CASES = (
    ("qwen3", "REPRESENTAX_QWEN3_RERANKER_CHECKPOINT"),
    ("contextual", "REPRESENTAX_CONTEXTUAL_RERANKER_CHECKPOINT"),
    ("mixedbread", "REPRESENTAX_MXBAI_RERANKER_CHECKPOINT"),
)


@eqx.filter_jit
def _logits_and_scores(model, batch):
    return model.logits(batch), model.score(batch)


@pytest.mark.parametrize(("name", "environment"), CASES)
def test_real_scores_and_preprocessing_match_sentence_transformers(
    name: str,
    environment: str,
) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Qwen reranker parity requires a GPU")
    checkpoint_value = os.environ.get(environment)
    oracle_directory = os.environ.get("REPRESENTAX_QWEN_RERANKER_ORACLES")
    if checkpoint_value is None or oracle_directory is None:
        pytest.skip(f"set {environment} and REPRESENTAX_QWEN_RERANKER_ORACLES")
    checkpoint = Path(checkpoint_value)
    oracle = np.load(Path(oracle_directory) / f"{name}.npz")
    length = int(oracle["input_ids"].shape[1])
    model, processor = load_qwen_reranker(
        checkpoint,
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(length,),
        rematerialization="none",
    )
    pairs = (
        (
            "Which planet is known as the Red Planet?",
            "Mars is often called the Red Planet because of iron oxide.",
        ),
        (
            "Which planet is known as the Red Planet?",
            "Venus has a dense carbon-dioxide atmosphere.",
        ),
    )
    batch = processor(pairs)
    np.testing.assert_array_equal(batch.input_ids, oracle["input_ids"])
    np.testing.assert_array_equal(batch.attention_mask, oracle["attention_mask"])
    with jax.default_matmul_precision("highest"):
        logits, scores = _logits_and_scores(model, batch)
    logit_result = numerical_result(logits, oracle["logits"])
    score_result = numerical_result(scores, oracle["scores"])
    print(
        {
            "checkpoint": name,
            "logits": logit_result.__dict__,
            "scores": score_result.__dict__,
        }
    )
    assert_numerically_equivalent(
        logits,
        oracle["logits"],
        NumericalTolerance(absolute=0.35, relative=0.08, cosine=0.999),
    )
    assert_numerically_equivalent(
        scores,
        oracle["scores"],
        NumericalTolerance(absolute=0.35, relative=0.08, cosine=0.999),
    )
