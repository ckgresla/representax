"""Pinned Sentence Transformers parity for Jina v5 Omni Small text."""

from __future__ import annotations

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import Route
from representax.models.jina_v5 import (
    JinaV5TextBatch,
    JinaV5TextCheckpointAdapter,
)
from tests.models.acceptance import NumericalTolerance, assert_numerically_equivalent

pytestmark = pytest.mark.parity


@pytest.mark.skipif(
    "REPRESENTAX_JINA_V5_SMALL_CHECKPOINT" not in os.environ
    or "REPRESENTAX_JINA_V5_SMALL_ORACLE" not in os.environ,
    reason=(
        "set REPRESENTAX_JINA_V5_SMALL_CHECKPOINT and "
        "REPRESENTAX_JINA_V5_SMALL_ORACLE for the pinned text gate"
    ),
)
@pytest.mark.parametrize("rematerialization", ["none", "selective", "full"])
def test_pinned_sentence_transformers_text_forward_parity(rematerialization):
    reference = np.load(Path(os.environ["REPRESENTAX_JINA_V5_SMALL_ORACLE"]))
    model = JinaV5TextCheckpointAdapter(
        rematerialization=rematerialization,
    ).load(
        Path(os.environ["REPRESENTAX_JINA_V5_SMALL_CHECKPOINT"]),
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
    )
    batch = JinaV5TextBatch(
        input_ids=jnp.asarray(reference["input_ids"]),
        attention_mask=jnp.asarray(reference["attention_mask"]),
        position_ids=jnp.asarray(reference["position_ids"]),
    )

    with jax.default_matmul_precision("highest"):
        actual = eqx.filter_jit(
            lambda candidate, values: candidate.encode(
                values,
                route=Route.GENERIC,
            )
        )(model, batch)

    assert_numerically_equivalent(
        np.asarray(actual, dtype=np.float32),
        reference["pooled"],
        NumericalTolerance(absolute=3e-3, relative=2e-2, cosine=0.9998),
    )
