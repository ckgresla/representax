"""Real-checkpoint acceptance for native GTE-ModernColBERT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import jax
import numpy as np
import optax
import pytest

from representax.core import Route, encode_late_interaction
from representax.integrations import (
    GTE_MODERN_COLBERT_MODEL_ID,
    GTE_MODERN_COLBERT_REVISION,
    load_late_interaction_text_model,
)
from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models import LateInteractionTextEncoder
from representax.tasks.late_interaction import LateInteractionTask
from representax.tasks.retrieval import retrieval_batch
from representax.train import build_train_step, init_train_state
from tests.models.acceptance import NumericalTolerance, assert_numerically_equivalent

_ORACLE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "late_interaction"
    / "gte-modern-colbert-v1"
)


def _load_model():
    return load_late_interaction_text_model(
        GTE_MODERN_COLBERT_MODEL_ID,
        revision=GTE_MODERN_COLBERT_REVISION,
        local_files_only=True,
        rematerialization="full",
    )


@pytest.mark.parity
def test_real_checkpoint_preprocessing_projection_and_mask_match_pylate():
    metadata = json.loads((_ORACLE / "metadata.json").read_text())
    reference = np.load(_ORACLE / "oracle.npz")
    model, processor = _load_model()

    for name, texts, route in (
        ("query", metadata["queries"], Route.QUERY),
        ("document", metadata["documents"], Route.DOCUMENT),
    ):
        batch = processor(texts, route=route)
        np.testing.assert_array_equal(batch.input_ids, reference[f"{name}_input_ids"])
        np.testing.assert_array_equal(
            batch.attention_mask,
            reference[f"{name}_attention_mask"],
        )
        native = encode_late_interaction(model, batch, route=route)
        np.testing.assert_array_equal(native.valid, reference[f"{name}_valid"])
        assert_numerically_equivalent(
            np.asarray(native.values),
            reference[f"{name}_normalized"],
            NumericalTolerance(
                absolute=2e-4,
                relative=5e-4,
                cosine=0.9999998,
            ),
        )


@pytest.mark.parity
@pytest.mark.runtime
def test_real_checkpoint_trains_three_updates_and_exports_exactly(tmp_path):
    metadata = json.loads((_ORACLE / "metadata.json").read_text())
    model, processor = _load_model()
    query = processor(metadata["queries"], route=Route.QUERY)
    document = processor(metadata["documents"], route=Route.DOCUMENT)
    batch = retrieval_batch(
        query=query,
        document=document,
        positive_mask=np.eye(2, dtype=np.bool_),
    )
    task = LateInteractionTask(temperature=0.07)
    optimizer = optax.adamw(learning_rate=1e-6)
    state = init_train_state(model, optimizer)
    initial_projection = np.asarray(model.projection.weight)
    step = build_train_step(task, optimizer, max_grad_norm=1.0)
    losses = []
    for iteration in range(3):
        result = step(state, batch, jax.random.fold_in(jax.random.key(17), iteration))
        state = result.state
        losses.append(float(result.metrics.loss))
    assert np.isfinite(losses).all()
    trained = cast(LateInteractionTextEncoder, state.model)
    assert not np.array_equal(
        initial_projection,
        np.asarray(trained.projection.weight),
    )

    checkpoint = resolve_hf_checkpoint(
        GTE_MODERN_COLBERT_MODEL_ID,
        revision=GTE_MODERN_COLBERT_REVISION,
        local_files_only=True,
    ).path
    exported = trained.save_to_hf(
        tmp_path / "gte-modern-colbert",
        source_checkpoint=checkpoint,
    )
    restored, _ = load_late_interaction_text_model(
        exported,
        local_files_only=True,
        rematerialization="full",
    )
    expected = encode_late_interaction(trained, query, route=Route.QUERY)
    actual = encode_late_interaction(restored, query, route=Route.QUERY)
    for left, right in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(left, right)
