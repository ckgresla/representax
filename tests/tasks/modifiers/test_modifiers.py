from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from representax.core import EncoderMetadata, Modality, Route
from representax.tasks.modifiers import (
    AdaptiveLayerTask,
    Matryoshka2dTask,
    MatryoshkaTask,
)
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import GradCache, build_train_step, init_train_state


class _LayerEncoder(eqx.Module):
    projection: jax.Array
    metadata: EncoderMetadata

    def encode(self, inputs, *, route, key=None):
        del route, key
        layerwise = self._project(inputs)
        return layerwise[-1]

    def encode_layers(self, inputs, *, route, key=None):
        del route, key
        return self._project(inputs)

    def _project(self, inputs):
        values = jnp.einsum("blf,df->bld", inputs, self.projection)
        values = jnp.transpose(values, (1, 0, 2))
        return values / jnp.maximum(
            jnp.linalg.norm(values, axis=-1, keepdims=True),
            1e-12,
        )


def _fixture():
    dimension = 6
    model = _LayerEncoder(
        projection=jnp.eye(dimension, dtype=jnp.float32),
        metadata=EncoderMetadata(
            model_id="tests/layer-encoder",
            revision="1",
            output_dimension=dimension,
            routes=frozenset(Route),
            modalities=frozenset({Modality.TEXT}),
        ),
    )
    query = jax.random.normal(jax.random.key(1), (5, 4, dimension))
    document = jax.random.normal(jax.random.key(2), (5, 4, dimension))
    batch = retrieval_batch(
        query=query,
        document=document,
        positive_mask=jnp.eye(5, dtype=jnp.bool_),
    )
    return model, batch


def test_matryoshka_reuses_one_representation_pass_and_preserves_raw_weights():
    model, batch = _fixture()
    base = MNRTask(scale=7.0)
    task = MatryoshkaTask(base, (3, 6), weights=(0.5, 2.0))

    output = task.loss(model, batch)
    representations = base.representations(model, batch)
    expected = sum(
        weight
        * base.loss_from_representations(
            jax.tree.map(
                lambda value, dim=dimension: (
                    value[..., :dim]
                    / jnp.linalg.norm(
                        value[..., :dim],
                        axis=-1,
                        keepdims=True,
                    )
                ),
                representations,
            ),
            batch,
        ).loss
        for dimension, weight in ((6, 2.0), (3, 0.5))
    )

    assert task.dimensions == (6, 3)
    assert jnp.allclose(output.loss, expected, rtol=2e-6, atol=2e-6)


def test_adaptive_layer_and_matryoshka_2d_run_through_compiled_train_step():
    model, batch = _fixture()
    base = MNRTask(scale=7.0)
    tasks = (
        AdaptiveLayerTask(base, layers_per_step=-1),
        Matryoshka2dTask(
            base,
            (6, 3),
            dimensions_per_step=-1,
            layers_per_step=-1,
        ),
    )

    for task in tasks:
        optimizer = optax.adamw(1e-3)
        state = init_train_state(model, optimizer)
        result = build_train_step(task, optimizer)(state, batch, jax.random.key(3))
        assert bool(result.metrics.numeric_finite)
        assert int(result.state.step) == 1
        updated_model = cast(_LayerEncoder, result.state.model)
        assert not jnp.array_equal(updated_model.projection, model.projection)


def test_random_modifier_selection_is_explicitly_keyed():
    model, batch = _fixture()
    task = MatryoshkaTask(
        MNRTask(scale=7.0),
        (6, 4, 2),
        dimensions_per_step=1,
    )

    try:
        task.loss(model, batch)
    except ValueError as error:
        assert "requires a JAX key" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("unkeyed random modifier selection must fail")

    output = task.loss(model, batch, key=jax.random.key(4))
    assert int(jnp.sum(output.metrics["selected_dimensions"])) == 1


def test_matryoshka_grad_cache_preserves_loss_and_optimizer_update():
    model, batch = _fixture()
    task = MatryoshkaTask(
        MNRTask(scale=7.0),
        (6, 3),
        weights=(1.0, 0.5),
    )
    optimizer = optax.adamw(1e-3)
    state = init_train_state(model, optimizer)
    direct = build_train_step(task, optimizer)(state, batch, jax.random.key(9))
    cached = build_train_step(
        task,
        optimizer,
        execution=GradCache(
            query_chunk_size=2,
            document_chunk_size=3,
            loss_row_chunk_size=2,
        ),
    )(state, batch, jax.random.key(9))

    np.testing.assert_allclose(
        direct.metrics.loss,
        cached.metrics.loss,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        cast(_LayerEncoder, direct.state.model).projection,
        cast(_LayerEncoder, cached.state.model).projection,
        rtol=2e-5,
        atol=2e-5,
    )
