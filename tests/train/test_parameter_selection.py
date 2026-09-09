"""Independent adapter insertion, parameter selection, and sparse embedding rows."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.config import BatchConfig, LoRAConfig, TrainingConfig
from representax.models.components import EmbeddingRows, Linear, embedding_lookup
from representax.train.job import prepare_model


class Model(eqx.Module):
    tower: Linear
    connector: Linear
    embedding: jax.Array | EmbeddingRows
    counter: jax.Array

    @classmethod
    def init(cls):
        return cls(
            Linear.init(4, 4, key=jax.random.key(0), scale=0.1, dtype=jnp.float32),
            Linear.init(4, 4, key=jax.random.key(1), scale=0.1, dtype=jnp.float32),
            jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 32,
            jnp.asarray(1),
        )


@pytest.mark.parametrize("adapter", [None, LoRAConfig(rank=2, alpha=4)])
def test_explicit_filter_is_independent_of_adapter_insertion(adapter):
    model, selected = prepare_model(
        Model.init(),
        adapter=adapter,
        key=jax.random.key(2),
        trainable_pattern=r"\.connector\.|\.lora_[ab]$",
    )
    assert selected.connector.weight and not selected.tower.weight
    assert not selected.embedding and not selected.counter
    if adapter:
        assert selected.tower.lora_a and selected.tower.lora_b
    else:
        assert isinstance(model.tower, Linear)


def test_defaults_and_explicit_all_selection():
    _, plain = prepare_model(Model.init(), adapter=None, key=jax.random.key(2))
    assert plain is eqx.is_inexact_array
    adapter = LoRAConfig(rank=2, alpha=4)
    _, selected = prepare_model(Model.init(), adapter=adapter, key=jax.random.key(2))
    assert selected.tower.lora_a and not selected.tower.weight
    _, selected = prepare_model(
        Model.init(), adapter=adapter, key=jax.random.key(2), trainable_pattern=".*"
    )
    assert selected.tower.lora_a and selected.tower.weight
    assert not selected.counter


@pytest.mark.parametrize("pattern", ["missing", r"\.counter$"])
def test_empty_parameter_selection_fails(pattern):
    with pytest.raises(ValueError, match="matched no"):
        prepare_model(
            Model.init(), adapter=None, key=jax.random.key(2), trainable_pattern=pattern
        )


def test_embedding_rows_match_dense_gradients_and_keep_unselected_rows_frozen():
    original = Model.init()
    model, selected = prepare_model(
        original,
        adapter=LoRAConfig(rank=2, alpha=4),
        key=jax.random.key(2),
        trainable_pattern=r"\.connector\.|\.lora_[ab]$",
        trainable_embedding_rows={".embedding": (2, 5)},
    )
    ids = jnp.array([2, 5, 2, 1, 7])
    np.testing.assert_array_equal(
        embedding_lookup(model.embedding, ids), original.embedding[ids]
    )
    assert not selected.embedding.base and selected.embedding.rows
    trainable, frozen = eqx.partition(model, selected)
    optimizer = optax.adamw(1e-2, weight_decay=0.1)
    state = optimizer.init(trainable)
    assert sum(
        x.size for x in jax.tree.leaves(state) if eqx.is_inexact_array(x)
    ) == 2 * sum(x.size for x in jax.tree.leaves(trainable) if eqx.is_inexact_array(x))

    def loss(candidate):
        complete = eqx.combine(candidate, frozen)
        values = embedding_lookup(complete.embedding, ids)
        return jnp.square(complete.connector(complete.tower(values)) - 1).sum()

    for _ in range(2):
        gradients = eqx.filter_grad(loss)(trainable)
        updates, state = optimizer.update(gradients, state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
    trained = eqx.combine(trainable, frozen)
    np.testing.assert_array_equal(trained.embedding.base, original.embedding)
    np.testing.assert_array_equal(trained.tower.weight, original.tower.weight)
    assert not np.array_equal(trained.connector.weight, original.connector.weight)
    assert not np.array_equal(
        trained.embedding.rows, original.embedding[jnp.array([2, 5])]
    )
    merged = trained.embedding.merge()
    np.testing.assert_array_equal(
        merged[jnp.array([0, 1, 3, 4, 6, 7])],
        original.embedding[jnp.array([0, 1, 3, 4, 6, 7])],
    )
    np.testing.assert_array_equal(embedding_lookup(trained.embedding, ids), merged[ids])
    dense_gradient = jax.grad(
        lambda table: jnp.square(embedding_lookup(table, ids)).sum()
    )(original.embedding)
    sparse_gradient = eqx.filter_grad(
        lambda table: jnp.square(embedding_lookup(table, ids)).sum()
    )(model.embedding)
    np.testing.assert_array_equal(
        sparse_gradient.rows, dense_gradient[jnp.array([2, 5])]
    )


@pytest.mark.parametrize("rows", [(2, 2), (), (8,), (-1,)])
def test_invalid_embedding_rows_fail(rows):
    with pytest.raises(ValueError):
        EmbeddingRows.from_array(Model.init().embedding, rows)


def test_selection_config_validation_and_roundtrip():
    config = TrainingConfig(
        global_batch_size=4,
        max_steps=2,
        seed=7,
        batch=BatchConfig(micro_batch_size=4),
        trainable_pattern=r"\.connector\.",
        trainable_embedding_rows={".embedding": (2, 5)},
    )
    assert TrainingConfig.model_validate_json(config.model_dump_json()) == config
    data = config.model_dump()
    data["trainable_pattern"] = "["
    with pytest.raises(ValueError, match="invalid trainable_pattern"):
        TrainingConfig.model_validate(data)
