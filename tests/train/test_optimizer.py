"""Declarative optimizer construction and train-state initialization tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.config import ComponentConfig, OptimizationConfig
from representax.models import DenseEncoder
from representax.train import build_optimizer, init_train_state, make_train_state


def test_build_optimizer_initializes_adamw_state_from_config():
    model = DenseEncoder(4, 3, key=jax.random.key(0))
    config = OptimizationConfig(
        optimizer=ComponentConfig(
            target="optax.adamw",
            parameters={"learning_rate": 1e-3, "weight_decay": 0.0},
        )
    )

    optimizer = build_optimizer(config)
    state = init_train_state(model, optimizer)

    parameters = eqx.filter(model, eqx.is_inexact_array)
    adam_state = state.optimizer_state[0]
    assert jax.tree.structure(adam_state.mu) == jax.tree.structure(parameters)
    assert jax.tree.structure(adam_state.nu) == jax.tree.structure(parameters)
    assert int(adam_state.count) == 0
    for moment in (adam_state.mu, adam_state.nu):
        for leaf in jax.tree.leaves(moment):
            if eqx.is_inexact_array(leaf):
                np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


def test_make_train_state_remains_a_compatibility_alias():
    assert make_train_state is init_train_state


@pytest.mark.parametrize(
    ("target", "error", "message"),
    [
        ("adamw", ValueError, "dotted import path"),
        ("optax.missing_optimizer", AttributeError, "has no attribute"),
        ("optax.__version__", TypeError, "is not callable"),
        ("builtins.dict", TypeError, "must return an Optax"),
    ],
)
def test_build_optimizer_rejects_invalid_targets(target, error, message):
    config = OptimizationConfig(optimizer=ComponentConfig(target=target))

    with pytest.raises(error, match=message):
        build_optimizer(config)
