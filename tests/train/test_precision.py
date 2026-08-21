"""Explicit mixed-precision boundary contracts."""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from representax.config import PrecisionConfig
from representax.core import EncoderMetadata, Modality, Route, encode
from representax.models import DenseEncoder
from representax.precision import (
    model_for_compute,
    precision_context,
    resolve_precision_policy,
)
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import GradCache, build_train_step, init_train_state


class _IdentityEncoder(eqx.Module):
    scale: jax.Array
    metadata: EncoderMetadata

    def __init__(self) -> None:
        self.scale = jnp.asarray(1.0, dtype=jnp.float32)
        self.metadata = EncoderMetadata(
            model_id="representax/test-identity",
            revision="1",
            output_dimension=2,
            routes=frozenset(Route),
            modalities=frozenset(Modality),
        )

    def encode(
        self,
        inputs: jax.Array,
        *,
        route: Route,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del route, key
        return inputs * self.scale


def _inexact_dtypes(tree: Any) -> set[jnp.dtype]:
    return {leaf.dtype for leaf in jax.tree.leaves(tree) if eqx.is_inexact_array(leaf)}


def _assert_bfloat16_equivalent(actual: Any, expected: Any) -> None:
    """Accept one BF16 ULP across differently shaped compiled reductions."""

    actual_arrays = [leaf for leaf in jax.tree.leaves(actual) if eqx.is_array(leaf)]
    expected_arrays = [leaf for leaf in jax.tree.leaves(expected) if eqx.is_array(leaf)]
    assert len(actual_arrays) == len(expected_arrays)
    for actual_leaf, expected_leaf in zip(
        actual_arrays,
        expected_arrays,
        strict=True,
    ):
        if jnp.issubdtype(actual_leaf.dtype, jnp.inexact):
            np.testing.assert_allclose(
                actual_leaf,
                expected_leaf,
                rtol=2e-2,
                atol=2e-3,
            )
        else:
            np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _pairwise_batch():
    return pairwise_batch(
        left=jnp.asarray([[1.002, 0.0], [0.0, 0.998]], dtype=jnp.float32),
        right=jnp.asarray([[0.91, 0.09], [0.11, 0.89]], dtype=jnp.float32),
        labels=jnp.asarray([0.92, 0.84], dtype=jnp.float32),
    )


def _retrieval_batch():
    return retrieval_batch(
        query=jnp.eye(4, dtype=jnp.float32),
        document=jnp.asarray(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.0, 0.1, 0.8, 0.1],
                [0.1, 0.0, 0.1, 0.8],
            ],
            dtype=jnp.float32,
        ),
        positive_mask=jnp.eye(4, dtype=jnp.bool_),
    )


def test_bfloat16_policy_casts_only_the_transient_model_and_model_inputs():
    policy = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    master = _IdentityEncoder()
    compute = model_for_compute(master, policy)
    inputs = jnp.asarray([[1.002, 0.998]], dtype=jnp.float32)

    with precision_context(policy):
        representations = encode(compute, inputs)

    assert _inexact_dtypes(master) == {jnp.dtype(jnp.float32)}
    assert _inexact_dtypes(compute) == {jnp.dtype(jnp.bfloat16)}
    assert representations.dtype == jnp.dtype(jnp.float32)
    np.testing.assert_array_equal(
        representations,
        inputs.astype(jnp.bfloat16).astype(jnp.float32),
    )


def test_bfloat16_model_boundary_lowers_compute_but_returns_fp32():
    policy = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    model = DenseEncoder(2, 2, key=jax.random.key(2), normalize=False)
    inputs = jnp.asarray([[1.002, 0.998]], dtype=jnp.float32)

    @eqx.filter_jit
    def compiled(candidate: DenseEncoder, values: jax.Array) -> jax.Array:
        with precision_context(policy):
            return encode(candidate, values)

    stablehlo = cast(Any, compiled).lower(model, inputs).as_text()
    output = compiled(model, inputs)

    dot_lines = [
        line for line in stablehlo.splitlines() if "stablehlo.dot_general" in line
    ]
    assert dot_lines
    assert all("bf16" in line for line in dot_lines)
    assert output.dtype == jnp.dtype(jnp.float32)

    def objective(candidate: DenseEncoder) -> jax.Array:
        with precision_context(policy):
            return jnp.sum(encode(candidate, inputs))

    gradients = eqx.filter_grad(objective)(model)
    assert _inexact_dtypes(gradients) == {jnp.dtype(jnp.float32)}


def test_bfloat16_step_keeps_master_optimizer_gradients_and_loss_fp32():
    policy = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    model = DenseEncoder(2, 2, key=jax.random.key(3), normalize=False)
    optimizer = optax.adamw(1e-3, weight_decay=1e-2)
    state = init_train_state(model, optimizer, precision=policy)
    result = build_train_step(
        CosineRegressionTask(),
        optimizer,
        precision=policy,
    )(state, _pairwise_batch(), None)

    assert _inexact_dtypes(result.state.model) == {jnp.dtype(jnp.float32)}
    assert _inexact_dtypes(result.state.optimizer_state) == {jnp.dtype(jnp.float32)}
    assert result.metrics.loss.dtype == jnp.dtype(jnp.float32)
    assert all(
        not eqx.is_inexact_array(value) or value.dtype == jnp.dtype(jnp.float32)
        for value in jax.tree.leaves(result.metrics)
    )


def test_bfloat16_gradient_accumulation_matches_full_batch_update():
    policy = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    model = DenseEncoder(2, 2, key=jax.random.key(5), normalize=False)
    optimizer = optax.adamw(1e-3, weight_decay=0.0)
    state = init_train_state(model, optimizer, precision=policy)
    direct = build_train_step(
        CosineRegressionTask(),
        optimizer,
        max_grad_norm=None,
        precision=policy,
    )(state, _pairwise_batch(), None)
    accumulated = build_train_step(
        CosineRegressionTask(),
        optimizer,
        max_grad_norm=None,
        gradient_accumulation_steps=2,
        precision=policy,
    )(state, _pairwise_batch(), None)

    _assert_bfloat16_equivalent(accumulated, direct)


def test_bfloat16_grad_cache_matches_direct_update():
    policy = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    model = DenseEncoder(4, 3, key=jax.random.key(7), normalize=False)
    optimizer = optax.adamw(1e-3, weight_decay=0.0)
    state = init_train_state(model, optimizer, precision=policy)
    task = MNRTask(scale=5.0, symmetric=True)
    batch = _retrieval_batch()
    direct = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        precision=policy,
    )(state, batch, None)
    cached = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        execution=GradCache(query_chunk_size=2, document_chunk_size=2),
        precision=policy,
    )(state, batch, None)

    _assert_bfloat16_equivalent(cached, direct)
