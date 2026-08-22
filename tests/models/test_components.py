"""Reusable native model-component contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from representax.models import (
    LayerNorm,
    Linear,
    RMSNorm,
    embedding_lookup,
    l2_normalize,
    mean_pool,
)


def test_linear_uses_hugging_face_weight_orientation():
    linear = Linear(
        weight=jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        bias=jnp.asarray([0.5, -0.5]),
    )
    actual = linear(jnp.asarray([[2.0, 1.0]]))
    np.testing.assert_allclose(actual, [[4.5, 9.5]])


def test_layer_norm_matches_jax_reference_with_fp32_statistics():
    layer_norm = LayerNorm(
        weight=jnp.asarray([1.5, 0.5, 2.0]),
        bias=jnp.asarray([0.1, -0.2, 0.3]),
        epsilon=1e-5,
    )
    value = jnp.asarray([[1.0, 2.0, 4.0]], dtype=jnp.float32)
    expected = jax.nn.standardize(value, axis=-1, epsilon=1e-5)
    assert layer_norm.bias is not None
    expected = expected * layer_norm.weight + layer_norm.bias
    np.testing.assert_allclose(layer_norm(value), expected, rtol=1e-6, atol=1e-6)


def test_rms_norm_matches_transformers_bfloat16_rounding_boundary():
    value = jnp.asarray([[0.125, -0.875, 1.75]], dtype=jnp.bfloat16)
    weight = jnp.asarray([0.8125, 1.125, -0.6875], dtype=jnp.bfloat16)
    norm = RMSNorm(weight=weight, epsilon=1e-6)
    numeric = value.astype(jnp.float32)
    normalized = (
        numeric
        * jax.lax.rsqrt(jnp.mean(jnp.square(numeric), axis=-1, keepdims=True) + 1e-6)
    ).astype(jnp.bfloat16)
    expected = normalized * weight
    np.testing.assert_array_equal(norm(value), expected)


def test_embedding_lookup_accumulates_repeated_table_gradients():
    table = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    indices = jnp.asarray([[1, 1, 3]])
    gradient = jax.grad(lambda value: jnp.sum(embedding_lookup(value, indices)))(table)
    expected = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(gradient, expected)


def test_embedding_lookup_supports_forward_mode_autodiff():
    table = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    tangent = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    indices = jnp.asarray([[1, 1, 3]])

    _, actual = jax.jvp(
        lambda value: embedding_lookup(value, indices),
        (table,),
        (tangent,),
    )

    np.testing.assert_array_equal(actual, tangent[indices])


def test_pooling_and_normalization_ignore_padding():
    hidden = jnp.asarray(
        [
            [[3.0, 4.0], [0.0, 10.0], [100.0, 100.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = jnp.asarray([[1, 1, 0], [0, 0, 0]])
    pooled = mean_pool(hidden, mask)
    normalized = l2_normalize(pooled)
    np.testing.assert_allclose(pooled[0], [1.5, 7.0])
    np.testing.assert_allclose(jnp.linalg.norm(normalized[0]), 1.0)
    np.testing.assert_array_equal(normalized[1], jnp.zeros((2,)))
