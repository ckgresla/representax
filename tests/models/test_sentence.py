"""Dense sentence-composition numerical contracts."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from representax.models import SentenceDense, SentenceNormalize, SentencePooling
from representax.models.components import Linear


def test_pooling_supports_every_released_padded_mode_in_canonical_order():
    hidden = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            [[9.0, 9.0], [2.0, 6.0], [4.0, 8.0]],
        ]
    )
    mask = jnp.asarray([[1, 1, 0], [0, 1, 1]])
    pooling = SentencePooling(
        input_dimension=2,
        modes=(
            "cls",
            "max",
            "mean",
            "mean_sqrt_len_tokens",
            "weightedmean",
            "lasttoken",
        ),
    )

    actual = pooling(hidden, mask)
    expected = np.asarray(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
                2.0,
                3.0,
                4.0 / np.sqrt(2.0),
                6.0 / np.sqrt(2.0),
                7.0 / 3.0,
                10.0 / 3.0,
                3.0,
                4.0,
            ],
            [
                2.0,
                6.0,
                4.0,
                8.0,
                3.0,
                7.0,
                6.0 / np.sqrt(2.0),
                14.0 / np.sqrt(2.0),
                16.0 / 5.0,
                36.0 / 5.0,
                4.0,
                8.0,
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert pooling.output_dimension == 12


def test_dense_projection_and_normalization_compose_as_plain_equinox_modules():
    dense = SentenceDense(
        linear=Linear(
            weight=jnp.asarray([[1.0, 2.0], [-1.0, 1.0]]),
            bias=jnp.asarray([0.5, -0.5]),
        ),
        activation="tanh",
    )
    normalized = SentenceNormalize()(dense(jnp.asarray([[2.0, 1.0]])))

    expected = np.tanh(np.asarray([[4.5, -1.5]], dtype=np.float32))
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-6)
    assert dense.output_dimension == 2
