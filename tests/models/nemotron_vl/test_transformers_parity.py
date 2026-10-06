"""Real Llama Nemotron VL preprocessing and inference parity."""

from __future__ import annotations

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from representax.core import Route
from representax.models.nemotron_vl import load_nemotron_vl
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.performance

_CHECKPOINTS = {
    "embedding": "REPRESENTAX_NEMOTRON_VL_EMBED_CHECKPOINT",
    "reranking": "REPRESENTAX_NEMOTRON_VL_RERANK_CHECKPOINT",
}
_ORACLES = {
    "embedding": "REPRESENTAX_NEMOTRON_VL_EMBED_ORACLE",
    "reranking": "REPRESENTAX_NEMOTRON_VL_RERANK_ORACLE",
}


def _path(environment: str, description: str) -> Path:
    value = os.environ.get(environment)
    if value is None:
        pytest.skip(f"set {environment} for real {description} acceptance")
    assert value is not None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


@eqx.filter_jit
def _encode(model, batch):
    return model.encode(batch, route=Route.GENERIC)


@pytest.mark.parametrize("mode", ("embedding", "reranking"))
def test_real_preprocessing_and_inference_matches_upstream(mode):
    checkpoint = _path(_CHECKPOINTS[mode], "Llama Nemotron VL checkpoint")
    oracle = np.load(_path(_ORACLES[mode], "Llama Nemotron VL oracle"))
    image = Image.fromarray(oracle["source_pixels"])
    query = "Which item contains a chart?"
    document = "A deterministic chart."
    if mode == "embedding":
        values = {
            "query": (query, Route.QUERY),
            "text": (document, Route.DOCUMENT),
            "image": (image, Route.DOCUMENT),
            "composed": (
                {"image": image, "text": document},
                Route.DOCUMENT,
            ),
        }
    else:
        values = {
            "text": (
                {"query": query, "text": document},
                Route.GENERIC,
            ),
            "image": (
                {"query": query, "image": image},
                Route.GENERIC,
            ),
            "composed": (
                {"query": query, "image": image, "text": document},
                Route.GENERIC,
            ),
        }
    lengths = tuple(
        sorted({int(oracle[f"{case}__input_ids"].shape[1]) for case in values})
    )
    tiles = tuple(
        sorted(
            {
                int(oracle[f"{case}__pixel_values"].shape[0])
                for case in values
                if f"{case}__pixel_values" in oracle
            }
        )
    )
    model, processor = load_nemotron_vl(
        checkpoint,
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=lengths,
        tile_count_buckets=tiles,
    )
    for case, (value, route) in values.items():
        batch = processor((value,), route=route)
        np.testing.assert_array_equal(
            np.asarray(batch.input_ids), oracle[f"{case}__input_ids"]
        )
        np.testing.assert_array_equal(
            np.asarray(batch.attention_mask), oracle[f"{case}__attention_mask"]
        )
        pixel_key = f"{case}__pixel_values"
        if pixel_key in oracle:
            assert batch.pixel_values is not None
            np.testing.assert_array_equal(
                np.asarray(batch.pixel_values).astype(np.float32),
                oracle[pixel_key],
            )
        else:
            assert batch.pixel_values is None
        with jax.default_matmul_precision("highest"):
            actual = _encode(model, batch)
        expected = oracle[f"{case}__output"].reshape(actual.shape)
        result = numerical_result(actual, expected)
        print({"mode": mode, "case": case, **result.__dict__})
        tolerance = (
            NumericalTolerance(absolute=1e-1, relative=3e-2, cosine=0.999)
            if mode == "embedding"
            else NumericalTolerance(absolute=2e-1, relative=5e-2, cosine=0.999)
        )
        assert_numerically_equivalent(actual, expected, tolerance)
