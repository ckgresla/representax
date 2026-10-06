"""Real BidirLM Omni processor and Sentence Transformers inference parity."""

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
from representax.models.bidirlm_omni import load_bidirlm_omni
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.performance


def _path(environment: str, description: str) -> Path:
    value = os.environ.get(environment)
    if value is None:
        pytest.skip(f"set {environment} for real {description} acceptance")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


@eqx.filter_jit
def _encode(model, batch):
    return model.encode(batch, route=Route.GENERIC)


def test_real_preprocessing_and_inference_matches_sentence_transformers() -> None:
    checkpoint = _path("REPRESENTAX_BIDIRLM_OMNI_CHECKPOINT", "BidirLM Omni checkpoint")
    oracle = np.load(_path("REPRESENTAX_BIDIRLM_OMNI_ORACLE", "BidirLM Omni oracle"))
    image = Image.fromarray(oracle["source_pixels"])
    video = tuple(Image.fromarray(frame) for frame in oracle["source_video"])
    audio = {"array": oracle["source_audio"], "sampling_rate": 16_000}
    values = {
        "text": "A precise red square.",
        "image": image,
        "video": {"video": video},
        "audio": audio,
        "composed": {
            "image": image,
            "audio": audio,
            "text": "Compare these signals.",
        },
    }
    lengths = tuple(sorted(oracle[f"{name}__input_ids"].shape[1] for name in values))
    patch_counts = tuple(
        sorted(
            {
                oracle[field].shape[0]
                for name in values
                for field in (
                    f"{name}__pixel_values",
                    f"{name}__pixel_values_videos",
                )
                if field in oracle
            }
        )
    )
    model, processor = load_bidirlm_omni(
        checkpoint,
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=lengths,
        patch_count_buckets=patch_counts,
        audio_chunk_buckets=(1,),
        rematerialization="none",
    )
    for name, value in values.items():
        batch = processor((value,))
        valid = int(np.asarray(batch.attention_mask).sum())
        np.testing.assert_array_equal(
            np.asarray(batch.input_ids)[0, :valid],
            oracle[f"{name}__input_ids"][0],
        )
        np.testing.assert_array_equal(
            np.asarray(batch.attention_mask)[0, :valid],
            oracle[f"{name}__attention_mask"][0],
        )
        pixel_field = next(
            (
                field
                for field in (
                    f"{name}__pixel_values",
                    f"{name}__pixel_values_videos",
                )
                if field in oracle
            ),
            None,
        )
        if pixel_field is not None:
            assert batch.pixel_values is not None
            count = oracle[pixel_field].shape[0]
            np.testing.assert_array_equal(
                np.asarray(batch.pixel_values)[:count],
                oracle[pixel_field],
            )
        if f"{name}__input_features" in oracle:
            assert batch.input_features is not None
            frames = oracle[f"{name}__input_features"].shape[-1]
            np.testing.assert_array_equal(
                np.asarray(batch.input_features)[0, :, :frames],
                oracle[f"{name}__input_features"][0],
            )
        with jax.default_matmul_precision("highest"):
            actual = _encode(model, batch)
        expected = oracle[f"{name}__output"]
        result = numerical_result(actual, expected)
        print({"case": name, **result.__dict__})
        assert_numerically_equivalent(
            actual,
            expected,
            NumericalTolerance(absolute=0.25, relative=0.025, cosine=0.9995),
        )
