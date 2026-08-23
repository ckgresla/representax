"""Real LLaVA-NeXT preprocessing and inference acceptance."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import Route
from representax.models.llava_next import load_llava_next
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.performance

_ENVIRONMENTS = {
    "bge-s1": "REPRESENTAX_BGE_VL_MLLM_S1_CHECKPOINT",
    "bge-s2": "REPRESENTAX_BGE_VL_MLLM_S2_CHECKPOINT",
    "bge-v15-zs": "REPRESENTAX_BGE_VL_V15_ZS_CHECKPOINT",
    "bge-v15-mmeb": "REPRESENTAX_BGE_VL_V15_MMEB_CHECKPOINT",
    "e5-v": "REPRESENTAX_E5_V_CHECKPOINT",
}


def _checkpoint(variant: str) -> Path:
    value = os.environ.get(_ENVIRONMENTS[variant])
    if value is None:
        pytest.skip(f"set {_ENVIRONMENTS[variant]} for real LLaVA-NeXT acceptance")
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _oracle(variant: str, checkpoint: Path, output: Path):
    is_e5 = variant == "e5"
    python = os.environ.get(
        "REPRESENTAX_E5_SENTENCE_TRANSFORMERS_PYTHON"
        if is_e5
        else "REPRESENTAX_SENTENCE_TRANSFORMERS_PYTHON",
        sys.executable,
    )
    expected_version = os.environ.get(
        "REPRESENTAX_E5_SENTENCE_TRANSFORMERS_VERSION"
        if is_e5
        else "REPRESENTAX_SENTENCE_TRANSFORMERS_VERSION",
        "5.4.0" if is_e5 else "5.6.1",
    )
    environment: dict[str, str] = {
        **dict(os.environ),
        "PYTHONPATH": str(Path.cwd()),
        "CUDA_VISIBLE_DEVICES": os.environ.get(
            "REPRESENTAX_ORACLE_CUDA_VISIBLE_DEVICES", "5"
        ),
    }
    environment.pop("LD_LIBRARY_PATH", None)
    subprocess.run(
        [
            python,
            "-m",
            "tests.models.llava_next.sentence_transformers_oracle",
            "--variant",
            variant,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--expected-sentence-transformers-version",
            expected_version,
        ],
        check=True,
        cwd=Path.cwd(),
        env=environment,
    )
    return np.load(output)


@eqx.filter_jit
def _encode(model, batch):
    return model.encode(batch, route=Route.QUERY)


@pytest.mark.parametrize("variant", tuple(_ENVIRONMENTS))
def test_real_preprocessing_and_inference_matches_sentence_transformers(
    variant, tmp_path
):
    checkpoint = _checkpoint(variant)
    family = "e5" if variant == "e5-v" else "bge"
    reference = _oracle(family, checkpoint, tmp_path / f"{variant}.npz")
    from PIL import Image

    image = Image.fromarray(reference["source_pixels"])
    values = {
        "text": "A chart with deterministic colored pixels.",
        "image": image,
        "composed": {
            "text": "A chart with deterministic colored pixels.",
            "image": image,
        },
    }
    sequence_buckets = tuple(
        sorted({int(reference[f"{case}__input_ids"].shape[1]) for case in values})
    )
    tile_buckets = tuple(
        sorted(
            {
                int(reference[f"{case}__pixel_values"].shape[1])
                for case in values
                if f"{case}__pixel_values" in reference
            }
        )
    )
    model, processor = load_llava_next(
        checkpoint,
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=sequence_buckets,
        image_count_buckets=(1,),
        tile_count_buckets=tile_buckets,
    )
    batches = {}
    for case, value in values.items():
        batch = processor(
            (value,),
            route=Route.QUERY,
            instruction=(
                "Retrieve the visual document that answers the question."
                if family == "bge"
                else None
            ),
        )
        batches[case] = batch
        np.testing.assert_array_equal(
            np.asarray(batch.input_ids), reference[f"{case}__input_ids"]
        )
        np.testing.assert_array_equal(
            np.asarray(batch.attention_mask),
            reference[f"{case}__attention_mask"],
        )
        pixel_key = f"{case}__pixel_values"
        if pixel_key in reference:
            assert batch.pixel_values is not None
            np.testing.assert_allclose(
                np.asarray(batch.pixel_values), reference[pixel_key], rtol=0, atol=0
            )
        else:
            assert batch.pixel_values is None
    batch = batches["composed"]
    actual = _encode(model, batch)
    expected = reference["embedding"].reshape(actual.shape)
    print({"variant": variant, **numerical_result(actual, expected).__dict__})
    assert_numerically_equivalent(
        actual,
        expected,
        NumericalTolerance(absolute=2e-2, relative=1e-1, cosine=0.995),
    )
