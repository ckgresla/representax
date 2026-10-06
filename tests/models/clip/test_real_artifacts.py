"""Real BGE-VL and legacy CLIP acceptance against Sentence Transformers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import Modality, Route
from representax.data import Artifact
from representax.models.clip import load_clip
from tests.models.acceptance import NumericalTolerance, assert_numerically_equivalent

pytestmark = pytest.mark.parity


def _checkpoint(variable: str) -> Path:
    value = os.environ.get(variable)
    if value is None:
        pytest.skip(f"set {variable} for real CLIP artifact parity")
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _upstream_python() -> str:
    value = os.environ.get("REPRESENTAX_SENTENCE_TRANSFORMERS_PYTHON")
    if value is None:
        pytest.skip("set REPRESENTAX_SENTENCE_TRANSFORMERS_PYTHON for parity")
    return value


def _reference(
    checkpoint: Path,
    output: Path,
    *,
    composition: bool,
) -> np.lib.npyio.NpzFile:
    command: list[str] = [
        _upstream_python(),
        "-m",
        "tests.models.clip.sentence_transformers_oracle",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    if composition:
        command.append("--composition")
    environment: dict[str, str] = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(Path.cwd()),
    }
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("USE_TORCH", None)
    subprocess.run(command, check=True, cwd=Path.cwd(), env=environment)
    return np.load(output)


def _image():
    from PIL import Image

    pixels = (
        np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3) % 251
    ).astype(np.uint8)
    return Image.fromarray(pixels)


@pytest.mark.parametrize(
    ("variable", "composition", "parameter_dtype", "tolerance"),
    (
        (
            "REPRESENTAX_BGE_VL_CHECKPOINT",
            True,
            jnp.float32,
            NumericalTolerance(absolute=5e-4, relative=2e-3, cosine=0.99999),
        ),
        (
            "REPRESENTAX_CLIP_CHECKPOINT",
            False,
            jnp.float32,
            NumericalTolerance(absolute=4e-3, relative=2e-3, cosine=0.99999),
        ),
    ),
)
def test_real_processor_inference_and_hf_layout(
    variable,
    composition,
    parameter_dtype,
    tolerance,
    tmp_path,
):
    checkpoint = _checkpoint(variable)
    reference = _reference(
        checkpoint,
        tmp_path / f"{variable}.npz",
        composition=composition,
    )
    model, processor = load_clip(
        checkpoint,
        local_files_only=True,
        parameter_dtype=parameter_dtype,
        compute_dtype=parameter_dtype,
    )
    image = _image()
    caption = "A deterministic caption."
    text_batch = processor([Artifact.text(caption)])
    image_batch = processor([Artifact.inline(Modality.IMAGE, image)])
    np.testing.assert_array_equal(text_batch.input_ids, reference["input_ids"])
    np.testing.assert_array_equal(
        text_batch.attention_mask,
        reference["attention_mask"],
    )
    np.testing.assert_allclose(
        image_batch.pixel_values,
        reference["pixel_values"],
        rtol=0,
        atol=0,
    )

    @eqx.filter_jit
    def encode(candidate, batch):
        return candidate.encode(batch, route=Route.GENERIC)

    batches = {"text": text_batch, "image": image_batch}
    if composition:
        batches["composed"] = processor(
            [
                {
                    "text": Artifact.text(caption),
                    "image": Artifact.inline(Modality.IMAGE, image),
                }
            ]
        )
    for name, batch in batches.items():
        actual = encode(model, batch)
        jax.block_until_ready(actual)
        result = assert_numerically_equivalent(
            np.asarray(actual), reference[name], tolerance
        )
        print(variable, name, result)
