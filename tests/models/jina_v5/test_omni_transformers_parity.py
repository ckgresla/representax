"""Released-checkpoint parity for every Jina v5 Omni execution path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import Route
from representax.models.jina_v5 import load_jina_v5_omni

pytestmark = [pytest.mark.parity, pytest.mark.performance]

_CHECKPOINT_ENVIRONMENTS = {
    "nano": "REPRESENTAX_JINA_V5_OMNI_NANO_CHECKPOINT",
    "small": "REPRESENTAX_JINA_V5_OMNI_SMALL_CHECKPOINT",
}


def _cases() -> dict[str, object]:
    image = np.arange(56 * 56 * 3, dtype=np.uint8).reshape(56, 56, 3)
    video = np.stack((image, image[::-1]))
    audio = np.sin(np.arange(16_000, dtype=np.float32) / 100)
    return {
        "text": "one two",
        "image": {"image": image},
        "audio": {"audio": audio},
        "video": {"video": video},
        "fused": {
            "text": "one",
            "image": image,
            "audio": audio,
            "video": video,
        },
    }


@pytest.mark.parametrize("variant", ["nano", "small"])
def test_all_paths_match_transformers_in_float32(variant: str, tmp_path: Path) -> None:
    environment = _CHECKPOINT_ENVIRONMENTS[variant]
    value = os.environ.get(environment)
    if value is None:
        pytest.skip(f"set {environment} for Jina v5 Omni parity")
    checkpoint = Path(value)
    oracle = tmp_path / f"{variant}-oracle.npz"
    subprocess.run(
        [
            sys.executable,
            "tests/models/jina_v5/omni_transformers_oracle.py",
            str(checkpoint),
            str(oracle),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
    )
    reference = np.load(oracle)
    model, processor = load_jina_v5_omni(
        checkpoint,
        local_files_only=True,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        sequence_length_buckets=(128,),
        patch_count_buckets=(8,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(32,),
        image_min_pixels=56 * 56,
        image_max_pixels=56 * 56,
        video_min_pixels=56 * 56,
        video_max_pixels=56 * 56,
        rematerialization="none",
    )
    encode = jax.jit(
        lambda candidate, batch: candidate.encode(batch, route=Route.QUERY)
    )
    with jax.default_matmul_precision("highest"):
        for name, sample in _cases().items():
            batch = processor([sample], route=Route.QUERY)
            actual = np.asarray(encode(model, batch))
            np.testing.assert_allclose(
                actual,
                reference[name],
                atol=1e-4,
                rtol=1e-4,
                err_msg=f"{variant} {name} path",
            )
            cosine = np.sum(actual * reference[name], axis=-1)
            np.testing.assert_allclose(cosine, 1.0, atol=5e-7, rtol=0)
