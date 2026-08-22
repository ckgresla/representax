"""Real-checkpoint parity for official multimodal Sentence Transformers models."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from representax.core import Route
from representax.core.sharding import activation_sharding
from representax.models.qwen2_vl import (
    Qwen2VLEncoder,
    load_qwen2_vl_embedding,
    load_qwen2_vl_reranker,
    vision_layout,
)
from representax.train import fsdp_parameter_specs, place_model
from tests.models.acceptance import NumericalTolerance, assert_numerically_equivalent

pytestmark = pytest.mark.performance

_CHECKPOINT_ENVIRONMENTS = {
    "bge": "REPRESENTAX_BGE_VL_SCREENSHOT_CHECKPOINT",
    "nomic3": "REPRESENTAX_NOMIC_MULTIMODAL_3B_CHECKPOINT",
    "nomic7": "REPRESENTAX_NOMIC_MULTIMODAL_7B_CHECKPOINT",
    "jina": "REPRESENTAX_JINA_RERANKER_M0_CHECKPOINT",
}


@eqx.filter_jit
def _encode(model, batch, *, route: Route):
    return model.encode(batch, route=route)


@eqx.filter_jit
def _score(model, batch):
    return model.score(batch)


def _fsdp_encode(model, batch, *, route: Route, mesh):
    replicated = NamedSharding(mesh, P())
    batch = jax.tree.map(
        lambda value: (
            jax.device_put(value, replicated) if eqx.is_array(value) else value
        ),
        batch,
        is_leaf=lambda value: value is None,
    )

    def encode(candidate, values):
        with activation_sharding(mesh, None):
            return candidate.encode(values, route=route)

    return eqx.filter_jit(encode)(model, batch)


def _checkpoint(variant: str) -> Path:
    environment = _CHECKPOINT_ENVIRONMENTS[variant]
    value = os.environ.get(environment)
    if value is None:
        pytest.skip(f"set {environment} for real-checkpoint acceptance")
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _oracle(
    variant: str, checkpoint: Path, output: Path
) -> tuple[np.lib.npyio.NpzFile, dict[str, object]]:
    precomputed_directory = os.environ.get("REPRESENTAX_QWEN2_VL_ORACLE_DIRECTORY")
    if precomputed_directory is not None:
        oracle_variant = "nomic7-fp32" if variant == "nomic7" else variant
        precomputed = Path(precomputed_directory) / f"{oracle_variant}-upstream.npz"
        if precomputed.is_file():
            return (
                np.load(precomputed),
                json.loads(precomputed.with_suffix(".json").read_text()),
            )
    oracle_variant = "nomic" if variant.startswith("nomic") else variant
    env: dict[str, str] = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "CUDA_VISIBLE_DEVICES": os.environ.get(
            "REPRESENTAX_ORACLE_CUDA_VISIBLE_DEVICES", "5"
        ),
    }
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run(
        [
            os.environ.get("REPRESENTAX_SENTENCE_TRANSFORMERS_PYTHON", sys.executable),
            "-m",
            "tests.models.qwen2_vl.sentence_transformers_oracle",
            "--variant",
            oracle_variant,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path.cwd(),
        env=env,
    )
    return np.load(output), json.loads(output.with_suffix(".json").read_text())


def _assert_processor_prefix(batch, reference, prefix: str, config) -> None:
    tokens = reference[f"{prefix}_input_ids"]
    mask = reference[f"{prefix}_attention_mask"]
    np.testing.assert_array_equal(
        np.asarray(batch.input_ids)[:, -tokens.shape[1] :], tokens
    )
    np.testing.assert_array_equal(
        np.asarray(batch.attention_mask)[:, -mask.shape[1] :], mask
    )
    if f"{prefix}_pixel_values" in reference:
        expected_pixels = reference[f"{prefix}_pixel_values"]
        assert batch.pixel_values is not None
        grids = []
        for name in ("image_grid_thw", "video_grid_thw"):
            key = f"{prefix}_{name}"
            if key in reference:
                grids.extend(reference[key].tolist())
        layout = vision_layout(
            grids,
            config.vision,
            patch_bucket=batch.pixel_values.shape[0],
        )
        padded = np.pad(
            expected_pixels,
            ((0, batch.pixel_values.shape[0] - expected_pixels.shape[0]), (0, 0)),
        )
        np.testing.assert_array_equal(
            np.asarray(batch.pixel_values), padded[layout["patch_order"]]
        )


@pytest.mark.parametrize("variant", ("bge", "nomic3", "nomic7"))
def test_real_embedding_preprocessing_and_inference(variant, tmp_path):
    checkpoint = _checkpoint(variant)
    reference, metadata = _oracle(variant, checkpoint, tmp_path / f"{variant}.npz")
    from PIL import Image

    image = Image.fromarray(reference["source_pixels"])
    sequence_buckets = tuple(
        sorted(
            {
                int(reference[name].shape[1])
                for name in reference.files
                if name.endswith("_input_ids")
            }
        )
    )
    patch_buckets = tuple(
        sorted(
            {
                int(reference[name].shape[0])
                for name in reference.files
                if name.endswith("_pixel_values")
            }
        )
    )
    load_options: dict[str, Any] = {
        "local_files_only": True,
        "cache_directory": os.environ.get("HF_HUB_CACHE"),
        "parameter_dtype": (
            jnp.float32 if variant in {"bge", "nomic7"} else jnp.bfloat16
        ),
        "compute_dtype": (
            jnp.float32 if variant in {"bge", "nomic7"} else jnp.bfloat16
        ),
        "sequence_length_buckets": sequence_buckets,
        "patch_count_buckets": patch_buckets,
    }
    mesh = None
    if variant == "nomic7":
        if len(jax.devices()) < 2:
            pytest.skip("Nomic 7B acceptance requires two visible GPUs")
        with jax.default_device(jax.devices("cpu")[0]):
            model, processor = load_qwen2_vl_embedding(checkpoint, **load_options)
        mesh = jax.make_mesh(
            (2,),
            ("fsdp",),
            axis_types=(AxisType.Explicit,),
            devices=jax.devices()[:2],
        )
        specs = fsdp_parameter_specs(
            model,
            mesh,
            axis_name="fsdp",
            minimum_elements=2**18,
        )
        model = cast(Qwen2VLEncoder, place_model(model, mesh, specs))
    else:
        model, processor = load_qwen2_vl_embedding(checkpoint, **load_options)
    if variant == "bge":
        batch = processor(
            [
                {
                    "text": "A deterministic chart about quarterly revenue.",
                    "image": image,
                }
            ],
            route=Route.QUERY,
        )
        _assert_processor_prefix(batch, reference, "query", model.config)
        actual = _encode(model, batch, route=Route.QUERY)
        tolerance = NumericalTolerance(absolute=1e-3, relative=4e-3, cosine=0.99999)
        assert_numerically_equivalent(actual, reference["query_embedding"], tolerance)
        assert "embed_tokens" in str(metadata["upstream_forward_error"])
        return

    text_batch = processor(["A deterministic chart about quarterly revenue."])
    image_batch = processor([image])
    _assert_processor_prefix(text_batch, reference, "text", model.config)
    _assert_processor_prefix(image_batch, reference, "image", model.config)
    tolerance = NumericalTolerance(absolute=1e-2, relative=8e-2, cosine=0.997)
    encode = (
        (lambda values, route: _fsdp_encode(model, values, route=route, mesh=mesh))
        if mesh is not None
        else (lambda values, route: _encode(model, values, route=route))
    )
    assert_numerically_equivalent(
        encode(text_batch, Route.QUERY),
        reference["text_embedding"],
        tolerance,
    )
    assert_numerically_equivalent(
        encode(image_batch, Route.DOCUMENT),
        reference["image_embedding"],
        tolerance,
    )


def test_real_jina_preprocessing_and_inference(tmp_path):
    checkpoint = _checkpoint("jina")
    reference, _ = _oracle("jina", checkpoint, tmp_path / "jina.npz")
    from PIL import Image

    image = Image.fromarray(reference["source_pixels"])
    sequence_buckets = tuple(
        sorted(
            {
                int(reference[name].shape[1]) + 1
                for name in reference.files
                if name.endswith("_input_ids")
            }
        )
    )
    patch_buckets = tuple(
        sorted(
            {
                int(reference[name].shape[0])
                for name in reference.files
                if name.endswith("_pixel_values")
            }
        )
    )
    model, processor = load_qwen2_vl_reranker(
        checkpoint,
        local_files_only=True,
        cache_directory=os.environ.get("HF_HUB_CACHE"),
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=sequence_buckets,
        patch_count_buckets=patch_buckets,
    )
    pairs = {
        "text": (
            "Which report contains revenue growth?",
            "The annual report contains a quarterly revenue chart.",
        ),
        "image": ("Which report contains revenue growth?", image),
    }
    for name, pair in pairs.items():
        batch = processor([pair])
        # The native processor appends the checkpoint's score token explicitly;
        # the remote forward appends the same token after ST preprocessing.
        token_count = reference[f"{name}_input_ids"].shape[1]
        np.testing.assert_array_equal(
            np.asarray(batch.input_ids)[:, -(token_count + 1) : -1],
            reference[f"{name}_input_ids"],
        )
        actual = _score(model, batch)
        np.testing.assert_allclose(
            actual,
            reference[f"{name}_score"],
            rtol=2e-2,
            atol=5e-3,
        )
