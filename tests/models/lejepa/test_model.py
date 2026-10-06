"""Canonical LeJEPA model composition and reference parity."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import encode
from representax.models.lejepa import (
    LeJEPAModel,
    LeJEPAMulticropImages,
    LeJEPAProjectionMLP,
    LeJEPAViTBackbone,
    LeJEPAViTConfig,
)
from representax.tasks.jepa import JEPABatch, LeJEPATask


def _tiny_config(*, projection_dimension: int = 7) -> LeJEPAViTConfig:
    return LeJEPAViTConfig(
        image_size=16,
        local_image_size=8,
        patch_size=4,
        hidden_size=12,
        depth=2,
        heads=3,
        projector_bottleneck=8,
        projector_hidden_size=24,
        projection_dimension=projection_dimension,
        drop_path_rate=0.0,
    )


def _multicrop_inputs(batch: int = 2) -> LeJEPAMulticropImages:
    pixels = jax.random.normal(
        jax.random.key(1),
        (batch, 8, 3, 16, 16),
    )
    sizes = jnp.tile(jnp.asarray((16, 16, 8, 8, 8, 8, 8, 8)), (batch, 1))
    return LeJEPAMulticropImages(pixels, sizes)


def _flatten(inputs: LeJEPAMulticropImages) -> LeJEPAMulticropImages:
    return jax.tree.map(
        lambda value: (
            value.reshape((value.shape[0] * value.shape[1], *value.shape[2:]))
            if isinstance(value, jax.Array)
            else value
        ),
        inputs,
    )


def test_training_projector_is_separate_from_public_evaluation_encode() -> None:
    config = _tiny_config()
    model = LeJEPAModel.init(
        config,
        key=jax.random.key(2),
        rematerialization="none",
    )
    inputs = _multicrop_inputs()
    images = inputs.pixel_values[:, 0]
    evaluation = encode(model, images)
    projection = model.project(_flatten(inputs), key=None)

    zero_projector = jax.tree.map(
        lambda value: jnp.zeros_like(value) if eqx.is_array(value) else value,
        model.projector,
    )
    changed = eqx.tree_at(lambda value: value.projector, model, zero_projector)

    np.testing.assert_array_equal(encode(changed, images), evaluation)
    assert projection.shape == (16, 7)
    assert changed.project(_flatten(inputs), key=None).shape == (16, 7)
    assert not np.allclose(projection, changed.project(_flatten(inputs), key=None))
    assert evaluation.shape == (2, 24)


def test_grouped_multicrop_matches_explicit_per_view_backbone_calls() -> None:
    config = _tiny_config()
    model = LeJEPAModel.init(
        config,
        key=jax.random.key(3),
        rematerialization="none",
    )
    grouped = _multicrop_inputs()
    flat = _flatten(grouped)
    actual = model.project(flat, key=None)

    features = []
    for sample in range(grouped.pixel_values.shape[0]):
        for view in range(8):
            pixels = grouped.pixel_values[sample, view]
            if view >= 2:
                pixels = pixels[:, :8, :8]
            features.append(model.backbone(pixels[None], key=None)[0])
    expected = model.projector(jnp.stack(features))

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_grouped_multicrop_preserves_bfloat16_scan_carry_with_drop_path() -> None:
    model = LeJEPAModel.init(
        _tiny_config(),
        key=jax.random.key(31),
        parameter_dtype=jnp.bfloat16,
        rematerialization="none",
    )
    inputs = _flatten(_multicrop_inputs(batch=1))
    inputs = eqx.tree_at(
        lambda value: value.pixel_values,
        inputs,
        inputs.pixel_values.astype(jnp.bfloat16),
    )

    projected = model.project(inputs, key=jax.random.key(32))

    assert projected.shape == (8, 7)
    assert jnp.all(jnp.isfinite(projected))


def test_task_uses_explicit_project_protocol_without_metadata_width_coercion() -> None:
    model = LeJEPAModel.init(
        _tiny_config(),
        key=jax.random.key(4),
        rematerialization="none",
    )
    inputs = _multicrop_inputs()
    batch = JEPABatch(inputs, jnp.ones((2, 8), dtype=jnp.bool_))

    projections = LeJEPATask(slices=4, knots=3).representations(
        model,
        batch,
        key=None,
    )

    assert projections.shape == (2, 8, 7)
    assert model.metadata.output_dimension == 24


def test_vit_large_keeps_512_projector_and_2048_evaluation_width() -> None:
    config = LeJEPAViTConfig.vit_large_patch16()
    projector = LeJEPAProjectionMLP.init(
        config,
        key=jax.random.key(5),
        dtype=jnp.float32,
    )
    projected = projector(jnp.ones((2, 1024), dtype=jnp.float32))
    abstract = eqx.filter_eval_shape(
        LeJEPAModel.init,
        config,
        key=jax.random.key(6),
        rematerialization="none",
    )

    assert config.depth == 24
    assert config.hidden_size == 1024
    assert projected.shape == (2, 512)
    assert abstract.metadata.output_dimension == 2048


def test_vit_base_profile_matches_timm_architecture_contract() -> None:
    config = LeJEPAViTConfig.vit_base_patch16()

    assert config.image_size == 224
    assert config.local_image_size == 98
    assert config.patch_size == 16
    assert config.hidden_size == 768
    assert config.depth == 12
    assert config.heads == 12
    assert config.mlp_ratio == 4.0
    assert config.projection_dimension == 512


@pytest.mark.parity
def test_native_backbone_matches_exact_timm_vit_state() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    import torch.nn as nn
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(7)
    reference = VisionTransformer(
        img_size=16,
        patch_size=4,
        in_chans=3,
        num_classes=0,
        global_pool="token",
        embed_dim=12,
        depth=2,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        class_token=True,
        dynamic_img_size=True,
        dynamic_img_pad=False,
        drop_path_rate=0.1,
        weight_init="",
    ).eval()

    class ReferenceModel(nn.Module):
        def __init__(self, backbone) -> None:
            super().__init__()
            self.backbone = backbone
            self.projector = nn.Sequential(
                nn.Linear(12, 8),
                nn.Sequential(
                    nn.Linear(8, 24),
                    nn.BatchNorm1d(24),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.0),
                    nn.Linear(24, 24),
                    nn.BatchNorm1d(24),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.0),
                    nn.Linear(24, 7),
                    nn.Dropout(0.0),
                ),
            )
            self.evaluation_norm = nn.LayerNorm(24, eps=1e-6)

    reference_model = ReferenceModel(reference)
    native_model = LeJEPAModel.from_timm_state_dict(
        _tiny_config(),
        reference_model.state_dict(),
        rematerialization="none",
    )
    native = LeJEPAViTBackbone.from_timm_state_dict(
        _tiny_config(),
        reference.state_dict(),
        rematerialization="none",
    )
    pixels = np.random.default_rng(8).normal(size=(2, 3, 16, 16)).astype(np.float32)

    with torch.no_grad():
        expected = reference(torch.from_numpy(pixels)).numpy()
    actual = np.asarray(native(jnp.asarray(pixels), key=None))

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=5e-4)
    features = np.random.default_rng(9).normal(size=(16, 12)).astype(np.float32)
    evaluation_features = (
        np.random.default_rng(10).normal(size=(4, 24)).astype(np.float32)
    )
    reference_model.projector.train()
    with torch.no_grad():
        expected_projected = reference_model.projector(
            torch.from_numpy(features)
        ).numpy()
        expected_normalized = reference_model.evaluation_norm(
            torch.from_numpy(evaluation_features)
        ).numpy()

    np.testing.assert_allclose(
        native_model.projector(jnp.asarray(features)),
        expected_projected,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        native_model.evaluation_norm(jnp.asarray(evaluation_features)),
        expected_normalized,
        rtol=2e-5,
        atol=2e-5,
    )
