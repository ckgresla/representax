from __future__ import annotations

from pathlib import Path
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.models.vjepa2_1 import (
    VJEPA2_1Config,
    VJEPA2_1Model,
    load_reference_state,
)
from representax.models.vjepa2_1.processing import _resize
from representax.tasks.jepa import VJEPA2_1Batch, VJEPA2_1Task
from representax.train import build_train_step, init_train_state

FIXTURE = Path(__file__).parents[1] / "fixtures" / "vjepa2_1" / "reference.npz"


def reference_model(reference: np.lib.npyio.NpzFile) -> VJEPA2_1Model:
    encoder = {
        name.removeprefix("encoder::"): reference[name]
        for name in reference.files
        if name.startswith("encoder::")
    }
    predictor = {
        name.removeprefix("predictor::"): reference[name]
        for name in reference.files
        if name.startswith("predictor::")
    }
    config = VJEPA2_1Config(
        image_size=8,
        patch_size=4,
        video_frames=4,
        tubelet_size=2,
        hidden_size=12,
        depth=12,
        heads=2,
        predictor_hidden_size=12,
        predictor_depth=4,
        predictor_heads=2,
        supervision_layers=(2, 5, 8, 11),
    )
    return load_reference_state(
        VJEPA2_1Model.init(
            config,
            key=jax.random.key(0),
            rematerialization="full",
        ),
        encoder=encoder,
        predictor=predictor,
    )


def reference_batch(reference: np.lib.npyio.NpzFile) -> VJEPA2_1Batch:
    context = jnp.asarray(reference["context_ids"])
    target = jnp.asarray(reference["target_ids"])
    return VJEPA2_1Batch(
        pixels=jnp.asarray(reference["pixels"]),
        context_ids=context[:, None],
        target_ids=target[:, None],
        context_valid=jnp.ones((*context.shape[:1], 1, context.shape[1]), dtype=bool),
        target_valid=jnp.ones((*target.shape[:1], 1, target.shape[1]), dtype=bool),
    )


@pytest.mark.parity
def test_official_forward_loss_gradients_update_and_ema() -> None:
    reference = np.load(FIXTURE)
    model = reference_model(reference)
    batch = reference_batch(reference)
    context_ids = batch.context_ids[:, 0]
    target_ids = batch.target_ids[:, 0]

    context_features = model.online(batch.pixels, context_ids)
    predicted_target, predicted_context = model.predictor(
        context_features,
        context_ids,
        target_ids,
        is_video=True,
    )
    np.testing.assert_allclose(
        context_features,
        reference["context_features"],
        rtol=2e-5,
        atol=4e-6,
    )
    np.testing.assert_allclose(
        predicted_target,
        reference["predicted_target"],
        rtol=3e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        predicted_context,
        reference["predicted_context"],
        rtol=3e-5,
        atol=5e-6,
    )

    task = VJEPA2_1Task(context_weight=0.5, ema_start=0.9, ema_end=0.9)
    output = task.loss(model, batch)
    np.testing.assert_allclose(output.loss, reference["loss"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        output.metrics["prediction"],
        reference["prediction_loss"],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        output.metrics["context"],
        reference["context_loss"],
        rtol=2e-6,
        atol=2e-6,
    )

    trainable_filter = model.training_filter()
    trainable, frozen = eqx.partition(model, trainable_filter)

    def loss_fn(selected):
        return task.loss(eqx.combine(selected, frozen), batch).loss

    _, gradients = eqx.filter_value_and_grad(loss_fn)(trainable)
    np.testing.assert_allclose(
        np.asarray(gradients.online.video_patch_weight),
        reference["gradient_encoder_patch"],
        rtol=3e-4,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(gradients.online.layers.layer(0).attention.qkv.weight),
        reference["gradient_encoder_qkv0"],
        rtol=5e-4,
        atol=8e-5,
    )
    np.testing.assert_allclose(
        np.asarray(gradients.predictor.layers.layer(0).attention.qkv.weight),
        reference["gradient_predictor_qkv0"],
        rtol=5e-4,
        atol=8e-5,
    )
    np.testing.assert_allclose(
        np.asarray(gradients.predictor.target_projection.weight),
        reference["gradient_predictor_projection"],
        rtol=5e-4,
        atol=3e-5,
    )

    optimizer = optax.adamw(
        1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=0.01,
    )
    state = init_train_state(
        model,
        optimizer,
        trainable_filter=trainable_filter,
    )
    result = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        trainable_filter=trainable_filter,
    )(state, batch, jax.random.key(1))
    updated_model = cast(VJEPA2_1Model, result.state.model)
    np.testing.assert_allclose(
        updated_model.online.video_patch_weight,
        reference["updated_encoder_patch"],
        rtol=2e-4,
        atol=6e-6,
    )
    np.testing.assert_allclose(
        updated_model.predictor.target_projection.weight,
        reference["updated_predictor_projection"],
        rtol=2e-4,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        updated_model.target.video_patch_weight,
        reference["updated_target_patch"],
        rtol=2e-4,
        atol=8e-7,
    )


@pytest.mark.parity
def test_official_image_forward() -> None:
    reference = np.load(FIXTURE)
    model = reference_model(reference)
    context_ids = jnp.asarray(reference["image_context_ids"])
    target_ids = jnp.asarray(reference["image_target_ids"])
    context_features = model.online(
        jnp.asarray(reference["image_pixels"]),
        context_ids,
    )
    predicted_target, predicted_context = model.predictor(
        context_features,
        context_ids,
        target_ids,
        is_video=False,
    )
    np.testing.assert_allclose(
        context_features,
        reference["image_context_features"],
        rtol=2e-5,
        atol=4e-6,
    )
    np.testing.assert_allclose(
        predicted_target,
        reference["image_predicted_target"],
        rtol=3e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        predicted_context,
        reference["image_predicted_context"],
        rtol=3e-5,
        atol=1e-5,
    )


@pytest.mark.parity
def test_load_complete_official_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    reference = np.load(FIXTURE)
    encoder = {
        name.removeprefix("encoder::"): torch.from_numpy(reference[name].copy())
        for name in reference.files
        if name.startswith("encoder::")
    }
    predictor = {
        name.removeprefix("predictor::"): torch.from_numpy(reference[name].copy())
        for name in reference.files
        if name.startswith("predictor::")
    }
    checkpoint = tmp_path / "official.pth.tar"
    torch.save(
        {
            "encoder": encoder,
            "predictor": predictor,
            "target_encoder": encoder,
        },
        checkpoint,
    )
    expected = reference_model(reference)
    actual = VJEPA2_1Model.load_from_reference(
        str(checkpoint),
        expected.online.config,
        key=jax.random.key(97),
        rematerialization="full",
    )
    assert eqx.tree_equal(actual, expected)


@pytest.mark.parity
def test_resize_matches_official_bilinear_kernel() -> None:
    reference = np.load(FIXTURE)
    np.testing.assert_allclose(
        _resize(reference["raw_resize_frames"], 8, 8),
        reference["resized_frames"],
        rtol=1e-6,
        atol=2e-5,
    )
