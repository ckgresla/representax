from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from representax.core import Route
from representax.models.clip import (
    CLIPBatch,
    CLIPCheckpointAdapter,
    CLIPConfig,
    CLIPEncoder,
    CLIPTextConfig,
    CLIPVisionConfig,
    clip_weight_names,
)
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state


def tiny_config() -> CLIPConfig:
    return CLIPConfig(
        text=CLIPTextConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=8,
            hidden_activation="quick_gelu",
            layer_norm_epsilon=1e-5,
            attention_dropout=0.0,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
        ),
        vision=CLIPVisionConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
            num_channels=3,
            hidden_activation="quick_gelu",
            layer_norm_epsilon=1e-5,
            attention_dropout=0.0,
        ),
        projection_dimension=6,
    )


def _batches() -> tuple[CLIPBatch, CLIPBatch, CLIPBatch]:
    input_ids = jnp.asarray(((0, 4, 5, 2, 1, 1), (0, 3, 2, 1, 1, 1)))
    attention_mask = input_ids != 1
    pixels = jnp.arange(2 * 3 * 8 * 8, dtype=jnp.float32).reshape((2, 3, 8, 8)) / 255
    text = CLIPBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        text_valid=jnp.ones((2,), dtype=bool),
    )
    image = CLIPBatch(
        pixel_values=pixels,
        image_valid=jnp.ones((2,), dtype=bool),
    )
    composed = CLIPBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        text_valid=jnp.ones((2,), dtype=bool),
        pixel_values=pixels,
        image_valid=jnp.ones((2,), dtype=bool),
    )
    return text, image, composed


def test_text_image_and_composed_encoding_are_normalized() -> None:
    model = CLIPEncoder.init(
        tiny_config(),
        key=jax.random.key(1),
        rematerialization="none",
        normalize_output=True,
    )
    for batch in _batches():
        encoded = model.encode(batch, route=Route.GENERIC)
        assert encoded.shape == (2, 6)
        np.testing.assert_allclose(jnp.linalg.norm(encoded, axis=-1), 1, atol=1e-5)


def test_checkpoint_state_dict_round_trip_is_exact() -> None:
    config = tiny_config()
    model = CLIPEncoder.init(
        config,
        key=jax.random.key(2),
        rematerialization="none",
        normalize_output=True,
    )
    adapter = CLIPCheckpointAdapter(rematerialization="none")
    state = adapter.state_dict(model)
    assert set(state) == set(clip_weight_names(config))
    restored = adapter.from_state_dict(config, state)
    for name, value in adapter.state_dict(restored).items():
        np.testing.assert_array_equal(value, state[name])


def test_export_preserves_legacy_sentence_transformers_layout(tmp_path) -> None:
    model = CLIPEncoder.init(
        tiny_config(),
        key=jax.random.key(21),
        rematerialization="none",
    )
    nested = tmp_path / "0_CLIPModel"
    nested.mkdir()
    (nested / "config.json").write_text("{}\n")
    export = CLIPCheckpointAdapter().save(model, tmp_path)
    assert export == tmp_path
    assert (nested / "model.safetensors").is_file()
    assert not (tmp_path / "model.safetensors").exists()


def test_dual_encoder_runs_three_generic_training_steps() -> None:
    text, image, _ = _batches()
    model = CLIPEncoder.init(
        tiny_config(),
        key=jax.random.key(3),
        rematerialization="none",
        normalize_output=True,
    )
    batch = pairwise_batch(
        left=text,
        right=image,
        labels=np.asarray((0.8, 0.2), dtype=np.float32),
    )
    optimizer = optax.adamw(1e-3)
    state = init_train_state(model, optimizer)
    step = build_train_step(CosineRegressionTask(), optimizer)
    update_norms = []
    for _ in range(3):
        result = step(state, batch, None)
        state = result.state
        update_norms.append(float(result.metrics.update_global_norm))
        assert bool(result.metrics.numeric_finite)
    assert int(state.step) == 3
    assert all(value > 0 for value in update_norms)


def test_composed_gradient_is_finite() -> None:
    model = CLIPEncoder.init(
        tiny_config(),
        key=jax.random.key(4),
        rematerialization="none",
        normalize_output=True,
    )
    _, _, batch = _batches()

    def loss(candidate):
        return jnp.sum(candidate.encode(batch, route=Route.GENERIC))

    _, gradients = eqx.filter_value_and_grad(loss)(model)
    leaves = [value for value in jax.tree.leaves(gradients) if eqx.is_array(value)]
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in leaves)


def test_compute_dtype_is_independent_of_master_parameter_dtype() -> None:
    model = CLIPEncoder.init(
        tiny_config(),
        key=jax.random.key(5),
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        rematerialization="none",
        normalize_output=True,
    )
    text, image, composed = _batches()
    for batch in (text, image, composed):
        output = model.encode(batch, route=Route.GENERIC)
        assert output.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(output)))
