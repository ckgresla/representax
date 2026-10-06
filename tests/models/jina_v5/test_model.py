"""Native Jina v5 Small text-family contracts and pinned parity."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.core import encode
from representax.models.jina_v5 import (
    JinaV5TextBatch,
    JinaV5TextCheckpointAdapter,
    JinaV5TextConfig,
    jina_v5_text_weight_names,
)


def tiny_config() -> JinaV5TextConfig:
    return JinaV5TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dimension=4,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        norm_epsilon=1e-6,
        pad_token_id=0,
        output_dimension=6,
    )


def _synthetic_state(config: JinaV5TextConfig) -> dict[str, jax.Array]:
    names = jina_v5_text_weight_names(config)
    shapes = {
        "language_model.embed_tokens.weight": (
            config.vocab_size,
            config.hidden_size,
        ),
        "language_model.norm.weight": (config.hidden_size,),
    }
    attention = config.num_attention_heads * config.head_dimension
    key_value = config.num_key_value_heads * config.head_dimension
    for index in range(config.num_hidden_layers):
        prefix = f"language_model.layers.{index}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (config.hidden_size,),
                prefix + "post_attention_layernorm.weight": (config.hidden_size,),
                prefix + "self_attn.q_proj.weight": (
                    attention,
                    config.hidden_size,
                ),
                prefix + "self_attn.k_proj.weight": (
                    key_value,
                    config.hidden_size,
                ),
                prefix + "self_attn.v_proj.weight": (
                    key_value,
                    config.hidden_size,
                ),
                prefix + "self_attn.o_proj.weight": (
                    config.hidden_size,
                    attention,
                ),
                prefix + "self_attn.q_norm.weight": (config.head_dimension,),
                prefix + "self_attn.k_norm.weight": (config.head_dimension,),
                prefix + "mlp.gate_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                prefix + "mlp.up_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                prefix + "mlp.down_proj.weight": (
                    config.hidden_size,
                    config.intermediate_size,
                ),
            }
        )
    assert frozenset(shapes) == names
    keys = jax.random.split(jax.random.key(12), len(shapes))
    state = {}
    for key, (name, shape) in zip(keys, sorted(shapes.items()), strict=True):
        if name.endswith("norm.weight") or "layernorm.weight" in name:
            state[name] = jnp.ones(shape)
        else:
            state[name] = 0.02 * jax.random.normal(key, shape)
    return state


def test_jina_v5_small_shape_scan_and_state_dict_round_trip():
    config = tiny_config()
    adapter = JinaV5TextCheckpointAdapter()
    state = _synthetic_state(config)
    model = adapter.from_state_dict(
        config,
        state,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        model_id="test/jina-v5",
        revision="test",
    )
    batch = JinaV5TextBatch(
        input_ids=jnp.asarray([[1, 2, 3, 0], [4, 5, 0, 0]]),
        attention_mask=jnp.asarray([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )
    compiled = eqx.filter_jit(lambda candidate, values: encode(candidate, values))
    representations = compiled(model, batch)

    assert representations.shape == (2, 6)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(representations), axis=-1),
        1.0,
        rtol=1e-5,
        atol=1e-5,
    )
    exported = adapter.state_dict(model)
    assert exported.keys() == state.keys()
    for name in state:
        np.testing.assert_array_equal(exported[name], state[name])
