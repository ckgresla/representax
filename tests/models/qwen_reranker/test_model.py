"""Fast contracts for native Qwen2 and Qwen3 text rerankers."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import Route
from representax.models.qwen_reranker import (
    QwenGeneration,
    QwenReranker,
    QwenRerankerBatch,
    QwenRerankerCheckpointAdapter,
    QwenRerankerConfig,
    qwen_reranker_weight_names,
)


def tiny_config(
    generation: QwenGeneration,
    *,
    tied: bool = True,
) -> QwenRerankerConfig:
    return QwenRerankerConfig(
        generation=generation,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dimension=4,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        norm_epsilon=1e-6,
        pad_token_id=0,
        true_token_id=7,
        false_token_id=3,
        tie_word_embeddings=tied,
    )


@pytest.mark.parametrize(
    ("generation", "activation", "expected"),
    (
        ("qwen2", None, "sigmoid"),
        ("qwen3", "torch.nn.modules.linear.Identity", "identity"),
    ),
)
def test_checkpoint_metadata_defines_architecture_and_inference_activation(
    generation: QwenGeneration,
    activation: str | None,
    expected: str,
    tmp_path: Path,
) -> None:
    config = {
        "model_type": generation,
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "eos_token_id": 0,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    score = tmp_path / "1_LogitScore"
    score.mkdir()
    (score / "config.json").write_text(
        json.dumps({"true_token_id": 7, "false_token_id": 3})
    )
    sentence_config = {"model_type": "CrossEncoder"}
    if activation is not None:
        sentence_config["activation_fn"] = activation
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps(sentence_config)
    )

    loaded = QwenRerankerConfig.from_checkpoint(tmp_path)

    assert loaded.generation == generation
    assert loaded.score_activation == expected


@pytest.mark.parametrize("generation", ("qwen2", "qwen3"))
@pytest.mark.parametrize("tied", (False, True))
def test_score_is_exactly_the_consumed_final_position_logits(
    generation: QwenGeneration,
    tied: bool,
) -> None:
    config = tiny_config(generation, tied=tied)
    model = QwenReranker.init(
        config,
        key=jax.random.key(13),
        rematerialization="none",
    )
    batch = QwenRerankerBatch(
        input_ids=jnp.asarray(((0, 1, 2, 4), (5, 6, 7, 8))),
        attention_mask=jnp.asarray(((0, 1, 1, 1), (1, 1, 1, 1))),
    )
    hidden = model.hidden_states(batch)[:, -1].astype(jnp.float32)
    head = model.text.token_embedding if model.lm_head is None else model.lm_head
    full_logits = hidden @ head.astype(jnp.float32).T
    expected = (
        full_logits[:, config.true_token_id] - full_logits[:, config.false_token_id]
    )
    np.testing.assert_allclose(model.logits(batch), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.score(batch), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        model.encode(batch, route=Route.GENERIC),
        expected[:, None],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("generation", ("qwen2", "qwen3"))
def test_checkpoint_roundtrip_covers_every_executed_tensor(
    generation: QwenGeneration,
) -> None:
    config = tiny_config(generation, tied=False)
    adapter = QwenRerankerCheckpointAdapter(rematerialization="selective")
    model = QwenReranker.init(config, key=jax.random.key(17))
    state = adapter.state_dict(model)
    assert frozenset(state) == qwen_reranker_weight_names(config)
    restored = adapter.from_state_dict(
        config,
        state,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    assert restored.rematerialization == "selective"
    for name, expected in state.items():
        np.testing.assert_array_equal(adapter.state_dict(restored)[name], expected)


def test_untied_single_logit_score_has_finite_input_and_parameter_gradients() -> None:
    config = tiny_config("qwen3", tied=False).model_copy(
        update={"false_token_id": None}
    )
    model = QwenReranker.init(
        config,
        key=jax.random.key(19),
        rematerialization="none",
    )
    batch = QwenRerankerBatch(
        input_ids=jnp.asarray(((1, 2, 3, 4),)),
        attention_mask=jnp.ones((1, 4), dtype=bool),
    )

    def loss(candidate):
        return jnp.square(candidate.score(batch)).mean()

    gradients = jax.grad(loss)(model)
    leaves = [leaf for leaf in jax.tree.leaves(gradients) if leaf is not None]
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(float(jnp.linalg.norm(leaf)) > 0 for leaf in leaves)
