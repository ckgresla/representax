"""Qwen3 scalar reward-model loading, training, and export lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors.numpy import save_file

from representax.core import score_logits
from representax.models.qwen_reranker import (
    QwenReranker,
    QwenRerankerBatch,
    QwenRerankerCheckpointAdapter,
    QwenRerankerConfig,
)
from representax.models.qwen_reward import QwenRewardCheckpointAdapter
from representax.tasks.reward_modeling import PairwiseRewardBatch, PairwiseRewardTask


def _hf_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
        "initializer_range": 0.02,
    }


def _source_checkpoint(path: Path) -> Path:
    config = QwenRerankerConfig.from_hf_config(
        _hf_config(), true_token_id=0, false_token_id=None
    ).model_copy(update={"tie_word_embeddings": True})
    backbone = QwenReranker.init(config, key=jax.random.key(1))
    state = QwenRerankerCheckpointAdapter().state_dict(backbone)
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_hf_config()))
    save_file(
        {name: np.asarray(value) for name, value in state.items()},
        path / "model.safetensors",
    )
    return path


def _batch() -> QwenRerankerBatch:
    return QwenRerankerBatch(
        input_ids=jnp.asarray([[2, 3, 4, 0], [5, 6, 7, 8]], dtype=jnp.int32),
        attention_mask=jnp.asarray([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=jnp.int32),
    )


def test_qwen_reward_loads_trains_and_exports(tmp_path: Path) -> None:
    source = _source_checkpoint(tmp_path / "source")
    adapter = QwenRewardCheckpointAdapter()
    model = adapter.load(
        source,
        head_key=jax.random.key(2),
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    initial = score_logits(model, _batch())
    assert initial.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(initial)))

    rejected = QwenRerankerBatch(
        input_ids=jnp.asarray([[2, 4, 3, 0], [5, 7, 6, 8]], dtype=jnp.int32),
        attention_mask=_batch().attention_mask,
    )
    reward_batch = PairwiseRewardBatch(
        chosen=_batch(),
        rejected=rejected,
        margins=jnp.zeros((2,), dtype=jnp.float32),
        valid=jnp.ones((2,), dtype=jnp.bool_),
    )
    task = PairwiseRewardTask()

    def loss(candidate):
        return task.loss(candidate, reward_batch).loss

    value, gradients = eqx.filter_value_and_grad(loss)(model)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradients.score_head.weight)))
    optimizer = optax.adamw(1e-3)
    parameters = eqx.filter(model, eqx.is_inexact_array)
    state = optimizer.init(parameters)
    updates, _ = optimizer.update(gradients, state, parameters)
    updated = eqx.apply_updates(model, updates)
    assert not np.array_equal(
        np.asarray(updated.score_head.weight), np.asarray(model.score_head.weight)
    )

    exported = adapter.save(updated, tmp_path / "exported", source_checkpoint=source)
    exported_config = json.loads((exported / "config.json").read_text())
    assert exported_config["architectures"] == ["Qwen3ForSequenceClassification"]
    assert exported_config["num_labels"] == 1
    assert exported_config["pad_token_id"] == 0
    reloaded = adapter.load(
        exported,
        head_key=jax.random.key(99),
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    np.testing.assert_array_equal(
        np.asarray(score_logits(reloaded, _batch())),
        np.asarray(score_logits(updated, _batch())),
    )
