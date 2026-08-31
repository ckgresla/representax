"""Fast contracts for native BERT sequence-classification scorers."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from representax.models.bert import (
    BertConfig,
    BertEncoder,
    BertScorer,
    BertScorerCheckpointAdapter,
)
from representax.models.bert.scoring_loading import _pair_processor
from representax.models.components import Linear


def _config() -> BertConfig:
    return BertConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        type_vocab_size=2,
    )


def _model() -> BertScorer:
    return BertScorer(
        backbone=BertEncoder.init(_config(), key=jax.random.key(1)),
        classifier=Linear.init(
            8,
            1,
            key=jax.random.key(2),
            scale=0.02,
            dtype=jnp.float32,
            bias=True,
        ),
        classifier_dropout_probability=0.1,
    )


def test_scalar_logits_and_checkpoint_roundtrip(tmp_path) -> None:
    model = _model()
    inputs = model.backbone.make_batch(
        input_ids=jnp.asarray(((1, 2, 3, 0), (4, 5, 0, 0))),
        attention_mask=jnp.asarray(((1, 1, 1, 0), (1, 1, 0, 0))),
        token_type_ids=jnp.zeros((2, 4), dtype=jnp.int32),
    )
    assert model.logits(inputs).shape == (2, 1)

    adapter = BertScorerCheckpointAdapter(rematerialization="none")
    target = adapter.save(model, tmp_path / "checkpoint")
    restored = adapter.load(target)
    expected = adapter.state_dict(model)
    actual = adapter.state_dict(restored)
    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    config = json.loads((target / "config.json").read_text())
    assert config["architectures"] == ["BertForSequenceClassification"]
    assert config["num_labels"] == 1


class _Tokenizer:
    pad_token_id = 0

    def __call__(
        self,
        queries,
        documents,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        del padding, truncation, max_length, return_tensors
        size = len(queries)
        assert size == len(documents)
        return {
            "input_ids": np.tile(np.asarray((1, 2, 3)), (size, 1)),
            "attention_mask": np.ones((size, 3), dtype=np.int32),
            "token_type_ids": np.tile(np.asarray((0, 0, 1)), (size, 1)),
        }


def test_pair_processor_uses_one_static_sequence_bucket() -> None:
    processor = _pair_processor(
        _Tokenizer(),
        maximum_length=8,
        sequence_length_buckets=(4, 8),
    )
    batch = processor((("query", "document"), ("other", "passage")))
    assert batch.input_ids.shape == (2, 4)
    np.testing.assert_array_equal(batch.token_type_ids[:, :3], ((0, 0, 1),) * 2)
