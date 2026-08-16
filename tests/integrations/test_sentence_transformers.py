"""Torch-free Sentence Transformers artifact and host-embedding contracts."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from safetensors.numpy import save_file

from representax import Route
from representax.integrations import (
    load_sentence_transformer,
    load_sentence_transformer_artifact,
    load_sentence_transformer_modules,
)
from representax.models.bert import BertCheckpointAdapter, BertConfig, BertEncoder


class _Tokenizer:
    all_special_ids = (0, 1, 2)

    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        assert padding in (True, "max_length")
        assert truncation is True
        assert return_tensors == "np"
        rows = []
        for text in texts:
            tokens = [1, *(3 + ord(character) % 10 for character in text), 2]
            tokens = tokens[:max_length]
            if tokens[-1] != 2:
                tokens[-1] = 2
            rows.append(tokens)
        width = max_length if padding == "max_length" else max(map(len, rows))
        input_ids = np.zeros((len(rows), width), dtype=np.int32)
        attention_mask = np.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = row
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _checkpoint(path: Path, *, include_prompt: bool = True) -> Path:
    config = BertConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        type_vocab_size=2,
        hidden_dropout_probability=0.0,
        attention_dropout_probability=0.0,
    )
    model = BertEncoder.init(config, key=jax.random.key(7))
    BertCheckpointAdapter(rematerialization="none").save(model, path)
    _write_json(
        path / "modules.json",
        [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer",
            },
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
            {
                "idx": 2,
                "name": "2",
                "path": "2_Dense",
                "type": "sentence_transformers.models.Dense",
            },
            {
                "idx": 3,
                "name": "3",
                "path": "3_Normalize",
                "type": "sentence_transformers.models.Normalize",
            },
        ],
    )
    _write_json(
        path / "config_sentence_transformers.json",
        {
            "model_type": "SentenceTransformer",
            "prompts": {"query": "q: ", "document": "d: "},
            "similarity_fn_name": "cosine",
        },
    )
    _write_json(
        path / "sentence_bert_config.json",
        {"max_seq_length": 8, "do_lower_case": False},
    )
    _write_json(
        path / "1_Pooling" / "config.json",
        {
            "word_embedding_dimension": 4,
            "pooling_mode_mean_tokens": True,
            "include_prompt": include_prompt,
        },
    )
    _write_json(
        path / "2_Dense" / "config.json",
        {
            "in_features": 4,
            "out_features": 3,
            "bias": True,
            "activation_function": "torch.nn.modules.linear.Identity",
            "module_input_name": "sentence_embedding",
            "module_output_name": "sentence_embedding",
        },
    )
    save_file(
        {
            "linear.weight": np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            "linear.bias": np.zeros((3,), dtype=np.float32),
        },
        path / "2_Dense" / "model.safetensors",
    )
    return path


def test_standard_dense_graph_loads_without_upstream_runtime(tmp_path):
    checkpoint = _checkpoint(tmp_path / "sentence-model")

    loaded = load_sentence_transformer_artifact(checkpoint)

    assert tuple(
        module.kind for module in load_sentence_transformer_modules(checkpoint)
    ) == (
        "Transformer",
        "Pooling",
        "Dense",
        "Normalize",
    )
    assert loaded.encoder.metadata.output_dimension == 3
    assert loaded.encoder.pooling.modes == ("mean",)
    assert loaded.max_sequence_length == 8
    assert loaded.prompts == {"query": "q: ", "document": "d: "}


def test_host_embed_uses_fixed_shapes_routes_and_partial_batch_padding(tmp_path):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    model = load_sentence_transformer(checkpoint, processor=_Tokenizer())

    output = model.embed(["a", "bc", "def"], route=Route.QUERY, batch_size=2)

    assert output.shape == (3, 3)
    np.testing.assert_allclose(
        np.linalg.norm(output, axis=1),
        np.ones((3,)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        model.similarity(output, output),
        output @ output.T,
        rtol=1e-6,
        atol=1e-6,
    )


def test_prompt_exclusion_uses_a_distinct_pooling_mask(tmp_path):
    checkpoint = _checkpoint(tmp_path / "sentence-model", include_prompt=False)
    model = load_sentence_transformer(checkpoint, processor=_Tokenizer())

    batch = model.preprocess(["abc"], route=Route.QUERY)

    assert hasattr(batch, "pooling_mask")
    np.testing.assert_array_equal(
        np.asarray(batch.pooling_mask),
        [[0, 0, 0, 0, 1, 1, 1, 1]],
    )


def test_unknown_module_is_rejected_without_dynamic_import(tmp_path):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    modules = json.loads((checkpoint / "modules.json").read_text())
    modules[-1]["type"] = "organization.remote_code.ArbitraryModule"
    _write_json(checkpoint / "modules.json", modules)

    with pytest.raises(ValueError, match="unsupported or executable"):
        load_sentence_transformer_artifact(checkpoint)
