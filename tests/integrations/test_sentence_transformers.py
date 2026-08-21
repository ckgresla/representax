"""Torch-free Sentence Transformers artifact and host-embedding contracts."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from safetensors.numpy import save_file

from representax import Route
from representax.config import ComponentConfig, DataConfig
from representax.data import Artifact, mix, source
from representax.integrations import (
    load_sentence_transformer,
    load_sentence_transformer_artifact,
    load_sentence_transformer_modules,
)
from representax.models.bert import BertCheckpointAdapter, BertConfig, BertEncoder
from representax.models.mpnet import (
    MPNetCheckpointAdapter,
    MPNetConfig,
    MPNetEncoder,
)
from representax.models.sentence import SentenceEncoder
from representax.tasks.pairwise import PairwiseCollator
from representax.tasks.retrieval import RetrievalCollator
from representax.train import build_batches


class _Tokenizer:
    def __init__(self, *, start_id: int = 1, pad_id: int = 0) -> None:
        self.start_id = start_id
        self.pad_id = pad_id
        self.all_special_ids = (start_id, pad_id, 2)

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
            tokens = [
                self.start_id,
                *(3 + ord(character) % 10 for character in text),
                2,
            ]
            tokens = tokens[:max_length]
            if tokens[-1] != 2:
                tokens[-1] = 2
            rows.append(tokens)
        width = max_length if padding == "max_length" else max(map(len, rows))
        input_ids = np.full(
            (len(rows), width),
            self.pad_id,
            dtype=np.int32,
        )
        attention_mask = np.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = row
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        padding,
        truncation,
        max_length,
        return_tensors,
        return_dict,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        assert return_dict is True
        texts = [f"User: {row[0]['content'][0]['text']}" for row in messages]
        return self(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def identity(record):
    return record


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


def _mpnet_checkpoint(path: Path) -> Path:
    config = MPNetConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=10,
        relative_attention_num_buckets=32,
        hidden_dropout_probability=0.0,
        attention_dropout_probability=0.0,
    )
    model = MPNetEncoder.init(config, key=jax.random.key(11))
    MPNetCheckpointAdapter(rematerialization="none").save(model, path)
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
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            },
        ],
    )
    _write_json(
        path / "config_sentence_transformers.json",
        {"model_type": "SentenceTransformer", "similarity_fn_name": "cosine"},
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
        },
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
    model, processor = SentenceEncoder.load_from_hf(
        checkpoint,
        local_files_only=True,
        processor=_Tokenizer(),
    )
    assert model.metadata.output_dimension == 3
    assert processor.data_contract()["max_sequence_length"] == 8


def test_sentence_pair_collator_has_fixed_shapes_and_a_stable_contract(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    loaded = load_sentence_transformer(checkpoint, processor=_Tokenizer())
    collator = PairwiseCollator(
        processor=loaded.processor,
        pad_to_size=4,
    )

    batch = collator(
        [
            {"sentence1": "left", "sentence2": "right", "score": 0.75},
            {"sentence1": "near", "sentence2": "far", "score": 0.25},
        ]
    )

    assert batch.left.input_ids.shape == (4, 8)
    assert batch.right.input_ids.shape == (4, 8)
    np.testing.assert_array_equal(batch.valid, [True, True, False, False])
    np.testing.assert_array_equal(batch.labels, [0.75, 0.25, 0.0, 0.0])
    assert collator.data_contract()["processor"]["max_sequence_length"] == 8


def test_retrieval_pair_collator_builds_aligned_static_mnr_batch(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    loaded = load_sentence_transformer(checkpoint, processor=_Tokenizer())
    collator = RetrievalCollator(processor=loaded.processor)

    batch = collator(
        [
            {"query": "first question", "positive": "first answer"},
            {"query": "second question", "positive": "second answer"},
        ]
    )

    assert batch.query.input_ids.shape == (2, 8)
    assert batch.document.input_ids.shape == (2, 8)
    np.testing.assert_array_equal(batch.positive_mask, np.eye(2, dtype=bool))
    np.testing.assert_array_equal(batch.query_valid, [True, True])
    np.testing.assert_array_equal(batch.document_valid, [True, True])
    contract = collator.data_contract()
    assert contract["schema_version"] == "representax-retrieval-collator-v1"
    assert contract["processor"]["max_sequence_length"] == 8


def test_model_loader_owns_the_processor_and_collators_reuse_it(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    tokenizer_loads = []

    def load_tokenizer(*_args, **_kwargs):
        tokenizer_loads.append(True)
        return _Tokenizer()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        load_tokenizer,
    )
    loaded = load_sentence_transformer(
        checkpoint,
        local_files_only=True,
    )
    collator = RetrievalCollator(processor=loaded.processor)

    batch = collator(
        [
            {"query": "first question", "positive": "first answer"},
            {"query": "second question", "positive": "second answer"},
        ]
    )

    assert tokenizer_loads == [True]
    assert batch.query.input_ids.shape == (2, 8)
    assert batch.document.input_ids.shape == (2, 8)
    assert collator.data_contract()["processor"]["max_sequence_length"] == 8


def test_sentence_processor_admits_only_configured_static_length_buckets(tmp_path):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    model = load_sentence_transformer(
        checkpoint,
        processor=_Tokenizer(),
        sequence_length_buckets=(8, 4, 8),
    )

    short = model.preprocess(("a", "bc"))
    long = model.preprocess(("abcdefghijk", "z"))

    assert model.processor.data_contract()["sequence_length_buckets"] == [4, 8]
    assert short.input_ids.shape == (2, 4)
    assert long.input_ids.shape == (2, 8)
    assert model.processor.data_contract()["sequence_length_buckets"] == [4, 8]
    np.testing.assert_array_equal(short.input_ids[:, -1], [0, 2])
    np.testing.assert_array_equal(long.input_ids[0, -1], 2)


def test_sentence_processor_rejects_buckets_beyond_the_model_limit(tmp_path):
    checkpoint = _checkpoint(tmp_path / "sentence-model")

    with pytest.raises(ValueError, match="cannot exceed max_sequence_length"):
        load_sentence_transformer(
            checkpoint,
            processor=_Tokenizer(),
            sequence_length_buckets=(4, 16),
        )


def test_native_grain_loader_injects_one_processor_into_task_collation(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"query": "first question", "positive": "first answer"},
                {"query": "second question", "positive": "second answer"},
            )
        )
        + "\n"
    )
    tokenizer_loads = []

    def load_tokenizer(*_args, **_kwargs):
        tokenizer_loads.append(True)
        return _Tokenizer()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        load_tokenizer,
    )
    loaded = load_sentence_transformer(
        checkpoint,
        local_files_only=True,
    )
    loader = build_batches(
        DataConfig(
            distribution=mix(
                source(records.as_uri(), map=identity),
                shuffle=False,
            ),
            collate=ComponentConfig(
                target="representax.tasks.retrieval.RetrievalCollator"
            ),
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        batch_size=2,
        processor=loaded.processor,
    )

    batch = next(iter(loader))

    assert tokenizer_loads == [True]
    assert batch.query.input_ids.shape == (2, 8)
    assert batch.document.input_ids.shape == (2, 8)


def test_native_grain_path_preserves_processor_static_shape_buckets(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    records = tmp_path / "bucketed-records.jsonl"
    records.write_text(
        json.dumps({"query": "a", "positive": "bc"})
        + "\n"
        + json.dumps({"query": "d", "positive": "ef"})
        + "\n"
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    loaded = load_sentence_transformer(
        checkpoint,
        local_files_only=True,
        sequence_length_buckets=(4, 8),
    )
    data_config = DataConfig(
        distribution=mix(
            source(records.as_uri(), map=identity),
            shuffle=False,
        ),
        collate=ComponentConfig(target="representax.tasks.retrieval.RetrievalCollator"),
        num_threads=0,
        prefetch_buffer_size=0,
    )
    loader = build_batches(
        data_config,
        batch_size=2,
        processor=loaded.processor,
    )

    batch = next(iter(loader))

    # Route prompts are applied before admission and move these rows to bucket 8.
    assert batch.query.input_ids.shape == (2, 8)
    assert batch.document.input_ids.shape == (2, 8)
    processor_contract = loader.data_contract["source"]["implementations"][
        "batch_mapper"
    ]["implementation"]
    assert processor_contract["state_sha256"].startswith("sha256:")

    fixed = load_sentence_transformer(
        checkpoint,
        local_files_only=True,
        sequence_length_buckets=(8,),
    )
    fixed_loader = build_batches(
        data_config,
        batch_size=2,
        processor=fixed.processor,
    )
    assert loader.data_fingerprint != fixed_loader.data_fingerprint


def test_generic_processor_is_the_shared_static_preprocessing_boundary(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    loaded = load_sentence_transformer(checkpoint, processor=_Tokenizer())

    batch = loaded.processor(("first", "second"))

    assert batch.input_ids.shape == (2, 8)
    assert batch.attention_mask.shape == (2, 8)
    assert loaded.processor.data_contract()["schema_version"] == (
        "representax-text-processor-v1"
    )


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


def test_sentence_processor_accepts_raw_text_artifacts(tmp_path):
    checkpoint = _checkpoint(tmp_path / "sentence-model")
    loaded = load_sentence_transformer(checkpoint, processor=_Tokenizer())

    batch = loaded.processor(
        (Artifact.text("first"), Artifact.text("second")),
        route=Route.DOCUMENT,
    )

    assert batch.input_ids.shape == (2, 8)


def test_mpnet_backbone_uses_its_native_token_contract(tmp_path):
    checkpoint = _mpnet_checkpoint(tmp_path / "sentence-mpnet")
    model = load_sentence_transformer(
        checkpoint,
        processor=_Tokenizer(start_id=0, pad_id=1),
    )

    output = model.embed(["a", "bc", "def"], batch_size=2)

    assert isinstance(model.model.backbone, MPNetEncoder)
    assert output.shape == (3, 4)
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
