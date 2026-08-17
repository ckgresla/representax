"""Inference-ready native and Hugging Face export contracts."""

import equinox as eqx
import jax
import numpy as np

from representax.config import (
    BatchConfig,
    ComponentConfig,
    DataConfig,
    ExportConfig,
    HuggingFaceExportConfig,
    JobConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from representax.data import mix, source
from representax.export import export_inference_bundle, load_inference_bundle
from representax.models.bert import BertCheckpointAdapter, BertConfig, BertEncoder
from representax.tasks.pairwise import CosineRegressionConfig, PairwiseConfig


def build_tiny_bert(*, key):
    return BertEncoder.init(
        BertConfig(
            vocab_size=16,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=8,
            type_vocab_size=2,
            hidden_dropout_probability=0.0,
            attention_dropout_probability=0.0,
        ),
        key=key,
        rematerialization="none",
    )


def _job(source_checkpoint) -> JobConfig:
    return JobConfig(
        name="bert-export",
        model=ModelConfig(target="tests.test_export.build_tiny_bert"),
        task=PairwiseConfig(),
        loss=CosineRegressionConfig(),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.sgd",
                parameters={"learning_rate": 0.01},
            )
        ),
        data=DataConfig(
            recipe=mix(source("memory://unused", map="tests.test_export.identity"))
        ),
        training=TrainingConfig(
            global_batch_size=1,
            max_steps=1,
            seed=13,
            batch=BatchConfig(micro_batch_size=1),
        ),
        export=ExportConfig(
            huggingface=HuggingFaceExportConfig(
                source_checkpoint=str(source_checkpoint),
                adapter=ComponentConfig(
                    target="representax.models.BertCheckpointAdapter",
                    parameters={"rematerialization": "none"},
                ),
            )
        ),
    )


def identity(record):
    return record


def test_inference_bundle_verifies_native_and_huggingface_reload(tmp_path):
    model = build_tiny_bert(key=jax.random.key(13))
    source_checkpoint = tmp_path / "source"
    adapter = BertCheckpointAdapter(rematerialization="none")
    adapter.save(model, source_checkpoint)
    (source_checkpoint / "tokenizer.json").write_text("{}\n")
    job = _job(source_checkpoint)

    bundle = export_inference_bundle(
        model,
        job,
        tmp_path / "bundle",
        iteration=7,
    )

    assert bundle.huggingface_path is not None
    assert (bundle.huggingface_path / "tokenizer.json").is_file()
    native, restored_job = load_inference_bundle(bundle.path)
    huggingface = adapter.load(bundle.huggingface_path)
    assert restored_job == job
    for restored in (native, huggingface):
        for actual, expected in zip(
            jax.tree.leaves(eqx.filter(restored, eqx.is_array)),
            jax.tree.leaves(eqx.filter(model, eqx.is_array)),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
