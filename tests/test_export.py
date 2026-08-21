"""Inference-ready native and Hugging Face export contracts."""

from typing import cast

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
    QuantizedLoRAConfig,
    TrainingConfig,
)
from representax.data import mix, source
from representax.export import export_inference_bundle, load_inference_bundle
from representax.models import merge_quantized_lora
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


def _job(source_checkpoint, *, quantized_adapter: bool = False) -> JobConfig:
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
            distribution=mix(
                source("memory://unused", map="tests.test_export.identity")
            )
        ),
        training=TrainingConfig(
            global_batch_size=1,
            max_steps=1,
            seed=13,
            batch=BatchConfig(micro_batch_size=1),
            adapter=(
                QuantizedLoRAConfig(rank=2, alpha=4.0) if quantized_adapter else None
            ),
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


def test_quantized_adapter_bundle_reloads_native_and_merges_huggingface(tmp_path):
    source = build_tiny_bert(key=jax.random.key(13))
    source_checkpoint = tmp_path / "source"
    adapter = BertCheckpointAdapter(rematerialization="none")
    adapter.save(source, source_checkpoint)
    job = _job(source_checkpoint, quantized_adapter=True)
    from representax.train.job import apply_configured_adapter

    model, _ = apply_configured_adapter(
        source,
        job,
        key=jax.random.fold_in(jax.random.key(job.training.seed), 1),
    )

    bundle = export_inference_bundle(
        model,
        job,
        tmp_path / "adapter-bundle",
        iteration=3,
    )
    native, restored_job = load_inference_bundle(bundle.path)

    assert restored_job == job
    for actual, expected in zip(
        jax.tree.leaves(eqx.filter(native, eqx.is_array)),
        jax.tree.leaves(eqx.filter(model, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert bundle.huggingface_path is not None
    huggingface = adapter.load(bundle.huggingface_path)
    expected = adapter.state_dict(cast(BertEncoder, merge_quantized_lora(model)))
    actual = adapter.state_dict(huggingface)
    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
