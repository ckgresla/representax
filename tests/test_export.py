"""Inference-ready native and Hugging Face export contracts."""

import json
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
    LoRAConfig,
    ModelConfig,
    OptimizationConfig,
    PrecisionConfig,
    QuantizedLoRAConfig,
    TrainingConfig,
)
from representax.data import mix, source
from representax.export import export_inference_bundle, load_inference_bundle
from representax.models import merge_quantized_lora
from representax.models.bert import BertCheckpointAdapter, BertConfig, BertEncoder
from representax.models.jina_v5 import JinaV5TextCheckpointAdapter
from representax.models.modernvbert import (
    ModernVBERTCheckpointAdapter,
    ModernVBERTEncoder,
)
from representax.tasks.pairwise import CosineRegressionConfig, PairwiseConfig
from tests.models.jina_v5.test_model import _synthetic_state
from tests.models.jina_v5.test_model import tiny_config as tiny_jina
from tests.models.modernvbert.test_model import tiny_multimodal_config


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


def build_tiny_bfloat16_bert(*, key):
    return jax.tree.map(
        lambda value: (
            value.astype(jax.numpy.bfloat16) if eqx.is_inexact_array(value) else value
        ),
        build_tiny_bert(key=key),
    )


def build_tiny_jina(*, key):
    del key
    config = tiny_jina()
    return JinaV5TextCheckpointAdapter().from_state_dict(
        config,
        _synthetic_state(config),
        model_id="test/jina-v5",
        revision="fixture",
    )


def build_tiny_modernvbert(*, key):
    return ModernVBERTEncoder.init(tiny_multimodal_config(), key=key)


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


def _family_job(source_checkpoint, *, model_target: str, adapter_target: str):
    job = _job(source_checkpoint)
    return job.model_copy(
        update={
            "model": ModelConfig(target=model_target),
            "export": ExportConfig(
                huggingface=HuggingFaceExportConfig(
                    source_checkpoint=str(source_checkpoint),
                    adapter=ComponentConfig(target=adapter_target),
                )
            ),
        }
    )


def identity(record):
    return record


def _save_safetensors(path, state):
    from safetensors.numpy import save_file

    path.mkdir()
    save_file(
        {name: np.asarray(value) for name, value in state.items()},
        path / "model.safetensors",
    )


def _assert_array_trees_equal(actual, expected):
    for left, right in zip(
        jax.tree.leaves(eqx.filter(actual, eqx.is_array)),
        jax.tree.leaves(eqx.filter(expected, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


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
    from representax.train.job import prepare_model

    model, _ = prepare_model(
        source,
        adapter=job.training.adapter,
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


def test_unquantized_adapter_bundle_reloads_native(tmp_path):
    source = build_tiny_bert(key=jax.random.key(17))
    source_checkpoint = tmp_path / "source"
    adapter = BertCheckpointAdapter(rematerialization="none")
    adapter.save(source, source_checkpoint)
    base_job = _job(source_checkpoint)
    job = base_job.model_copy(
        update={
            "training": base_job.training.model_copy(
                update={"adapter": LoRAConfig(rank=2, alpha=4.0)}
            )
        }
    )
    restored_config = JobConfig.model_validate_json(job.model_dump_json())
    assert type(restored_config.training.adapter) is LoRAConfig
    from representax.train.job import prepare_model

    model, _ = prepare_model(
        source,
        adapter=job.training.adapter,
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
    _assert_array_trees_equal(native, model)


def test_mixed_precision_full_model_bundle_reloads_master_parameters(tmp_path):
    base_job = _job(tmp_path / "unused")
    job = base_job.model_copy(
        update={
            "model": ModelConfig(target="tests.test_export.build_tiny_bfloat16_bert"),
            "training": base_job.training.model_copy(
                update={"precision": PrecisionConfig.bfloat16_mixed()}
            ),
            "export": ExportConfig(),
        }
    )
    source = build_tiny_bfloat16_bert(key=jax.random.key(job.training.seed))
    from representax.precision import prepare_master_model, resolve_precision_policy

    trained = prepare_master_model(
        source,
        resolve_precision_policy(job.training.precision),
    )
    assert all(
        leaf.dtype == jax.numpy.float32
        for leaf in jax.tree.leaves(eqx.filter(trained, eqx.is_inexact_array))
    )

    bundle = export_inference_bundle(
        trained,
        job,
        tmp_path / "mixed-precision-bundle",
        iteration=1,
    )
    restored, restored_job = load_inference_bundle(bundle.path)

    assert restored_job == job
    _assert_array_trees_equal(restored, trained)


def test_jina_v5_loads_local_hf_and_exports_exact_native_and_hf_bundles(tmp_path):
    from representax.integrations import load_jina_v5_text_encoder

    model = build_tiny_jina(key=jax.random.key(3))
    adapter = JinaV5TextCheckpointAdapter()
    source_checkpoint = tmp_path / "jina-source"
    _save_safetensors(source_checkpoint, adapter.state_dict(model))
    config = tiny_jina()
    (source_checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl_text",
                "output_dimension": config.output_dimension,
                "text_config": {
                    "vocab_size": config.vocab_size,
                    "hidden_size": config.hidden_size,
                    "intermediate_size": config.intermediate_size,
                    "num_hidden_layers": config.num_hidden_layers,
                    "num_attention_heads": config.num_attention_heads,
                    "num_key_value_heads": config.num_key_value_heads,
                    "head_dim": config.head_dimension,
                    "max_position_embeddings": config.max_position_embeddings,
                    "rope_parameters": {"rope_theta": config.rope_theta},
                    "rms_norm_eps": config.norm_epsilon,
                    "pad_token_id": config.pad_token_id,
                },
            }
        )
        + "\n"
    )
    loaded = load_jina_v5_text_encoder(
        source_checkpoint,
        local_files_only=True,
        parameter_dtype="float32",
        compute_dtype="float32",
    )
    _assert_array_trees_equal(loaded, model)

    job = _family_job(
        source_checkpoint,
        model_target="tests.test_export.build_tiny_jina",
        adapter_target="representax.models.JinaV5TextCheckpointAdapter",
    )
    bundle = export_inference_bundle(model, job, tmp_path / "jina-bundle", iteration=1)
    native, _ = load_inference_bundle(bundle.path)
    assert bundle.huggingface_path is not None
    huggingface = adapter.load(bundle.huggingface_path)
    _assert_array_trees_equal(native, model)
    _assert_array_trees_equal(huggingface, model)


def test_modernvbert_loads_local_hf_and_exports_exact_native_and_hf_bundles(
    tmp_path,
):
    from representax.integrations import load_modernvbert_encoder

    model = build_tiny_modernvbert(key=jax.random.key(5))
    adapter = ModernVBERTCheckpointAdapter()
    source_checkpoint = tmp_path / "modernvbert-source"
    _save_safetensors(source_checkpoint, adapter.state_dict(model))
    config = tiny_multimodal_config()
    (source_checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "modernvbert",
                "image_token_id": config.image_token_id,
                "pixel_shuffle_factor": config.pixel_shuffle_factor,
                "text_config": {
                    "vocab_size": config.text.vocab_size,
                    "hidden_size": config.text.hidden_size,
                    "intermediate_size": config.text.intermediate_size,
                    "num_hidden_layers": config.text.num_hidden_layers,
                    "num_attention_heads": config.text.num_attention_heads,
                    "layer_types": list(config.text.layer_types),
                    "local_attention": config.text.local_attention,
                    "rope_parameters": {
                        "full_attention": {
                            "rope_theta": config.text.full_attention_rope_theta
                        },
                        "sliding_attention": {
                            "rope_theta": config.text.sliding_attention_rope_theta
                        },
                    },
                    "layer_norm_eps": config.text.norm_epsilon,
                    "max_position_embeddings": config.text.max_position_embeddings,
                },
                "vision_config": {
                    "hidden_size": config.vision.hidden_size,
                    "intermediate_size": config.vision.intermediate_size,
                    "num_hidden_layers": config.vision.num_hidden_layers,
                    "num_attention_heads": config.vision.num_attention_heads,
                    "num_channels": config.vision.num_channels,
                    "image_size": config.vision.image_size,
                    "patch_size": config.vision.patch_size,
                    "layer_norm_eps": config.vision.norm_epsilon,
                    "hidden_act": config.vision.hidden_activation,
                },
            }
        )
        + "\n"
    )
    loaded = load_modernvbert_encoder(source_checkpoint, local_files_only=True)
    _assert_array_trees_equal(loaded, model)

    job = _family_job(
        source_checkpoint,
        model_target="tests.test_export.build_tiny_modernvbert",
        adapter_target="representax.models.ModernVBERTCheckpointAdapter",
    )
    bundle = export_inference_bundle(
        model, job, tmp_path / "modernvbert-bundle", iteration=2
    )
    native, _ = load_inference_bundle(bundle.path)
    assert bundle.huggingface_path is not None
    huggingface = adapter.load(bundle.huggingface_path)
    _assert_array_trees_equal(native, model)
    _assert_array_trees_equal(huggingface, model)
