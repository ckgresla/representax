"""MLM semantics, sparse-head gradients, accumulation and sharding."""

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.config import JobConfig
from representax.models.modernvbert import (
    ModernBERTMaskedLM,
    ModernVBERTTextBatch,
    ModernVBERTTextCheckpointAdapter,
)
from representax.tasks import build_task
from representax.tasks.masked_language_modeling import (
    MaskedLanguageModelConfig,
    MaskedLanguageModelLossConfig,
    MaskedLanguageModelTask,
    mask_tokens,
    masked_language_model_batch,
    masked_language_model_loss_terms,
)
from representax.train import build_train_step, init_train_state
from tests.models.modernvbert.test_model import tiny_config


def fixture():
    model = ModernBERTMaskedLM.init(tiny_config(), key=jax.random.key(1))
    inputs = ModernVBERTTextBatch(
        input_ids=jnp.asarray([[1, 2, 3, 0], [3, 2, 1, 4], [5, 2, 7, 0], [6, 4, 2, 1]]),
        attention_mask=jnp.asarray(
            [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
        ),
    )
    batch = masked_language_model_batch(
        inputs=inputs,
        positions=jnp.asarray([[1, 0], [1, 2], [1, 0], [0, 0]]),
        labels=jnp.asarray([[5, -100], [3, 8], [7, -100], [-100, -100]]),
    )
    return model, batch


def test_loss_and_gradient_match_independent_numpy_reference():
    logits = np.random.default_rng(7).normal(size=(3, 4, 5)).astype(np.float32)
    labels = np.asarray([[2, 0, -100, 3], [-100, 1, -100, -100], [3, 2, 1, 0]])
    rows = np.asarray([True, True, False])
    valid = (labels != -100) & rows[:, None]
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    expected = -np.log(probabilities[valid, labels[valid]]).mean()
    gradient = probabilities.copy()
    for b, t in zip(*np.where(valid), strict=True):
        gradient[b, t, labels[b, t]] -= 1
    gradient *= valid[..., None] / valid.sum()

    def loss(values):
        return masked_language_model_loss_terms(
            values, jnp.asarray(labels), row_valid=jnp.asarray(rows)
        ).loss

    actual, actual_grad = jax.jit(jax.value_and_grad(loss))(jnp.asarray(logits))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_grad, gradient, rtol=1e-6, atol=1e-7)


def test_empty_targets_and_ignored_nan_logits_have_zero_gradients():
    logits = jnp.full((2, 3, 5), jnp.nan)
    labels = jnp.full((2, 3), -100, dtype=jnp.int32)
    value, gradient = jax.jit(
        jax.value_and_grad(
            lambda values: masked_language_model_loss_terms(values, labels).loss
        )
    )(logits)
    np.testing.assert_array_equal(value, 0.0)
    np.testing.assert_array_equal(gradient, 0.0)


def test_sparse_prediction_matches_full_sequence_loss_and_parameter_gradients():
    model, batch = fixture()
    task = MaskedLanguageModelTask()
    all_positions = jnp.broadcast_to(jnp.arange(4), (4, 4))
    full_labels = jnp.full((4, 4), -100, dtype=jnp.int32)
    for row in range(4):
        for col in range(2):
            if int(batch.labels[row, col]) != -100:
                full_labels = full_labels.at[row, batch.positions[row, col]].set(
                    batch.labels[row, col]
                )
    full = replace(batch, labels=full_labels, positions=all_positions)
    sparse_loss, sparse_grad = eqx.filter_value_and_grad(
        lambda m: task.loss(m, batch).loss
    )(model)
    dense_loss, dense_grad = eqx.filter_value_and_grad(
        lambda m: task.loss(m, full).loss
    )(model)
    np.testing.assert_allclose(sparse_loss, dense_loss, rtol=1e-6, atol=1e-6)
    for a, b in zip(
        jax.tree.leaves(sparse_grad), jax.tree.leaves(dense_grad), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
    assert np.linalg.norm(sparse_grad.encoder.tower.token_embedding) > 0
    assert model.predict_masked(batch.inputs, batch.positions).shape == (4, 2, 17)
    # No separately owned vocabulary decoder weight / optimizer buffer.
    vocab_leaves = [x for x in jax.tree.leaves(model) if x.shape == (17, 8)]
    assert len(vocab_leaves) == 1


@pytest.mark.parametrize("empty_first", [False, True])
def test_accumulation_weights_unequal_target_counts_including_empty_microbatch(
    empty_first,
):
    model, batch = fixture()
    if empty_first:
        batch = replace(batch, labels=batch.labels.at[:2].set(-100))
    task = MaskedLanguageModelTask()
    optimizer = optax.sgd(1e-3)
    state = init_train_state(model, optimizer)
    key = jax.random.key(2)
    direct = build_train_step(task, optimizer, max_grad_norm=None)(state, batch, key)
    accumulated = build_train_step(
        task, optimizer, max_grad_norm=None, gradient_accumulation_steps=2
    )(state, batch, key)
    assert bool(accumulated.metrics.numeric_finite)
    np.testing.assert_allclose(direct.metrics.loss, accumulated.metrics.loss, rtol=1e-6)
    for a, b in zip(
        jax.tree.leaves(direct.state), jax.tree.leaves(accumulated.state), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


def test_registry_and_json_job_roundtrip_support_accumulation_not_gradcache():
    from tests.tasks.test_registry import _job_data

    data = _job_data()
    data.update(task=MaskedLanguageModelConfig(), loss=MaskedLanguageModelLossConfig())
    job = JobConfig.model_validate(data)
    assert JobConfig.model_validate_json(job.model_dump_json()) == job
    assert isinstance(build_task(job.task, job.loss), MaskedLanguageModelTask)
    payload = job.model_dump(mode="json")
    payload["training"]["batch"] = {
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 2,
    }
    JobConfig.model_validate(payload)
    payload["training"]["grad_cache"] = {"micro_batch_size": 2}
    with pytest.raises(ValueError):
        JobConfig.model_validate(payload)


def test_masking_preserves_special_padding_and_original_arrays():
    ids = np.arange(40, dtype=np.int32).reshape(4, 10)
    attention = np.ones_like(ids, dtype=bool)
    attention[0] = False
    special = np.zeros_like(attention)
    special[:, :2] = True
    original = ids.copy()
    kwargs: dict[str, Any] = dict(
        attention_mask=attention,
        special_tokens_mask=special,
        vocabulary_size=50,
        mask_token_id=49,
        seed=7,
        probability=0.5,
    )
    corrupted, positions, labels = mask_tokens(ids, **kwargs)
    repeated = mask_tokens(ids, **kwargs)
    for a, b in zip((corrupted, positions, labels), repeated, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ids, original)
    np.testing.assert_array_equal(
        corrupted[~attention | special], ids[~attention | special]
    )
    assert (labels[0] == -100).all()
    for row in range(1, 4):
        chosen = positions[row, labels[row] != -100]
        assert len(chosen) == len(set(chosen)) == 4
        np.testing.assert_array_equal(labels[row, :4], ids[row, chosen])


def test_invalid_batch_and_loss_shapes_rejected():
    _, batch = fixture()
    with pytest.raises(TypeError):
        replace(batch, positions=batch.positions.astype(jnp.float32))
    with pytest.raises(TypeError):
        replace(batch, valid=jnp.ones((4,), dtype=jnp.int32))
    with pytest.raises(ValueError):
        masked_language_model_loss_terms(jnp.zeros((4, 3, 17)), batch.labels)


def collate_mlm_rows(examples):
    return masked_language_model_batch(
        inputs=ModernVBERTTextBatch(
            input_ids=np.asarray(
                [row["input_ids"] for row in examples], dtype=np.int32
            ),
            attention_mask=np.asarray(
                [row["attention_mask"] for row in examples], dtype=np.int32
            ),
        ),
        positions=np.asarray([row["positions"] for row in examples], dtype=np.int32),
        labels=np.asarray([row["labels"] for row in examples], dtype=np.int32),
    )


@pytest.mark.runtime
def test_mlm_job_resume_and_native_export_reload(tmp_path):
    import json

    from representax import load_inference_bundle
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        ExportConfig,
        ModelConfig,
        OptimizationConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.train import run_job

    _, batch = fixture()
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "input_ids": np.asarray(batch.inputs.input_ids[i]).tolist(),
                    "attention_mask": np.asarray(
                        batch.inputs.attention_mask[i]
                    ).tolist(),
                    "positions": np.asarray(batch.positions[i]).tolist(),
                    "labels": np.asarray(batch.labels[i]).tolist(),
                }
            )
            for i in range(4)
        )
        + "\n"
    )
    job = JobConfig(
        name="mlm-lifecycle",
        model=ModelConfig(
            target="representax.models.modernvbert:ModernBERTMaskedLM.init",
            parameters={"config": tiny_config().model_dump(mode="json")},
        ),
        task=MaskedLanguageModelConfig(),
        loss=MaskedLanguageModelLossConfig(),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 0.001, "weight_decay": 0.0},
            )
        ),
        data=DataConfig(
            distribution=mix(source(str(path), map=identity), shuffle=False),
            collate=ComponentConfig(
                target="tests.tasks.test_masked_language_modeling:collate_mlm_rows"
            ),
            num_threads=1,
            prefetch_buffer_size=1,
        ),
        training=TrainingConfig(
            global_batch_size=4,
            max_steps=2,
            seed=7,
            batch=BatchConfig(micro_batch_size=2, gradient_accumulation_steps=2),
        ),
        checkpointing=CheckpointConfig(every=1, keep=2, asynchronous=False),
        export=ExportConfig(),
    )
    run = tmp_path / "resumed"
    partial = run_job(job, run, stop_after=1)
    assert partial.completed_iterations == 1
    resumed = run_job(job, run, resume=True)
    direct = run_job(job, tmp_path / "direct")
    assert resumed.completed_iterations == direct.completed_iterations == 2
    for a, b in zip(
        jax.tree.leaves(resumed.state), jax.tree.leaves(direct.state), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    assert resumed.inference_bundle is not None
    restored, restored_job = load_inference_bundle(resumed.inference_bundle)
    assert restored_job == job
    assert isinstance(restored, ModernBERTMaskedLM)
    assert isinstance(resumed.state.model, ModernBERTMaskedLM)
    np.testing.assert_array_equal(
        restored.predict_masked(batch.inputs, batch.positions),
        resumed.state.model.predict_masked(batch.inputs, batch.positions),
    )


@pytest.mark.distributed
@pytest.mark.parametrize("devices", [2, 4])
@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_sharded_unequal_token_counts_match_single_device(devices, strategy):
    from representax.train import ShardingPlan

    if jax.device_count() < devices:
        pytest.skip(
            "requires multiple devices; on CPU set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=4"
        )
    model, batch = fixture()
    # First shard has no labels; averaging device means would be incorrect.
    batch = replace(batch, labels=batch.labels.at[:2].set(-100))
    optimizer = optax.sgd(1e-3)
    state = init_train_state(model, optimizer)
    task = MaskedLanguageModelTask()
    key = jax.random.key(3)
    expected = build_train_step(task, optimizer, max_grad_norm=None)(state, batch, key)
    mesh = jax.make_mesh((devices,), ("data",), devices=jax.devices()[:devices])
    plan = (
        ShardingPlan.ddp(state, optimizer, mesh, axis_name="data")
        if strategy == "ddp"
        else ShardingPlan.fsdp(
            state,
            optimizer,
            mesh,
            parameter_axis_name="data",
            minimum_parameter_elements=1,
        )
    )
    actual = build_train_step(task, optimizer, plan=plan, max_grad_norm=None)(
        plan.place_state(state), plan.place_batch(batch), key
    )
    np.testing.assert_allclose(actual.metrics.loss, expected.metrics.loss, rtol=1e-6)
    for a, b in zip(
        jax.tree.leaves(actual.state), jax.tree.leaves(expected.state), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


@pytest.mark.distributed
@pytest.mark.parametrize("devices", [2, 4])
@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_fsdp_with_data_sharding_and_accumulation_matches_global_update(
    devices, strategy
):
    from representax.train import ShardingPlan

    if jax.device_count() < devices:
        pytest.skip("requires multiple devices")
    model, batch = fixture()
    batch = jax.tree.map(lambda x: jnp.tile(x, (4,) + (1,) * (x.ndim - 1)), batch)
    optimizer = optax.adamw(1e-4, weight_decay=0.01)
    state = init_train_state(model, optimizer)
    task = MaskedLanguageModelTask()
    key = jax.random.key(3)
    with jax.default_matmul_precision("highest"):
        expected = build_train_step(task, optimizer)(state, batch, key)
        mesh = jax.make_mesh((devices,), ("data",), devices=jax.devices()[:devices])
        plan = (
            ShardingPlan.ddp(state, optimizer, mesh, axis_name="data")
            if strategy == "ddp"
            else ShardingPlan.fsdp(
                state,
                optimizer,
                mesh,
                parameter_axis_name="data",
                data_axis_name="data",
                minimum_parameter_elements=1,
            )
        )
        actual = build_train_step(
            task,
            optimizer,
            plan=plan,
            gradient_accumulation_steps=2,
        )(plan.place_state(state), plan.place_batch(batch), plan.place_replicated(key))
        jax.block_until_ready(actual)
    assert bool(actual.metrics.numeric_finite)
    assert not bool(actual.metrics.skipped_update)
    np.testing.assert_allclose(actual.metrics.loss, expected.metrics.loss, rtol=1e-6)
    for a, b in zip(
        jax.tree.leaves(actual.state), jax.tree.leaves(expected.state), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


@pytest.mark.parity
def test_fp32_mlm_logits_loss_and_all_parameter_gradients_match_transformers():
    torch = pytest.importorskip("torch")
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    torch.set_num_threads(1)
    model, batch = fixture()
    config = ModernBertConfig.from_dict(
        dict(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32,
            local_attention=4,
            global_attn_every_n_layers=2,
            global_rope_theta=10000.0,
            local_rope_theta=1000.0,
            norm_eps=1e-5,
            classifier_bias=False,
            norm_bias=False,
            decoder_bias=True,
            classifier_activation="gelu",
            attention_dropout=0.0,
            embedding_dropout=0.0,
            mlp_dropout=0.0,
            classifier_dropout=0.0,
            attention_bias=False,
            mlp_bias=False,
            pad_token_id=0,
            sparse_prediction=False,
        )
    )
    config._attn_implementation = "eager"
    reference = ModernBertForMaskedLM(config).float().eval()
    adapter = ModernVBERTTextCheckpointAdapter(weight_prefix="model.")
    weights = adapter.state_dict(model.encoder)
    weights.update(
        {
            "head.dense.weight": model.dense.weight.T,
            "head.norm.weight": model.norm.weight,
            "decoder.weight": model.encoder.tower.token_embedding,
            "decoder.bias": model.decoder_bias,
        }
    )
    reference.load_state_dict(
        {k: torch.tensor(np.asarray(v).copy()) for k, v in weights.items()}, strict=True
    )
    output = reference(
        input_ids=torch.tensor(np.asarray(batch.inputs.input_ids), dtype=torch.long),
        attention_mask=torch.tensor(
            np.asarray(batch.inputs.attention_mask), dtype=torch.long
        ),
    ).logits
    positions = torch.tensor(np.asarray(batch.positions), dtype=torch.long)
    selected = output[torch.arange(4)[:, None], positions]
    labels = torch.tensor(np.asarray(batch.labels), dtype=torch.long)
    ref_loss = torch.nn.functional.cross_entropy(
        selected.reshape(-1, 17), labels.reshape(-1), ignore_index=-100
    )
    ref_loss.backward()
    task = MaskedLanguageModelTask()
    with jax.default_matmul_precision("highest"):
        logits = model.predict_masked(batch.inputs, batch.positions)
        loss, grads = eqx.filter_value_and_grad(lambda m: task.loss(m, batch).loss)(
            model
        )
    np.testing.assert_allclose(logits, selected.detach().numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(loss, ref_loss.detach().numpy(), rtol=1e-6, atol=1e-6)
    mapped_grads = adapter.state_dict(grads.encoder)
    mapped_grads.update(
        {
            "head.dense.weight": grads.dense.weight.T,
            "head.norm.weight": grads.norm.weight,
            "decoder.bias": grads.decoder_bias,
        }
    )
    for name, parameter in reference.named_parameters():
        np.testing.assert_allclose(
            mapped_grads[name],
            parameter.grad.numpy(),
            rtol=3e-4,
            atol=3e-6,
            err_msg=name,
        )


@pytest.mark.distributed
@pytest.mark.parametrize("empty", [False, True])
def test_deferred_ddp_selected_parameters_preserve_gradients_and_frozen_weights(empty):
    from representax.train import ShardingPlan

    if jax.device_count() < 2:
        pytest.skip("requires two devices")
    model, batch = fixture()
    batch = jax.tree.map(lambda x: jnp.tile(x, (4,) + (1,) * (x.ndim - 1)), batch)
    if empty:
        batch = replace(batch, labels=jnp.full_like(batch.labels, -100))
    else:
        batch = replace(batch, labels=batch.labels.at[:8].set(-100))
    selection = jax.tree.map(lambda _: False, model)
    selection = eqx.tree_at(lambda m: m.dense.weight, selection, True)
    optimizer = optax.sgd(1e-2)
    state = init_train_state(model, optimizer, trainable_filter=selection)
    mesh = jax.make_mesh((2,), ("data",), devices=jax.devices()[:2])
    plan = ShardingPlan.ddp(state, optimizer, mesh, trainable_filter=selection)
    kwargs = dict(trainable_filter=selection, max_grad_norm=None)
    expected = build_train_step(MaskedLanguageModelTask(), optimizer, **kwargs)(
        state, batch, None
    )
    actual = build_train_step(
        MaskedLanguageModelTask(),
        optimizer,
        plan=plan,
        gradient_accumulation_steps=4,
        **kwargs,
    )(plan.place_state(state), plan.place_batch(batch), None)
    np.testing.assert_allclose(
        actual.metrics.loss, expected.metrics.loss, atol=1e-6, rtol=1e-6
    )
    for a, b in zip(
        jax.tree.leaves(actual.state), jax.tree.leaves(expected.state), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        (np.asarray(actual.state.model.dense.weight) - np.asarray(model.dense.weight))
        / 1e-2,
        (np.asarray(expected.state.model.dense.weight) - np.asarray(model.dense.weight))
        / 1e-2,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        actual.state.model.encoder.tower.token_embedding,
        model.encoder.tower.token_embedding,
    )
