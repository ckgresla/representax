from __future__ import annotations

import json
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.config import (
    BatchConfig,
    CheckpointConfig,
    ComponentConfig,
    DataConfig,
    ExportConfig,
    JobConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from representax.core import Route
from representax.data import mix, source
from representax.export import export_inference_bundle, load_inference_bundle
from representax.models.vjepa2_1 import VJEPA2_1Config, VJEPA2_1Model
from representax.tasks import build_task
from representax.tasks.jepa import (
    VJEPA2_1Batch,
    VJEPA2_1DenseConfig,
    VJEPA2_1Task,
    VJEPA2_1TaskConfig,
    mask_distance_weights,
)
from representax.train import (
    CheckpointManager,
    build_train_step,
    init_train_state,
    run_job,
    training_checkpointables,
)


def tiny_model() -> VJEPA2_1Model:
    config = VJEPA2_1Config(
        image_size=8,
        patch_size=4,
        video_frames=4,
        tubelet_size=2,
        hidden_size=12,
        depth=2,
        heads=2,
        predictor_hidden_size=12,
        predictor_depth=2,
        predictor_heads=2,
        supervision_layers=(0, 1),
    )
    return VJEPA2_1Model.init(
        config,
        key=jax.random.key(0),
        rematerialization="none",
    )


def build_tiny_vjepa(*, key, activation_rematerialization="none"):
    config = VJEPA2_1Config(
        image_size=8,
        patch_size=4,
        video_frames=4,
        tubelet_size=2,
        hidden_size=12,
        depth=2,
        heads=2,
        predictor_hidden_size=12,
        predictor_depth=2,
        predictor_heads=2,
        supervision_layers=(0, 1),
    )
    return VJEPA2_1Model.init(
        config,
        key=key,
        rematerialization=activation_rematerialization,
    )


def tiny_batch() -> VJEPA2_1Batch:
    return VJEPA2_1Batch(
        pixels=jax.random.normal(jax.random.key(1), (2, 3, 8, 8)),
        context_ids=jnp.asarray([[[0, 1]], [[0, 2]]]),
        target_ids=jnp.asarray([[[2, 3]], [[1, 3]]]),
        context_valid=jnp.ones((2, 1, 2), dtype=jnp.bool_),
        target_valid=jnp.ones((2, 1, 2), dtype=jnp.bool_),
    )


def test_mask_distance_matches_reference_geometry() -> None:
    context = jnp.asarray([[[0, 1]]])
    target = jnp.asarray([[[2, 3]]])
    actual = mask_distance_weights(
        context,
        target,
        grid_height=2,
        grid_width=2,
    )
    np.testing.assert_allclose(actual, jnp.ones_like(actual))


def test_registry_builds_distinct_vjepa_task() -> None:
    task = build_task(
        VJEPA2_1TaskConfig(),
        VJEPA2_1DenseConfig(context_weight=0.25, ema_start=0.9, ema_end=0.99),
    )
    assert isinstance(task, VJEPA2_1Task)
    assert task.context_weight == 0.25
    assert task.ema_start == 0.9
    assert task.ema_end == 0.99


def test_compiled_step_excludes_and_updates_ema_target() -> None:
    model = tiny_model()
    batch = tiny_batch()
    task = VJEPA2_1Task(ema_start=0.9, ema_end=0.9)
    trainable_filter = model.training_filter()
    assert not any(bool(value) for value in jax.tree.leaves(trainable_filter.target))
    optimizer = optax.adamw(1e-3)
    state = init_train_state(
        model,
        optimizer,
        trainable_filter=trainable_filter,
    )
    result = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        trainable_filter=trainable_filter,
    )(state, batch, jax.random.key(2))
    previous_model = cast(VJEPA2_1Model, state.model)
    updated_model = cast(VJEPA2_1Model, result.state.model)
    assert bool(result.metrics.numeric_finite)
    assert int(result.state.step) == 1
    target_delta = sum(
        float(jnp.sum(jnp.abs(before - after)))
        for before, after in zip(
            jax.tree.leaves(previous_model.target),
            jax.tree.leaves(updated_model.target),
            strict=True,
        )
    )
    assert target_delta > 0.0
    expected = jax.tree.map(
        lambda old, online: 0.9 * old + 0.1 * online,
        previous_model.target,
        updated_model.online,
    )
    assert eqx.tree_equal(updated_model.target, expected, rtol=1e-6, atol=1e-6)


def test_gradient_accumulation_rejects_batch_global_vjepa_objective() -> None:
    model = tiny_model()
    optimizer = optax.adamw(1e-3)
    with np.testing.assert_raises_regex(TypeError, "does not support exact"):
        build_train_step(
            VJEPA2_1Task(),
            optimizer,
            gradient_accumulation_steps=2,
            trainable_filter=model.training_filter(),
        )


def test_vjepa_model_is_a_clean_reloadable_native_bundle(tmp_path) -> None:
    model = tiny_model()
    job = JobConfig(
        name="tiny-vjepa-export",
        model=ModelConfig(target="tests.tasks.test_vjepa2_1.build_tiny_vjepa"),
        task=VJEPA2_1TaskConfig(),
        loss=VJEPA2_1DenseConfig(),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 1e-3},
            )
        ),
        data=DataConfig(
            distribution=mix(
                source(
                    "memory://unused",
                    map="representax.data.identity",
                )
            )
        ),
        training=TrainingConfig(
            global_batch_size=1,
            max_steps=1,
            seed=0,
            batch=BatchConfig(micro_batch_size=1),
            activation_rematerialization="none",
        ),
        export=ExportConfig(),
    )
    bundle = export_inference_bundle(
        model,
        job,
        tmp_path / "vjepa",
        iteration=1,
    )
    restored, restored_job = load_inference_bundle(bundle.path)
    assert restored_job == job
    assert eqx.tree_equal(model, restored)


def test_async_checkpoint_restores_online_predictor_target_and_optimizer(
    tmp_path,
) -> None:
    model = tiny_model()
    task = VJEPA2_1Task(ema_start=0.9, ema_end=0.9)
    optimizer = optax.adamw(1e-3)
    trainable_filter = model.training_filter()
    initial = init_train_state(
        model,
        optimizer,
        trainable_filter=trainable_filter,
    )
    updated = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        trainable_filter=trainable_filter,
    )(initial, tiny_batch(), jax.random.key(2)).state

    manager = CheckpointManager(
        tmp_path / "run",
        scientific_fingerprint="sha256:vjepa-scientific",
        data_fingerprint="sha256:vjepa-data",
        asynchronous=True,
    )
    manager.save(
        1,
        training_checkpointables(
            state=updated,
            iteration=1,
            rng=jax.random.key(3),
            data_state={"next_index": 2},
            logging_cursor={
                "events_bytes": 0,
                "metrics_bytes": 0,
                "optimizer_step": 1,
                "sequence": 4,
            },
        ),
    )
    manager.close()

    resumed = CheckpointManager(
        tmp_path / "run",
        scientific_fingerprint="sha256:vjepa-scientific",
        data_fingerprint="sha256:vjepa-data",
    )
    restored = resumed.restore_training_state(initial)
    resumed.close()
    assert restored.iteration == 1
    assert restored.data_state == {"next_index": 2}
    assert eqx.tree_equal(restored.state, updated)


@pytest.mark.runtime
def test_job_config_trains_from_disk_and_exports_reloadable_vjepa(tmp_path) -> None:
    config = tiny_model().online.config
    data = tmp_path / "images.jsonl"
    rows = []
    for index in range(4):
        image = np.full((8, 8, 3), index * 16, dtype=np.uint8)
        rows.append(json.dumps({"artifact": image.tolist()}))
    data.write_text("\n".join(rows) + "\n")
    job = JobConfig(
        name="vjepa-disk-to-model",
        model=ModelConfig(
            target="representax.models.vjepa2_1.load_vjepa2_1",
            parameters={
                "config": config.model_dump(mode="json"),
                "modality": "image",
                "training": False,
            },
        ),
        task=VJEPA2_1TaskConfig(),
        loss=VJEPA2_1DenseConfig(ema_start=0.9, ema_end=0.9),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 1e-3},
            ),
            max_gradient_norm=None,
        ),
        data=DataConfig(
            distribution=mix(
                source(str(data), map="representax.data.identity"),
                shuffle=False,
            ),
            collate=ComponentConfig(
                target="representax.models.vjepa2_1.VJEPA2_1Collator",
                parameters={
                    "config": config.model_dump(mode="json"),
                    "patterns": [
                        {
                            "spatial_scale": [0.5, 0.5],
                            "temporal_scale": [1.0, 1.0],
                            "aspect_ratio": [1.0, 1.0],
                        }
                    ],
                    "seed": 23,
                },
            ),
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        training=TrainingConfig(
            global_batch_size=2,
            max_steps=2,
            seed=29,
            batch=BatchConfig(micro_batch_size=2),
            activation_rematerialization="none",
        ),
        checkpointing=CheckpointConfig(every=1, keep=2, asynchronous=True),
        export=ExportConfig(),
    )
    result = run_job(job, tmp_path / "run")
    assert result.inference_bundle is not None
    restored, restored_job = load_inference_bundle(result.inference_bundle)
    assert result.completed_iterations == 2
    assert restored_job == job
    assert isinstance(restored, VJEPA2_1Model)
    assert restored.encode(
        jnp.zeros((1, 3, 8, 8), dtype=jnp.float32),
        route=Route.GENERIC,
    ).shape == (1, config.hidden_size)
