#!/usr/bin/env python3
"""Accept the real Qwen3-0.6B reward-model training and artifact lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from representax.core import score_logits
from representax.models.qwen_reward import (
    QWEN3_REWARD_0_6B_MODEL_ID,
    QWEN3_REWARD_0_6B_REVISION,
    QwenRewardCheckpointAdapter,
    QwenRewardModel,
    load_qwen_reward_model,
)
from representax.tasks.reward_modeling import PairwiseRewardCollator, PairwiseRewardTask


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--transformers-python", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()
    model, processor = load_qwen_reward_model(
        QWEN3_REWARD_0_6B_MODEL_ID,
        revision=QWEN3_REWARD_0_6B_REVISION,
        cache_directory=arguments.cache_directory,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        head_seed=17,
        sequence_length_buckets=(32,),
    )
    model = eqx.filter_shard(model, jax.devices()[0])
    checkpoint = Path(processor.data_contract()["checkpoint"])
    collator = PairwiseRewardCollator(processor=processor)
    batch = collator(
        [
            {
                "chosen": "Human: Explain gravity. Assistant: Gravity attracts mass.",
                "rejected": "Human: Explain gravity. Assistant: Gravity is a color.",
            },
            {
                "chosen": "Human: What is 2+2? Assistant: 4.",
                "rejected": "Human: What is 2+2? Assistant: 5.",
            },
        ]
    )
    padded = np.asarray(batch.chosen.attention_mask) == 0
    padded_ids = np.asarray(batch.chosen.input_ids)[padded]
    if padded_ids.size and not np.all(padded_ids == model.config.backbone.pad_token_id):
        raise AssertionError("reward processing must use one checkpoint pad token")
    task = PairwiseRewardTask()

    def objective(candidate: QwenRewardModel) -> jax.Array:
        return task.loss(candidate, batch).loss

    value_and_gradient = eqx.filter_jit(eqx.filter_value_and_grad(cast(Any, objective)))
    loss, gradients = value_and_gradient(model)
    gradient_leaves = [
        np.asarray(value)
        for value in jax.tree.leaves(eqx.filter(gradients, eqx.is_array))
    ]
    finite_gradients = all(np.all(np.isfinite(value)) for value in gradient_leaves)
    nonzero_gradient_leaves = sum(bool(np.any(value != 0)) for value in gradient_leaves)
    head_gradient_norm = float(jnp.linalg.norm(gradients.score_head.weight))
    if not finite_gradients or nonzero_gradient_leaves < 2 or head_gradient_norm == 0:
        raise AssertionError("reward backbone and scalar head must receive gradients")

    parameters = eqx.filter(model, eqx.is_inexact_array)
    optimizer = optax.sgd(1e-4)
    optimizer_state = optimizer.init(parameters)
    updates, _ = optimizer.update(gradients, optimizer_state, parameters)
    updated = eqx.apply_updates(model, updates)
    head_update_norm = float(
        jnp.linalg.norm(updated.score_head.weight - model.score_head.weight)
    )
    if head_update_norm == 0:
        raise AssertionError("optimizer must update the scalar head")

    adapter = QwenRewardCheckpointAdapter()
    export = adapter.save(
        updated,
        arguments.artifact_directory,
        source_checkpoint=checkpoint,
    )
    reloaded = adapter.load(
        export,
        head_key=jax.random.key(999),
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        model_id=QWEN3_REWARD_0_6B_MODEL_ID,
        revision=QWEN3_REWARD_0_6B_REVISION,
    )
    expected = np.asarray(score_logits(updated, batch.chosen))
    actual = np.asarray(score_logits(reloaded, batch.chosen))
    reload_exact = bool(np.array_equal(expected, actual))
    if not reload_exact:
        raise AssertionError("native reward export/reload changed logits")
    transformers_output = arguments.artifact_directory.parent / "transformers.npz"
    transformers_inputs = (
        arguments.artifact_directory.parent / "transformers-inputs.npz"
    )
    np.savez(
        transformers_inputs,
        input_ids=np.asarray(batch.chosen.input_ids),
        attention_mask=np.asarray(batch.chosen.attention_mask),
    )
    subprocess.run(
        [
            str(arguments.transformers_python),
            "-m",
            "tests.models.qwen_reward.transformers_reload",
            "--checkpoint",
            str(export),
            "--inputs",
            str(transformers_inputs),
            "--output",
            str(transformers_output),
            "--dtype",
            "bfloat16",
            "--device",
            "cuda",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    with np.load(transformers_output) as result:
        transformers_logits = np.asarray(result["logits"])
    transformers_reload_finite = bool(
        transformers_logits.shape == expected[:, None].shape
        and np.all(np.isfinite(transformers_logits))
    )
    if not transformers_reload_finite:
        raise AssertionError("Transformers reward reload did not produce finite logits")
    transformers_difference = float(
        np.max(np.abs(transformers_logits[:, 0] - expected))
    )
    if transformers_difference > 0.02:
        raise AssertionError(
            "native and Transformers BF16 reward logits exceed tolerance: "
            f"{transformers_difference}"
        )
    _atomic_json(
        arguments.summary,
        {
            "schema_version": "representax-qwen-reward-acceptance-v1",
            "model_id": QWEN3_REWARD_0_6B_MODEL_ID,
            "revision": QWEN3_REWARD_0_6B_REVISION,
            "device": str(jax.devices()[0]),
            "loss": float(loss),
            "finite_gradients": finite_gradients,
            "nonzero_gradient_leaves": nonzero_gradient_leaves,
            "head_gradient_norm": head_gradient_norm,
            "head_update_norm": head_update_norm,
            "native_reload_exact": reload_exact,
            "transformers_reload_finite": transformers_reload_finite,
            "transformers_native_max_absolute_difference": transformers_difference,
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_directory": str(export.resolve()),
        },
    )


if __name__ == "__main__":
    main()
