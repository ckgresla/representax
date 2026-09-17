"""Reproduce whether a historical late-interaction local scope is honored.

Run with PYTHONPATH pointing to the historical checkout and four CPU devices.
This diagnostic does not modify the task or training implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from representax.core import Route, encode_late_interaction
from representax.tasks.late_interaction import LateInteractionTask, late_interaction_loss_terms
from representax.train import GradCache, ShardingPlan, build_train_step, init_train_state
from tests.train.test_distributed_grad_cache import _LateInteractionEncoder, _late_interaction_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    model = _LateInteractionEncoder(key=jax.random.key(103))
    batch = _late_interaction_batch()
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=0)
    initial = init_train_state(model, optimizer)
    execution = GradCache(query_chunk_size=2, document_chunk_size=2, loss_row_chunk_size=2)
    queries = encode_late_interaction(model, batch.query, route=Route.QUERY)
    documents = encode_late_interaction(model, batch.document, route=Route.DOCUMENT)
    global_loss = float(late_interaction_loss_terms(
        queries, documents, batch.positive_mask, temperature=0.2
    ).loss)
    reports = []
    for devices in (2, 4):
        count = 8 // devices
        local_losses = []
        for index in range(devices):
            rows = slice(index * count, (index + 1) * count)
            local_losses.append(float(late_interaction_loss_terms(
                jax.tree.map(lambda x: x[rows], queries),
                jax.tree.map(lambda x: x[rows], documents),
                batch.positive_mask[rows, rows], temperature=0.2,
            ).loss))
        mesh = jax.make_mesh((devices,), ("data",), devices=jax.devices()[:devices])
        plan = ShardingPlan.ddp(initial, optimizer, mesh, axis_name="data")
        step = build_train_step(
            LateInteractionTask(temperature=0.2, negative_scope="local"),
            optimizer, plan=plan, execution=execution, donate_state=False,
        )
        result = step(plan.place_state(initial), plan.place_batch(batch),
                      jax.device_put(jax.random.key(104), plan.replicated_sharding))
        observed = float(result.metrics.loss)
        expected_local = float(np.mean(local_losses))
        reports.append({
            "devices": devices, "configured_scope": "local", "observed_loss": observed,
            "global_loss": global_loss, "expected_local_loss": expected_local,
            "matches_global": bool(np.isclose(observed, global_loss, rtol=1e-5, atol=1e-6)),
            "matches_local": bool(np.isclose(observed, expected_local, rtol=1e-5, atol=1e-6)),
        })
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
