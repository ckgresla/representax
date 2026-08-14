"""Compile one complete Representax update in an installed environment."""

from __future__ import annotations

import importlib.metadata
import json

import jax
import jax.numpy as jnp
import optax

from representax.models import DenseEncoder
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import build_train_step, make_train_state


def main() -> None:
    model = DenseEncoder(4, 3, key=jax.random.key(0))
    optimizer = optax.adamw(learning_rate=1e-3)
    state = make_train_state(model, optimizer)
    batch = retrieval_batch(
        query=jnp.eye(4, dtype=jnp.float32),
        document=jnp.roll(jnp.eye(4, dtype=jnp.float32), 1, axis=0),
        positive_mask=jnp.eye(4, dtype=jnp.bool_),
    )
    step = build_train_step(MNRTask(scale=5.0, symmetric=True), optimizer)
    result = step(state, batch, jax.random.key(1))
    jax.block_until_ready((result.metrics.loss, result.state.step))
    assert bool(result.metrics.numeric_finite)
    assert not bool(result.metrics.skipped_update)
    assert int(result.state.step) == 1
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "device_count": jax.device_count(),
                "jax": jax.__version__,
                "loss": float(result.metrics.loss),
                "representax": importlib.metadata.version("representax"),
                "status": "passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
