"""One generic compiled optimization boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from representax.core import Task, evaluate_loss

from .state import StepMetrics, StepResult, TrainState


def tree_global_norm(tree: Any) -> jax.Array:
    """Compute an FP32 L2 norm across every inexact array leaf."""

    squares = [
        jnp.sum(jnp.square(leaf.astype(jnp.float32)))
        for leaf in jax.tree.leaves(tree)
        if eqx.is_inexact_array(leaf)
    ]
    if not squares:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.stack(squares), dtype=jnp.float32))


def tree_all_finite(*trees: Any) -> jax.Array:
    """Return one scalar finite check over all inexact leaves."""

    checks = [
        jnp.all(jnp.isfinite(leaf.astype(jnp.float32)))
        for tree in trees
        for leaf in jax.tree.leaves(tree)
        if eqx.is_inexact_array(leaf)
    ]
    if not checks:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(checks))


def _scale_arrays(tree: Any, scale: jax.Array) -> Any:
    return jax.tree.map(
        lambda leaf: leaf * scale if eqx.is_inexact_array(leaf) else leaf,
        tree,
        is_leaf=lambda value: value is None,
    )


def _select_arrays(condition: jax.Array, proposed: Any, previous: Any) -> Any:
    return jax.tree.map(
        lambda new, old: (
            jnp.where(condition, new, old)
            if eqx.is_array(new) and eqx.is_array(old)
            else new
        ),
        proposed,
        previous,
        is_leaf=lambda value: value is None,
    )


def make_train_state(
    model: eqx.Module,
    optimizer: optax.GradientTransformationExtraArgs,
) -> TrainState:
    """Initialize Optax against only trainable inexact model leaves."""

    parameters = eqx.filter(model, eqx.is_inexact_array)
    return TrainState(
        model=model,
        optimizer_state=optimizer.init(parameters),
        step=jnp.asarray(0, dtype=jnp.int32),
    )


TrainStep = Callable[[TrainState, Any, jax.Array | None], StepResult]


def build_train_step(
    task: Task[Any],
    optimizer: optax.GradientTransformationExtraArgs,
    *,
    max_grad_norm: float | None = 1.0,
) -> TrainStep:
    """Build a compiled task-generic optimizer update.

    The task and optimizer are closed-over static program structure. Model,
    optimizer state, batch, and random key remain explicit JAX inputs.
    """

    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or None")

    @eqx.filter_jit
    def train_step(
        state: TrainState,
        batch: Any,
        key: jax.Array | None = None,
    ) -> StepResult:
        def loss_fn(model: eqx.Module):
            output = evaluate_loss(task, model, batch, key=key)
            return output.loss, output.metrics

        (loss, task_metrics), gradients = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(state.model)
        gradient_norm = tree_global_norm(gradients)
        if max_grad_norm is None:
            coefficient = jnp.asarray(1.0, dtype=jnp.float32)
        else:
            coefficient = jnp.minimum(
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.asarray(max_grad_norm, dtype=jnp.float32)
                / (gradient_norm + jnp.asarray(1e-6, dtype=jnp.float32)),
            )
        clipped_gradients = _scale_arrays(gradients, coefficient)
        clipped_gradient_norm = tree_global_norm(clipped_gradients)
        parameters = eqx.filter(state.model, eqx.is_inexact_array)
        updates, proposed_optimizer_state = optimizer.update(
            clipped_gradients,
            state.optimizer_state,
            parameters,
        )
        proposed_model = eqx.apply_updates(state.model, updates)
        finite = tree_all_finite(
            loss,
            task_metrics,
            clipped_gradients,
            updates,
            proposed_model,
            proposed_optimizer_state,
        )
        model = _select_arrays(finite, proposed_model, state.model)
        optimizer_state = _select_arrays(
            finite, proposed_optimizer_state, state.optimizer_state
        )
        new_state = TrainState(
            model=model,
            optimizer_state=optimizer_state,
            step=state.step + finite.astype(jnp.int32),
        )
        return StepResult(
            state=new_state,
            metrics=StepMetrics(
                loss=loss,
                task=task_metrics,
                gradient_global_norm=gradient_norm,
                clipped_gradient_global_norm=clipped_gradient_norm,
                update_global_norm=tree_global_norm(updates),
                numeric_finite=finite,
                skipped_update=~finite,
            ),
        )

    return train_step
