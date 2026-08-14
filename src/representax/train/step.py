"""One generic compiled optimization boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from representax.core import Task

from .execution import Direct, LossExecution
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
    execution: LossExecution | None = None,
    donate_state: bool = False,
) -> TrainStep:
    """Build a compiled task-generic optimizer update.

    The task and optimizer are closed-over static program structure. Model,
    optimizer state, batch, and random key remain explicit JAX inputs. State
    donation is opt-in because callers may retain the old state for comparison,
    retry, or branching. Orbax asynchronous checkpointing is compatible with
    donation: its blocking device-to-host snapshot completes before save returns.
    """

    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or None")
    resolved_execution = Direct() if execution is None else execution
    resolved_execution.validate(task)
    donation = "all-except-first" if donate_state else "none"

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(
        model: eqx.Module,
        batch: Any,
        key: jax.Array | None,
    ) -> tuple[jax.Array, Any]:
        output = resolved_execution.evaluate(task, model, batch, key=key)
        return output.loss, output.metrics

    @eqx.filter_jit(donate=donation)
    def compiled_step(
        inputs: tuple[Any, jax.Array | None],
        state: TrainState,
    ) -> StepResult:
        batch, key = inputs
        (loss, task_metrics), gradients = loss_fn(state.model, batch, key)
        gradient_norm = tree_global_norm(gradients)
        if max_grad_norm is None:
            clipped_gradients = gradients
            clipped_gradient_norm = gradient_norm
        else:
            coefficient = jnp.minimum(
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.asarray(max_grad_norm, dtype=jnp.float32)
                / (gradient_norm + jnp.asarray(1e-6, dtype=jnp.float32)),
            )
            clipped_gradients = optax.tree.scale(coefficient, gradients)
            clipped_gradient_norm = gradient_norm * coefficient

        finite = tree_all_finite(
            loss,
            task_metrics,
            gradients,
            gradient_norm,
        )
        parameters = eqx.filter(state.model, eqx.is_inexact_array)
        updates, optimizer_state = optimizer.update(
            clipped_gradients,
            state.optimizer_state,
            parameters,
        )
        model = eqx.apply_updates(state.model, updates)
        proposed_state = TrainState(
            model=model,
            optimizer_state=optimizer_state,
            step=state.step + jnp.asarray(1, dtype=jnp.int32),
        )
        new_state = optax.tree.where(finite, proposed_state, state)
        update_norm = jnp.where(
            finite,
            tree_global_norm(updates),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return StepResult(
            state=new_state,
            metrics=StepMetrics(
                loss=loss,
                task=task_metrics,
                gradient_global_norm=gradient_norm,
                clipped_gradient_global_norm=clipped_gradient_norm,
                update_global_norm=update_norm,
                numeric_finite=finite,
                skipped_update=~finite,
            ),
        )

    def train_step(
        state: TrainState,
        batch: Any,
        key: jax.Array | None = None,
    ) -> StepResult:
        return compiled_step((batch, key), state)

    return train_step
