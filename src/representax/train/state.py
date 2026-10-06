"""Durable training state and measured step outputs."""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import optax


class TrainState(eqx.Module):
    """Native Equinox model, Optax state, and absolute optimizer step."""

    model: eqx.Module
    optimizer_state: optax.OptState
    step: jax.Array


class StepMetrics(eqx.Module):
    """Numerical facts retained from one compiled optimizer update."""

    loss: jax.Array
    task: Mapping[str, jax.Array]
    gradient_global_norm: jax.Array
    clipped_gradient_global_norm: jax.Array
    update_global_norm: jax.Array
    numeric_finite: jax.Array
    skipped_update: jax.Array


class StepResult(eqx.Module):
    """Updated state and metrics returned by the compiled step."""

    state: TrainState
    metrics: StepMetrics
