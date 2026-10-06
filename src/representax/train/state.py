"""Durable training state and measured step outputs."""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import optax
from jaxtyping import Array, Bool, Float, Int


class TrainState(eqx.Module):
    """Native Equinox model, Optax state, and absolute optimizer step."""

    model: eqx.Module
    optimizer_state: optax.OptState
    step: Int[Array, ""]


class StepMetrics(eqx.Module):
    """Numerical facts retained from one compiled optimizer update."""

    loss: Float[Array, ""]
    task: Mapping[str, Array]
    gradient_global_norm: Float[Array, ""]
    clipped_gradient_global_norm: Float[Array, ""]
    update_global_norm: Float[Array, ""]
    numeric_finite: Bool[Array, ""]
    skipped_update: Bool[Array, ""]


class StepResult(eqx.Module):
    """Updated state and metrics returned by the compiled step."""

    state: TrainState
    metrics: StepMetrics
