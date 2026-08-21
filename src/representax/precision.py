"""Explicit numerical precision boundaries for compiled representation learning."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

TreeT = TypeVar("TreeT")


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Resolved dtypes for stored state, forward compute, and objectives."""

    parameter_dtype: jnp.dtype
    compute_dtype: jnp.dtype
    activation_dtype: jnp.dtype
    accumulation_dtype: jnp.dtype
    loss_dtype: jnp.dtype

    @property
    def communication_dtype(self) -> jnp.dtype:
        """Dtype communicated when transient compute parameters are sharded."""

        return self.compute_dtype


FP32_POLICY = PrecisionPolicy(
    parameter_dtype=jnp.dtype(jnp.float32),
    compute_dtype=jnp.dtype(jnp.float32),
    activation_dtype=jnp.dtype(jnp.float32),
    accumulation_dtype=jnp.dtype(jnp.float32),
    loss_dtype=jnp.dtype(jnp.float32),
)

_DTYPES = {
    "float32": jnp.dtype(jnp.float32),
    "bfloat16": jnp.dtype(jnp.bfloat16),
}
_ACTIVE_PRECISION: ContextVar[PrecisionPolicy | None] = ContextVar(
    "representax_precision",
    default=None,
)


def resolve_precision_policy(config: Any) -> PrecisionPolicy:
    """Resolve a validated serializable config into JAX dtype objects."""

    return PrecisionPolicy(
        parameter_dtype=_DTYPES[config.parameter_dtype],
        compute_dtype=_DTYPES[config.compute_dtype],
        activation_dtype=_DTYPES[config.activation_dtype],
        accumulation_dtype=_DTYPES[config.accumulation_dtype],
        loss_dtype=_DTYPES[config.loss_dtype],
    )


@contextmanager
def precision_context(policy: PrecisionPolicy) -> Iterator[None]:
    """Make one static policy visible at model/objective call boundaries."""

    token = _ACTIVE_PRECISION.set(policy)
    try:
        yield
    finally:
        _ACTIVE_PRECISION.reset(token)


def cast_floating_tree(tree: TreeT, dtype: jnp.dtype) -> TreeT:
    """Cast only floating or complex array leaves in an arbitrary PyTree."""

    return jax.tree.map(
        lambda value: value.astype(dtype) if eqx.is_inexact_array(value) else value,
        tree,
    )


def prepare_master_model(model: TreeT, policy: PrecisionPolicy) -> TreeT:
    """Store trainable model arrays in the policy's master-parameter dtype."""

    return cast_floating_tree(model, policy.parameter_dtype)


def model_for_compute(model: TreeT, policy: PrecisionPolicy) -> TreeT:
    """Create the transient model view consumed by the forward program."""

    return cast_floating_tree(model, policy.compute_dtype)


def active_model_for_compute(model: TreeT) -> TreeT:
    """Create a compute view at the model-use site inside the transformed graph."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return model
    return model_for_compute(model, policy)


def compute_parameter(value: Float[Array, ...]) -> Float[Array, ...]:
    """Cast a parameter used outside the shared encoder entry point."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return value
    return value.astype(policy.compute_dtype)


def active_compute_dtype(fallback: Any) -> jnp.dtype:
    """Resolve a model's inference default against the active training policy."""

    policy = _ACTIVE_PRECISION.get()
    return jnp.dtype(fallback) if policy is None else policy.compute_dtype


def activation_inputs(inputs: Any) -> Any:
    """Cast model inputs without touching labels or other task-owned values."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return inputs
    return cast_floating_tree(inputs, policy.activation_dtype)


def objective_output(
    value: Float[Array, ...],
) -> Float[Array, ...]:
    """Cross from model activations into FP32-sensitive task computation."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return value
    return value.astype(policy.accumulation_dtype)


def loss_value(value: Float[Array, ""]) -> Float[Array, ""]:
    """Apply the explicit scalar-loss dtype at the optimizer boundary."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return value
    return value.astype(policy.loss_dtype)


def accumulated_values(tree: Any) -> Any:
    """Cast floating metric leaves to the configured reduction dtype."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return tree
    return cast_floating_tree(tree, policy.accumulation_dtype)


__all__ = [
    "FP32_POLICY",
    "PrecisionPolicy",
    "active_compute_dtype",
    "active_model_for_compute",
    "accumulated_values",
    "activation_inputs",
    "cast_floating_tree",
    "compute_parameter",
    "loss_value",
    "model_for_compute",
    "objective_output",
    "precision_context",
    "prepare_master_model",
    "resolve_precision_policy",
]
