"""Private validation shared by model-native task payloads."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core


def payload_row_count(payload: Any, *, name: str) -> int:
    """Return the logical example count for a model-native payload.

    Packed multimodal payloads may contain patch-, frame-, or chunk-major arrays
    whose leading dimensions intentionally differ from the example dimension.
    Such payloads expose an explicit ``batch_size`` property; ordinary payloads
    retain the stricter common-leading-dimension inference.
    """

    explicit = getattr(payload, "batch_size", None)
    if explicit is not None:
        if not isinstance(explicit, int) or explicit <= 0:
            raise ValueError(f"{name} payload batch_size must be a positive integer")
        return explicit

    leaves = [value for value in jax.tree.leaves(payload) if eqx.is_array(value)]
    if not leaves:
        raise ValueError(f"{name} payload must contain arrays")
    if any(value.ndim == 0 for value in leaves):
        raise ValueError(f"{name} payload arrays must have a batch dimension")
    row_count = leaves[0].shape[0]
    if any(value.shape[0] != row_count for value in leaves):
        raise ValueError(f"{name} payload arrays must have the same row count")
    return row_count


def asarray(value: Any, *, dtype: Any | None = None) -> Any:
    """Preserve JAX callers while keeping ordinary collation on the host."""

    if isinstance(value, (jax.Array, jax_core.Tracer)):
        return jnp.asarray(value, dtype=dtype)
    return np.asarray(value, dtype=dtype)


def ones(shape: tuple[int, ...], *, dtype: Any, like: Any) -> Any:
    """Construct defaults in the same host/JAX domain as an existing array."""

    if isinstance(like, (jax.Array, jax_core.Tracer)):
        return jnp.ones(shape, dtype=dtype)
    return np.ones(shape, dtype=dtype)


__all__ = ["asarray", "ones", "payload_row_count"]
