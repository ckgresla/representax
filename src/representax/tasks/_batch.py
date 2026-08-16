"""Private validation shared by model-native task payloads."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax


def payload_row_count(payload: Any, *, name: str) -> int:
    """Return a payload's common leading dimension or fail at the host boundary."""

    leaves = [value for value in jax.tree.leaves(payload) if eqx.is_array(value)]
    if not leaves:
        raise ValueError(f"{name} payload must contain arrays")
    if any(value.ndim == 0 for value in leaves):
        raise ValueError(f"{name} payload arrays must have a batch dimension")
    row_count = leaves[0].shape[0]
    if any(value.shape[0] != row_count for value in leaves):
        raise ValueError(f"{name} payload arrays must have the same row count")
    return row_count


__all__ = ["payload_row_count"]
