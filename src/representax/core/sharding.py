"""Trace-local sharding annotations for global JAX programs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@dataclass(frozen=True, slots=True)
class _ActivationSharding:
    mesh: Mesh
    data_axis_name: str | None

    def sharding(self, rank: int) -> NamedSharding:
        spec = P() if rank <= 0 else P(self.data_axis_name, *((None,) * (rank - 1)))
        return NamedSharding(self.mesh, spec)

    @property
    def automatic(self) -> bool:
        return any(axis_type is AxisType.Auto for axis_type in self.mesh.axis_types)

    def scanned_sharding(self, rank: int) -> NamedSharding:
        if rank <= 1:
            spec = P()
        else:
            spec = P(None, self.data_axis_name, *((None,) * (rank - 2)))
        return NamedSharding(self.mesh, spec)

    @property
    def data_axis_size(self) -> int:
        if self.data_axis_name is None:
            return 1
        return int(self.mesh.shape[self.data_axis_name])


_ACTIVE: ContextVar[_ActivationSharding | None] = ContextVar(
    "representax_activation_sharding",
    default=None,
)


@contextmanager
def activation_sharding(
    mesh: Mesh,
    data_axis_name: str | None,
) -> Iterator[None]:
    """Annotate activations while a distributed step is being traced."""

    token = _ACTIVE.set(_ActivationSharding(mesh, data_axis_name))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def activation_out_sharding(rank: int) -> NamedSharding | None:
    """Return the active batch-leading layout, or no annotation locally."""

    policy = _ACTIVE.get()
    return None if policy is None or policy.automatic else policy.sharding(rank)


def scanned_out_sharding(rank: int) -> NamedSharding | None:
    """Return a scan-leading replicated, inner-batch sharded layout."""

    policy = _ACTIVE.get()
    return None if policy is None or policy.automatic else policy.scanned_sharding(rank)


def replicated_out_sharding() -> NamedSharding | None:
    """Return the active fully replicated layout."""

    policy = _ACTIVE.get()
    return (
        None if policy is None or policy.automatic else NamedSharding(policy.mesh, P())
    )


def active_data_axis_size() -> int:
    """Return the physical data-axis size active for the traced program."""

    policy = _ACTIVE.get()
    return 1 if policy is None else policy.data_axis_size


def constrain_activation(value: jax.Array) -> jax.Array:
    """Keep one activation in its batch-leading distributed layout."""

    policy = _ACTIVE.get()
    if policy is None:
        return value
    sharding = policy.sharding(value.ndim)
    if policy.automatic:
        return jax.lax.with_sharding_constraint(value, sharding)
    return jax.reshard(value, sharding)


def replicate(value: jax.Array) -> jax.Array:
    """Replicate a value at an explicit global-program boundary."""

    policy = _ACTIVE.get()
    if policy is None:
        return value
    sharding = NamedSharding(policy.mesh, P())
    if policy.automatic:
        return jax.lax.with_sharding_constraint(value, sharding)
    return jax.reshard(value, sharding)


def batch_to_scan(
    value: jax.Array,
    *,
    local_chunk_size: int,
    pad_mode: str = "constant",
    pad_value: int | float = 0,
) -> jax.Array:
    """Arrange a sharded global batch as replicated scan steps.

    ``local_chunk_size`` retains its per-device meaning. Each scan slice has a
    global batch of ``data_axis_size * local_chunk_size`` whose leading axis is
    sharded over the data mesh; the scan axis itself is replicated.
    """

    axis_size = active_data_axis_size()
    batch_size = value.shape[0]
    if batch_size % axis_size:
        raise ValueError("global batch size must be divisible by the data axis size")
    local_batch_size = batch_size // axis_size
    chunk_count = (local_batch_size + local_chunk_size - 1) // local_chunk_size
    padded_local_size = chunk_count * local_chunk_size
    local_padding = padded_local_size - local_batch_size

    device_major = jnp.reshape(
        value,
        (axis_size, local_batch_size, *value.shape[1:]),
        out_sharding=activation_out_sharding(value.ndim + 1),
    )
    if local_padding:
        widths = (
            (0, 0),
            (0, local_padding),
            *((0, 0),) * (value.ndim - 1),
        )
        if pad_mode == "constant":
            device_major = jnp.pad(
                device_major,
                widths,
                mode=pad_mode,
                constant_values=pad_value,
            )
        else:
            device_major = jnp.pad(device_major, widths, mode=pad_mode)
    device_chunks = jnp.reshape(
        device_major,
        (axis_size, chunk_count, local_chunk_size, *value.shape[1:]),
        out_sharding=activation_out_sharding(value.ndim + 2),
    )
    scan_major = jnp.transpose(
        device_chunks,
        (1, 0, 2, *range(3, device_chunks.ndim)),
    )
    chunks = jnp.reshape(
        scan_major,
        (chunk_count, axis_size * local_chunk_size, *value.shape[1:]),
        out_sharding=scanned_out_sharding(value.ndim + 1),
    )
    policy = _ACTIVE.get()
    if policy is not None and policy.automatic:
        chunks = jax.lax.with_sharding_constraint(
            chunks,
            policy.scanned_sharding(chunks.ndim),
        )
    return chunks


def scan_to_batch(
    value: jax.Array,
    *,
    batch_size: int,
    local_chunk_size: int,
) -> jax.Array:
    """Invert :func:`batch_to_scan`, discarding per-device padding."""

    axis_size = active_data_axis_size()
    local_batch_size = batch_size // axis_size
    chunk_count = value.shape[0]
    padded_local_size = chunk_count * local_chunk_size
    device_major = jnp.transpose(
        jnp.reshape(
            value,
            (chunk_count, axis_size, local_chunk_size, *value.shape[2:]),
            out_sharding=scanned_out_sharding(value.ndim + 1),
        ),
        (1, 0, 2, *range(3, value.ndim + 1)),
    )
    unpadded = jnp.reshape(
        device_major,
        (axis_size, padded_local_size, *value.shape[2:]),
        out_sharding=activation_out_sharding(value.ndim),
    )[:, :local_batch_size]
    result = jnp.reshape(
        unpadded,
        (batch_size, *value.shape[2:]),
        out_sharding=activation_out_sharding(value.ndim - 1),
    )
    return constrain_activation(result)


__all__ = [
    "activation_out_sharding",
    "activation_sharding",
    "active_data_axis_size",
    "batch_to_scan",
    "constrain_activation",
    "replicate",
    "replicated_out_sharding",
    "scan_to_batch",
    "scanned_out_sharding",
]
