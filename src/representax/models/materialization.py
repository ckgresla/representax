"""Model-side utilities for bounded FSDP parameter materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def _axis_names(axis: Any) -> tuple[str, ...]:
    if axis is None:
        return ()
    if isinstance(axis, tuple):
        return tuple(str(name) for name in axis)
    return (str(axis),)


def _materializing_all_gather(
    value: jax.Array,
    *,
    axis_name: str,
    array_axis: int,
    average_gradient: bool,
    axis_size: int,
) -> jax.Array:
    """Materialize one shard with topology-correct gradient aggregation."""

    if not average_gradient:
        return jax.lax.all_gather(
            value,
            axis_name,
            axis=array_axis,
            tiled=True,
        )

    @jax.custom_vjp
    def gather(local_value: jax.Array) -> jax.Array:
        return jax.lax.all_gather(
            local_value,
            axis_name,
            axis=array_axis,
            tiled=True,
        )

    def gather_fwd(local_value: jax.Array) -> tuple[jax.Array, None]:
        return (
            jax.lax.all_gather(
                local_value,
                axis_name,
                axis=array_axis,
                tiled=True,
            ),
            None,
        )

    def gather_bwd(_: None, cotangent: jax.Array) -> tuple[jax.Array]:
        gradient_shard = jax.lax.psum_scatter(
            cotangent,
            axis_name,
            scatter_dimension=array_axis,
            tiled=True,
        )
        return (gradient_shard / axis_size,)

    gather.defvjp(gather_fwd, gather_bwd)
    return gather(value)


@dataclass(frozen=True)
class FSDPMaterializer:
    """Materialize full parameters while retaining sharded storage and gradients."""

    axis_sizes: tuple[tuple[str, int], ...]
    data_axis_name: str | None
    bucket_bytes: int = 256 * 2**20

    def __post_init__(self) -> None:
        if self.bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be positive")

    @property
    def _axis_size_map(self) -> dict[str, int]:
        return dict(self.axis_sizes)

    def parameter(self, value: jax.Array, spec: P) -> jax.Array:
        """AllGather one parameter according to its at-rest partition spec."""

        gathered = value
        axis_sizes = self._axis_size_map
        for array_axis, axis in enumerate(tuple(spec)):
            for axis_name in _axis_names(axis):
                gathered = _materializing_all_gather(
                    gathered,
                    axis_name=axis_name,
                    array_axis=array_axis,
                    average_gradient=axis_name != self.data_axis_name,
                    axis_size=axis_sizes[axis_name],
                )
        return gathered

    def tree(self, value: Any, specs: Any) -> Any:
        """Materialize a subtree with bounded dtype/mesh-axis gather buckets."""

        parameters, structure = jax.tree.flatten(
            value,
            is_leaf=lambda item: item is None,
        )
        parameter_specs = jax.tree.leaves(
            specs,
            is_leaf=lambda item: item is None or isinstance(item, P),
        )
        if len(parameters) != len(parameter_specs):  # pragma: no cover
            raise AssertionError("parameter and partition-spec trees must match")

        materialized = list(parameters)
        grouped: dict[
            tuple[str, str, bool],
            list[tuple[int, jax.Array, int, int]],
        ] = {}
        for index, (parameter, spec) in enumerate(
            zip(parameters, parameter_specs, strict=True)
        ):
            if parameter is None:
                continue
            if spec is None:  # pragma: no cover - matching trees are required
                raise AssertionError("an array parameter requires a partition spec")
            sharded_axes = [
                (array_axis, axis_name)
                for array_axis, axis in enumerate(tuple(spec))
                for axis_name in _axis_names(axis)
            ]
            if not sharded_axes:
                continue
            if len(sharded_axes) != 1:
                materialized[index] = self.parameter(parameter, spec)
                continue
            array_axis, axis_name = sharded_axes[0]
            size_bytes = int(parameter.size * parameter.dtype.itemsize)
            key = (
                str(parameter.dtype),
                axis_name,
                axis_name != self.data_axis_name,
            )
            grouped.setdefault(key, []).append(
                (index, parameter, array_axis, size_bytes)
            )

        for (_dtype, axis_name, average_gradient), entries in grouped.items():
            buckets: list[list[tuple[int, jax.Array, int, int]]] = []
            bucket_sizes: list[int] = []
            for entry in entries:
                size_bytes = entry[-1]
                if not buckets or bucket_sizes[-1] + size_bytes > self.bucket_bytes:
                    buckets.append([])
                    bucket_sizes.append(0)
                buckets[-1].append(entry)
                bucket_sizes[-1] += size_bytes
            for bucket in buckets:
                moved = [
                    jnp.moveaxis(parameter, array_axis, 0)
                    for _, parameter, array_axis, _ in bucket
                ]
                flattened = [parameter.reshape(-1) for parameter in moved]
                offsets: list[tuple[int, int]] = []
                offset = 0
                for parameter in flattened:
                    next_offset = offset + parameter.size
                    offsets.append((offset, next_offset))
                    offset = next_offset
                packed = jnp.concatenate(flattened)
                axis_size = self._axis_size_map[axis_name]
                gathered = _materializing_all_gather(
                    packed,
                    axis_name=axis_name,
                    array_axis=0,
                    average_gradient=average_gradient,
                    axis_size=axis_size,
                ).reshape((axis_size, packed.size))
                for entry, local_parameter, (start, stop) in zip(
                    bucket,
                    moved,
                    offsets,
                    strict=True,
                ):
                    index, _parameter, array_axis, _size_bytes = entry
                    global_parameter = gathered[:, start:stop].reshape(
                        (
                            axis_size * local_parameter.shape[0],
                            *local_parameter.shape[1:],
                        )
                    )
                    materialized[index] = jnp.moveaxis(
                        global_parameter,
                        0,
                        array_axis,
                    )

        return jax.tree.unflatten(structure, materialized)

    def scanned(self, value: eqx.Module, specs: eqx.Module) -> DeferredFSDPModule:
        """Defer a depth-major module stack until each scan iteration."""

        def layer_spec(spec: P) -> P:
            axes = tuple(spec)
            if axes and _axis_names(axes[0]):
                raise ValueError(
                    "FSDP scanned modules cannot shard their leading depth axis"
                )
            return P(*axes[1:]) if axes else P()

        per_layer_specs = jax.tree.map(
            layer_spec,
            specs,
            is_leaf=lambda item: isinstance(item, P),
        )
        return DeferredFSDPModule(
            value=value,
            specs=cast(eqx.Module, per_layer_specs),
            axis_sizes=self.axis_sizes,
            data_axis_name=self.data_axis_name,
            bucket_bytes=self.bucket_bytes,
        )


class DeferredFSDPModule(eqx.Module):
    """A depth-major stack whose current scan slice is gathered on demand."""

    value: eqx.Module
    specs: eqx.Module = eqx.field(static=True)
    axis_sizes: tuple[tuple[str, int], ...] = eqx.field(static=True)
    data_axis_name: str | None = eqx.field(static=True)
    bucket_bytes: int = eqx.field(static=True)

    def materialize(self) -> eqx.Module:
        return cast(
            eqx.Module,
            FSDPMaterializer(
                axis_sizes=self.axis_sizes,
                data_axis_name=self.data_axis_name,
                bucket_bytes=self.bucket_bytes,
            ).tree(self.value, self.specs),
        )


def materialize_deferred(value: Any) -> Any:
    """Materialize a scan slice when it carries deferred FSDP parameters."""

    return value.materialize() if isinstance(value, DeferredFSDPModule) else value


__all__ = [
    "DeferredFSDPModule",
    "FSDPMaterializer",
    "materialize_deferred",
]
