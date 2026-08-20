"""Named training-state and batch sharding configurations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from representax.core import Task
from representax.core.sharding import activation_sharding
from representax.models.materialization import (
    FSDPMaterializer,
    GlobalFSDPMaterializer,
)

from .execution import ExecutionContext, LossExecution
from .state import StepResult, TrainState
from .step import TrainStep, _build_train_step_body


def _named_shardings(mesh: Mesh, specs: Any) -> Any:
    return jax.tree.map(
        lambda spec: NamedSharding(mesh, spec),
        specs,
        is_leaf=lambda value: isinstance(value, P),
    )


def _axis_names(axis: Any) -> tuple[str, ...]:
    if axis is None:
        return ()
    if isinstance(axis, tuple):
        return tuple(str(name) for name in axis)
    return (str(axis),)


def _spec_axis_names(spec: P) -> tuple[str, ...]:
    return tuple(name for axis in tuple(spec) for name in _axis_names(axis))


def fsdp_partition_spec(
    shape: tuple[int, ...],
    *,
    axis_name: str,
    axis_size: int,
    minimum_elements: int = 1024,
) -> P:
    """Shard the largest divisible dimension of one substantial parameter."""

    if axis_size <= 1 or np.prod(shape, dtype=np.int64) < minimum_elements:
        return P()
    candidates = sorted(range(len(shape)), key=shape.__getitem__, reverse=True)
    selected = next(
        (dimension for dimension in candidates if shape[dimension] % axis_size == 0),
        None,
    )
    if selected is None:
        return P()
    axes: list[str | None] = [None] * len(shape)
    axes[selected] = axis_name
    return P(*axes)


def parameter_specs_from_rules(
    model: eqx.Module,
    rules: Sequence[tuple[str, P]],
    *,
    default: P | None = None,
) -> eqx.Module:
    """Resolve ordered path regular expressions into a model-shaped spec tree."""

    compiled = tuple((re.compile(pattern), spec) for pattern, spec in rules)
    default = P() if default is None else default

    def resolve(path: tuple[Any, ...], value: Any) -> P:
        if not eqx.is_inexact_array(value):
            return P()
        name = jax.tree_util.keystr(path)
        return next(
            (spec for pattern, spec in compiled if pattern.search(name)),
            default,
        )

    return cast(eqx.Module, jax.tree_util.tree_map_with_path(resolve, model))


def _gradient_specs(model: eqx.Module, parameter_specs: eqx.Module) -> eqx.Module:
    """Mirror Equinox's trainable-parameter filter in a layout tree."""

    return cast(
        eqx.Module,
        jax.tree.map(
            lambda value, spec: spec if eqx.is_inexact_array(value) else None,
            model,
            parameter_specs,
            is_leaf=lambda value: isinstance(value, P),
        ),
    )


@dataclass(frozen=True)
class ShardingPlan:
    """Resolved layouts for one compiled global training program.

    The plan is model- and task-neutral. ``parameter_specs`` selects the physical
    at-rest layout; batch, optimizer-state, and result layouts follow from the
    same declaration. Full-model execution coalesces compatible parameter shards
    and annotates the packed materialization boundary so JAX derives compact
    gather and gradient-transpose communication. The optional layer boundary
    retains an explicit model hook and collectives solely to shorten the gathered
    parameter live range. A separate optional data axis supports hybrid
    data/FSDP execution.
    """

    mesh: Mesh
    strategy: str
    parameter_specs: eqx.Module
    gradient_specs: eqx.Module
    optimizer_specs: optax.OptState
    data_axis_name: str | None
    parameter_axis_names: tuple[str, ...]
    materialization_boundary: Literal["model", "layer"] = "model"
    materialization_bucket_bytes: int = 256 * 2**20
    rematerialize_gathers: bool = True
    gradient_bucket_bytes: int = 256 * 2**20

    def __post_init__(self) -> None:
        mesh_axes = set(self.mesh.axis_names)
        if self.data_axis_name is not None and self.data_axis_name not in mesh_axes:
            raise ValueError(
                f"data axis {self.data_axis_name!r} is absent from mesh axes "
                f"{self.mesh.axis_names!r}"
            )
        if not self.parameter_axis_names:
            raise ValueError("a sharding plan requires at least one parameter axis")
        unknown = set(self.parameter_axis_names) - mesh_axes
        if unknown:
            raise ValueError(
                f"parameter axes are absent from the mesh: {sorted(unknown)}"
            )
        if len(set(self.parameter_axis_names)) != len(self.parameter_axis_names):
            raise ValueError("parameter axis names must be unique")
        if self.gradient_bucket_bytes <= 0:
            raise ValueError("gradient_bucket_bytes must be positive")
        if self.materialization_boundary not in {"model", "layer"}:
            raise ValueError(
                "materialization_boundary must be either 'model' or 'layer'"
            )
        if self.materialization_bucket_bytes <= 0:
            raise ValueError("materialization_bucket_bytes must be positive")

    @classmethod
    def fsdp(
        cls,
        state: TrainState,
        optimizer: optax.GradientTransformationExtraArgs,
        mesh: Mesh,
        *,
        parameter_axis_name: str = "fsdp",
        data_axis_name: str | None = None,
        minimum_parameter_elements: int = 1024,
        parameter_specs: eqx.Module | None = None,
        materialization_boundary: Literal["model", "layer"] = "model",
        materialization_bucket_bytes: int = 256 * 2**20,
        rematerialize_gathers: bool = True,
        gradient_bucket_bytes: int = 256 * 2**20,
    ) -> ShardingPlan:
        """Resolve the named fully-sharded data parallelism preset."""

        if minimum_parameter_elements <= 0:
            raise ValueError("minimum_parameter_elements must be positive")
        if parameter_axis_name not in mesh.axis_names:
            raise ValueError(
                f"parameter axis {parameter_axis_name!r} is absent from mesh axes "
                f"{mesh.axis_names!r}"
            )
        axis_size = int(mesh.shape[parameter_axis_name])
        if axis_size <= 1:
            raise ValueError("FSDP parameter axis must contain more than one device")
        if parameter_specs is None:
            parameter_specs = jax.tree.map(
                lambda value: (
                    fsdp_partition_spec(
                        tuple(value.shape),
                        axis_name=parameter_axis_name,
                        axis_size=axis_size,
                        minimum_elements=minimum_parameter_elements,
                    )
                    if eqx.is_inexact_array(value)
                    else P()
                ),
                state.model,
            )
        cls._validate_parameter_specs(
            state.model,
            parameter_specs,
            mesh=mesh,
            data_axis_name=data_axis_name,
            parameter_axis_names=(parameter_axis_name,),
        )
        cls._validate_materialization_boundary(
            state.model,
            parameter_specs,
            materialization_boundary,
        )
        gradient_specs = _gradient_specs(state.model, parameter_specs)
        optimizer_specs = optax.tree.map_params(
            optimizer,
            lambda _value, spec: spec,
            state.optimizer_state,
            gradient_specs,
            transform_non_params=lambda _value: P(),
        )
        return cls(
            mesh=mesh,
            strategy="fsdp",
            parameter_specs=parameter_specs,
            gradient_specs=gradient_specs,
            optimizer_specs=optimizer_specs,
            data_axis_name=data_axis_name,
            parameter_axis_names=(parameter_axis_name,),
            materialization_boundary=materialization_boundary,
            materialization_bucket_bytes=materialization_bucket_bytes,
            rematerialize_gathers=rematerialize_gathers,
            gradient_bucket_bytes=gradient_bucket_bytes,
        )

    @classmethod
    def custom(
        cls,
        state: TrainState,
        optimizer: optax.GradientTransformationExtraArgs,
        mesh: Mesh,
        parameter_specs: eqx.Module,
        *,
        parameter_axis_names: tuple[str, ...],
        data_axis_name: str | None,
        materialization_boundary: Literal["model", "layer"] = "model",
        materialization_bucket_bytes: int = 256 * 2**20,
        rematerialize_gathers: bool = True,
        gradient_bucket_bytes: int = 256 * 2**20,
        strategy: str = "custom",
    ) -> ShardingPlan:
        """Build a plan from an exact user-supplied model-shaped layout tree."""

        cls._validate_parameter_specs(
            state.model,
            parameter_specs,
            mesh=mesh,
            data_axis_name=data_axis_name,
            parameter_axis_names=parameter_axis_names,
        )
        cls._validate_materialization_boundary(
            state.model,
            parameter_specs,
            materialization_boundary,
        )
        gradient_specs = _gradient_specs(state.model, parameter_specs)
        optimizer_specs = optax.tree.map_params(
            optimizer,
            lambda _value, spec: spec,
            state.optimizer_state,
            gradient_specs,
            transform_non_params=lambda _value: P(),
        )
        return cls(
            mesh=mesh,
            strategy=strategy,
            parameter_specs=parameter_specs,
            gradient_specs=gradient_specs,
            optimizer_specs=optimizer_specs,
            data_axis_name=data_axis_name,
            parameter_axis_names=parameter_axis_names,
            materialization_boundary=materialization_boundary,
            materialization_bucket_bytes=materialization_bucket_bytes,
            rematerialize_gathers=rematerialize_gathers,
            gradient_bucket_bytes=gradient_bucket_bytes,
        )

    @classmethod
    def ddp(
        cls,
        state: TrainState,
        optimizer: optax.GradientTransformationExtraArgs,
        mesh: Mesh,
        *,
        axis_name: str = "data",
    ) -> ShardingPlan:
        """Resolve the named replicated-state data parallelism preset."""

        parameter_specs = cast(
            eqx.Module,
            jax.tree.map(lambda _value: P(), state.model),
        )
        return cls.custom(
            state,
            optimizer,
            mesh,
            parameter_specs,
            parameter_axis_names=(axis_name,),
            data_axis_name=axis_name,
            materialization_boundary="model",
            materialization_bucket_bytes=256 * 2**20,
            rematerialize_gathers=False,
            gradient_bucket_bytes=256 * 2**20,
            strategy="ddp",
        )

    @staticmethod
    def _validate_parameter_specs(
        model: eqx.Module,
        specs: eqx.Module,
        *,
        mesh: Mesh,
        data_axis_name: str | None,
        parameter_axis_names: tuple[str, ...],
    ) -> None:
        if jax.tree.structure(model) != jax.tree.structure(specs):
            raise ValueError("parameter specs must match the model PyTree")
        allowed = set(parameter_axis_names)
        for path, value, spec in zip(
            (path for path, _ in jax.tree.flatten_with_path(model)[0]),
            jax.tree.leaves(model),
            jax.tree.leaves(specs, is_leaf=lambda item: isinstance(item, P)),
            strict=True,
        ):
            if not isinstance(spec, P):
                raise TypeError(
                    "parameter spec "
                    f"{jax.tree_util.keystr(path)} is not a PartitionSpec"
                )
            if len(spec) > value.ndim:
                raise ValueError(
                    f"parameter spec {jax.tree_util.keystr(path)} has rank {len(spec)} "
                    f"for a rank-{value.ndim} array"
                )
            names = _spec_axis_names(spec)
            unknown = set(names) - allowed
            if unknown:
                raise ValueError(
                    f"parameter spec {jax.tree_util.keystr(path)} uses unsupported "
                    f"materialization axes {sorted(unknown)}"
                )
            for dimension, axis in enumerate(tuple(spec)):
                axis_names = _axis_names(axis)
                partitions = int(np.prod([mesh.shape[name] for name in axis_names]))
                if partitions > 1 and value.shape[dimension] % partitions:
                    raise ValueError(
                        f"parameter {jax.tree_util.keystr(path)} dimension {dimension} "
                        f"of size {value.shape[dimension]} is not divisible by "
                        f"partition count {partitions}"
                    )

    @staticmethod
    def _validate_materialization_boundary(
        model: eqx.Module,
        specs: eqx.Module,
        boundary: Literal["model", "layer"],
    ) -> None:
        if boundary != "layer":
            return
        has_parameter_shards = any(
            _spec_axis_names(spec)
            for spec in jax.tree.leaves(
                specs,
                is_leaf=lambda value: isinstance(value, P),
            )
        )
        if has_parameter_shards and not callable(
            getattr(model, "fsdp_materialize", None)
        ):
            raise NotImplementedError(
                f"{type(model).__name__} does not implement layer-boundary FSDP; "
                "name its layer stack in fsdp_materialize() or configure "
                "materialization_boundary='model'"
            )

    @property
    def state_specs(self) -> TrainState:
        return TrainState(
            model=self.parameter_specs,
            optimizer_state=self.optimizer_specs,
            step=cast(Any, P()),
        )

    @property
    def state_shardings(self) -> TrainState:
        return _named_shardings(self.mesh, self.state_specs)

    @property
    def replicated_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, P())

    @property
    def batch_spec(self) -> P:
        return P() if self.data_axis_name is None else P(self.data_axis_name)

    @property
    def batch_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, self.batch_spec)

    def place_state(self, state: TrainState) -> TrainState:
        """Place parameters and Optax state directly in their at-rest layouts."""

        def place(value: Any, sharding: NamedSharding) -> Any:
            if not eqx.is_array(value):
                return value
            if sharding.is_fully_addressable:
                return jax.device_put(value, sharding)
            if tuple(sharding.spec):
                raise NotImplementedError(
                    "multi-host placement of sharded parameters requires "
                    "process-local checkpoint restoration"
                )
            return jax.make_array_from_process_local_data(
                sharding,
                value,
                global_shape=value.shape,
            )

        return jax.tree.map(place, state, self.state_shardings)

    def place_replicated(self, tree: Any) -> Any:
        """Place an array PyTree as replicas, including from every host."""

        def place(value: Any) -> Any:
            if not eqx.is_array(value):
                return value
            if self.replicated_sharding.is_fully_addressable:
                return jax.device_put(value, self.replicated_sharding)
            return jax.make_array_from_process_local_data(
                self.replicated_sharding,
                value,
                global_shape=value.shape,
            )

        return jax.tree.map(place, tree, is_leaf=lambda value: value is None)

    def place_batch(self, batch: Any) -> Any:
        """Shard example rows on the data axis and replicate over FSDP axes."""

        return jax.tree.map(
            lambda value: (
                jax.device_put(value, self.batch_sharding)
                if eqx.is_array(value)
                else value
            ),
            batch,
            is_leaf=lambda value: value is None,
        )

    def materialize_layer_model(self, model: eqx.Module) -> eqx.Module:
        """Materialize parameters through a model's explicit layer boundary."""

        if self.materialization_boundary != "layer":  # pragma: no cover
            raise AssertionError("explicit materialization is reserved for layer FSDP")

        materializer = FSDPMaterializer(
            axis_sizes=tuple(
                (str(name), int(self.mesh.shape[name]))
                for name in self.parameter_axis_names
            ),
            data_axis_name=self.data_axis_name,
            bucket_bytes=self.materialization_bucket_bytes,
        )
        bounded_materialize = getattr(model, "fsdp_materialize", None)
        if not callable(bounded_materialize):  # pragma: no cover - plan invariant
            raise AssertionError("layer materialization capability disappeared")
        return cast(
            eqx.Module,
            bounded_materialize(self.parameter_specs, materializer),
        )

    @property
    def has_sharded_parameters(self) -> bool:
        """Whether any trainable model leaf has non-replicated at-rest storage."""

        return any(
            _spec_axis_names(spec)
            for spec in jax.tree.leaves(
                self.parameter_specs,
                is_leaf=lambda value: isinstance(value, P),
            )
        )

    @property
    def requires_internal_annotations(self) -> bool:
        """Whether explicit inference or parameter shards need layout guidance."""

        return self.has_sharded_parameters or any(
            axis_type is AxisType.Explicit for axis_type in self.mesh.axis_types
        )

    def global_norm(self, tree: Any) -> jax.Array:
        """Compute one FP32 norm over local shards without materializing them."""

        total = jnp.asarray(0.0, dtype=jnp.float32)
        for value, spec in zip(
            jax.tree.leaves(tree),
            jax.tree.leaves(
                self.gradient_specs,
                is_leaf=lambda item: isinstance(item, P),
            ),
            strict=True,
        ):
            if not eqx.is_inexact_array(value):
                continue
            squared = jnp.sum(jnp.square(value.astype(jnp.float32)))
            used_axes = _spec_axis_names(spec)
            if used_axes:
                squared = jax.lax.psum(squared, used_axes)
            total = total + squared
        return jnp.sqrt(total)

    def synchronize_gradients(self, gradients: Any) -> Any:
        """Average dtype-compatible gradient buckets over replicated mesh axes."""

        mesh_axes = tuple(
            name for name in self.mesh.axis_names if int(self.mesh.shape[name]) > 1
        )
        gradient_leaves, structure = jax.tree.flatten(
            gradients,
            is_leaf=lambda value: value is None,
        )
        spec_leaves, spec_structure = jax.tree.flatten(
            self.gradient_specs,
            is_leaf=lambda value: value is None or isinstance(value, P),
        )
        if structure != spec_structure:  # pragma: no cover - construction invariant
            raise AssertionError("gradient and sharding structures diverged")

        grouped: dict[tuple[tuple[str, ...], np.dtype[Any]], list[list[int]]] = {}
        grouped_bytes: dict[
            tuple[tuple[str, ...], np.dtype[Any]],
            list[int],
        ] = {}
        synchronized = list(gradient_leaves)
        for index, (gradient, spec) in enumerate(
            zip(gradient_leaves, spec_leaves, strict=True)
        ):
            if gradient is None or not eqx.is_inexact_array(gradient):
                continue
            sharded_axes = set(_spec_axis_names(spec))
            replicated_axes = tuple(
                name for name in mesh_axes if name not in sharded_axes
            )
            if not replicated_axes:
                continue
            key = (replicated_axes, np.dtype(gradient.dtype))
            size_bytes = int(gradient.size * np.dtype(gradient.dtype).itemsize)
            buckets = grouped.setdefault(key, [])
            bucket_sizes = grouped_bytes.setdefault(key, [])
            if (
                not buckets
                or bucket_sizes[-1] + size_bytes > self.gradient_bucket_bytes
            ):
                buckets.append([])
                bucket_sizes.append(0)
            buckets[-1].append(index)
            bucket_sizes[-1] += size_bytes

        for (replicated_axes, _dtype), buckets in grouped.items():
            divisor = float(
                np.prod([self.mesh.shape[name] for name in replicated_axes])
            )
            for indices in buckets:
                values = [gradient_leaves[index] for index in indices]
                if len(values) == 1:
                    collective_input = jax.lax.optimization_barrier(values[0])
                    synchronized[indices[0]] = (
                        jax.lax.psum(collective_input, replicated_axes) / divisor
                    )
                    continue
                sizes = [value.size for value in values]
                packed = jnp.concatenate([value.reshape(-1) for value in values])
                packed = jax.lax.optimization_barrier(packed)
                packed = jax.lax.psum(packed, replicated_axes) / divisor
                offsets = np.cumsum(sizes[:-1], dtype=np.int64).tolist()
                parts = jnp.split(packed, offsets)
                for index, value, part in zip(indices, values, parts, strict=True):
                    synchronized[index] = part.reshape(value.shape)

        return jax.tree.unflatten(structure, synchronized)

    def all_finite(self, *trees: Any) -> jax.Array:
        """Require finite loss, metrics, and every local parameter-gradient shard."""

        checks = [
            jnp.all(jnp.isfinite(leaf.astype(jnp.float32)))
            for tree in trees
            for leaf in jax.tree.leaves(tree)
            if eqx.is_inexact_array(leaf)
        ]
        finite = jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)
        collective_axes = tuple(
            name for name in self.mesh.axis_names if int(self.mesh.shape[name]) > 1
        )
        if collective_axes:
            finite = jax.lax.pmin(finite.astype(jnp.int32), collective_axes).astype(
                jnp.bool_
            )
        return finite

    def replicate_metrics(self, metrics: Any) -> Any:
        """Make scalar reports explicitly invariant over every mesh axis."""

        collective_axes = tuple(
            name for name in self.mesh.axis_names if int(self.mesh.shape[name]) > 1
        )
        if not collective_axes:
            return metrics

        def replicate(value: jax.Array) -> jax.Array:
            if jnp.issubdtype(value.dtype, jnp.bool_):
                replicated = value.astype(jnp.int32)
                for axis_name in collective_axes:
                    replicated = jax.lax.pmin(replicated, axis_name)
                return replicated.astype(jnp.bool_)
            replicated = value
            for axis_name in collective_axes:
                replicated = jax.lax.pmean(replicated, axis_name)
            return replicated

        return jax.tree.map(replicate, metrics)


def build_sharded_train_step(
    task: Task[Any],
    optimizer: optax.GradientTransformationExtraArgs,
    plan: ShardingPlan,
    *,
    max_grad_norm: float | None = 1.0,
    execution: LossExecution,
    donate_state: bool = False,
) -> TrainStep:
    """Compile one ordinary task update from a resolved sharding plan.

    Pure FSDP replicates the scientific batch over the parameter mesh. A data
    axis shards rows for any execution strategy that supplies the task's exact
    distributed loss semantics; the sharding plan remains task-neutral.
    """

    if plan.has_sharded_parameters and plan.materialization_boundary == "layer":
        return _build_layer_sharded_train_step(
            task,
            optimizer,
            plan,
            max_grad_norm=max_grad_norm,
            execution=execution,
            donate_state=donate_state,
        )

    global_materializer = GlobalFSDPMaterializer(
        mesh=plan.mesh,
        bucket_bytes=plan.materialization_bucket_bytes,
    )

    def materialize_model(model: eqx.Module) -> eqx.Module:
        return cast(eqx.Module, global_materializer.tree(model, plan.parameter_specs))

    unannotated_train_step = _build_train_step_body(
        task,
        optimizer,
        max_grad_norm=max_grad_norm,
        execution=execution,
        context=ExecutionContext(),
        materialize_model=materialize_model if plan.has_sharded_parameters else None,
    )

    def train_step_body(
        state: TrainState,
        batch: Any,
        key: jax.Array | None,
    ) -> StepResult:
        if not plan.requires_internal_annotations:
            return unannotated_train_step(state, batch, key)
        with activation_sharding(plan.mesh, plan.data_axis_name):
            return unannotated_train_step(state, batch, key)

    donate_argnums = (0,) if donate_state else ()
    return jax.jit(
        train_step_body,
        in_shardings=(
            plan.state_shardings,
            plan.batch_sharding,
            plan.replicated_sharding,
        ),
        out_shardings=StepResult(
            state=plan.state_shardings,
            metrics=cast(Any, plan.replicated_sharding),
        ),
        donate_argnums=donate_argnums,
    )


def _build_layer_sharded_train_step(
    task: Task[Any],
    optimizer: optax.GradientTransformationExtraArgs,
    plan: ShardingPlan,
    *,
    max_grad_norm: float | None,
    execution: LossExecution,
    donate_state: bool,
) -> TrainStep:
    """Compile opt-in layer-bounded FSDP with explicit gather transposition."""

    materialize_model = plan.materialize_layer_model
    if plan.rematerialize_gathers:
        materialize_model = jax.checkpoint(
            materialize_model,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
    train_step_body = _build_train_step_body(
        task,
        optimizer,
        max_grad_norm=max_grad_norm,
        execution=execution,
        context=ExecutionContext(data_axis_name=plan.data_axis_name),
        materialize_model=materialize_model,
        synchronize_gradients=plan.synchronize_gradients,
        norm_fn=plan.global_norm,
        finite_fn=plan.all_finite,
    )

    def mapped_train_step_body(
        state: TrainState,
        batch: Any,
        key: jax.Array | None,
    ) -> StepResult:
        result = train_step_body(state, batch, key)
        return StepResult(
            state=result.state,
            metrics=plan.replicate_metrics(result.metrics),
        )

    mapped_step = jax.shard_map(
        mapped_train_step_body,
        mesh=plan.mesh,
        in_specs=(plan.state_specs, plan.batch_spec, P()),
        out_specs=StepResult(state=plan.state_specs, metrics=cast(Any, P())),
        check_vma=True,
    )
    donate_argnums = (0,) if donate_state else ()
    return jax.jit(mapped_step, donate_argnums=donate_argnums)
