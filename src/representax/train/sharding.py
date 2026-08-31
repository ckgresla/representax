"""Named training-state and batch sharding configurations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import jax
import numpy as np
import optax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from representax.core import Task
from representax.core.sharding import activation_sharding
from representax.precision import FP32_POLICY, PrecisionPolicy

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

    # Replicate scalar/vector state. Sharding a normalization scale or bias on
    # the same mesh axis that carries the activation batch would give an
    # elementwise operation two unrelated uses of one mesh resource. Matrix and
    # embedding parameters instead expose a contraction/gather boundary where
    # XLA can derive ordinary FSDP communication.
    if (
        axis_size <= 1
        or len(shape) < 2
        or np.prod(shape, dtype=np.int64) < minimum_elements
    ):
        return P()
    selected = max(
        (dimension for dimension, size in enumerate(shape) if size % axis_size == 0),
        key=shape.__getitem__,
        default=None,
    )
    if selected is None:
        return P()
    axes: list[str | None] = [None] * len(shape)
    axes[selected] = axis_name
    return P(*axes)


def fsdp_parameter_specs(
    model: eqx.Module,
    mesh: Mesh,
    *,
    axis_name: str = "fsdp",
    minimum_elements: int = 1024,
) -> eqx.Module:
    """Resolve the named FSDP preset into a model-shaped layout tree."""

    if minimum_elements <= 0:
        raise ValueError("minimum_elements must be positive")
    if axis_name not in mesh.axis_names:
        raise ValueError(f"FSDP axis {axis_name!r} is absent from the mesh")
    axis_size = int(mesh.shape[axis_name])
    if axis_size <= 1:
        raise ValueError("FSDP parameter axis must contain more than one device")
    return cast(
        eqx.Module,
        jax.tree.map(
            lambda value: (
                fsdp_partition_spec(
                    tuple(value.shape),
                    axis_name=axis_name,
                    axis_size=axis_size,
                    minimum_elements=minimum_elements,
                )
                if eqx.is_array(value)
                else P()
            ),
            model,
        ),
    )


def place_model(model: eqx.Module, mesh: Mesh, specs: eqx.Module) -> eqx.Module:
    """Place a model directly into an explicit model-shaped sharding layout."""

    if jax.tree.structure(model) != jax.tree.structure(specs):
        raise ValueError("parameter specs must match the model PyTree")
    shardings = _named_shardings(mesh, specs)

    def place(value: Any, sharding: NamedSharding) -> Any:
        if not eqx.is_array(value):
            return value
        if sharding.is_fully_addressable:
            # Slice host-resident checkpoints before device transfer. Passing
            # one complete multi-GiB array to device_put lets its internal
            # _multi_slice executable stage every shard on one GPU before
            # distribution, defeating FSDP precisely when it is needed most.
            host_value = np.asarray(value)
            indices = sharding.addressable_devices_indices_map(host_value.shape)
            local_arrays = [
                jax.device_put(host_value[indices[device]], device)
                for device in sharding.addressable_devices
            ]
            return jax.make_array_from_single_device_arrays(
                host_value.shape,
                sharding,
                local_arrays,
            )
        if tuple(sharding.spec):
            raise NotImplementedError(
                "multi-host model placement requires process-local "
                "checkpoint restoration"
            )
        return jax.make_array_from_process_local_data(
            sharding,
            value,
            global_shape=value.shape,
        )

    return cast(eqx.Module, jax.tree.map(place, model, shardings))


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
        if not eqx.is_array(value):
            return P()
        name = jax.tree_util.keystr(path)
        return next(
            (spec for pattern, spec in compiled if pattern.search(name)),
            default,
        )

    return cast(eqx.Module, jax.tree_util.tree_map_with_path(resolve, model))


def _resolved_trainable_filter(model: eqx.Module, filter_spec: Any) -> Any:
    if callable(filter_spec):
        return jax.tree.map(filter_spec, model)
    if jax.tree.structure(model) != jax.tree.structure(filter_spec):
        raise ValueError("trainable filter must match the model PyTree")
    return filter_spec


def _gradient_specs(
    model: eqx.Module,
    parameter_specs: eqx.Module,
    trainable_filter: Any,
) -> eqx.Module:
    """Mirror the selected optimizer parameters in a layout tree."""

    selected = _resolved_trainable_filter(model, trainable_filter)

    return cast(
        eqx.Module,
        jax.tree.map(
            lambda spec, trainable: spec if trainable else None,
            parameter_specs,
            selected,
            is_leaf=lambda value: isinstance(value, P),
        ),
    )


@dataclass(frozen=True)
class ShardingPlan:
    """Resolved layouts for one compiled global training program.

    The plan is model- and task-neutral. ``parameter_specs`` selects the physical
    at-rest layout; batch, optimizer-state, and result layouts follow from the
    same declaration. Shared model primitives annotate their exact parameter-use
    and activation layouts so JAX derives communication and its autodiff
    transpose. A separate optional data axis supports hybrid data/FSDP execution.
    """

    mesh: Mesh
    strategy: str
    parameter_specs: eqx.Module
    optimizer_specs: optax.OptState
    data_axis_name: str | None
    parameter_axis_names: tuple[str, ...]

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
        trainable_filter: Any = eqx.is_inexact_array,
    ) -> ShardingPlan:
        """Resolve the named fully-sharded data parallelism preset."""

        if parameter_specs is None:
            parameter_specs = fsdp_parameter_specs(
                state.model,
                mesh,
                axis_name=parameter_axis_name,
                minimum_elements=minimum_parameter_elements,
            )
        cls._validate_parameter_specs(
            state.model,
            parameter_specs,
            mesh=mesh,
            data_axis_name=data_axis_name,
            parameter_axis_names=(parameter_axis_name,),
        )
        gradient_specs = _gradient_specs(
            state.model,
            parameter_specs,
            trainable_filter,
        )
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
            optimizer_specs=optimizer_specs,
            data_axis_name=data_axis_name,
            parameter_axis_names=(parameter_axis_name,),
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
        strategy: str = "custom",
        trainable_filter: Any = eqx.is_inexact_array,
    ) -> ShardingPlan:
        """Build a plan from an exact user-supplied model-shaped layout tree."""

        cls._validate_parameter_specs(
            state.model,
            parameter_specs,
            mesh=mesh,
            data_axis_name=data_axis_name,
            parameter_axis_names=parameter_axis_names,
        )
        gradient_specs = _gradient_specs(
            state.model,
            parameter_specs,
            trainable_filter,
        )
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
            optimizer_specs=optimizer_specs,
            data_axis_name=data_axis_name,
            parameter_axis_names=parameter_axis_names,
        )

    @classmethod
    def ddp(
        cls,
        state: TrainState,
        optimizer: optax.GradientTransformationExtraArgs,
        mesh: Mesh,
        *,
        axis_name: str = "data",
        trainable_filter: Any = eqx.is_inexact_array,
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
            strategy="ddp",
            trainable_filter=trainable_filter,
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
                    f"parameter axes {sorted(unknown)}"
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
        """Whether activation or parameter layouts need internal guidance."""

        return (
            self.data_axis_name is not None
            or self.has_sharded_parameters
            or any(axis_type is AxisType.Explicit for axis_type in self.mesh.axis_types)
        )


def _build_train_step_from_sharding_plan(
    task: Task[Any],
    optimizer: optax.GradientTransformationExtraArgs,
    plan: ShardingPlan,
    *,
    max_grad_norm: float | None = 1.0,
    execution: LossExecution,
    donate_state: bool = False,
    gradient_accumulation_steps: int = 1,
    precision: PrecisionPolicy = FP32_POLICY,
    trainable_filter: Any = eqx.is_inexact_array,
) -> TrainStep:
    """Compile one ordinary task update from a resolved sharding plan.

    Parameter and batch layouts are independent: an optional data axis shards
    example rows, while parameter axes shard persistent train state. The global
    task program and scientific batch remain unchanged.
    """

    unannotated_train_step = _build_train_step_body(
        task,
        optimizer,
        max_grad_norm=max_grad_norm,
        execution=execution,
        context=ExecutionContext(),
        gradient_accumulation_steps=gradient_accumulation_steps,
        accumulation_split_sharding=(
            NamedSharding(plan.mesh, P(None, plan.data_axis_name))
            if gradient_accumulation_steps > 1 and plan.data_axis_name is not None
            else None
        ),
        precision=precision,
        trainable_filter=trainable_filter,
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
