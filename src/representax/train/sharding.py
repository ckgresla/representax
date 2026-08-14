"""Named training-state and batch sharding configurations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from representax.core import Task

from .execution import ExecutionContext, LossExecution
from .grad_cache import GradCache
from .step import TrainStep, _build_train_step_body


@dataclass(frozen=True)
class DataParallel:
    """Replicate training state and shard example rows over a named data axis."""

    mesh: Mesh
    axis_name: str = "data"

    def __post_init__(self) -> None:
        if self.axis_name not in self.mesh.axis_names:
            raise ValueError(
                f"data axis {self.axis_name!r} is absent from mesh axes "
                f"{self.mesh.axis_names!r}"
            )
        if len(self.mesh.axis_names) != 1:
            raise ValueError("DataParallel currently requires a one-dimensional mesh")

    @classmethod
    def from_devices(
        cls,
        devices: Sequence[jax.Device],
        *,
        axis_name: str = "data",
    ) -> DataParallel:
        """Build the standard replicated-data-parallel mesh."""

        if not devices:
            raise ValueError("DataParallel requires at least one device")
        mesh = Mesh(np.asarray(devices), (axis_name,))
        return cls(mesh=mesh, axis_name=axis_name)

    @property
    def world_size(self) -> int:
        return int(self.mesh.shape[self.axis_name])

    @property
    def replicated_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, P())

    @property
    def batch_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, P(self.axis_name))

    def place_replicated(self, tree: Any) -> Any:
        """Place every array leaf as a replica over the data mesh."""

        return jax.tree.map(
            lambda value: (
                jax.device_put(value, self.replicated_sharding)
                if eqx.is_array(value)
                else value
            ),
            tree,
            is_leaf=lambda value: value is None,
        )

    def place_batch(self, batch: Any) -> Any:
        """Shard every row-major batch leaf on its leading record axis."""

        return jax.tree.map(
            lambda value: (
                jax.device_put(value, self.batch_sharding)
                if eqx.is_array(value)
                else value
            ),
            batch,
            is_leaf=lambda value: value is None,
        )


def build_data_parallel_train_step(
    task: Task[Any],
    optimizer: optax.GradientTransformationExtraArgs,
    plan: DataParallel,
    *,
    max_grad_norm: float | None = 1.0,
    execution: LossExecution,
    donate_state: bool = False,
) -> TrainStep:
    """Compile one global-negative update over replicated training state.

    Batch leaves are sharded on their leading record axis. For a retrieval
    batch this means the positive relation is row-sharded while retaining its
    global document axis. GradCache gathers compact representations and relation
    rows, and reverse-mode transposition synchronizes replicated parameter
    gradients exactly once after encoder replay.
    """

    if not isinstance(execution, GradCache):
        raise TypeError("DataParallel currently requires GradCache execution")
    train_step_body = _build_train_step_body(
        task,
        optimizer,
        max_grad_norm=max_grad_norm,
        execution=execution,
        context=ExecutionContext(data_axis_name=plan.axis_name),
    )

    mapped_step = jax.shard_map(
        train_step_body,
        mesh=plan.mesh,
        in_specs=(P(), P(plan.axis_name), P()),
        out_specs=P(),
        check_vma=True,
    )
    donate_argnums = (0,) if donate_state else ()
    return jax.jit(mapped_step, donate_argnums=donate_argnums)
