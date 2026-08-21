"""Canonical construction and execution of a complete configured job."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

from representax.config import (
    ComponentConfig,
    CustomShardingConfig,
    DataConfig,
    DDPConfig,
    EmbeddingSimilarityEvaluatorConfig,
    FSDPConfig,
    JobConfig,
    ModelConfig,
)
from representax.data import ArtifactResolver, build_grain_iterator
from representax.evaluation import EmbeddingSimilarityEvaluator, LossEvaluator
from representax.precision import resolve_precision_policy
from representax.tasks import build_task

from .config import build_loss_execution
from .evaluation import EvaluationRunner
from .loop import TrainingRunResult, run_training
from .optimizer import build_optimizer
from .sharding import ShardingPlan, parameter_specs_from_rules
from .state import TrainState
from .step import TrainStep, build_train_step, init_train_state


def resolve_target(target: str) -> Callable[..., Any]:
    """Resolve one trusted dotted Python target from a serialized recipe."""

    module_name, separator, attribute_path = target.partition(":")
    if not separator:
        module_name, separator, attribute_path = target.rpartition(".")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"component target must be a dotted import path: {target!r}")
    try:
        value: Any = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            raise ImportError(
                f"could not import component module {module_name!r}"
            ) from error
        raise
    try:
        for attribute in attribute_path.split("."):
            value = getattr(value, attribute)
    except AttributeError as error:
        raise AttributeError(
            f"could not resolve component target {target!r}"
        ) from error
    if not callable(value):
        raise TypeError(f"component target {target!r} is not callable")
    return value


def build_component(config: ComponentConfig) -> Any:
    """Call a configured component factory with JSON-serializable parameters."""

    return resolve_target(config.target)(**config.parameters)


def build_model(
    config: ModelConfig,
    *,
    key: Any,
    activation_rematerialization: str | None = None,
) -> eqx.Module:
    """Construct a native model, injecting JAX runtime inputs when accepted."""

    factory = resolve_target(config.target)
    parameters = dict(config.parameters)
    signature = inspect.signature(factory)
    if "key" in signature.parameters and "key" not in parameters:
        parameters["key"] = key
    if (
        activation_rematerialization is not None
        and "rematerialization" in signature.parameters
        and "rematerialization" not in parameters
    ):
        parameters["rematerialization"] = activation_rematerialization
    model = factory(**parameters)
    if not isinstance(model, eqx.Module):
        raise TypeError(f"model target {config.target!r} must return an eqx.Module")
    return model


def build_collate(config: DataConfig) -> Callable[[Any], Any] | None:
    """Bind configured collation parameters without consuming an example batch."""

    if config.collate is None:
        return None
    collate = resolve_target(config.collate.target)
    if inspect.isclass(collate):
        instance = collate(**config.collate.parameters)
        if not callable(instance):
            raise TypeError("collate class must construct a callable instance")
        return instance
    if not config.collate.parameters:
        return collate
    return partial(collate, **config.collate.parameters)


def build_batches(
    config: DataConfig,
    *,
    batch_size: int,
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Any:
    """Materialize one reproducible Grain batch source from its data config."""

    return build_grain_iterator(
        config.recipe,
        batch_size=batch_size,
        batch_fn=build_collate(config),
        drop_remainder=config.drop_remainder,
        num_threads=config.num_threads,
        prefetch_buffer_size=config.prefetch_buffer_size,
        resolvers=resolvers,
        mappers=mappers,
    )


@dataclass(frozen=True)
class JobRuntime:
    """Fully constructed runtime boundary consumed by ``run_training``."""

    state: TrainState
    step: TrainStep
    batches: Any
    evaluation_runners: tuple[EvaluationRunner, ...]
    evaluation_batches: Callable[[], Any] | None
    place_batch: Callable[[Any], Any]


def build_job_runtime(
    job: JobConfig,
    *,
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
) -> JobRuntime:
    """Build every live JAX, Optax, task, and Grain object from one JobConfig."""

    key = jax.random.key(job.training.seed)
    model = build_model(
        job.model,
        key=jax.random.fold_in(key, 0),
        activation_rematerialization=job.training.activation_rematerialization,
    )
    task = build_task(job.task, job.loss, modifiers=job.loss_modifiers)
    optimizer = build_optimizer(job.optimization)
    execution = build_loss_execution(
        job.training.grad_cache,
        mega_batch_mining=job.training.mega_batch_mining,
    )
    precision = resolve_precision_policy(job.training.precision)
    state = init_train_state(model, optimizer, precision=precision)
    mesh_size = job.training.mesh.device_count
    if mesh_size > len(jax.devices()):
        raise ValueError(
            f"job mesh requires {mesh_size} devices; only {len(jax.devices())} visible"
        )
    plan: ShardingPlan | None = None
    place_batch = jax.device_put
    if mesh_size == 1:
        if job.training.grad_cache is None and job.training.mega_batch_mining is None:
            realized_batch_size = (
                job.training.batch.micro_batch_size
                * job.training.batch.gradient_accumulation_steps
            )
            if realized_batch_size != job.training.global_batch_size:
                raise ValueError(
                    "direct execution batch plan differs from global_batch_size: "
                    f"{realized_batch_size} != {job.training.global_batch_size}"
                )
    else:
        mesh = jax.make_mesh(
            job.training.mesh.axis_shapes,
            job.training.mesh.axis_names,
            axis_types=(
                (jax.sharding.AxisType.Auto,) * len(job.training.mesh.axis_names)
            ),
            devices=jax.devices()[:mesh_size],
        )
        sharding = job.training.sharding
        if isinstance(sharding, DDPConfig):
            plan = ShardingPlan.ddp(
                state,
                optimizer,
                mesh,
                axis_name=sharding.axis,
            )
            data_axis_name = sharding.axis
        elif isinstance(sharding, FSDPConfig):
            plan = ShardingPlan.fsdp(
                state,
                optimizer,
                mesh,
                parameter_axis_name=sharding.resolved_parameter_axis,
                data_axis_name=sharding.data_axis,
                minimum_parameter_elements=sharding.minimum_parameter_elements,
            )
            data_axis_name = sharding.data_axis
        elif isinstance(sharding, CustomShardingConfig):
            parameter_specs = parameter_specs_from_rules(
                state.model,
                tuple(
                    (rule.pattern, jax.sharding.PartitionSpec(*rule.axes))
                    for rule in sharding.parameter_rules
                ),
                default=jax.sharding.PartitionSpec(*sharding.default_parameter_axes),
            )
            plan = ShardingPlan.custom(
                state,
                optimizer,
                mesh,
                parameter_specs,
                parameter_axis_names=sharding.parameter_axes,
                data_axis_name=sharding.data_axis,
            )
            data_axis_name = sharding.data_axis
        else:  # pragma: no cover - the Pydantic discriminator closes the union
            raise TypeError(f"unsupported sharding config {type(sharding).__name__}")
        data_axis_size = (
            1 if data_axis_name is None else int(mesh.shape[data_axis_name])
        )
        realized_batch_size = job.training.batch.micro_batch_size * data_axis_size
        if realized_batch_size != job.training.global_batch_size:
            raise ValueError(
                "distributed batch plan differs from global_batch_size: "
                f"{realized_batch_size} != {job.training.global_batch_size}"
            )
        state = plan.place_state(state)
        place_batch = plan.place_batch
    step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=job.optimization.max_gradient_norm,
        execution=execution,
        donate_state=job.training.donate_buffers,
        gradient_accumulation_steps=job.training.batch.gradient_accumulation_steps,
        precision=precision,
    )
    batches = build_batches(
        job.data,
        batch_size=job.training.global_batch_size,
        resolvers=resolvers,
        mappers=mappers,
    )
    if job.evaluation is None:
        evaluation_runners: tuple[EvaluationRunner, ...] = ()
        evaluation_batches = None
    else:

        def build_evaluator(config: Any) -> Any:
            if config.kind == "loss":
                return LossEvaluator(task, name=config.name)
            if isinstance(config, EmbeddingSimilarityEvaluatorConfig):
                return EmbeddingSimilarityEvaluator(
                    name=config.name,
                    similarity_functions=config.similarity_functions,
                    main_similarity=config.main_similarity,
                    left_route=config.left_route,
                    right_route=config.right_route,
                )
            raise ValueError(f"unsupported evaluator kind {config.kind!r}")

        evaluation_runners = tuple(
            EvaluationRunner(build_evaluator(config), precision=precision)
            for config in job.evaluation.evaluators
        )

        def evaluation_batches() -> Any:
            if job.evaluation is None:  # pragma: no cover - closed-over invariant
                raise AssertionError("evaluation config disappeared")
            return build_batches(
                job.evaluation.data,
                batch_size=job.evaluation.batch_size,
                resolvers=resolvers,
                mappers=mappers,
            )

    return JobRuntime(
        state=state,
        step=step,
        batches=batches,
        evaluation_runners=evaluation_runners,
        evaluation_batches=evaluation_batches,
        place_batch=place_batch,
    )


def run_job(
    job: JobConfig,
    run_directory: str | Path,
    *,
    resume: bool = False,
    reporters: tuple[Any, ...] = (),
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
) -> TrainingRunResult:
    """Execute one validated configuration from artifacts to inference model."""

    runtime = build_job_runtime(job, resolvers=resolvers, mappers=mappers)
    result = run_training(
        state=runtime.state,
        step=runtime.step,
        batches=runtime.batches,
        job=job,
        run_directory=run_directory,
        resume=resume,
        reporters=reporters,
        place_batch=runtime.place_batch,
        evaluation_runners=runtime.evaluation_runners,
        evaluation_batches=runtime.evaluation_batches,
    )
    if job.export.enabled:
        from representax.export import export_inference_bundle

        bundle = export_inference_bundle(
            result.selected_model,
            job,
            result.run_directory / job.export.directory_name,
            iteration=(
                result.best_iteration
                if job.export.selection == "best"
                else result.completed_iterations
            ),
        )
        result = replace(result, inference_bundle=bundle.path)
    return result


__all__ = [
    "JobRuntime",
    "build_batches",
    "build_collate",
    "build_component",
    "build_job_runtime",
    "build_model",
    "resolve_target",
    "run_job",
]
