"""Canonical construction and execution of a complete configured job."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

from representax.config import (
    ClassificationEvaluatorConfig,
    ClassificationProbeEvaluatorConfig,
    ClusteringEvaluatorConfig,
    ComponentConfig,
    CustomShardingConfig,
    DataConfig,
    DDPConfig,
    FSDPConfig,
    InformationRetrievalEvaluatorConfig,
    JEPAEvaluatorConfig,
    JEPARepresentationEvaluatorConfig,
    JobConfig,
    LoRAConfig,
    ModelConfig,
    MSEEvaluatorConfig,
    PairClassificationEvaluatorConfig,
    ParaphraseMiningEvaluatorConfig,
    QuantizedLoRAConfig,
    RerankingEvaluatorConfig,
    RewardEvaluatorConfig,
    SimilarityEvaluatorConfig,
    TripletEvaluatorConfig,
)
from representax.data import ArtifactResolver, build_data_loader
from representax.evaluation import (
    ClassificationEvaluator,
    ClassificationProbeEvaluator,
    ClusteringEvaluator,
    InformationRetrievalEvaluator,
    JEPAEvaluator,
    JEPARepresentationEvaluator,
    LossEvaluator,
    MSEEvaluator,
    PairClassificationEvaluator,
    ParaphraseMiningEvaluator,
    RerankingEvaluator,
    RewardEvaluator,
    SequentialEvaluator,
    SimilarityEvaluator,
    TripletEvaluator,
)
from representax.models import apply_lora, apply_quantized_lora, lora_parameter_filter
from representax.models.processing import Processor
from representax.precision import resolve_precision_policy
from representax.tasks import build_task

from .config import build_loss_execution
from .evaluation import EvaluationRunner
from .loop import TrainingRunResult, run_training
from .optimizer import build_optimizer
from .sharding import ShardingPlan, parameter_specs_from_rules
from .state import TrainState
from .step import TrainStep, build_train_step, init_train_state
from .wandb import WandbReporter


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


def load_model(
    config: ModelConfig,
    *,
    key: Any,
    activation_rematerialization: str | None = None,
) -> tuple[eqx.Module, Processor | None]:
    """Load one native model and its optional host processor exactly once."""

    factory = resolve_target(config.target)
    parameters = dict(config.parameters)
    signature = inspect.signature(factory)
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "key" in signature.parameters and "key" not in parameters:
        parameters["key"] = key
    if (
        activation_rematerialization is not None
        and ("rematerialization" in signature.parameters or accepts_keywords)
        and "rematerialization" not in parameters
    ):
        parameters["rematerialization"] = activation_rematerialization
    result = factory(**parameters)
    if isinstance(result, tuple) and len(result) == 2:
        model, processor = result
    else:
        model, processor = result, None
    if not isinstance(model, eqx.Module):
        raise TypeError(
            f"model target {config.target!r} must return an eqx.Module or "
            "(eqx.Module, Processor)"
        )
    if processor is not None and not isinstance(processor, Processor):
        raise TypeError(f"model target {config.target!r} returned an invalid processor")
    return model, processor


def prepare_model(
    model: eqx.Module,
    *,
    adapter: QuantizedLoRAConfig | LoRAConfig | None,
    key: Any,
) -> tuple[eqx.Module, Any]:
    """Apply one scientific adapter recipe and return its trainable filter."""

    if adapter is None:
        training_filter = getattr(model, "training_filter", None)
        if callable(training_filter):
            return model, training_filter()
        return model, eqx.is_inexact_array
    apply = (
        apply_quantized_lora if isinstance(adapter, QuantizedLoRAConfig) else apply_lora
    )
    adapted = apply(
        model,
        rank=adapter.rank,
        alpha=adapter.alpha,
        key=key,
        target_pattern=adapter.target_pattern,
        initialization_scale=adapter.initialization_scale,
    )
    return adapted, lora_parameter_filter(adapted)


def build_collate(
    config: DataConfig,
    *,
    processor: Processor | None = None,
) -> Callable[[Any], Any] | None:
    """Bind task collation and inject the loaded model processor if accepted."""

    if config.collate is None:
        return None
    collate = resolve_target(config.collate.target)
    parameters: dict[str, Any] = dict(config.collate.parameters)
    if (
        processor is not None
        and "processor" in inspect.signature(collate).parameters
        and "processor" not in parameters
    ):
        parameters["processor"] = processor
    if inspect.isclass(collate):
        instance = collate(**parameters)
        if not callable(instance):
            raise TypeError("collate class must construct a callable instance")
        return instance
    if not parameters:
        return collate
    return partial(collate, **parameters)


def build_batches(
    config: DataConfig,
    *,
    batch_size: int,
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
    processor: Processor | None = None,
    measure_training_tokens: bool = False,
) -> Any:
    """Materialize one reproducible Grain batch source from its data config."""

    return build_data_loader(
        config.distribution,
        batch_size=batch_size,
        batch_fn=build_collate(config, processor=processor),
        drop_remainder=config.drop_remainder,
        num_threads=config.num_threads,
        prefetch_buffer_size=config.prefetch_buffer_size,
        host_memory_budget_bytes=config.host_memory_budget_bytes,
        measure_training_tokens=measure_training_tokens,
        resolvers=resolvers,
        mappers=mappers,
    )


@dataclass(frozen=True)
class JobRuntime:
    """Fully constructed runtime boundary consumed by ``run_training``."""

    state: TrainState
    processor: Processor | None
    step: TrainStep
    batches: Any
    evaluation_runners: tuple[EvaluationRunner, ...]
    evaluation_batches: Callable[[], Any] | None
    place_state: Callable[[TrainState], TrainState]
    place_batch: Callable[[Any], Any]
    startup_metrics: Mapping[str, float]


def build_job_runtime(
    job: JobConfig,
    *,
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
    place_initial_state: bool = True,
) -> JobRuntime:
    """Build every live JAX, Optax, task, and Grain object from one JobConfig."""

    startup_started = time.perf_counter()
    startup_metrics: dict[str, float] = {}
    key = jax.random.key(job.training.seed)
    phase_started = time.perf_counter()
    model, processor = load_model(
        job.model,
        key=jax.random.fold_in(key, 0),
        activation_rematerialization=job.training.activation_rematerialization,
    )
    startup_metrics["perf/model_load_seconds"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    model, trainable_filter = prepare_model(
        model,
        adapter=job.training.adapter,
        key=jax.random.fold_in(key, 1),
    )
    startup_metrics["perf/adapter_preparation_seconds"] = (
        time.perf_counter() - phase_started
    )
    phase_started = time.perf_counter()
    task = build_task(job.task, job.loss, modifiers=job.loss_modifiers)
    startup_metrics["perf/task_initialization_seconds"] = (
        time.perf_counter() - phase_started
    )
    phase_started = time.perf_counter()
    optimizer = build_optimizer(job.optimization)
    execution = build_loss_execution(
        job.training.grad_cache,
        mega_batch_mining=job.training.mega_batch_mining,
    )
    precision = resolve_precision_policy(job.training.precision)
    state = init_train_state(
        model,
        optimizer,
        precision=precision,
        trainable_filter=trainable_filter,
    )
    startup_metrics["perf/optimizer_initialization_seconds"] = (
        time.perf_counter() - phase_started
    )
    phase_started = time.perf_counter()
    mesh_size = job.training.mesh.device_count
    if mesh_size > len(jax.devices()):
        raise ValueError(
            f"job mesh requires {mesh_size} devices; only {len(jax.devices())} visible"
        )
    plan: ShardingPlan | None = None
    device = jax.devices()[0]

    def place_single_device_state(value: TrainState) -> TrainState:
        return jax.device_put(value, device)

    place_state: Callable[[TrainState], TrainState] = place_single_device_state
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
        if place_initial_state:
            state = place_state(state)
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
                trainable_filter=trainable_filter,
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
                trainable_filter=trainable_filter,
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
                trainable_filter=trainable_filter,
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
        if place_initial_state:
            state = plan.place_state(state)
        place_state = plan.place_state
        place_batch = plan.place_batch
    startup_metrics["perf/sharding_initialization_seconds"] = (
        time.perf_counter() - phase_started
    )
    phase_started = time.perf_counter()
    step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=job.optimization.max_gradient_norm,
        execution=execution,
        donate_state=job.training.donate_buffers,
        gradient_accumulation_steps=job.training.batch.gradient_accumulation_steps,
        precision=precision,
        trainable_filter=trainable_filter,
    )
    startup_metrics["perf/train_step_initialization_seconds"] = (
        time.perf_counter() - phase_started
    )
    phase_started = time.perf_counter()
    batches = build_batches(
        job.data,
        batch_size=job.training.global_batch_size,
        resolvers=resolvers,
        mappers=mappers,
        processor=processor,
        measure_training_tokens=job.logging.timing,
    )
    startup_metrics["perf/data_loader_initialization_seconds"] = (
        time.perf_counter() - phase_started
    )
    phase_started = time.perf_counter()
    if job.evaluation is None:
        evaluation_runners: tuple[EvaluationRunner, ...] = ()
        evaluation_batches = None
    else:

        def build_evaluator(config: Any) -> Any:
            if config.kind == "loss":
                return LossEvaluator(task, name=config.name)
            if isinstance(config, SimilarityEvaluatorConfig):
                return SimilarityEvaluator(
                    name=config.name,
                    similarity_functions=config.similarity_functions,
                    main_similarity=config.main_similarity,
                    left_route=config.left_route,
                    right_route=config.right_route,
                )
            if isinstance(config, ClassificationEvaluatorConfig):
                return ClassificationEvaluator(name=config.name, task=task)
            if isinstance(config, PairClassificationEvaluatorConfig):
                return PairClassificationEvaluator(
                    name=config.name,
                    similarity_functions=config.similarity_functions,
                    left_route=config.left_route,
                    right_route=config.right_route,
                )
            if isinstance(config, JEPARepresentationEvaluatorConfig):
                return JEPARepresentationEvaluator(
                    name=config.name,
                    inverse_regularization=config.inverse_regularization,
                    normalization=config.normalization,
                    max_iterations=config.max_iterations,
                    neighbors=config.neighbors,
                    seed=config.seed,
                    route=config.route,
                )
            if isinstance(config, ClassificationProbeEvaluatorConfig):
                return ClassificationProbeEvaluator(
                    name=config.name,
                    inverse_regularization=config.inverse_regularization,
                    normalization=config.normalization,
                    max_iterations=config.max_iterations,
                    seed=config.seed,
                    route=config.route,
                )
            if isinstance(config, ClusteringEvaluatorConfig):
                return ClusteringEvaluator(
                    name=config.name,
                    clusters=config.clusters,
                    normalization=config.normalization,
                    batch_size=config.batch_size,
                    max_iterations=config.max_iterations,
                    n_init=config.n_init,
                    seed=config.seed,
                    route=config.route,
                )
            if isinstance(config, MSEEvaluatorConfig):
                return MSEEvaluator(name=config.name, route=config.route)
            if isinstance(config, TripletEvaluatorConfig):
                return TripletEvaluator(
                    name=config.name,
                    distance=config.distance,
                    anchor_route=config.anchor_route,
                    positive_route=config.positive_route,
                    negative_route=config.negative_route,
                )
            if isinstance(config, RerankingEvaluatorConfig):
                return RerankingEvaluator(name=config.name, at_k=config.at_k)
            if isinstance(config, RewardEvaluatorConfig):
                return RewardEvaluator(
                    kind=config.mode,
                    name=config.name,
                    at_k=config.at_k,
                )
            if isinstance(config, JEPAEvaluatorConfig):
                return JEPAEvaluator(task=task, name=config.name)
            if isinstance(config, ParaphraseMiningEvaluatorConfig):
                return ParaphraseMiningEvaluator(
                    duplicate_pairs=config.duplicate_pairs,
                    name=config.name,
                    max_pairs=config.max_pairs,
                    block_size=config.block_size,
                    route=config.route,
                )
            if isinstance(config, InformationRetrievalEvaluatorConfig):
                return InformationRetrievalEvaluator(
                    relevant_documents=config.relevant_documents,
                    name=config.name,
                    score_functions=config.score_functions,
                    main_score_function=config.main_score_function,
                    accuracy_at_k=config.accuracy_at_k,
                    precision_recall_at_k=config.precision_recall_at_k,
                    mrr_at_k=config.mrr_at_k,
                    ndcg_at_k=config.ndcg_at_k,
                    map_at_k=config.map_at_k,
                    query_route=config.query_route,
                    document_route=config.document_route,
                )
            raise ValueError(f"unsupported evaluator kind {config.kind!r}")

        evaluators = tuple(
            build_evaluator(config) for config in job.evaluation.evaluators
        )
        evaluator = (
            evaluators[0]
            if len(evaluators) == 1
            else SequentialEvaluator(
                evaluators,
                primary_metric=job.evaluation.primary_metric,
            )
        )
        evaluation_runners = (EvaluationRunner(evaluator, precision=precision),)

        def evaluation_batches() -> Any:
            if job.evaluation is None:  # pragma: no cover - closed-over invariant
                raise AssertionError("evaluation config disappeared")
            return build_batches(
                job.evaluation.data,
                batch_size=job.evaluation.batch_size,
                resolvers=resolvers,
                mappers=mappers,
                processor=processor,
            )

    startup_metrics["perf/evaluation_initialization_seconds"] = (
        time.perf_counter() - phase_started
    )
    startup_metrics["perf/startup_seconds"] = time.perf_counter() - startup_started

    return JobRuntime(
        state=state,
        processor=processor,
        step=step,
        batches=batches,
        evaluation_runners=evaluation_runners,
        evaluation_batches=evaluation_batches,
        place_state=place_state,
        place_batch=place_batch,
        startup_metrics=startup_metrics,
    )


def run_job(
    job: JobConfig,
    run_directory: str | Path,
    *,
    resume: bool = False,
    reporters: tuple[Any, ...] = (),
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
    stop_after: int | None = None,
) -> TrainingRunResult:
    """Execute one validated configuration from artifacts to inference model."""

    runtime = build_job_runtime(
        job,
        resolvers=resolvers,
        mappers=mappers,
        place_initial_state=not resume,
    )
    configured_reporters = list(reporters)
    if job.logging.wandb is not None:
        configured_reporters.append(
            WandbReporter(
                job.logging.wandb,
                job=job,
                run_directory=run_directory,
                resume=resume,
            )
        )
    result = run_training(
        state=runtime.state,
        step=runtime.step,
        batches=runtime.batches,
        job=job,
        run_directory=run_directory,
        resume=resume,
        reporters=tuple(configured_reporters),
        place_state=runtime.place_state,
        place_batch=runtime.place_batch,
        evaluation_runners=runtime.evaluation_runners,
        evaluation_batches=runtime.evaluation_batches,
        startup_metrics=runtime.startup_metrics,
        export_inference=job.export.enabled,
        stop_after=stop_after,
    )
    return result


__all__ = [
    "JobRuntime",
    "build_batches",
    "build_collate",
    "build_component",
    "build_job_runtime",
    "load_model",
    "prepare_model",
    "resolve_target",
    "run_job",
]
