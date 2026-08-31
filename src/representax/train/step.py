"""One generic compiled optimization boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import Task
from representax.precision import (
    FP32_POLICY,
    PrecisionPolicy,
    accumulated_values,
    loss_value,
    precision_context,
    prepare_master_model,
)

from .execution import (
    _LOCAL_EXECUTION_CONTEXT,
    Direct,
    ExecutionContext,
    LossExecution,
)
from .state import StepMetrics, StepResult, TrainState

if TYPE_CHECKING:
    from .sharding import ShardingPlan


def tree_global_norm(tree: Any) -> Float[Array, ""]:
    """Compute an FP32 L2 norm across every inexact array leaf."""

    squares = [
        jnp.sum(jnp.square(leaf.astype(jnp.float32)))
        for leaf in jax.tree.leaves(tree)
        if eqx.is_inexact_array(leaf)
    ]
    if not squares:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.stack(squares), dtype=jnp.float32))


def tree_all_finite(*trees: Any) -> Bool[Array, ""]:
    """Return one scalar finite check over all inexact leaves."""

    checks = [
        jnp.all(jnp.isfinite(leaf.astype(jnp.float32)))
        for tree in trees
        for leaf in jax.tree.leaves(tree)
        if eqx.is_inexact_array(leaf)
    ]
    if not checks:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(checks))


def init_train_state(
    model: eqx.Module,
    optimizer: optax.GradientTransformationExtraArgs,
    *,
    precision: PrecisionPolicy = FP32_POLICY,
    trainable_filter: Any = eqx.is_inexact_array,
) -> TrainState:
    """Initialize FP32 master parameters and matching selected Optax state."""

    model = prepare_master_model(
        model,
        precision,
        trainable_filter=trainable_filter,
    )
    parameters = eqx.filter(model, trainable_filter)
    return TrainState(
        model=model,
        optimizer_state=optimizer.init(parameters),
        step=jnp.asarray(0, dtype=jnp.int32),
    )


# Compatibility alias for the public 0.0.1 API. New code should use the
# canonical Optax-style ``init_train_state`` name.
make_train_state = init_train_state


TrainStep = Callable[[TrainState, Any, PRNGKeyArray | None], StepResult]


def _validate_accumulation_batch(batch: Any, steps: int, *, task: Any) -> None:
    batch_size_for_accumulation = getattr(task, "accumulation_batch_size", None)
    if callable(batch_size_for_accumulation):
        batch_size = int(batch_size_for_accumulation(batch))
        if batch_size % steps != 0:
            raise ValueError(
                "logical batch size must be divisible by "
                f"gradient_accumulation_steps: {batch_size} % {steps} != 0"
            )
        return
    arrays = [leaf for leaf in jax.tree.leaves(batch) if eqx.is_array(leaf)]
    if not arrays:
        raise TypeError("gradient accumulation requires an array batch")
    scalar_leaves = [leaf for leaf in arrays if leaf.ndim == 0]
    if scalar_leaves:
        raise ValueError("every array batch leaf must have a leading example dimension")
    batch_size = arrays[0].shape[0]
    if any(leaf.shape[0] != batch_size for leaf in arrays[1:]):
        raise ValueError("every array batch leaf must have the same leading size")
    if batch_size % steps != 0:
        raise ValueError(
            "logical batch size must be divisible by gradient_accumulation_steps: "
            f"{batch_size} % {steps} != 0"
        )


def _split_batch_arrays(batch: Any, steps: int) -> tuple[Any, Any]:
    arrays, static = eqx.partition(batch, eqx.is_array)
    split = jax.tree.map(
        lambda leaf: leaf.reshape(
            steps,
            leaf.shape[0] // steps,
            *leaf.shape[1:],
        ),
        arrays,
    )
    return split, static


def _build_train_step_body(
    task: Task[Any],
    optimizer: optax.GradientTransformationExtraArgs,
    *,
    max_grad_norm: float | None = 1.0,
    execution: LossExecution | None = None,
    context: ExecutionContext = _LOCAL_EXECUTION_CONTEXT,
    gradient_accumulation_steps: int = 1,
    accumulation_split_sharding: Any | None = None,
    precision: PrecisionPolicy = FP32_POLICY,
    trainable_filter: Any = eqx.is_inexact_array,
) -> TrainStep:
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or None")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    accumulation_weight = getattr(task, "accumulation_weight", None)
    supports_gradient_accumulation = getattr(
        task,
        "supports_gradient_accumulation",
        True,
    )
    if gradient_accumulation_steps > 1 and not supports_gradient_accumulation:
        raise TypeError(
            f"{type(task).__name__} objective does not support exact gradient "
            "accumulation"
        )
    accumulation_metric_reductions = getattr(
        task,
        "accumulation_metric_reductions",
        None,
    )
    accumulation_loss_reduction = getattr(
        task,
        "accumulation_loss_reduction",
        "mean",
    )
    if accumulation_loss_reduction not in {"mean", "sum"}:
        raise ValueError("accumulation loss reduction must be 'mean' or 'sum'")
    accumulation_microbatch = getattr(task, "accumulation_microbatch", None)
    if gradient_accumulation_steps > 1 and not callable(accumulation_weight):
        raise TypeError(
            "gradient accumulation requires a task with accumulation_weight(batch)"
        )
    if gradient_accumulation_steps > 1 and not isinstance(
        accumulation_metric_reductions,
        dict,
    ):
        raise TypeError(
            "gradient accumulation requires task accumulation_metric_reductions"
        )
    if isinstance(accumulation_metric_reductions, dict):
        invalid_reductions = {
            name: reduction
            for name, reduction in accumulation_metric_reductions.items()
            if reduction not in {"mean", "sum", "root_mean_square"}
        }
        if invalid_reductions:
            raise ValueError(
                "accumulation metric reductions must be 'mean', 'sum', or "
                "'root_mean_square': "
                f"{invalid_reductions}"
            )
    metric_reductions = (
        cast(dict[str, str], accumulation_metric_reductions)
        if isinstance(accumulation_metric_reductions, dict)
        else {}
    )
    resolved_execution = Direct() if execution is None else execution
    resolved_execution.validate(task)

    def evaluate_loss(
        model: eqx.Module,
        batch: Any,
        key: PRNGKeyArray | None,
    ) -> tuple[Float[Array, ""], Any]:
        with precision_context(precision):
            output = resolved_execution.evaluate(
                task,
                model,
                batch,
                key=key,
                context=context,
            )
            return loss_value(output.loss), accumulated_values(output.metrics)

    full_parameter_training = trainable_filter is eqx.is_inexact_array

    @eqx.filter_value_and_grad(has_aux=True)
    def full_loss_fn(
        model: eqx.Module,
        batch: Any,
        key: PRNGKeyArray | None,
    ) -> tuple[Float[Array, ""], Any]:
        return evaluate_loss(model, batch, key)

    @eqx.filter_value_and_grad(has_aux=True)
    def selected_loss_fn(
        trainable_model: Any,
        frozen_model: Any,
        batch: Any,
        key: PRNGKeyArray | None,
    ) -> tuple[Float[Array, ""], Any]:
        return evaluate_loss(
            cast(eqx.Module, eqx.combine(trainable_model, frozen_model)),
            batch,
            key,
        )

    def train_step_body(
        state: TrainState,
        batch: Any,
        key: PRNGKeyArray | None,
    ) -> StepResult:
        if full_parameter_training:
            trainable_model = state.model
            frozen_model = None
        else:
            trainable_model, frozen_model = eqx.partition(
                state.model,
                trainable_filter,
            )

        def differentiated_loss(
            batch: Any,
            key: PRNGKeyArray | None,
        ) -> tuple[Any, Any]:
            if full_parameter_training:
                return full_loss_fn(
                    cast(eqx.Module, trainable_model),
                    batch,
                    key,
                )
            return selected_loss_fn(
                trainable_model,
                frozen_model,
                batch,
                key,
            )

        if gradient_accumulation_steps == 1:
            (loss, task_metrics), gradients = differentiated_loss(batch, key)
        else:
            if callable(accumulation_microbatch):
                split_arrays = batch_static = None
            else:
                split_arrays, batch_static = _split_batch_arrays(
                    batch,
                    gradient_accumulation_steps,
                )
                if accumulation_split_sharding is not None:
                    split_arrays = jax.tree.map(
                        lambda value: jax.reshard(
                            value,
                            accumulation_split_sharding,
                        ),
                        split_arrays,
                    )

            def evaluate_microbatch(index: Array) -> tuple[Any, Any, Array]:
                if callable(accumulation_microbatch):
                    microbatch = accumulation_microbatch(
                        batch,
                        index,
                        gradient_accumulation_steps,
                    )
                else:
                    microbatch_arrays = jax.tree.map(
                        lambda leaf: leaf[index],
                        split_arrays,
                    )
                    microbatch = eqx.combine(microbatch_arrays, batch_static)
                microbatch_key = None if key is None else jax.random.fold_in(key, index)
                loss_and_metrics, gradients = differentiated_loss(
                    microbatch,
                    microbatch_key,
                )
                if not callable(accumulation_weight):  # pragma: no cover
                    raise AssertionError("accumulation weight disappeared")
                weight = jnp.asarray(
                    accumulation_weight(microbatch),
                    dtype=jnp.float32,
                )
                return loss_and_metrics, gradients, weight

            (
                first_loss_and_metrics,
                first_gradients,
                first_weight,
            ) = evaluate_microbatch(jnp.asarray(0, dtype=jnp.int32))
            first_loss, first_metrics = first_loss_and_metrics
            first_loss_weight = (
                jnp.asarray(1.0, dtype=jnp.float32)
                if accumulation_loss_reduction == "sum"
                else first_weight
            )
            unknown_metrics = set(first_metrics) - set(metric_reductions)
            if unknown_metrics:
                raise ValueError(
                    "gradient accumulation has no reduction for task metrics: "
                    f"{sorted(unknown_metrics)}"
                )
            first_loss_and_metrics = (
                first_loss * first_loss_weight,
                {
                    name: (
                        value
                        if metric_reductions[name] == "sum"
                        else (
                            jnp.square(value) * first_weight
                            if metric_reductions[name] == "root_mean_square"
                            else value * first_weight
                        )
                    )
                    for name, value in first_metrics.items()
                },
            )
            first_gradients = jax.tree.map(
                lambda value: value * first_loss_weight,
                first_gradients,
            )

            def accumulate_microbatch(
                totals: tuple[Any, Any, Array],
                index: Array,
            ) -> tuple[tuple[Any, Any, Array], None]:
                (
                    (microbatch_loss, microbatch_metrics),
                    microbatch_gradients,
                    weight,
                ) = evaluate_microbatch(index)
                loss_weight = (
                    jnp.asarray(1.0, dtype=jnp.float32)
                    if accumulation_loss_reduction == "sum"
                    else weight
                )
                return (
                    (
                        (
                            totals[0][0] + microbatch_loss * loss_weight,
                            {
                                name: (
                                    totals[0][1][name] + value
                                    if metric_reductions[name] == "sum"
                                    else (
                                        totals[0][1][name] + jnp.square(value) * weight
                                        if metric_reductions[name] == "root_mean_square"
                                        else totals[0][1][name] + value * weight
                                    )
                                )
                                for name, value in microbatch_metrics.items()
                            },
                        ),
                        jax.tree.map(
                            lambda total, value: total + value * loss_weight,
                            totals[1],
                            microbatch_gradients,
                        ),
                        totals[2] + weight,
                    ),
                    None,
                )

            (loss_and_metrics, gradients, total_weight), _ = jax.lax.scan(
                accumulate_microbatch,
                (first_loss_and_metrics, first_gradients, first_weight),
                jnp.arange(1, gradient_accumulation_steps, dtype=jnp.int32),
            )
            reciprocal_weight = jnp.reciprocal(jnp.maximum(total_weight, 1.0))
            loss_total, metric_totals = loss_and_metrics
            loss = (
                loss_total
                if accumulation_loss_reduction == "sum"
                else loss_total * reciprocal_weight
            )
            task_metrics = {
                name: (
                    value
                    if metric_reductions[name] == "sum"
                    else (
                        jnp.sqrt(value * reciprocal_weight)
                        if metric_reductions[name] == "root_mean_square"
                        else value * reciprocal_weight
                    )
                )
                for name, value in metric_totals.items()
            }
            if accumulation_loss_reduction == "mean":
                gradients = jax.tree.map(
                    lambda value: value * reciprocal_weight,
                    gradients,
                )
        gradient_norm = tree_global_norm(gradients)
        if max_grad_norm is None:
            clipped_gradients = gradients
            clipped_gradient_norm = gradient_norm
        else:
            coefficient = jnp.minimum(
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.asarray(max_grad_norm, dtype=jnp.float32)
                / (gradient_norm + jnp.asarray(1e-6, dtype=jnp.float32)),
            )
            clipped_gradients = optax.tree.scale(coefficient, gradients)
            clipped_gradient_norm = gradient_norm * coefficient

        finite = tree_all_finite(
            loss,
            task_metrics,
            gradients,
            gradient_norm,
        )
        if full_parameter_training:
            parameters, static_model = eqx.partition(
                state.model,
                eqx.is_inexact_array,
            )
            updates, optimizer_state = optimizer.update(
                clipped_gradients,
                state.optimizer_state,
                parameters,
            )
            parameters = optax.apply_updates(parameters, updates)
            model = cast(eqx.Module, eqx.combine(parameters, static_model))
        else:
            updates, optimizer_state = optimizer.update(
                clipped_gradients,
                state.optimizer_state,
                cast(Any, trainable_model),
            )
            trainable_model = optax.apply_updates(
                cast(Any, trainable_model),
                updates,
            )
            model = cast(eqx.Module, eqx.combine(trainable_model, frozen_model))
        post_update_model = getattr(task, "post_update_model", None)
        if callable(post_update_model):
            model = cast(
                eqx.Module,
                post_update_model(
                    state.model,
                    model,
                    step=state.step,
                ),
            )
        proposed_state = TrainState(
            model=model,
            optimizer_state=optimizer_state,
            step=state.step + jnp.asarray(1, dtype=jnp.int32),
        )
        new_state = optax.tree.where(finite, proposed_state, state)
        update_norm = jnp.where(
            finite,
            tree_global_norm(updates),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return StepResult(
            state=new_state,
            metrics=StepMetrics(
                loss=loss,
                task=task_metrics,
                gradient_global_norm=gradient_norm,
                clipped_gradient_global_norm=clipped_gradient_norm,
                update_global_norm=update_norm,
                numeric_finite=finite,
                skipped_update=~finite,
            ),
        )

    return train_step_body


def build_train_step(
    task: Task[Any],
    optimizer: optax.GradientTransformationExtraArgs,
    *,
    plan: ShardingPlan | None = None,
    max_grad_norm: float | None = 1.0,
    execution: LossExecution | None = None,
    donate_state: bool = False,
    gradient_accumulation_steps: int = 1,
    precision: PrecisionPolicy = FP32_POLICY,
    trainable_filter: Any = eqx.is_inexact_array,
) -> TrainStep:
    """Build one compiled task-generic optimizer update.

    The task and optimizer are closed-over static program structure. Model,
    optimizer state, batch, and random key remain explicit JAX inputs. Exact
    microbatch accumulation is expressed as one compiled ``lax.scan``, uses the
    task's exact reduction denominator, and performs one Optax update; callers
    must only enable it for objectives that decompose over examples. State
    donation is opt-in because callers may retain the old state for comparison,
    retry, or branching. Orbax asynchronous checkpointing is compatible with
    donation: its blocking device-to-host snapshot completes before save returns.
    An optional resolved sharding plan changes physical layouts and collective
    boundaries without selecting a different trainer or scientific program.
    """

    if plan is not None:
        from .sharding import _build_train_step_from_sharding_plan

        sharded_step = _build_train_step_from_sharding_plan(
            task,
            optimizer,
            plan,
            max_grad_norm=max_grad_norm,
            execution=Direct() if execution is None else execution,
            donate_state=donate_state,
            gradient_accumulation_steps=gradient_accumulation_steps,
            precision=precision,
            trainable_filter=trainable_filter,
        )

        def train_step(
            state: TrainState,
            batch: Any,
            key: PRNGKeyArray | None = None,
        ) -> StepResult:
            if gradient_accumulation_steps > 1:
                _validate_accumulation_batch(
                    batch,
                    gradient_accumulation_steps,
                    task=task,
                )
            return sharded_step(state, batch, key)

        return train_step

    train_step_body = _build_train_step_body(
        task,
        optimizer,
        max_grad_norm=max_grad_norm,
        execution=execution,
        gradient_accumulation_steps=gradient_accumulation_steps,
        precision=precision,
        trainable_filter=trainable_filter,
    )
    donation = "all-except-first" if donate_state else "none"

    @eqx.filter_jit(donate=donation)
    def compiled_step(
        inputs: tuple[Any, PRNGKeyArray | None],
        state: TrainState,
    ) -> StepResult:
        batch, key = inputs
        return train_step_body(state, batch, key)

    def train_step(
        state: TrainState,
        batch: Any,
        key: PRNGKeyArray | None = None,
    ) -> StepResult:
        if gradient_accumulation_steps > 1:
            _validate_accumulation_batch(
                batch,
                gradient_accumulation_steps,
                task=task,
            )
        return compiled_step((batch, key), state)

    return train_step
