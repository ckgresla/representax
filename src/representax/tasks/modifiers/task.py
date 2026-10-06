"""Composable dimensional and layerwise representation objectives."""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import (
    EncodeFunction,
    Encoder,
    LossOutput,
    RepresentationTask,
    encode,
    encode_layers,
)


def _selected_mask(
    count: int,
    selected_per_step: int,
    key: PRNGKeyArray | None,
) -> Bool[Array, " item"]:
    if selected_per_step == -1 or selected_per_step >= count:
        return jnp.ones((count,), dtype=jnp.bool_)
    if key is None:
        raise ValueError("random modifier sampling requires a JAX key")
    selected = jax.random.choice(
        key,
        count,
        shape=(selected_per_step,),
        replace=False,
    )
    return jnp.zeros((count,), dtype=jnp.bool_).at[selected].set(True)


def _shrink(value: Array, dimension: int) -> Array:
    if not eqx.is_inexact_array(value) or value.ndim < 2:
        raise TypeError("task representations must be floating arrays with rank >= 2")
    if dimension > value.shape[-1]:
        raise ValueError("Matryoshka dimension exceeds representation dimension")
    truncated = value[..., :dimension].astype(jnp.float32)
    norm = jnp.linalg.norm(truncated, axis=-1, keepdims=True)
    return truncated / jnp.maximum(norm, jnp.asarray(1e-12, truncated.dtype))


def _dimension_batch(
    task: RepresentationTask[Any],
    batch: Any,
    *,
    dimension: int,
    full_dimension: int,
) -> Any:
    adapter = getattr(task, "dimension_batch", None)
    if not callable(adapter):
        return batch
    return adapter(batch, dimension=dimension, full_dimension=full_dimension)


class MatryoshkaTask(eqx.Module):
    """Apply one representation task at multiple normalized prefix dimensions."""

    task: Any
    dimensions: tuple[int, ...] = eqx.field(static=True)
    weights: tuple[float, ...] = eqx.field(static=True)
    dimensions_per_step: int = eqx.field(static=True, default=-1)

    def __init__(
        self,
        task: RepresentationTask[Encoder],
        dimensions: tuple[int, ...],
        *,
        weights: tuple[float, ...] | None = None,
        dimensions_per_step: int = -1,
    ) -> None:
        if not isinstance(task, RepresentationTask):
            raise TypeError("Matryoshka requires a representation task")
        if not dimensions or any(dimension <= 0 for dimension in dimensions):
            raise ValueError("Matryoshka dimensions must be positive and non-empty")
        resolved_weights = weights or tuple(1.0 for _ in dimensions)
        if len(resolved_weights) != len(dimensions):
            raise ValueError("Matryoshka weights must match dimensions")
        if any(not math.isfinite(weight) or weight <= 0 for weight in resolved_weights):
            raise ValueError("Matryoshka weights must be finite and positive")
        if dimensions_per_step == 0 or dimensions_per_step < -1:
            raise ValueError("dimensions_per_step must be -1 or positive")
        ordered = sorted(
            zip(dimensions, resolved_weights, strict=True),
            reverse=True,
        )
        self.task = task
        self.dimensions = tuple(dimension for dimension, _ in ordered)
        self.weights = tuple(float(weight) for _, weight in ordered)
        self.dimensions_per_step = dimensions_per_step

    def representations(
        self,
        model: Encoder,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> Any:
        return self.task.representations(
            model,
            batch,
            key=key,
            encode_fn=encode_fn,
        )

    def loss_from_representations(
        self,
        representations: Any,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
        row_chunk_size: int | None = None,
    ) -> LossOutput:
        leaves = [
            leaf for leaf in jax.tree.leaves(representations) if eqx.is_array(leaf)
        ]
        if not leaves:
            raise ValueError("Matryoshka task representations must contain arrays")
        full_dimension = leaves[0].shape[-1]
        if any(leaf.shape[-1] != full_dimension for leaf in leaves):
            raise ValueError("Matryoshka representations must share a final dimension")
        if self.dimensions[0] > full_dimension:
            raise ValueError("Matryoshka dimension exceeds representation dimension")

        selected = _selected_mask(
            len(self.dimensions),
            self.dimensions_per_step,
            key,
        )
        losses = []
        for dimension in self.dimensions:
            shrunk = jax.tree.map(
                lambda value, dim=dimension: _shrink(value, dim),
                representations,
            )
            dimension_batch = _dimension_batch(
                self.task,
                batch,
                dimension=dimension,
                full_dimension=full_dimension,
            )
            if row_chunk_size is None:
                dimension_output = self.task.loss_from_representations(
                    shrunk,
                    dimension_batch,
                )
            else:
                from representax.tasks.guided import GISTTask
                from representax.tasks.retrieval import MNRTask

                if not isinstance(self.task, (MNRTask, GISTTask)):
                    raise TypeError(
                        "score-row chunking requires an MNR or GIST base task"
                    )
                dimension_output = self.task.loss_from_representations(
                    shrunk,
                    dimension_batch,
                    row_chunk_size=row_chunk_size,
                )
            losses.append(dimension_output.loss)
        dimension_losses = jnp.stack(losses)
        weights = jnp.asarray(self.weights, dtype=jnp.float32)
        effective_weights = jnp.where(selected, weights, 0.0)
        return LossOutput(
            loss=jnp.sum(effective_weights * dimension_losses),
            metrics={
                "dimension_losses": dimension_losses,
                "selected_dimensions": selected,
            },
        )

    def loss(
        self,
        model: Encoder,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if self.dimensions_per_step == -1 or self.dimensions_per_step >= len(
            self.dimensions
        ):
            representation_key = selection_key = key
        elif key is None:
            raise ValueError("random Matryoshka sampling requires a JAX key")
        else:
            representation_key, selection_key = jax.random.split(key)
        representations = self.representations(
            model,
            batch,
            key=representation_key,
        )
        return self.loss_from_representations(
            representations,
            batch,
            key=selection_key,
        )


def _flatten_embeddings(representations: Any) -> Float[Array, "row representation"]:
    leaves = [leaf for leaf in jax.tree.leaves(representations) if eqx.is_array(leaf)]
    if not leaves:
        raise ValueError("adaptive-layer representations must contain arrays")
    dimension = leaves[0].shape[-1]
    if any(leaf.shape[-1] != dimension for leaf in leaves):
        raise ValueError("adaptive-layer representations must share a dimension")
    return jnp.concatenate(
        tuple(leaf.reshape((-1, dimension)) for leaf in leaves),
        axis=0,
    )


def _task_loss(
    task: RepresentationTask[Any],
    representations: Any,
    batch: Any,
    *,
    key: PRNGKeyArray | None,
) -> LossOutput:
    if isinstance(task, MatryoshkaTask):
        return task.loss_from_representations(representations, batch, key=key)
    return task.loss_from_representations(representations, batch)


class AdaptiveLayerTask(eqx.Module):
    """Apply a representation task to final and selected prior encoder depths."""

    task: Any
    layers_per_step: int = eqx.field(static=True, default=1)
    final_layer_weight: float = eqx.field(static=True, default=1.0)
    prior_layer_weight: float = eqx.field(static=True, default=1.0)
    kl_divergence_weight: float = eqx.field(static=True, default=1.0)
    kl_temperature: float = eqx.field(static=True, default=0.3)

    def __post_init__(self) -> None:
        if not isinstance(self.task, RepresentationTask):
            raise TypeError("adaptive-layer training requires a representation task")
        if self.layers_per_step == 0 or self.layers_per_step < -1:
            raise ValueError("layers_per_step must be -1 or positive")
        for name, value in (
            ("final_layer_weight", self.final_layer_weight),
            ("prior_layer_weight", self.prior_layer_weight),
            ("kl_divergence_weight", self.kl_divergence_weight),
            ("kl_temperature", self.kl_temperature),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def loss(
        self,
        model: Encoder,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if key is None:
            representation_key = selection_key = None
        else:
            representation_key, selection_key = jax.random.split(key)
        layerwise = self.task.representations(
            model,
            batch,
            key=representation_key,
            encode_fn=encode_layers,
        )
        leaves = [leaf for leaf in jax.tree.leaves(layerwise) if eqx.is_array(leaf)]
        if not leaves or any(leaf.ndim < 3 for leaf in leaves):
            raise ValueError("adaptive-layer representations must be layer-major")
        layer_count = leaves[0].shape[0]
        if layer_count < 2 or any(leaf.shape[0] != layer_count for leaf in leaves):
            raise ValueError("adaptive-layer representations must share layer depth")

        prior_count = layer_count - 1
        selected = _selected_mask(prior_count, self.layers_per_step, selection_key)
        if key is None:
            loss_keys = (None,) * layer_count
        else:
            assert selection_key is not None
            loss_keys = tuple(jax.random.split(selection_key, layer_count))

        final_representations = jax.tree.map(lambda value: value[-1], layerwise)
        final_output = _task_loss(
            self.task,
            final_representations,
            batch,
            key=loss_keys[-1],
        )
        final_embeddings = _flatten_embeddings(final_representations)
        if self.kl_temperature > 0:
            final_probabilities = jax.nn.softmax(
                final_embeddings / self.kl_temperature,
                axis=-1,
            )

        prior_losses = []
        kl_losses = []
        selected_count = jnp.maximum(jnp.sum(selected), 1).astype(jnp.float32)
        for layer_index in range(prior_count):
            layer_representations = jax.tree.map(
                lambda value, index=layer_index: value[index],
                layerwise,
            )
            layer_output = _task_loss(
                self.task,
                layer_representations,
                batch,
                key=loss_keys[layer_index],
            )
            prior_losses.append(layer_output.loss)
            if self.kl_temperature > 0:
                layer_embeddings = _flatten_embeddings(layer_representations)
                log_probabilities = jax.nn.log_softmax(
                    layer_embeddings / self.kl_temperature,
                    axis=-1,
                )
                kl_losses.append(
                    jnp.sum(
                        final_probabilities
                        * (jnp.log(final_probabilities) - log_probabilities)
                    )
                    / final_probabilities.shape[0]
                )

        prior_losses_array = jnp.stack(prior_losses)
        layer_denominators = jnp.arange(1, prior_count + 1, dtype=jnp.float32)
        prior_weights = jnp.where(
            selected,
            self.prior_layer_weight / layer_denominators / selected_count,
            0.0,
        )
        loss = self.final_layer_weight * final_output.loss + jnp.sum(
            prior_weights * prior_losses_array
        )
        if self.kl_temperature > 0:
            kl_losses_array = jnp.stack(kl_losses)
            kl_weight = self.kl_temperature * self.kl_divergence_weight
            loss = loss + jnp.sum(jnp.where(selected, kl_weight * kl_losses_array, 0.0))
        else:
            kl_losses_array = jnp.zeros((prior_count,), dtype=jnp.float32)
        return LossOutput(
            loss=loss,
            metrics={
                "final_layer_loss": final_output.loss,
                "prior_layer_losses": prior_losses_array,
                "layer_kl_divergences": kl_losses_array,
                "selected_prior_layers": selected,
            },
        )


class Matryoshka2dTask(eqx.Module):
    """Sentence Transformers 2D composition: dimensions within encoder depths."""

    task: AdaptiveLayerTask

    def __init__(
        self,
        task: RepresentationTask[Encoder],
        dimensions: tuple[int, ...],
        *,
        weights: tuple[float, ...] | None = None,
        dimensions_per_step: int = 1,
        layers_per_step: int = 1,
        final_layer_weight: float = 1.0,
        prior_layer_weight: float = 1.0,
        kl_divergence_weight: float = 1.0,
        kl_temperature: float = 0.3,
    ) -> None:
        self.task = AdaptiveLayerTask(
            task=MatryoshkaTask(
                task,
                dimensions,
                weights=weights,
                dimensions_per_step=dimensions_per_step,
            ),
            layers_per_step=layers_per_step,
            final_layer_weight=final_layer_weight,
            prior_layer_weight=prior_layer_weight,
            kl_divergence_weight=kl_divergence_weight,
            kl_temperature=kl_temperature,
        )

    def loss(
        self,
        model: Encoder,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        return self.task.loss(model, batch, key=key)
