"""Reference-matched dense V-JEPA 2.1 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import LossOutput

if TYPE_CHECKING:
    from representax.models.vjepa2_1 import VJEPA2_1Model


class VJEPA2_1Batch(eqx.Module):
    """Static image/video tensors and aligned context/target token selections."""

    pixels: Float[Array, "batch channel *media"]
    context_ids: Int[Array, "batch mask context"]
    target_ids: Int[Array, "batch mask target"]
    context_valid: Bool[Array, "batch mask context"]
    target_valid: Bool[Array, "batch mask target"]

    def __post_init__(self) -> None:
        if self.pixels.ndim not in (4, 5):
            raise ValueError("V-JEPA pixels must be images or videos")
        if self.context_ids.ndim != 3 or self.target_ids.ndim != 3:
            raise ValueError("V-JEPA token IDs must be [batch, mask, token]")
        if self.context_ids.shape != self.context_valid.shape:
            raise ValueError("context validity must align with context IDs")
        if self.target_ids.shape != self.target_valid.shape:
            raise ValueError("target validity must align with target IDs")
        if self.context_ids.shape[:2] != self.target_ids.shape[:2]:
            raise ValueError("context and target masks must share batch/mask axes")
        if self.pixels.shape[0] != self.context_ids.shape[0]:
            raise ValueError("pixels and masks must share the batch axis")


def mask_distance_weights(
    context_ids: Int[Array, "batch mask context"],
    target_ids: Int[Array, "batch mask target"],
    *,
    grid_height: int,
    grid_width: int,
    offset_scale: float = 1.0,
) -> Float[Array, "batch mask context"]:
    """Return Eq. 3 context weights from the nearest masked tubelet.

    The official implementation computes Euclidean distance, applies a square
    root so the decay is less aggressive, then divides the context error by the
    result. Consequently the final weight is the inverse fourth root of squared
    coordinate distance.
    """

    from representax.models.vjepa2_1.model import token_positions

    context = jnp.stack(
        token_positions(
            context_ids,
            grid_height=grid_height,
            grid_width=grid_width,
        ),
        axis=-1,
    ).astype(jnp.float32)
    target = jnp.stack(
        token_positions(
            target_ids,
            grid_height=grid_height,
            grid_width=grid_width,
        ),
        axis=-1,
    ).astype(jnp.float32)
    squared = jnp.sum(
        jnp.square(context[..., :, None, :] - target[..., None, :, :]),
        axis=-1,
    )
    euclidean = jnp.sqrt(jnp.min(squared, axis=-1)) * offset_scale
    softened = jnp.sqrt(euclidean)
    return jnp.reciprocal(jnp.maximum(softened, 1e-12))


def _gather_hierarchical(
    values: Array,
    token_ids: Array,
) -> Array:
    """Gather `[B,T,L*D]` target features for every `[B,M,N]` mask."""

    batch, masks, tokens = token_ids.shape
    repeated = jnp.broadcast_to(
        values[:, None],
        (batch, masks, *values.shape[1:]),
    )
    return jnp.take_along_axis(
        repeated,
        token_ids[..., None],
        axis=2,
    ).reshape((batch, masks, tokens, values.shape[-1]))


def dense_prediction_loss(
    prediction: Array,
    target: Array,
    valid: Array,
    *,
    token_weights: Array | None = None,
) -> Float[Array, ""]:
    """Exact valid-token-weighted L1 mean over deep-supervision features."""

    error = jnp.abs(prediction.astype(jnp.float32) - target.astype(jnp.float32))
    weights = valid.astype(jnp.float32)
    if token_weights is not None:
        weights = weights * token_weights.astype(jnp.float32)
    numerator = jnp.sum(error * weights[..., None])
    denominator = jnp.maximum(
        jnp.sum(valid.astype(jnp.float32)) * error.shape[-1],
        1.0,
    )
    return numerator / denominator


class VJEPA2_1Task(eqx.Module):
    """Dense masked/context prediction with a post-step EMA target encoder."""

    context_weight: float = eqx.field(static=True, default=0.5)
    ema_start: float = eqx.field(static=True, default=0.99925)
    ema_end: float = eqx.field(static=True, default=0.99925)
    ema_steps: int = eqx.field(static=True, default=1)
    offset_context_loss: bool = eqx.field(static=True, default=False)
    supports_gradient_accumulation: bool = eqx.field(static=True, default=False)

    def _momentum(self, step: Array) -> Array:
        progress = jnp.minimum(step.astype(jnp.float32) / max(self.ema_steps, 1), 1.0)
        return self.ema_start + progress * (self.ema_end - self.ema_start)

    def loss(
        self,
        model: VJEPA2_1Model,
        batch: VJEPA2_1Batch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        del key
        online_tokens, _ = model.online.tokenize(batch.pixels)
        target_tokens, target_ids = model.target.tokenize(batch.pixels)
        _, target_levels = model.target.encode_tokens(target_tokens, target_ids)
        target_levels = jax.vmap(
            lambda values: (
                (values - jnp.mean(values, axis=-1, keepdims=True))
                * jax.lax.rsqrt(
                    jnp.var(values, axis=-1, keepdims=True)
                    + model.target.config.layer_norm_epsilon
                )
            )
        )(target_levels)
        target_features = jax.lax.stop_gradient(
            jnp.concatenate(tuple(target_levels), axis=-1)
        )
        is_video = batch.pixels.ndim == 5

        def predict_one_mask(
            context_ids: Array,
            target_ids: Array,
            context_valid: Array,
            target_valid: Array,
        ) -> tuple[Array, Array]:
            context_tokens = jnp.take_along_axis(
                online_tokens,
                context_ids[..., None],
                axis=1,
            )
            _, context_levels = model.online.encode_tokens(
                context_tokens,
                context_ids,
                context_valid,
            )
            context_features = jnp.concatenate(tuple(context_levels), axis=-1)
            return model.predictor(
                context_features,
                context_ids,
                target_ids,
                is_video=is_video,
                context_valid=context_valid,
                target_valid=target_valid,
            )

        predicted_target, predicted_context = jax.vmap(
            predict_one_mask,
            in_axes=(1, 1, 1, 1),
            out_axes=1,
        )(
            batch.context_ids,
            batch.target_ids,
            batch.context_valid,
            batch.target_valid,
        )
        target_target = _gather_hierarchical(target_features, batch.target_ids)
        target_context = _gather_hierarchical(target_features, batch.context_ids)
        prediction_loss = dense_prediction_loss(
            predicted_target,
            target_target,
            batch.target_valid,
        )
        offset_scale = (
            1.0 / max(model.online.config.spatial_grid // 16, 1)
            if self.offset_context_loss
            else 1.0
        )
        distance_weights = mask_distance_weights(
            batch.context_ids,
            batch.target_ids,
            grid_height=model.online.config.spatial_grid,
            grid_width=model.online.config.spatial_grid,
            offset_scale=offset_scale,
        )
        context_loss = dense_prediction_loss(
            predicted_context,
            target_context,
            batch.context_valid,
            token_weights=distance_weights,
        )
        total = prediction_loss + self.context_weight * context_loss
        return LossOutput(
            loss=total,
            metrics={
                "prediction": prediction_loss,
                "context": context_loss,
                "context_weight": jnp.asarray(self.context_weight, jnp.float32),
                "valid_context_tokens": jnp.sum(batch.context_valid),
                "valid_target_tokens": jnp.sum(batch.target_valid),
            },
        )

    def post_update_model(
        self,
        previous_model: VJEPA2_1Model,
        optimized_model: VJEPA2_1Model,
        *,
        step: Array,
    ) -> VJEPA2_1Model:
        return previous_model.ema_update(optimized_model, self._momentum(step))


__all__ = [
    "VJEPA2_1Batch",
    "VJEPA2_1Task",
    "dense_prediction_loss",
    "mask_distance_weights",
]
