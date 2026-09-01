"""Pure-JAX LeJEPA invariance and SIGReg objectives."""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def invariance_loss(
    projections: Float[Array, "batch view dimension"],
    valid: Bool[Array, "batch view"],
    *,
    global_views: int,
) -> Float[Array, ""]:
    if global_views <= 0 or global_views > projections.shape[1]:
        raise ValueError("global_views must select a non-empty prefix of views")
    weights = valid.astype(jnp.float32)
    global_weights = weights[:, :global_views]
    center = jnp.sum(
        projections[:, :global_views] * global_weights[..., None],
        axis=1,
        keepdims=True,
    )
    center = center / jnp.maximum(
        jnp.sum(global_weights, axis=1, keepdims=True), 1.0
    )[..., None]
    errors = jnp.mean(jnp.square(projections - center), axis=-1)
    return jnp.sum(jnp.where(valid, errors, 0.0)) / jnp.maximum(
        jnp.sum(weights), 1.0
    )


def sigreg_loss(
    projections: Float[Array, "batch view dimension"],
    valid: Bool[Array, "batch view"],
    directions: Float[Array, "dimension slice"],
    *,
    knots: int = 17,
    max_frequency: float = 3.0,
) -> Float[Array, ""]:
    """Sketched isotropic-Gaussian regularization via characteristic functions."""

    if projections.shape[:2] != valid.shape:
        raise ValueError("SIGReg projections and validity must align")
    if directions.shape[0] != projections.shape[-1]:
        raise ValueError("SIGReg directions must match the projection dimension")
    t = jnp.linspace(0.0, max_frequency, knots, dtype=jnp.float32)
    dt = max_frequency / (knots - 1) if knots > 1 else max_frequency
    weights = jnp.full((knots,), 2.0 * dt, dtype=jnp.float32)
    weights = weights.at[jnp.asarray((0, knots - 1))].set(dt)
    phi = jnp.exp(-jnp.square(t) / 2.0)
    weights = weights * phi
    normalized = directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=0, keepdims=True), 1e-12
    )
    sliced = jnp.einsum("bvd,ds->vbs", projections, normalized)
    values = sliced[..., None] * t
    view_valid = valid.T.astype(jnp.float32)
    counts = jnp.maximum(jnp.sum(view_valid, axis=1), 1.0)
    cosine = jnp.sum(jnp.cos(values) * view_valid[:, :, None, None], axis=1)
    sine = jnp.sum(jnp.sin(values) * view_valid[:, :, None, None], axis=1)
    cosine = cosine / counts[:, None, None]
    sine = sine / counts[:, None, None]
    error = jnp.square(cosine - phi) + jnp.square(sine)
    statistic = jnp.sum(error * weights, axis=-1) * counts[:, None]
    return jnp.mean(statistic)


__all__ = ["invariance_loss", "sigreg_loss"]
