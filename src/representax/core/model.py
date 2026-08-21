"""Model-neutral representation encoder contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.precision import (
    activation_inputs,
    active_model_for_compute,
    objective_output,
)


class Route(StrEnum):
    """Semantic route through a representation model."""

    GENERIC = "generic"
    QUERY = "query"
    DOCUMENT = "document"


class Modality(StrEnum):
    """Input modalities understood by Representax."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FUSED = "fused"


class EncoderMetadata(eqx.Module):
    """Static identity and capabilities attached to an encoder tree."""

    model_id: str = eqx.field(static=True)
    revision: str = eqx.field(static=True)
    output_dimension: int = eqx.field(static=True)
    routes: frozenset[Route] = eqx.field(static=True)
    modalities: frozenset[Modality] = eqx.field(static=True)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.revision:
            raise ValueError("revision must be non-empty")
        if self.output_dimension <= 0:
            raise ValueError("output_dimension must be positive")
        if not self.routes:
            raise ValueError("an encoder must support at least one route")
        if not self.modalities:
            raise ValueError("an encoder must support at least one modality")


@runtime_checkable
class Encoder(Protocol):
    """Structural boundary implemented by native Equinox encoder trees."""

    metadata: EncoderMetadata

    def encode(
        self,
        inputs: Any,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]: ...


@runtime_checkable
class LayerwiseEncoder(Encoder, Protocol):
    """Encoder that exposes postprocessed representations at every depth."""

    def encode_layers(
        self,
        inputs: Any,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "layer batch representation"]: ...


def _metadata(model: Any) -> EncoderMetadata:
    if not isinstance(model, eqx.Module):
        raise TypeError("encoders must be Equinox module trees")
    metadata = getattr(model, "metadata", None)
    if not isinstance(metadata, EncoderMetadata):
        raise TypeError("encoders must expose EncoderMetadata")
    if not callable(getattr(model, "encode", None)):
        raise TypeError("encoders must implement encode")
    return metadata


def encode(
    model: Encoder,
    inputs: Any,
    *,
    route: Route = Route.GENERIC,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "batch representation"]:
    """Encode an array PyTree and enforce the shared representation contract."""

    metadata = _metadata(model)
    route = Route(route)
    if route not in metadata.routes:
        raise ValueError(f"{metadata.model_id} does not support route {route.value!r}")
    compute_model = active_model_for_compute(model)
    result = jnp.asarray(
        compute_model.encode(activation_inputs(inputs), route=route, key=key)
    )
    if result.ndim != 2:
        raise ValueError("encoder output must have shape [batch, dimension]")
    if not jnp.issubdtype(result.dtype, jnp.floating):
        raise TypeError("encoder output must have a floating dtype")
    if result.shape[1] != metadata.output_dimension:
        raise ValueError(
            f"{metadata.model_id} declares output_dimension="
            f"{metadata.output_dimension} but returned {result.shape[1]}"
        )
    return objective_output(result)


def encode_layers(
    model: LayerwiseEncoder,
    inputs: Any,
    *,
    route: Route = Route.GENERIC,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "layer batch representation"]:
    """Encode every available depth and enforce a stable layer-major contract."""

    metadata = _metadata(model)
    route = Route(route)
    if route not in metadata.routes:
        raise ValueError(f"{metadata.model_id} does not support route {route.value!r}")
    compute_model = active_model_for_compute(model)
    layerwise = getattr(compute_model, "encode_layers", None)
    if not callable(layerwise):
        raise TypeError(
            f"{metadata.model_id} does not expose layerwise representations"
        )
    result = jnp.asarray(layerwise(activation_inputs(inputs), route=route, key=key))
    if result.ndim != 3:
        raise ValueError(
            "layerwise encoder output must have shape [layer, batch, dimension]"
        )
    if not jnp.issubdtype(result.dtype, jnp.floating):
        raise TypeError("layerwise encoder output must have a floating dtype")
    if result.shape[0] < 2:
        raise ValueError("layerwise encoders must expose a prior and final layer")
    if result.shape[2] != metadata.output_dimension:
        raise ValueError(
            f"{metadata.model_id} declares output_dimension="
            f"{metadata.output_dimension} but returned {result.shape[2]}"
        )
    return objective_output(result)


class BoundEncoder(eqx.Module):
    """An encoder with a static semantic route, suitable for JAX transforms."""

    model: Encoder
    route: Route = eqx.field(static=True)

    def __call__(
        self,
        inputs: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        return encode(self.model, inputs, route=self.route, key=key)


def bind(model: Encoder, *, route: Route) -> BoundEncoder:
    """Bind one route without creating duplicate query/document APIs."""

    return BoundEncoder(model=model, route=Route(route))
