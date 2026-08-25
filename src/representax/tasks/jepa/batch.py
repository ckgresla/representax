"""Multi-view batches for JEPA-style self-supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool

if TYPE_CHECKING:
    from representax.models.processing import Processor


class JEPABatch(eqx.Module):
    views: Any
    valid: Bool[Array, "batch view"]

    def __post_init__(self) -> None:
        if self.valid.ndim != 2 or self.valid.dtype != jnp.bool_:
            raise TypeError("JEPA validity must be a boolean [batch, view] matrix")
        if self.valid.shape[1] < 2:
            raise ValueError("JEPA requires at least two views per sample")
        leaves = [
            leaf
            for leaf in jax.tree.leaves(self.views)
            if isinstance(leaf, (jax.Array, np.ndarray))
        ]
        if not leaves or any(leaf.shape[:2] != self.valid.shape for leaf in leaves):
            raise ValueError("JEPA view leaves must begin with [batch, view]")


class JEPACollator:
    """Apply the model processor to an already selected finite set of views."""

    def __init__(
        self,
        *,
        processor: Processor,
        views_per_sample: int,
        views_field: str = "views",
    ) -> None:
        if views_per_sample < 2:
            raise ValueError("views_per_sample must be at least two")
        self.processor = processor
        self.views_per_sample = views_per_sample
        self.views_field = views_field

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> JEPABatch:
        artifacts: list[Any] = []
        for example in examples:
            views = list(example[self.views_field])
            if len(views) != self.views_per_sample:
                raise ValueError("JEPA row has the wrong finite view count")
            artifacts.extend(views)
        processed = self.processor(artifacts)

        def reshape(value: Any) -> Any:
            if not isinstance(value, (jax.Array, np.ndarray)):
                return value
            return value.reshape(
                (len(examples), self.views_per_sample, *value.shape[1:])
            )

        return JEPABatch(
            views=jax.tree.map(reshape, processed),
            valid=jnp.ones((len(examples), self.views_per_sample), dtype=jnp.bool_),
        )


__all__ = ["JEPABatch", "JEPACollator"]
