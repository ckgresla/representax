"""Native token-level text encoders for late-interaction retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PRNGKeyArray

from representax.core import (
    EncoderMetadata,
    LateInteractionRepresentation,
    Route,
)

from .components import Linear
from .processing import Processor


class LateInteractionTextEncoder(eqx.Module):
    """A native text backbone followed by a token projection.

    The backbone owns contextualization. This composition owns the ColBERT
    projection and route-dependent scoring mask; shared normalization remains
    at the public ``encode_late_interaction`` boundary.
    """

    backbone: Any
    projection: Linear
    metadata: EncoderMetadata
    skip_token_ids: tuple[int, ...] = eqx.field(static=True, default=())
    query_expansion: bool = eqx.field(static=True, default=False)

    @classmethod
    def load_from_hf(
        cls,
        model_name_or_path: str | Path,
        **options: Any,
    ) -> tuple[LateInteractionTextEncoder, Processor]:
        """Load native weights and ColBERT preprocessing from one HF artifact."""

        from representax.integrations.late_interaction import (
            load_late_interaction_text_model,
        )

        return load_late_interaction_text_model(model_name_or_path, **options)

    def save_to_hf(
        self,
        directory: str | Path,
        *,
        source_checkpoint: str | Path,
    ) -> Path:
        """Export a PyLate-compatible checkpoint and verify it independently."""

        from representax.integrations.late_interaction import (
            save_late_interaction_text_model,
        )

        return save_late_interaction_text_model(
            self,
            directory,
            source_checkpoint=source_checkpoint,
        )

    def encode_late_interaction(
        self,
        inputs: Any,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> LateInteractionRepresentation:
        del key
        hidden_states = getattr(self.backbone, "hidden_states", None)
        if not callable(hidden_states):
            raise TypeError("late-interaction text backbones must expose hidden_states")
        input_ids: Int[Array, "batch sequence"] | None = getattr(
            inputs, "input_ids", None
        )
        attention_mask: (
            Bool[Array, "batch sequence"] | Int[Array, "batch sequence"] | None
        ) = getattr(inputs, "attention_mask", None)
        if input_ids is None or attention_mask is None:
            raise TypeError(
                "late-interaction text inputs must expose input_ids and attention_mask"
            )
        values = self.projection(hidden_states(inputs))
        valid = attention_mask.astype(jnp.bool_)
        if route is Route.QUERY and self.query_expansion:
            valid = jnp.ones_like(valid)
        elif route is Route.DOCUMENT and self.skip_token_ids:
            token_ids = jnp.asarray(input_ids, dtype=jnp.int32)
            skipped = jnp.isin(
                token_ids,
                jnp.asarray(self.skip_token_ids, dtype=jnp.int32),
            )
            valid = valid & ~skipped
        return LateInteractionRepresentation(values=values, valid=valid)


__all__ = ["LateInteractionTextEncoder"]
