"""Encoder MLM with additive, token-weighted gradient accumulation."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import LossOutput

from .batch import MaskedLanguageModelBatch
from .losses import masked_language_model_loss_terms


class MaskedLanguageModelTask(eqx.Module):
    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "accuracy": "mean",
        "valid_tokens": "sum",
    }

    def accumulation_weight(self, batch: MaskedLanguageModelBatch) -> Array:
        return jnp.sum((batch.labels != -100) & batch.valid[:, None]).astype(
            jnp.float32
        )

    def loss(
        self,
        model: Any,
        batch: MaskedLanguageModelBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        positions = jnp.where(
            (batch.labels != -100) & batch.valid[:, None], batch.positions, 0
        )
        terms = masked_language_model_loss_terms(
            model.predict_masked(batch.inputs, positions, key=key),
            batch.labels,
            row_valid=batch.valid,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "accuracy": terms.accuracy,
                "valid_tokens": jnp.sum(terms.token_valid),
            },
        )
