"""Runtime denoising autoencoder task."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import LossOutput, Route, encode

if TYPE_CHECKING:
    from representax.models import DenoisingAutoEncoder

from .batch import DenoisingBatch
from .losses import denoising_autoencoder_loss_terms


class DenoisingAutoEncoderTask(eqx.Module):
    """Reconstruct clean token sequences from damaged-input representations."""

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "valid_tokens": "sum",
        "valid_examples": "sum",
    }

    pad_token_id: int = eqx.field(static=True)
    route: Route = eqx.field(static=True, default=Route.GENERIC)

    def __post_init__(self) -> None:
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")

    def accumulation_weight(self, batch: DenoisingBatch) -> Array:
        targets = batch.target_input_ids[:, 1:]
        token_valid = (targets != self.pad_token_id) & batch.valid[:, None]
        return jnp.sum(token_valid).astype(jnp.float32)

    def loss(
        self,
        model: DenoisingAutoEncoder,
        batch: DenoisingBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if key is None:
            encoder_key = decoder_key = None
        else:
            encoder_key, decoder_key = jax.random.split(key)
        representation = encode(
            model.encoder,
            batch.damaged,
            route=self.route,
            key=encoder_key,
        )
        decoder_input_ids = batch.target_input_ids[:, :-1]
        target_ids = batch.target_input_ids[:, 1:]
        logits = model.decoder.decode(
            decoder_input_ids,
            encoder_memory=representation[:, None, :],
            key=decoder_key,
        )
        terms = denoising_autoencoder_loss_terms(
            logits,
            target_ids,
            pad_token_id=self.pad_token_id,
            row_valid=batch.valid,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "valid_tokens": jnp.sum(terms.token_valid),
                "valid_examples": jnp.sum(batch.valid),
            },
        )
