"""Native BERT sequence-classification scorer."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.models.components import Linear, dropout

from .model import BertBatch, BertEncoder


class BertScorer(eqx.Module):
    """A BERT pooler followed by a scalar sequence-classification head."""

    backbone: BertEncoder
    classifier: Linear
    classifier_dropout_probability: float = eqx.field(static=True)

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        """Load a scorer and its paired-text processor from one HF artifact."""

        from .scoring_loading import load_bert_scorer

        return load_bert_scorer(model_name_or_path, **options)

    def logits(
        self,
        inputs: BertBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch output"]:
        if key is None:
            backbone_key = classifier_key = None
        else:
            backbone_key, classifier_key = jax.random.split(key)
        pooled = self.backbone.pooler_output(inputs, key=backbone_key)
        pooled = dropout(
            pooled,
            self.classifier_dropout_probability,
            key=classifier_key,
        )
        return self.classifier(pooled).astype(jnp.float32)


__all__ = ["BertScorer"]
