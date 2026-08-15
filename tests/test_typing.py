"""Shape-and-dtype annotations remain concrete and introspectable."""

from typing import get_type_hints

import jax
import jax.numpy as jnp

from representax.core import encode
from representax.models.modernvbert.model import _scaled_dot_product_attention
from representax.tasks.retrieval import mnr_loss_terms


def test_attention_annotations_describe_rank_and_dtype():
    annotations = get_type_hints(_scaled_dot_product_attention)

    query = jnp.zeros((2, 8, 4, 16), dtype=jnp.bfloat16)
    assert isinstance(query, annotations["query"])
    assert not isinstance(query[..., 0], annotations["query"])
    assert not isinstance(query.astype(jnp.int32), annotations["query"])


def test_mnr_annotations_distinguish_embeddings_and_relations():
    annotations = get_type_hints(mnr_loss_terms)

    embeddings = jnp.zeros((4, 32), dtype=jnp.float32)
    positive_mask = jnp.eye(4, dtype=jnp.bool_)
    assert isinstance(embeddings, annotations["query_embeddings"])
    assert isinstance(positive_mask, annotations["positive_mask"])
    assert not isinstance(positive_mask.astype(jnp.int32), annotations["positive_mask"])


def test_encoder_annotation_accepts_typed_prng_keys():
    annotation = get_type_hints(encode)["key"]

    assert isinstance(jax.random.key(0), annotation)
    assert isinstance(None, annotation)
