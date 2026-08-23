"""Native scanned Equinox DistilBERT encoder."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.bert import BertLayer, BertLayerStack
from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    dropout,
    embedding_lookup,
    l2_normalize,
    mean_pool,
    rematerialize,
)
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import DistilBertConfig


class DistilBertBatch(eqx.Module):
    """Token IDs or embedded tokens accepted by native DistilBERT."""

    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    input_ids: Int[Array, "batch sequence"] | None = None
    inputs_embeds: Float[Array, "batch sequence hidden"] | None = None

    def __post_init__(self) -> None:
        if (self.input_ids is None) == (self.inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if self.attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if self.input_ids is not None:
            if self.input_ids.shape != self.attention_mask.shape:
                raise ValueError("input_ids and attention_mask must align")
            if not jnp.issubdtype(self.input_ids.dtype, jnp.integer):
                raise TypeError("input_ids must have an integer dtype")
        else:
            assert self.inputs_embeds is not None
            if self.inputs_embeds.ndim != 3:
                raise ValueError("inputs_embeds must have [batch, sequence, hidden]")
            if self.inputs_embeds.shape[:2] != self.attention_mask.shape:
                raise ValueError("inputs_embeds and attention_mask must align")


class DistilBertEmbeddings(eqx.Module):
    word: Float[Array, "vocabulary hidden"]
    position: Float[Array, "position hidden"]
    norm: LayerNorm

    @classmethod
    def init(
        cls,
        config: DistilBertConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> DistilBertEmbeddings:
        word_key, position_key = jax.random.split(key)
        word = config.initializer_range * jax.random.normal(
            word_key,
            (config.vocab_size, config.hidden_size),
            dtype=dtype,
        )
        word = word.at[config.pad_token_id].set(0)
        return cls(
            word=word,
            position=config.initializer_range
            * jax.random.normal(
                position_key,
                (config.max_position_embeddings, config.hidden_size),
                dtype=dtype,
            ),
            norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        batch: DistilBertBatch,
        *,
        config: DistilBertConfig,
        key: PRNGKeyArray | None,
    ) -> Float[Array, "batch sequence hidden"]:
        sequence = batch.attention_mask.shape[1]
        if sequence > config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if batch.input_ids is None:
            assert batch.inputs_embeds is not None
            hidden = batch.inputs_embeds
        else:
            hidden = embedding_lookup(self.word, batch.input_ids)
            hidden = jnp.where(
                (batch.input_ids == config.pad_token_id)[..., None],
                jax.lax.stop_gradient(hidden),
                hidden,
            )
        positions = jnp.arange(sequence, dtype=jnp.int32)[None, :]
        hidden = hidden + embedding_lookup(self.position, positions)
        return dropout(
            self.norm(hidden),
            config.hidden_dropout_probability,
            key=key,
        )


class DistilBertTower(eqx.Module):
    embeddings: DistilBertEmbeddings
    layers: BertLayerStack
    config: DistilBertConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: DistilBertConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> DistilBertTower:
        keys = jax.random.split(key, config.num_hidden_layers + 1)
        return cls(
            embeddings=DistilBertEmbeddings.init(config, key=keys[0], dtype=dtype),
            layers=BertLayerStack.from_layers(
                tuple(
                    BertLayer.init(config, key=keys[index + 1], dtype=dtype)
                    for index in range(config.num_hidden_layers)
                )
            ),
            config=config,
        )

    def __call__(
        self,
        batch: DistilBertBatch,
        *,
        key: PRNGKeyArray | None,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "batch sequence hidden"]:
        if key is None:
            embedding_key = None
            layer_keys = jax.random.split(jax.random.key(0), self.layers.depth)
        else:
            embedding_key, layer_key = jax.random.split(key)
            layer_keys = jax.random.split(layer_key, self.layers.depth)
        hidden = self.embeddings(batch, config=self.config, key=embedding_key)
        hidden = hidden.astype(compute_dtype)
        if self.layers.blocks is None:
            return hidden
        attention_mask = batch.attention_mask.astype(bool)
        training = key is not None

        def apply_layer(
            carry: Float[Array, "batch sequence hidden"],
            values: tuple[BertLayer, PRNGKeyArray],
        ) -> tuple[Float[Array, "batch sequence hidden"], None]:
            layer, layer_key = values
            output = layer(
                carry,
                attention_mask,
                config=self.config,
                key=layer_key if training else None,
                implementation=attention_implementation,
            )
            return output, None

        hidden, _ = jax.lax.scan(
            rematerialize(apply_layer, rematerialization),
            hidden,
            (self.layers.blocks, layer_keys),
        )
        return hidden


class DistilBertEncoder(eqx.Module):
    """DistilBERT token backbone with a standard representation fallback."""

    tower: DistilBertTower
    metadata: EncoderMetadata
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: DistilBertConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/distilbert",
        revision: str = "random-init",
    ) -> DistilBertEncoder:
        return cls(
            tower=DistilBertTower.init(config, key=key, dtype=parameter_dtype),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT}),
            ),
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )

    @staticmethod
    def make_batch(
        *,
        input_ids: Int[Array, "batch sequence"],
        attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
        token_type_ids: Int[Array, "batch sequence"] | None = None,
    ) -> DistilBertBatch:
        if token_type_ids is not None and bool(jnp.any(token_type_ids != 0)):
            raise ValueError("DistilBERT accepts only the tokenizer's zero segments")
        return DistilBertBatch(input_ids=input_ids, attention_mask=attention_mask)

    def hidden_states(
        self,
        inputs: DistilBertBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        if not isinstance(inputs, DistilBertBatch):
            raise TypeError("DistilBERT inputs must be DistilBertBatch")
        return self.tower(
            inputs,
            key=key,
            compute_dtype=active_compute_dtype(self.compute_dtype),
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def encode(
        self,
        inputs: DistilBertBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        return l2_normalize(
            mean_pool(self.hidden_states(inputs, key=key), inputs.attention_mask)
        )


__all__ = [
    "DistilBertBatch",
    "DistilBertEmbeddings",
    "DistilBertEncoder",
    "DistilBertTower",
]
