"""Native Equinox MPNet encoder compatible with Transformers 5.3."""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    activate,
    dot_product_attention,
    dropout,
    embedding_lookup,
    l2_normalize,
    mean_pool,
    rematerialize,
)
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import MPNetConfig


def create_mpnet_position_ids(
    input_ids: Int[Array, "batch sequence"],
    *,
    padding_index: int = 1,
) -> Int[Array, "batch sequence"]:
    """Create padding-aware MPNet positions exactly as Transformers does."""

    mask = (input_ids != padding_index).astype(input_ids.dtype)
    incremental = jnp.cumsum(mask, axis=1) * mask
    return incremental + padding_index


def mpnet_relative_position_bucket(
    relative_position: Int[Array, "*shape"],
    *,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> Int[Array, "*shape"]:
    """Map signed relative positions into MPNet's logarithmic buckets."""

    if num_buckets < 4 or num_buckets % 2:
        raise ValueError("num_buckets must be even and >= 4")
    if max_distance <= num_buckets // 4:
        raise ValueError("max_distance must exceed the exact bucket range")
    distance = -relative_position
    half = num_buckets // 2
    direction = (distance < 0).astype(jnp.int32) * half
    distance = jnp.abs(distance)
    max_exact = half // 2
    small = distance < max_exact
    safe_distance = jnp.maximum(distance, max_exact).astype(jnp.float32)
    logarithmic = max_exact + (
        jnp.log(safe_distance / max_exact)
        / math.log(max_distance / max_exact)
        * (half - max_exact)
    ).astype(jnp.int32)
    logarithmic = jnp.minimum(logarithmic, half - 1)
    return direction + jnp.where(small, distance, logarithmic).astype(jnp.int32)


class MPNetBatch(eqx.Module):
    """Token IDs or input embeddings and masks accepted by native MPNet."""

    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    input_ids: Int[Array, "batch sequence"] | None = None
    inputs_embeds: Float[Array, "batch sequence hidden"] | None = None
    position_ids: Int[Array, "#batch sequence"] | None = None

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
                raise ValueError(
                    "inputs_embeds must have shape [batch, sequence, hidden]"
                )
            if self.inputs_embeds.shape[:2] != self.attention_mask.shape:
                raise ValueError("inputs_embeds and attention_mask must align")
        if self.position_ids is not None:
            if self.position_ids.ndim != 2:
                raise ValueError("position_ids must have shape [batch, sequence]")
            if self.position_ids.shape[0] not in {1, self.attention_mask.shape[0]}:
                raise ValueError("position_ids batch must be one or match inputs")
            if self.position_ids.shape[1] != self.attention_mask.shape[1]:
                raise ValueError("position_ids and attention_mask must align")
            if not jnp.issubdtype(self.position_ids.dtype, jnp.integer):
                raise TypeError("position_ids must have an integer dtype")


class MPNetEmbeddings(eqx.Module):
    word: Float[Array, "vocabulary hidden"]
    position: Float[Array, "position hidden"]
    norm: LayerNorm

    @classmethod
    def init(
        cls,
        config: MPNetConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> MPNetEmbeddings:
        word_key, position_key = jax.random.split(key)

        def initialize(key: PRNGKeyArray, shape: tuple[int, int]) -> jax.Array:
            return config.initializer_range * jax.random.normal(key, shape, dtype=dtype)

        word = initialize(word_key, (config.vocab_size, config.hidden_size))
        word = word.at[config.pad_token_id].set(0)
        position = initialize(
            position_key,
            (config.max_position_embeddings, config.hidden_size),
        )
        position = position.at[config.pad_token_id].set(0)
        return cls(
            word=word,
            position=position,
            norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        batch: MPNetBatch,
        *,
        config: MPNetConfig,
        key: PRNGKeyArray | None,
    ) -> Float[Array, "batch sequence hidden"]:
        sequence = batch.attention_mask.shape[1]
        if sequence + config.pad_token_id >= config.max_position_embeddings:
            raise ValueError("sequence exceeds MPNet position embedding capacity")
        if batch.input_ids is None:
            assert batch.inputs_embeds is not None
            hidden = batch.inputs_embeds
            default_positions = jnp.arange(
                config.pad_token_id + 1,
                sequence + config.pad_token_id + 1,
                dtype=jnp.int32,
            )[None, :]
        else:
            hidden = embedding_lookup(self.word, batch.input_ids)
            padding = batch.input_ids == config.pad_token_id
            hidden = jnp.where(
                padding[..., None], jax.lax.stop_gradient(hidden), hidden
            )
            default_positions = create_mpnet_position_ids(
                batch.input_ids,
                padding_index=config.pad_token_id,
            )
        position_ids = (
            default_positions if batch.position_ids is None else batch.position_ids
        )
        positions = embedding_lookup(self.position, position_ids)
        position_padding = position_ids == config.pad_token_id
        positions = jnp.where(
            position_padding[..., None],
            jax.lax.stop_gradient(positions),
            positions,
        )
        hidden = self.norm(hidden + positions)
        return dropout(hidden, config.hidden_dropout_probability, key=key)


class MPNetSelfAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: MPNetConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> MPNetSelfAttention:
        keys = jax.random.split(key, 4)
        arguments = {
            "input_size": config.hidden_size,
            "output_size": config.hidden_size,
            "scale": config.initializer_range,
            "dtype": dtype,
            "bias": True,
        }
        return cls(
            query=Linear.init(key=keys[0], **arguments),
            key=Linear.init(key=keys[1], **arguments),
            value=Linear.init(key=keys[2], **arguments),
            output=Linear.init(key=keys[3], **arguments),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        attention_mask: Bool[Array, "batch sequence"],
        position_bias: Float[Array, "1 heads sequence sequence"],
        *,
        config: MPNetConfig,
        probability_key: PRNGKeyArray | None,
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        batch, sequence, _ = hidden.shape

        def project(
            projection: Linear,
        ) -> Float[Array, "batch sequence heads head"]:
            return projection(hidden).reshape(
                batch,
                sequence,
                config.num_attention_heads,
                config.head_dimension,
            )

        attended = dot_product_attention(
            project(self.query),
            project(self.key),
            project(self.value),
            attention_bias=position_bias,
            attention_mask=attention_mask[:, None, None, :],
            dropout_probability=config.attention_dropout_probability,
            dropout_key=probability_key,
            implementation=implementation,
        )
        return self.output(attended.reshape(batch, sequence, config.hidden_size))


class MPNetMLP(eqx.Module):
    input: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: MPNetConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> MPNetMLP:
        input_key, output_key = jax.random.split(key)
        return cls(
            input=Linear.init(
                config.hidden_size,
                config.intermediate_size,
                key=input_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            output=Linear.init(
                config.intermediate_size,
                config.hidden_size,
                key=output_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        *,
        config: MPNetConfig,
    ) -> Float[Array, "batch sequence hidden"]:
        return self.output(activate(self.input(hidden), config.hidden_activation))


class MPNetLayer(eqx.Module):
    attention: MPNetSelfAttention
    attention_norm: LayerNorm
    mlp: MPNetMLP
    output_norm: LayerNorm

    @classmethod
    def init(
        cls,
        config: MPNetConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> MPNetLayer:
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention=MPNetSelfAttention.init(config, key=attention_key, dtype=dtype),
            attention_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            mlp=MPNetMLP.init(config, key=mlp_key, dtype=dtype),
            output_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        attention_mask: Bool[Array, "batch sequence"],
        position_bias: Float[Array, "1 heads sequence sequence"],
        *,
        config: MPNetConfig,
        key: PRNGKeyArray | None,
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        if key is None:
            probability_key = attention_output_key = mlp_output_key = None
        else:
            probability_key, attention_output_key, mlp_output_key = jax.random.split(
                key, 3
            )
        attention = self.attention(
            hidden,
            attention_mask,
            position_bias,
            config=config,
            probability_key=probability_key,
            implementation=implementation,
        )
        attention = dropout(
            attention,
            config.hidden_dropout_probability,
            key=attention_output_key,
        )
        hidden = self.attention_norm(hidden + attention)
        output = dropout(
            self.mlp(hidden, config=config),
            config.hidden_dropout_probability,
            key=mlp_output_key,
        )
        return self.output_norm(hidden + output)


class MPNetLayerStack(eqx.Module):
    """Depth-major homogeneous MPNet layers executed by one scan."""

    blocks: MPNetLayer | None
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(cls, layers: tuple[MPNetLayer, ...]) -> MPNetLayerStack:
        if not layers:
            return cls(blocks=None, depth=0)
        blocks = jax.tree.map(lambda *values: jnp.stack(values), *layers)
        return cls(blocks=blocks, depth=len(layers))

    def layer(self, index: int) -> MPNetLayer:
        if self.blocks is None or not 0 <= index < self.depth:
            raise IndexError(index)
        return jax.tree.map(lambda value: value[index], self.blocks)


class MPNetTower(eqx.Module):
    embeddings: MPNetEmbeddings
    layers: MPNetLayerStack
    relative_attention_bias: Float[Array, "bucket heads"]
    pooler: Linear
    config: MPNetConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: MPNetConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> MPNetTower:
        keys = jax.random.split(key, config.num_hidden_layers + 3)
        layers = tuple(
            MPNetLayer.init(config, key=keys[index + 1], dtype=dtype)
            for index in range(config.num_hidden_layers)
        )
        relative_attention_bias = config.initializer_range * jax.random.normal(
            keys[-2],
            (config.relative_attention_num_buckets, config.num_attention_heads),
            dtype=dtype,
        )
        return cls(
            embeddings=MPNetEmbeddings.init(config, key=keys[0], dtype=dtype),
            layers=MPNetLayerStack.from_layers(layers),
            relative_attention_bias=relative_attention_bias,
            pooler=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=keys[-1],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            config=config,
        )

    def position_bias(
        self,
        sequence: int,
        *,
        dtype: jnp.dtype,
    ) -> Float[Array, "1 heads sequence sequence"]:
        positions = jnp.arange(sequence, dtype=jnp.int32)
        relative = positions[None, :] - positions[:, None]
        buckets = mpnet_relative_position_bucket(
            relative,
            num_buckets=self.config.relative_attention_num_buckets,
        )
        values = embedding_lookup(self.relative_attention_bias, buckets)
        return jnp.transpose(values, (2, 0, 1))[None].astype(dtype)

    def __call__(
        self,
        batch: MPNetBatch,
        *,
        key: PRNGKeyArray | None,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "batch sequence hidden"]:
        return self.all_hidden_states(
            batch,
            key=key,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )[-1]

    def all_hidden_states(
        self,
        batch: MPNetBatch,
        *,
        key: PRNGKeyArray | None,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "layer batch sequence hidden"]:
        """Return embedding output followed by every encoder-layer output."""

        if key is None:
            embedding_key = None
            layer_keys = jax.random.split(jax.random.key(0), self.layers.depth)
        else:
            embedding_key, layer_key = jax.random.split(key)
            layer_keys = jax.random.split(layer_key, self.layers.depth)
        hidden = self.embeddings(batch, config=self.config, key=embedding_key)
        hidden = hidden.astype(compute_dtype)
        attention_mask = batch.attention_mask.astype(bool)
        position_bias = self.position_bias(hidden.shape[1], dtype=compute_dtype)
        if self.layers.blocks is None:
            return hidden[None, ...]

        training = key is not None

        def apply_layer(
            carry: Float[Array, "batch sequence hidden"],
            values: tuple[MPNetLayer, PRNGKeyArray],
        ) -> tuple[
            Float[Array, "batch sequence hidden"],
            Float[Array, "batch sequence hidden"],
        ]:
            layer, layer_key = values
            output = layer(
                carry,
                attention_mask,
                position_bias,
                config=self.config,
                key=layer_key if training else None,
                implementation=attention_implementation,
            )
            return output, output

        executed_layer = rematerialize(apply_layer, rematerialization)
        _, layer_outputs = jax.lax.scan(
            executed_layer,
            hidden,
            (self.layers.blocks, layer_keys),
        )
        return jnp.concatenate((hidden[None, ...], layer_outputs), axis=0)

    def pool(
        self,
        hidden: Float[Array, "batch sequence hidden"],
    ) -> Float[Array, "batch hidden"]:
        return jnp.tanh(self.pooler(hidden[:, 0]))


class MPNetEncoder(eqx.Module):
    """Native bidirectional MPNet model plus representation pooling."""

    tower: MPNetTower
    metadata: EncoderMetadata
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: MPNetConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/mpnet",
        revision: str = "random-init",
    ) -> MPNetEncoder:
        return cls(
            tower=MPNetTower.init(config, key=key, dtype=parameter_dtype),
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

    def hidden_states(
        self,
        inputs: MPNetBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        if not isinstance(inputs, MPNetBatch):
            raise TypeError("MPNet inputs must be MPNetBatch")
        return self.tower(
            inputs,
            key=key,
            compute_dtype=active_compute_dtype(self.compute_dtype),
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def hidden_states_by_layer(
        self,
        inputs: MPNetBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "layer batch sequence hidden"]:
        """Expose the embedding output and every encoder layer for modifiers."""

        if not isinstance(inputs, MPNetBatch):
            raise TypeError("MPNet inputs must be MPNetBatch")
        return self.tower.all_hidden_states(
            inputs,
            key=key,
            compute_dtype=active_compute_dtype(self.compute_dtype),
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    @staticmethod
    def make_batch(
        *,
        input_ids: Int[Array, "batch sequence"],
        attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
        token_type_ids: Int[Array, "batch sequence"] | None = None,
    ) -> MPNetBatch:
        """Build this backbone's token-input contract at the host boundary."""

        if token_type_ids is not None:
            raise TypeError("MPNet does not accept token_type_ids")
        return MPNetBatch(input_ids=input_ids, attention_mask=attention_mask)

    def pooler_output(
        self,
        inputs: MPNetBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch hidden"]:
        return self.tower.pool(self.hidden_states(inputs, key=key))

    def encode(
        self,
        inputs: MPNetBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        hidden = self.hidden_states(inputs, key=key)
        return l2_normalize(mean_pool(hidden, inputs.attention_mask))

    def encode_layers(
        self,
        inputs: MPNetBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "layer batch representation"]:
        """Pool normalized representations for the embedding and every layer."""

        del route
        hidden = self.hidden_states_by_layer(inputs, key=key)
        pooled = jax.vmap(lambda value: mean_pool(value, inputs.attention_mask))(hidden)
        return l2_normalize(pooled)


__all__ = [
    "MPNetBatch",
    "MPNetEmbeddings",
    "MPNetEncoder",
    "MPNetLayer",
    "MPNetLayerStack",
    "MPNetMLP",
    "MPNetSelfAttention",
    "MPNetTower",
    "create_mpnet_position_ids",
    "mpnet_relative_position_bucket",
]
