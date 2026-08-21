"""Native Equinox BERT encoder compatible with Transformers 5.3."""

from __future__ import annotations

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

from .config import BertConfig


class BertBatch(eqx.Module):
    """Token IDs or input embeddings and masks accepted by native BERT."""

    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    input_ids: Int[Array, "batch sequence"] | None = None
    inputs_embeds: Float[Array, "batch sequence hidden"] | None = None
    token_type_ids: Int[Array, "batch sequence"] | None = None
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
        if self.token_type_ids is not None:
            if self.token_type_ids.shape != self.attention_mask.shape:
                raise ValueError("token_type_ids and attention_mask must align")
            if not jnp.issubdtype(self.token_type_ids.dtype, jnp.integer):
                raise TypeError("token_type_ids must have an integer dtype")
        if self.position_ids is not None:
            if self.position_ids.ndim != 2:
                raise ValueError("position_ids must have shape [batch, sequence]")
            if self.position_ids.shape[0] not in {1, self.attention_mask.shape[0]}:
                raise ValueError("position_ids batch must be one or match inputs")
            if self.position_ids.shape[1] != self.attention_mask.shape[1]:
                raise ValueError("position_ids and attention_mask must align")
            if not jnp.issubdtype(self.position_ids.dtype, jnp.integer):
                raise TypeError("position_ids must have an integer dtype")


class BertEmbeddings(eqx.Module):
    word: Float[Array, "vocabulary hidden"]
    position: Float[Array, "position hidden"]
    token_type: Float[Array, "type hidden"]
    norm: LayerNorm

    @classmethod
    def init(
        cls,
        config: BertConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> BertEmbeddings:
        word_key, position_key, token_type_key = jax.random.split(key, 3)

        def initialize(key: PRNGKeyArray, shape: tuple[int, int]) -> jax.Array:
            return config.initializer_range * jax.random.normal(key, shape, dtype=dtype)

        word = initialize(word_key, (config.vocab_size, config.hidden_size))
        word = word.at[config.pad_token_id].set(0)
        return cls(
            word=word,
            position=initialize(
                position_key,
                (config.max_position_embeddings, config.hidden_size),
            ),
            token_type=initialize(
                token_type_key,
                (config.type_vocab_size, config.hidden_size),
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
        batch: BertBatch,
        *,
        config: BertConfig,
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
            # Transformers constructs this table with ``padding_idx``. Preserve
            # the selected forward value while suppressing only that row's
            # embedding cotangent, exactly as PyTorch does.
            padding = batch.input_ids == config.pad_token_id
            hidden = jnp.where(
                padding[..., None],
                jax.lax.stop_gradient(hidden),
                hidden,
            )
        position_ids = batch.position_ids
        if position_ids is None:
            position_ids = jnp.arange(sequence)[None, :]
        token_type_ids = batch.token_type_ids
        if token_type_ids is None:
            token_type_ids = jnp.zeros(batch.attention_mask.shape, dtype=jnp.int32)
        hidden = hidden + embedding_lookup(self.position, position_ids)
        hidden = hidden + embedding_lookup(self.token_type, token_type_ids)
        hidden = self.norm(hidden)
        return dropout(hidden, config.hidden_dropout_probability, key=key)


class BertSelfAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: BertConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> BertSelfAttention:
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
        *,
        config: BertConfig,
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
            attention_mask=attention_mask[:, None, None, :],
            dropout_probability=config.attention_dropout_probability,
            dropout_key=probability_key,
            implementation=implementation,
        )
        return self.output(attended.reshape(batch, sequence, config.hidden_size))


class BertMLP(eqx.Module):
    input: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: BertConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> BertMLP:
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
        config: BertConfig,
    ) -> Float[Array, "batch sequence hidden"]:
        return self.output(activate(self.input(hidden), config.hidden_activation))


class BertLayer(eqx.Module):
    attention: BertSelfAttention
    attention_norm: LayerNorm
    mlp: BertMLP
    output_norm: LayerNorm

    @classmethod
    def init(
        cls,
        config: BertConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> BertLayer:
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention=BertSelfAttention.init(config, key=attention_key, dtype=dtype),
            attention_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            mlp=BertMLP.init(config, key=mlp_key, dtype=dtype),
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
        *,
        config: BertConfig,
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


class BertLayerStack(eqx.Module):
    """Depth-major homogeneous BERT layers executed by one scan."""

    blocks: BertLayer | None
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(cls, layers: tuple[BertLayer, ...]) -> BertLayerStack:
        if not layers:
            return cls(blocks=None, depth=0)
        blocks = jax.tree.map(lambda *values: jnp.stack(values), *layers)
        return cls(blocks=blocks, depth=len(layers))

    def layer(self, index: int) -> BertLayer:
        if self.blocks is None or not 0 <= index < self.depth:
            raise IndexError(index)
        return jax.tree.map(lambda value: value[index], self.blocks)


class BertTower(eqx.Module):
    embeddings: BertEmbeddings
    layers: BertLayerStack
    pooler: Linear
    config: BertConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: BertConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> BertTower:
        keys = jax.random.split(key, config.num_hidden_layers + 2)
        layers = tuple(
            BertLayer.init(config, key=keys[index + 1], dtype=dtype)
            for index in range(config.num_hidden_layers)
        )
        return cls(
            embeddings=BertEmbeddings.init(config, key=keys[0], dtype=dtype),
            layers=BertLayerStack.from_layers(layers),
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

    def __call__(
        self,
        batch: BertBatch,
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
        attention_mask = batch.attention_mask.astype(bool)
        if self.layers.blocks is None:
            return hidden

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

        executed_layer = rematerialize(apply_layer, rematerialization)
        hidden, _ = jax.lax.scan(
            executed_layer,
            hidden,
            (self.layers.blocks, layer_keys),
        )
        return hidden

    def all_hidden_states(
        self,
        batch: BertBatch,
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
        if self.layers.blocks is None:
            return hidden[None, ...]
        attention_mask = batch.attention_mask.astype(bool)
        training = key is not None

        def apply_layer(
            carry: Float[Array, "batch sequence hidden"],
            values: tuple[BertLayer, PRNGKeyArray],
        ) -> tuple[
            Float[Array, "batch sequence hidden"],
            Float[Array, "batch sequence hidden"],
        ]:
            layer, layer_key = values
            output = layer(
                carry,
                attention_mask,
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


class BertEncoder(eqx.Module):
    """Native bidirectional BERT model plus representation pooling."""

    tower: BertTower
    metadata: EncoderMetadata
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: BertConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/bert",
        revision: str = "random-init",
    ) -> BertEncoder:
        return cls(
            tower=BertTower.init(config, key=key, dtype=parameter_dtype),
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
        inputs: BertBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        if not isinstance(inputs, BertBatch):
            raise TypeError("BERT inputs must be BertBatch")
        return self.tower(
            inputs,
            key=key,
            compute_dtype=active_compute_dtype(self.compute_dtype),
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def hidden_states_by_layer(
        self,
        inputs: BertBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "layer batch sequence hidden"]:
        """Expose the embedding output and every encoder layer for modifiers."""

        if not isinstance(inputs, BertBatch):
            raise TypeError("BERT inputs must be BertBatch")
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
    ) -> BertBatch:
        """Build this backbone's token-input contract at the host boundary."""

        return BertBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def pooler_output(
        self,
        inputs: BertBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch hidden"]:
        return self.tower.pool(self.hidden_states(inputs, key=key))

    def encode(
        self,
        inputs: BertBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        hidden = self.hidden_states(inputs, key=key)
        return l2_normalize(mean_pool(hidden, inputs.attention_mask))

    def encode_layers(
        self,
        inputs: BertBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "layer batch representation"]:
        """Pool normalized representations for the embedding and every layer."""

        del route
        if not isinstance(inputs, BertBatch):
            raise TypeError("BERT inputs must be BertBatch")
        hidden = self.hidden_states_by_layer(inputs, key=key)
        pooled = jax.vmap(lambda value: mean_pool(value, inputs.attention_mask))(hidden)
        return l2_normalize(pooled)


__all__ = [
    "BertBatch",
    "BertEmbeddings",
    "BertEncoder",
    "BertLayer",
    "BertLayerStack",
    "BertMLP",
    "BertSelfAttention",
    "BertTower",
]
