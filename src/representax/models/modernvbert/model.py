"""Equinox implementation of the ModernVBERT ModernBERT text tower.

The module graph is written from the published Transformers architecture and
validated against a pinned Transformers oracle.  PyTorch is not part of the
execution path.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    dot_product_attention,
    embedding_lookup,
    l2_normalize,
    mean_pool,
    rematerialize,
)
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import ModernVBERTTextConfig


class ModernVBERTTextBatch(eqx.Module):
    """Token IDs or input embeddings plus their valid-token mask."""

    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    input_ids: Int[Array, "batch sequence"] | None = None
    inputs_embeds: Float[Array, "batch sequence hidden"] | None = None
    position_ids: Int[Array, "#batch sequence"] | None = None

    def __post_init__(self) -> None:
        input_ids = self.input_ids
        inputs_embeds = self.inputs_embeds
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if self.attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence]")
            if not jnp.issubdtype(input_ids.dtype, jnp.integer):
                raise TypeError("input_ids must have an integer dtype")
            token_shape = input_ids.shape
        else:
            if inputs_embeds is None:  # pragma: no cover - guarded above
                raise AssertionError("inputs_embeds must be present")
            if inputs_embeds.ndim != 3:
                raise ValueError(
                    "inputs_embeds must have shape [batch, sequence, hidden]"
                )
            token_shape = inputs_embeds.shape[:2]
        if token_shape != self.attention_mask.shape:
            raise ValueError("token inputs and attention_mask must align")
        if self.position_ids is not None:
            if self.position_ids.ndim != 2:
                raise ValueError("position_ids must have shape [batch, sequence]")
            if self.position_ids.shape[0] not in {
                1,
                self.attention_mask.shape[0],
            }:
                raise ValueError(
                    "position_ids batch must be one or match attention_mask"
                )
            if self.position_ids.shape[1] != self.attention_mask.shape[1]:
                raise ValueError("position_ids and attention_mask must align")
            if not jnp.issubdtype(self.position_ids.dtype, jnp.integer):
                raise TypeError("position_ids must have an integer dtype")


def _rotary_frequencies(
    head_dimension: int,
    theta: float,
    position_ids: Int[Array, "#batch sequence"],
) -> tuple[
    Float[Array, "#batch sequence head"],
    Float[Array, "#batch sequence head"],
]:
    inverse_frequency = 1.0 / (
        theta ** (jnp.arange(0, head_dimension, 2, dtype=jnp.float32) / head_dimension)
    )
    frequency = position_ids.astype(jnp.float32)[..., None] * inverse_frequency
    embedding = jnp.concatenate((frequency, frequency), axis=-1)
    return jnp.cos(embedding), jnp.sin(embedding)


def _rotate_half(
    value: Float[Array, "*batch head"],
) -> Float[Array, "*batch head"]:
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def _apply_rope(
    value: Float[Array, "batch sequence heads head"],
    cosine: Float[Array, "#batch sequence 1 head"],
    sine: Float[Array, "#batch sequence 1 head"],
) -> Float[Array, "batch sequence heads head"]:
    value32 = value.astype(jnp.float32)
    rotated = value32 * cosine + _rotate_half(value32) * sine
    return rotated.astype(value.dtype)


def _scaled_dot_product_attention(
    query: Float[Array, "batch target_sequence heads head"],
    key: Float[Array, "batch source_sequence heads head"],
    value: Float[Array, "batch source_sequence heads head"],
    attention_mask: Bool[Array, "batch source_sequence"],
    *,
    local_radius: int | None,
    implementation: AttentionImplementation,
) -> Float[Array, "batch target_sequence heads head"]:
    """Run exact full or symmetric local attention through JAX's primitive."""

    local_window = None if local_radius is None else (local_radius, local_radius)
    return dot_product_attention(
        query,
        key,
        value,
        attention_mask=attention_mask[:, None, None, :].astype(bool),
        local_window_size=local_window,
        implementation=implementation,
    )


class FusedSelfAttention(eqx.Module):
    qkv: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> FusedSelfAttention:
        qkv_key, output_key = jax.random.split(key)
        return cls(
            qkv=Linear.init(
                config.hidden_size,
                3 * config.hidden_size,
                key=qkv_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
            output=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=output_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        *,
        config: ModernVBERTTextConfig,
        attention_mask: Bool[Array, "batch sequence"],
        position_ids: Int[Array, "#batch sequence"],
        sliding_attention: Bool[Array, ""],
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        batch, sequence, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch,
            sequence,
            3,
            config.num_attention_heads,
            config.head_dimension,
        )
        query, key, value = (qkv[:, :, index] for index in range(3))

        def attend(
            operands: tuple[
                Float[Array, "batch sequence heads head"],
                Float[Array, "batch sequence heads head"],
                Float[Array, "batch sequence heads head"],
            ],
            *,
            theta: float,
            local_radius: int | None,
        ) -> Float[Array, "batch sequence heads head"]:
            branch_query, branch_key, branch_value = operands
            cosine, sine = _rotary_frequencies(
                config.head_dimension,
                theta,
                position_ids,
            )
            cosine = cosine[:, :, None, :]
            sine = sine[:, :, None, :]
            branch_query = _apply_rope(branch_query, cosine, sine)
            branch_key = _apply_rope(branch_key, cosine, sine)
            return _scaled_dot_product_attention(
                branch_query,
                branch_key,
                branch_value,
                attention_mask,
                local_radius=local_radius,
                implementation=implementation,
            )

        operands = (query, key, value)
        attended = jax.lax.cond(
            sliding_attention,
            lambda values: attend(
                values,
                theta=config.sliding_attention_rope_theta,
                local_radius=config.local_attention // 2,
            ),
            lambda values: attend(
                values,
                theta=config.full_attention_rope_theta,
                local_radius=None,
            ),
            operands,
        )
        return self.output(attended.reshape(batch, sequence, config.hidden_size))


class GatedMLP(eqx.Module):
    input: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> GatedMLP:
        input_key, output_key = jax.random.split(key)
        return cls(
            input=Linear.init(
                config.hidden_size,
                2 * config.intermediate_size,
                key=input_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
            output=Linear.init(
                config.intermediate_size,
                config.hidden_size,
                key=output_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
    ) -> Float[Array, "batch sequence hidden"]:
        value, gate = jnp.split(self.input(hidden), 2, axis=-1)
        return self.output(jax.nn.gelu(value, approximate=False) * gate)


class ModernVBERTTextBlock(eqx.Module):
    attention: FusedSelfAttention
    attention_norm: LayerNorm | None
    mlp_norm: LayerNorm
    mlp: GatedMLP
    sliding_attention: Bool[Array, ""]

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        index: int,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> ModernVBERTTextBlock:
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention=FusedSelfAttention.init(
                config,
                key=attention_key,
                dtype=dtype,
            ),
            # ModernBERT's first layer is the intentional pre-norm exception.
            attention_norm=(
                None
                if index == 0
                else LayerNorm.init(
                    config.hidden_size,
                    epsilon=config.norm_epsilon,
                    dtype=dtype,
                )
            ),
            mlp_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
            ),
            mlp=GatedMLP.init(config, key=mlp_key, dtype=dtype),
            sliding_attention=jnp.asarray(
                config.layer_types[index] == "sliding_attention"
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        *,
        config: ModernVBERTTextConfig,
        attention_mask: Bool[Array, "batch sequence"],
        position_ids: Int[Array, "#batch sequence"],
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        attention_input = (
            hidden if self.attention_norm is None else self.attention_norm(hidden)
        )
        hidden = hidden + self.attention(
            attention_input,
            config=config,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sliding_attention=self.sliding_attention,
            implementation=implementation,
        )
        return hidden + self.mlp(self.mlp_norm(hidden))


class _ModernVBERTTextScanBlock(eqx.Module):
    """The structurally uniform portion of one scanned text block."""

    attention: FusedSelfAttention
    mlp_norm: LayerNorm
    mlp: GatedMLP
    sliding_attention: Bool[Array, ""]


class ModernVBERTTextLayerStack(eqx.Module):
    """A depth-major PyTree of homogeneous ModernVBERT text blocks."""

    first_block: _ModernVBERTTextScanBlock | None
    blocks: _ModernVBERTTextScanBlock | None
    attention_norms: LayerNorm | None
    depth: int = eqx.field(static=True)

    @classmethod
    def from_blocks(
        cls,
        blocks: tuple[ModernVBERTTextBlock, ...],
    ) -> ModernVBERTTextLayerStack:
        if not blocks:
            return cls(
                first_block=None,
                blocks=None,
                attention_norms=None,
                depth=0,
            )
        if blocks[0].attention_norm is not None:
            raise ValueError("ModernVBERT layer zero must omit attention_norm")
        if any(block.attention_norm is None for block in blocks[1:]):
            raise ValueError("ModernVBERT layers after zero require attention_norm")
        scan_blocks = tuple(
            _ModernVBERTTextScanBlock(
                attention=block.attention,
                mlp_norm=block.mlp_norm,
                mlp=block.mlp,
                sliding_attention=block.sliding_attention,
            )
            for block in blocks
        )
        first_block = scan_blocks[0]
        remaining_blocks = scan_blocks[1:]
        stacked = (
            None
            if not remaining_blocks
            else jax.tree.map(lambda *leaves: jnp.stack(leaves), *remaining_blocks)
        )
        norms = tuple(block.attention_norm for block in blocks[1:])
        stacked_norms = (
            None
            if not norms
            else jax.tree.map(lambda *leaves: jnp.stack(leaves), *norms)
        )
        return cls(
            first_block=first_block,
            blocks=stacked,
            attention_norms=stacked_norms,
            depth=len(blocks),
        )

    def __len__(self) -> int:
        return self.depth

    @overload
    def __getitem__(self, index: int) -> ModernVBERTTextBlock: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ModernVBERTTextBlock, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> ModernVBERTTextBlock | tuple[ModernVBERTTextBlock, ...]:
        if isinstance(index, slice):
            positions = range(*index.indices(self.depth))
            return tuple(self[position] for position in positions)
        if not 0 <= index < self.depth:
            raise IndexError(index)
        if self.first_block is None:
            raise IndexError(index)
        if index == 0:
            block = self.first_block
            attention_norm = None
        else:
            if (  # pragma: no cover - invalid tree
                self.blocks is None or self.attention_norms is None
            ):
                raise AssertionError("post-zero layers require attention norms")
            block = jax.tree.map(lambda leaf: leaf[index - 1], self.blocks)
            attention_norm = jax.tree.map(
                lambda leaf: leaf[index - 1], self.attention_norms
            )
        return ModernVBERTTextBlock(
            attention=block.attention,
            attention_norm=attention_norm,
            mlp_norm=block.mlp_norm,
            mlp=block.mlp,
            sliding_attention=block.sliding_attention,
        )

    def __iter__(self) -> Iterator[ModernVBERTTextBlock]:
        return (self[index] for index in range(self.depth))


class ModernVBERTTextTower(eqx.Module):
    token_embedding: Float[Array, "vocabulary hidden"]
    embedding_norm: LayerNorm
    layers: ModernVBERTTextLayerStack
    final_norm: LayerNorm
    config: ModernVBERTTextConfig = eqx.field(static=True)

    def token_embeddings(
        self,
        input_ids: Int[Array, "batch sequence"],
    ) -> Float[Array, "batch sequence hidden"]:
        """Gather token embeddings with repeated-token gradient accumulation."""

        return embedding_lookup(self.token_embedding, input_ids)

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype = jnp.float32,
    ) -> ModernVBERTTextTower:
        keys = jax.random.split(key, config.num_hidden_layers + 1)
        token_embedding = config.initializer_range * jax.random.normal(
            keys[0], (config.vocab_size, config.hidden_size), dtype=dtype
        )
        layers = tuple(
            ModernVBERTTextBlock.init(
                config,
                index,
                key=keys[index + 1],
                dtype=dtype,
            )
            for index in range(config.num_hidden_layers)
        )
        return cls(
            token_embedding=token_embedding,
            embedding_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
            ),
            layers=ModernVBERTTextLayerStack.from_blocks(layers),
            final_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
            ),
            config=config,
        )

    def __call__(
        self,
        batch: ModernVBERTTextBatch,
        *,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "batch sequence hidden"]:
        return self.all_hidden_states(
            batch,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )[-1]

    def all_hidden_states(
        self,
        batch: ModernVBERTTextBatch,
        *,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "layer batch sequence hidden"]:
        """Return embedding output followed by every encoder-layer output."""

        if batch.input_ids is not None:
            hidden = self.token_embeddings(batch.input_ids)
        elif batch.inputs_embeds is not None:
            hidden = batch.inputs_embeds
        else:  # pragma: no cover - rejected by ModernVBERTTextBatch
            raise AssertionError("token inputs must be present")
        hidden = hidden.astype(compute_dtype)
        attention_mask = batch.attention_mask.astype(bool)
        batch_size, sequence, _ = hidden.shape
        if hidden.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "input embedding dimension does not match ModernVBERT hidden_size"
            )
        if sequence > self.config.max_position_embeddings:
            raise ValueError("input sequence exceeds max_position_embeddings")
        if batch.position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(sequence)[None, :], (batch_size, sequence)
            )
        elif batch.position_ids.shape[0] == 1 and batch_size != 1:
            position_ids = jnp.broadcast_to(batch.position_ids, (batch_size, sequence))
        else:
            position_ids = batch.position_ids
        hidden = self.embedding_norm(hidden)
        initial_hidden = hidden
        if self.layers.first_block is not None:

            def apply_block(
                carry: Float[Array, "batch sequence hidden"],
                layer: _ModernVBERTTextScanBlock,
                attention_input: Float[Array, "batch sequence hidden"],
            ) -> Float[Array, "batch sequence hidden"]:
                output = carry + layer.attention(
                    attention_input,
                    config=self.config,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    sliding_attention=layer.sliding_attention,
                    implementation=attention_implementation,
                )
                return output + layer.mlp(layer.mlp_norm(output))

            def apply_first_layer(
                carry: Float[Array, "batch sequence hidden"],
                layer: _ModernVBERTTextScanBlock,
            ) -> Float[Array, "batch sequence hidden"]:
                return apply_block(carry, layer, carry)

            first_output = rematerialize(
                apply_first_layer,
                rematerialization,
            )(hidden, self.layers.first_block)

            def apply_layer(
                carry: Float[Array, "batch sequence hidden"],
                values: tuple[_ModernVBERTTextScanBlock, LayerNorm],
            ) -> tuple[
                Float[Array, "batch sequence hidden"],
                Float[Array, "batch sequence hidden"],
            ]:
                layer, attention_norm = values
                output = apply_block(carry, layer, attention_norm(carry))
                return output, output

            if self.layers.blocks is None:
                layer_outputs = first_output[None, ...]
            else:
                if self.layers.attention_norms is None:  # pragma: no cover
                    raise AssertionError("post-zero layers require attention norms")
                executed_layer = rematerialize(apply_layer, rematerialization)
                _, remaining_outputs = jax.lax.scan(
                    executed_layer,
                    first_output,
                    (self.layers.blocks, self.layers.attention_norms),
                )
                layer_outputs = jnp.concatenate(
                    (first_output[None, ...], remaining_outputs),
                    axis=0,
                )
            final_output = self.final_norm(layer_outputs[-1])
            layer_outputs = jnp.concatenate(
                (layer_outputs[:-1], final_output[None, ...]),
                axis=0,
            )
            return jnp.concatenate((initial_hidden[None, ...], layer_outputs), axis=0)
        return self.final_norm(initial_hidden)[None, ...]


class ModernVBERTTextEncoder(eqx.Module):
    """Text-only Representax encoder backed by a native ModernBERT tower."""

    tower: ModernVBERTTextTower
    metadata: EncoderMetadata
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/modernvbert-text",
        revision: str = "random-init",
    ) -> ModernVBERTTextEncoder:
        return cls(
            tower=ModernVBERTTextTower.init(config, key=key, dtype=parameter_dtype),
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
    ) -> ModernVBERTTextBatch:
        """Build this backbone's token-input contract at the host boundary."""

        if token_type_ids is not None:
            raise TypeError("ModernVBERT does not accept token_type_ids")
        return ModernVBERTTextBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def hidden_states(
        self,
        inputs: ModernVBERTTextBatch,
    ) -> Float[Array, "batch sequence hidden"]:
        if not isinstance(inputs, ModernVBERTTextBatch):
            raise TypeError("ModernVBERT text inputs must be ModernVBERTTextBatch")
        return self.tower(
            inputs,
            compute_dtype=active_compute_dtype(self.compute_dtype),
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def hidden_states_by_layer(
        self,
        inputs: ModernVBERTTextBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "layer batch sequence hidden"]:
        """Expose the embedding output and every text layer for modifiers."""

        del key
        if not isinstance(inputs, ModernVBERTTextBatch):
            raise TypeError("ModernVBERT text inputs must be ModernVBERTTextBatch")
        return self.tower.all_hidden_states(
            inputs,
            compute_dtype=active_compute_dtype(self.compute_dtype),
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def encode(
        self,
        inputs: ModernVBERTTextBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route, key
        hidden = self.hidden_states(inputs)
        return l2_normalize(mean_pool(hidden, inputs.attention_mask))

    def encode_layers(
        self,
        inputs: ModernVBERTTextBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "layer batch representation"]:
        """Pool normalized representations for every text-layer depth."""

        del route, key
        hidden = self.hidden_states_by_layer(inputs)
        pooled = jax.vmap(lambda value: mean_pool(value, inputs.attention_mask))(hidden)
        return l2_normalize(pooled)
