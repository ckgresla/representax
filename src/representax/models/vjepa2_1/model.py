"""Native Equinox V-JEPA 2.1 image/video encoder and dense predictor."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    dot_product_attention,
    rematerialize,
)
from representax.planning import RematerializationPolicy

from .config import VJEPA2_1Config


def token_positions(
    token_ids: Int[Array, "*batch token"],
    *,
    grid_height: int,
    grid_width: int,
) -> tuple[Array, Array, Array]:
    """Map flattened tubelet IDs to temporal, row, and column coordinates."""

    per_frame = grid_height * grid_width
    frame = token_ids // per_frame
    spatial = token_ids - frame * per_frame
    row = spatial // grid_width
    column = spatial - row * grid_width
    return frame, row, column


def _rotate(
    values: Float[Array, "batch head token rotary"],
    positions: Array,
) -> Array:
    width = values.shape[-1]
    frequencies = 1.0 / (
        10_000 ** (jnp.arange(width // 2, dtype=values.dtype) / (width / 2.0))
    )
    angles = positions.astype(values.dtype)[..., None] * frequencies
    sine = jnp.repeat(jnp.sin(angles), 2, axis=-1)
    cosine = jnp.repeat(jnp.cos(angles), 2, axis=-1)
    pairs = values.reshape((*values.shape[:-1], width // 2, 2))
    rotated = jnp.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(values.shape)
    return values * cosine + rotated * sine


def apply_3d_rope(
    values: Float[Array, "batch head token head_dimension"],
    token_ids: Int[Array, "batch token"],
    *,
    grid_height: int,
    grid_width: int,
) -> Array:
    """Apply the reference temporal/height/width rotary partition."""

    head_dimension = values.shape[-1]
    axis_width = 2 * ((head_dimension // 3) // 2)
    frame, row, column = token_positions(
        token_ids,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    positions = (frame, row, column)
    pieces = []
    offset = 0
    for position in positions:
        pieces.append(
            _rotate(
                values[..., offset : offset + axis_width],
                position[:, None, :],
            )
        )
        offset += axis_width
    if offset < head_dimension:
        pieces.append(values[..., offset:])
    return jnp.concatenate(pieces, axis=-1)


class VJEPA2_1Attention(eqx.Module):
    qkv: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        scale: float,
        dtype: jnp.dtype,
    ) -> VJEPA2_1Attention:
        qkv_key, output_key = jax.random.split(key)
        return cls(
            qkv=Linear.init(
                hidden_size,
                3 * hidden_size,
                key=qkv_key,
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            output=Linear.init(
                hidden_size,
                hidden_size,
                key=output_key,
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        hidden: Array,
        token_ids: Array,
        token_valid: Array | None,
        *,
        heads: int,
        grid_height: int,
        grid_width: int,
        implementation: AttentionImplementation,
    ) -> Array:
        batch, tokens, hidden_size = hidden.shape
        head_dimension = hidden_size // heads
        qkv = self.qkv(hidden).reshape((batch, tokens, 3, heads, head_dimension))
        query, key, value = jnp.moveaxis(qkv, 2, 0)
        query = jnp.swapaxes(query, 1, 2)
        key = jnp.swapaxes(key, 1, 2)
        value = jnp.swapaxes(value, 1, 2)
        query = apply_3d_rope(
            query,
            token_ids,
            grid_height=grid_height,
            grid_width=grid_width,
        )
        key = apply_3d_rope(
            key,
            token_ids,
            grid_height=grid_height,
            grid_width=grid_width,
        )
        attended = dot_product_attention(
            jnp.swapaxes(query, 1, 2),
            jnp.swapaxes(key, 1, 2),
            jnp.swapaxes(value, 1, 2),
            attention_mask=(
                None if token_valid is None else token_valid[:, None, None, :]
            ),
            implementation=implementation,
        )
        return self.output(attended.reshape((batch, tokens, hidden_size)))


class VJEPA2_1MLP(eqx.Module):
    up: Linear
    down: Linear

    @classmethod
    def init(
        cls,
        hidden_size: int,
        intermediate_size: int,
        *,
        key: PRNGKeyArray,
        scale: float,
        dtype: jnp.dtype,
    ) -> VJEPA2_1MLP:
        up_key, down_key = jax.random.split(key)
        return cls(
            up=Linear.init(
                hidden_size,
                intermediate_size,
                key=up_key,
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            down=Linear.init(
                intermediate_size,
                hidden_size,
                key=down_key,
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(self, hidden: Array) -> Array:
        return self.down(jax.nn.gelu(self.up(hidden), approximate=False))


class VJEPA2_1Layer(eqx.Module):
    attention_norm: LayerNorm
    attention: VJEPA2_1Attention
    mlp_norm: LayerNorm
    mlp: VJEPA2_1MLP

    @classmethod
    def init(
        cls,
        hidden_size: int,
        heads: int,
        mlp_ratio: float,
        *,
        epsilon: float,
        scale: float,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> VJEPA2_1Layer:
        del heads
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention_norm=LayerNorm.init(
                hidden_size, epsilon=epsilon, dtype=dtype, bias=True
            ),
            attention=VJEPA2_1Attention.init(
                hidden_size,
                key=attention_key,
                scale=scale,
                dtype=dtype,
            ),
            mlp_norm=LayerNorm.init(
                hidden_size, epsilon=epsilon, dtype=dtype, bias=True
            ),
            mlp=VJEPA2_1MLP.init(
                hidden_size,
                int(hidden_size * mlp_ratio),
                key=mlp_key,
                scale=scale,
                dtype=dtype,
            ),
        )

    def __call__(
        self,
        hidden: Array,
        token_ids: Array,
        token_valid: Array | None,
        *,
        heads: int,
        grid_height: int,
        grid_width: int,
        implementation: AttentionImplementation,
    ) -> Array:
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            token_ids,
            token_valid,
            heads=heads,
            grid_height=grid_height,
            grid_width=grid_width,
            implementation=implementation,
        )
        return hidden + self.mlp(self.mlp_norm(hidden))


class VJEPA2_1LayerStack(eqx.Module):
    layers: VJEPA2_1Layer
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(cls, layers: tuple[VJEPA2_1Layer, ...]) -> VJEPA2_1LayerStack:
        compute_layers = tuple(
            jax.tree.map(
                lambda value: (
                    value.input_major() if isinstance(value, Linear) else value
                ),
                layer,
                is_leaf=lambda value: isinstance(value, Linear),
            )
            for layer in layers
        )
        return cls(
            layers=jax.tree.map(lambda *values: jnp.stack(values), *compute_layers),
            depth=len(layers),
        )

    def layer(self, index: int) -> VJEPA2_1Layer:
        if not 0 <= index < self.depth:
            raise IndexError(index)
        layer = jax.tree.map(lambda value: value[index], self.layers)
        return jax.tree.map(
            lambda value: value.output_major() if isinstance(value, Linear) else value,
            layer,
            is_leaf=lambda value: isinstance(value, Linear),
        )

    def __call__(
        self,
        hidden: Array,
        token_ids: Array,
        token_valid: Array | None = None,
        *,
        heads: int,
        grid_height: int,
        grid_width: int,
        supervision_layers: tuple[int, ...],
        norms: Any,
        implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> tuple[Array, Array]:
        levels = len(supervision_layers)
        selected = jnp.zeros((levels, *hidden.shape), dtype=hidden.dtype)

        def apply_layer(carry, index):
            current, outputs = carry
            layer = jax.tree.map(
                lambda value: jax.lax.dynamic_index_in_dim(
                    value, index, axis=0, keepdims=False
                ),
                self.layers,
            )
            current = layer(
                current,
                token_ids,
                token_valid,
                heads=heads,
                grid_height=grid_height,
                grid_width=grid_width,
                implementation=implementation,
            )
            matches = index == jnp.asarray(supervision_layers, dtype=index.dtype)
            outputs = jnp.where(matches[:, None, None, None], current[None], outputs)
            return (current, outputs), None

        (hidden, selected), _ = jax.lax.scan(
            rematerialize(apply_layer, rematerialization),
            (hidden, selected),
            jnp.arange(self.depth, dtype=jnp.int32),
        )
        normalized = jax.vmap(lambda norm, value: norm(value))(norms, selected)
        return hidden, normalized


class VJEPA2_1Encoder(eqx.Module):
    image_patch_weight: Array
    image_patch_bias: Array
    video_patch_weight: Array
    video_patch_bias: Array
    image_modality_embedding: Array
    video_modality_embedding: Array
    layers: VJEPA2_1LayerStack
    supervision_norms: LayerNorm
    config: VJEPA2_1Config = eqx.field(static=True)
    implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: VJEPA2_1Config,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype = jnp.float32,
        implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
    ) -> VJEPA2_1Encoder:
        keys = jax.random.split(key, config.depth + 6)
        scale = config.initializer_range
        image_shape = (
            config.hidden_size,
            config.channels,
            config.patch_size,
            config.patch_size,
        )
        video_shape = (
            config.hidden_size,
            config.channels,
            config.tubelet_size,
            config.patch_size,
            config.patch_size,
        )
        layers = tuple(
            VJEPA2_1Layer.init(
                config.hidden_size,
                config.heads,
                config.mlp_ratio,
                epsilon=config.layer_norm_epsilon,
                scale=scale,
                key=layer_key,
                dtype=dtype,
            )
            for layer_key in keys[6:]
        )
        return cls(
            image_patch_weight=scale * jax.random.normal(keys[0], image_shape, dtype),
            image_patch_bias=jnp.zeros((config.hidden_size,), dtype=dtype),
            video_patch_weight=scale * jax.random.normal(keys[1], video_shape, dtype),
            video_patch_bias=jnp.zeros((config.hidden_size,), dtype=dtype),
            image_modality_embedding=1e-6
            * jax.random.normal(keys[2], (config.hidden_size,), dtype),
            video_modality_embedding=1e-6
            * jax.random.normal(keys[3], (config.hidden_size,), dtype),
            layers=VJEPA2_1LayerStack.from_layers(layers),
            supervision_norms=jax.tree.map(
                lambda *values: jnp.stack(values),
                *(
                    LayerNorm.init(
                        config.hidden_size,
                        epsilon=config.layer_norm_epsilon,
                        dtype=dtype,
                        bias=True,
                    )
                    for _ in config.supervision_layers
                ),
            ),
            config=config,
            implementation=implementation,
            rematerialization=rematerialization,
        )

    def tokenize(self, pixels: Array) -> tuple[Array, Array]:
        if pixels.ndim == 4:
            tokens = jax.lax.conv_general_dilated(
                pixels,
                self.image_patch_weight,
                window_strides=(self.config.patch_size,) * 2,
                padding="VALID",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            tokens = tokens + self.image_patch_bias[None, :, None, None]
            tokens = jnp.moveaxis(tokens, 1, -1).reshape(
                (pixels.shape[0], -1, self.config.hidden_size)
            )
            return tokens + self.image_modality_embedding, jnp.arange(
                tokens.shape[1], dtype=jnp.int32
            )[None].repeat(pixels.shape[0], axis=0)
        if pixels.ndim == 5:
            tokens = jax.lax.conv_general_dilated(
                pixels,
                self.video_patch_weight,
                window_strides=(
                    self.config.tubelet_size,
                    self.config.patch_size,
                    self.config.patch_size,
                ),
                padding="VALID",
                dimension_numbers=("NCTHW", "OITHW", "NCTHW"),
            )
            tokens = tokens + self.video_patch_bias[None, :, None, None, None]
            tokens = jnp.moveaxis(tokens, 1, -1).reshape(
                (pixels.shape[0], -1, self.config.hidden_size)
            )
            return tokens + self.video_modality_embedding, jnp.arange(
                tokens.shape[1], dtype=jnp.int32
            )[None].repeat(pixels.shape[0], axis=0)
        raise ValueError("V-JEPA pixels must be [B,C,H,W] or [B,C,T,H,W]")

    def encode_tokens(
        self,
        tokens: Array,
        token_ids: Array,
        token_valid: Array | None = None,
    ) -> tuple[Array, Array]:
        return self.layers(
            tokens,
            token_ids,
            token_valid,
            heads=self.config.heads,
            grid_height=self.config.spatial_grid,
            grid_width=self.config.spatial_grid,
            supervision_layers=self.config.supervision_layers,
            norms=self.supervision_norms,
            implementation=self.implementation,
            rematerialization=self.rematerialization,
        )

    def __call__(self, pixels: Array, token_ids: Array | None = None) -> Array:
        tokens, all_ids = self.tokenize(pixels)
        if token_ids is not None:
            tokens = jnp.take_along_axis(tokens, token_ids[..., None], axis=1)
            all_ids = token_ids
        _, levels = self.encode_tokens(tokens, all_ids)
        return jnp.concatenate(tuple(levels), axis=-1)


class VJEPA2_1Predictor(eqx.Module):
    fuse_in: Linear
    fuse_out: Linear
    mask_tokens: Array
    image_modality_embedding: Array
    video_modality_embedding: Array
    layers: VJEPA2_1LayerStack
    final_norm: LayerNorm
    target_projection: Linear
    context_projection: Linear
    config: VJEPA2_1Config = eqx.field(static=True)
    implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: VJEPA2_1Config,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype = jnp.float32,
        implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        mask_token_count: int = 2,
    ) -> VJEPA2_1Predictor:
        keys = jax.random.split(key, config.predictor_depth + 9)
        scale = config.initializer_range
        layers = tuple(
            VJEPA2_1Layer.init(
                config.predictor_hidden_size,
                config.predictor_heads,
                config.mlp_ratio,
                epsilon=config.layer_norm_epsilon,
                scale=scale,
                key=layer_key,
                dtype=dtype,
            )
            for layer_key in keys[9:]
        )
        output_size = len(config.supervision_layers) * config.hidden_size
        return cls(
            fuse_in=Linear.init(
                output_size,
                config.hidden_size,
                key=keys[0],
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            fuse_out=Linear.init(
                config.hidden_size,
                config.predictor_hidden_size,
                key=keys[1],
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            mask_tokens=jnp.zeros(
                (mask_token_count, config.predictor_hidden_size), dtype=dtype
            ),
            image_modality_embedding=1e-6
            * jax.random.normal(keys[2], (config.predictor_hidden_size,), dtype=dtype),
            video_modality_embedding=1e-6
            * jax.random.normal(keys[3], (config.predictor_hidden_size,), dtype=dtype),
            layers=VJEPA2_1LayerStack.from_layers(layers),
            final_norm=LayerNorm.init(
                config.predictor_hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            target_projection=Linear.init(
                config.predictor_hidden_size,
                output_size,
                key=keys[4],
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            context_projection=Linear.init(
                config.predictor_hidden_size,
                output_size,
                key=keys[5],
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            config=config,
            implementation=implementation,
            rematerialization=rematerialization,
        )

    def __call__(
        self,
        context_features: Array,
        context_ids: Array,
        target_ids: Array,
        *,
        is_video: bool,
        mask_index: int = 1,
        context_valid: Array | None = None,
        target_valid: Array | None = None,
    ) -> tuple[Array, Array]:
        context = self.fuse_out(jax.nn.gelu(self.fuse_in(context_features)))
        mask_token = self.mask_tokens[mask_index % self.mask_tokens.shape[0]]
        targets = jnp.broadcast_to(
            mask_token,
            (*target_ids.shape, self.config.predictor_hidden_size),
        )
        hidden = jnp.concatenate((context, targets), axis=1)
        token_ids = jnp.concatenate((context_ids, target_ids), axis=1)
        token_valid = (
            None
            if context_valid is None or target_valid is None
            else jnp.concatenate((context_valid, target_valid), axis=1)
        )
        order = jnp.argsort(token_ids, axis=1)
        hidden = jnp.take_along_axis(hidden, order[..., None], axis=1)
        sorted_ids = jnp.take_along_axis(token_ids, order, axis=1)
        if token_valid is not None:
            token_valid = jnp.take_along_axis(token_valid, order, axis=1)
        modality = (
            self.video_modality_embedding if is_video else self.image_modality_embedding
        )
        hidden = hidden + modality
        hidden, _ = self.layers(
            hidden,
            sorted_ids,
            token_valid,
            heads=self.config.predictor_heads,
            grid_height=self.config.spatial_grid,
            grid_width=self.config.spatial_grid,
            supervision_layers=(self.config.predictor_depth - 1,),
            norms=jax.tree.map(lambda value: value[None], self.final_norm),
            implementation=self.implementation,
            rematerialization=self.rematerialization,
        )
        hidden = self.final_norm(hidden)
        inverse = jnp.argsort(order, axis=1)
        hidden = jnp.take_along_axis(hidden, inverse[..., None], axis=1)
        context_count = context_ids.shape[1]
        return (
            self.target_projection(hidden[:, context_count:]),
            self.context_projection(hidden[:, :context_count]),
        )


class VJEPA2_1Model(eqx.Module):
    """Online encoder, predictor, and stop-gradient EMA target encoder."""

    online: VJEPA2_1Encoder
    predictor: VJEPA2_1Predictor
    target: VJEPA2_1Encoder
    metadata: EncoderMetadata = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: VJEPA2_1Config,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype = jnp.float32,
        implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
    ) -> VJEPA2_1Model:
        encoder_key, predictor_key = jax.random.split(key)
        online = VJEPA2_1Encoder.init(
            config,
            key=encoder_key,
            dtype=dtype,
            implementation=implementation,
            rematerialization=rematerialization,
        )
        predictor = VJEPA2_1Predictor.init(
            config,
            key=predictor_key,
            dtype=dtype,
            implementation=implementation,
            rematerialization=rematerialization,
        )
        target = jax.tree.map(lambda value: jnp.array(value), online)
        return cls(
            online=online,
            predictor=predictor,
            target=target,
            metadata=EncoderMetadata(
                model_id="facebook/vjepa2-1",
                revision="204698b45b3712590f06245fbfba32d3be539812",
                output_dimension=config.hidden_size,
                routes=frozenset({Route.GENERIC}),
                modalities=frozenset({Modality.IMAGE, Modality.VIDEO}),
            ),
        )

    @classmethod
    def load_from_reference(
        cls,
        path: str,
        config: VJEPA2_1Config,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype = jnp.float32,
        implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
    ) -> VJEPA2_1Model:
        """Convert one official Meta training checkpoint into native Equinox."""

        from .reference import load_reference_checkpoint

        template = cls.init(
            config,
            key=key,
            dtype=dtype,
            implementation=implementation,
            rematerialization=rematerialization,
        )
        return load_reference_checkpoint(template, path)

    def training_filter(self) -> Any:
        selected = jax.tree.map(eqx.is_inexact_array, self)
        frozen_target = jax.tree.map(lambda _value: False, self.target)
        return eqx.tree_at(lambda model: model.target, selected, frozen_target)

    def ema_update(self, optimized: VJEPA2_1Model, momentum: Array) -> VJEPA2_1Model:
        target = jax.tree.map(
            lambda old, online: momentum * old + (1.0 - momentum) * online,
            self.target,
            optimized.online,
        )
        return eqx.tree_at(lambda model: model.target, optimized, target)

    def encode(
        self,
        inputs: Any,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        del route, key
        pixels = getattr(inputs, "pixels", inputs)
        hierarchical = self.target(pixels)
        levels = len(self.target.config.supervision_layers)
        final = hierarchical[..., (levels - 1) * self.metadata.output_dimension :]
        return jnp.mean(final, axis=1)


__all__ = [
    "VJEPA2_1Encoder",
    "VJEPA2_1Model",
    "VJEPA2_1Predictor",
    "apply_3d_rope",
    "token_positions",
]
