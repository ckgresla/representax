"""Native LeJEPA ViT backbone and training-only projection composition."""

from __future__ import annotations

from collections.abc import Mapping
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
from representax.precision import active_compute_dtype, compute_parameter

from .config import LeJEPAViTConfig


class LeJEPAMulticropImages(eqx.Module):
    """Padded multicrop images whose encode path is the training projector."""

    pixel_values: Float[Array, "batch channel height width"]
    crop_sizes: Int[Array, " batch"]
    view_count: int = eqx.field(static=True, default=8)
    global_views: int = eqx.field(static=True, default=2)

    def __post_init__(self) -> None:
        if self.pixel_values.ndim not in (4, 5):
            raise ValueError(
                "LeJEPA pixels must be flat images or "
                "[batch, view, channel, height, width]"
            )
        leading = self.pixel_values.shape[:-3]
        if self.crop_sizes.shape != leading:
            raise ValueError("crop_sizes must align with the image leading dimensions")
        if not jnp.issubdtype(self.crop_sizes.dtype, jnp.integer):
            raise TypeError("crop_sizes must be an integer vector")
        if self.view_count <= 1 or not 0 < self.global_views < self.view_count:
            raise ValueError(
                "LeJEPA multicrop metadata requires global and local views"
            )
        if (
            self.pixel_values.ndim == 5
            and self.pixel_values.shape[1] != self.view_count
        ):
            raise ValueError("LeJEPA view axis differs from view_count")
        if self.pixel_values.ndim == 4 and self.pixel_values.shape[0] % self.view_count:
            raise ValueError(
                "flattened LeJEPA images must contain complete view groups"
            )


def _stochastic_depth(
    value: Array,
    probability: Array,
    *,
    key: PRNGKeyArray | None,
) -> Array:
    if key is None:
        return value
    keep_probability = jnp.asarray(1.0 - probability, dtype=value.dtype)
    keep = jax.random.bernoulli(
        key,
        keep_probability,
        (value.shape[0],) + (1,) * (value.ndim - 1),
    )
    return jnp.where(keep, value / keep_probability, 0.0).astype(value.dtype)


class LeJEPAAttention(eqx.Module):
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
    ) -> LeJEPAAttention:
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
        *,
        heads: int,
        implementation: AttentionImplementation,
    ) -> tuple[Array, Array]:
        batch, tokens, hidden_size = hidden.shape
        head_dimension = hidden_size // heads
        qkv = self.qkv(hidden).reshape((batch, tokens, 3, heads, head_dimension))
        query, key, value = jnp.moveaxis(qkv, 2, 0)
        attended = dot_product_attention(
            query,
            key,
            value,
            implementation=implementation,
        )
        return self.output(attended.reshape((batch, tokens, hidden_size)))


class LeJEPAMLP(eqx.Module):
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
    ) -> LeJEPAMLP:
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


class LeJEPABlock(eqx.Module):
    attention_norm: LayerNorm
    attention: LeJEPAAttention
    mlp_norm: LayerNorm
    mlp: LeJEPAMLP

    @classmethod
    def init(
        cls,
        config: LeJEPAViTConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> LeJEPABlock:
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            attention=LeJEPAAttention.init(
                config.hidden_size,
                key=attention_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
            mlp_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            mlp=LeJEPAMLP.init(
                config.hidden_size,
                config.intermediate_size,
                key=mlp_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
        )

    def __call__(
        self,
        hidden: Array,
        *,
        heads: int,
        implementation: AttentionImplementation,
        drop_path_probability: Array,
        key: PRNGKeyArray | None,
    ) -> Array:
        attention_key = mlp_key = None
        if key is not None:
            attention_key, mlp_key = jax.random.split(key)
        attention = self.attention(
            self.attention_norm(hidden),
            heads=heads,
            implementation=implementation,
        )
        hidden = hidden + _stochastic_depth(
            attention,
            drop_path_probability,
            key=attention_key,
        )
        return hidden + _stochastic_depth(
            self.mlp(self.mlp_norm(hidden)),
            drop_path_probability,
            key=mlp_key,
        )


class LeJEPABlockStack(eqx.Module):
    blocks: LeJEPABlock
    depth: int = eqx.field(static=True)

    @classmethod
    def from_blocks(cls, blocks: tuple[LeJEPABlock, ...]) -> LeJEPABlockStack:
        if not blocks:
            raise ValueError("LeJEPA ViTs require at least one transformer block")
        compute_blocks = tuple(
            jax.tree.map(
                lambda value: (
                    value.input_major() if isinstance(value, Linear) else value
                ),
                block,
                is_leaf=lambda value: isinstance(value, Linear),
            )
            for block in blocks
        )
        return cls(
            blocks=jax.tree.map(lambda *values: jnp.stack(values), *compute_blocks),
            depth=len(blocks),
        )

    def layer(self, index: int) -> LeJEPABlock:
        if not 0 <= index < self.depth:
            raise IndexError(index)
        block = jax.tree.map(lambda value: value[index], self.blocks)
        return jax.tree.map(
            lambda value: value.output_major() if isinstance(value, Linear) else value,
            block,
            is_leaf=lambda value: isinstance(value, Linear),
        )

    def __call__(
        self,
        hidden: Array,
        *,
        config: LeJEPAViTConfig,
        implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
        key: PRNGKeyArray | None,
    ) -> Array:
        keys = jax.random.split(jax.random.key(0) if key is None else key, self.depth)
        training = key is not None
        rates = jnp.linspace(0.0, config.drop_path_rate, self.depth)

        def apply_block(carry, values):
            index, block_key, rate = values
            current, _ = carry
            block = jax.tree.map(
                lambda value: jax.lax.dynamic_index_in_dim(
                    value,
                    index,
                    axis=0,
                    keepdims=False,
                ),
                self.blocks,
            )
            output = block(
                current,
                heads=config.heads,
                implementation=implementation,
                drop_path_probability=rate,
                key=block_key if training else None,
            )
            return (output, current[:, 0]), None

        (hidden, penultimate_cls), _ = jax.lax.scan(
            rematerialize(apply_block, rematerialization),
            (hidden, hidden[:, 0]),
            (jnp.arange(self.depth, dtype=jnp.int32), keys, rates),
        )
        return hidden, penultimate_cls


class LeJEPAViTBackbone(eqx.Module):
    patch_weight: Float[Array, "hidden channel patch_height patch_width"]
    patch_bias: Float[Array, " hidden"]
    class_token: Float[Array, " hidden"]
    position_embedding: Float[Array, "position hidden"]
    blocks: LeJEPABlockStack
    final_norm: LayerNorm
    config: LeJEPAViTConfig = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: LeJEPAViTConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
    ) -> LeJEPAViTBackbone:
        keys = jax.random.split(key, config.depth + 3)
        return cls(
            patch_weight=config.initializer_range
            * jax.random.normal(
                keys[0],
                (
                    config.hidden_size,
                    config.channels,
                    config.patch_size,
                    config.patch_size,
                ),
                dtype,
            ),
            patch_bias=jnp.zeros((config.hidden_size,), dtype=dtype),
            class_token=config.initializer_range
            * jax.random.normal(keys[1], (config.hidden_size,), dtype),
            position_embedding=config.initializer_range
            * jax.random.normal(
                keys[2],
                (config.patch_count + 1, config.hidden_size),
                dtype,
            ),
            blocks=LeJEPABlockStack.from_blocks(
                tuple(
                    LeJEPABlock.init(config, key=block_key, dtype=dtype)
                    for block_key in keys[3:]
                )
            ),
            final_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            config=config,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )

    @classmethod
    def from_timm_state_dict(
        cls,
        config: LeJEPAViTConfig,
        state: Mapping[str, Any],
        *,
        dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
    ) -> LeJEPAViTBackbone:
        """Convert one exact timm VisionTransformer state dictionary."""

        def array(name: str) -> Array:
            value = state[name]
            detach = getattr(value, "detach", None)
            if callable(detach):
                value = detach().cpu().numpy()
            return jnp.asarray(value, dtype=dtype)

        def linear(prefix: str) -> Linear:
            return Linear(
                weight=array(prefix + ".weight"), bias=array(prefix + ".bias")
            )

        def norm(prefix: str) -> LayerNorm:
            return LayerNorm(
                weight=array(prefix + ".weight"),
                bias=array(prefix + ".bias"),
                epsilon=config.layer_norm_epsilon,
            )

        blocks = []
        for index in range(config.depth):
            prefix = f"blocks.{index}"
            blocks.append(
                LeJEPABlock(
                    attention_norm=norm(prefix + ".norm1"),
                    attention=LeJEPAAttention(
                        qkv=linear(prefix + ".attn.qkv"),
                        output=linear(prefix + ".attn.proj"),
                    ),
                    mlp_norm=norm(prefix + ".norm2"),
                    mlp=LeJEPAMLP(
                        up=linear(prefix + ".mlp.fc1"),
                        down=linear(prefix + ".mlp.fc2"),
                    ),
                )
            )
        position = array("pos_embed")
        class_token = array("cls_token")
        return cls(
            patch_weight=array("patch_embed.proj.weight"),
            patch_bias=array("patch_embed.proj.bias"),
            class_token=class_token.reshape(-1, config.hidden_size)[0],
            position_embedding=position.reshape(-1, config.hidden_size),
            blocks=LeJEPABlockStack.from_blocks(tuple(blocks)),
            final_norm=norm("norm"),
            config=config,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )

    def _positions(self, grid_height: int, grid_width: int, dtype: jnp.dtype) -> Array:
        if (grid_height, grid_width) == (self.config.grid_size,) * 2:
            return self.position_embedding.astype(dtype)
        prefix = self.position_embedding[:1]
        patches = self.position_embedding[1:].reshape(
            (1, self.config.grid_size, self.config.grid_size, self.config.hidden_size)
        )
        patches = jax.image.resize(
            patches,
            (1, grid_height, grid_width, self.config.hidden_size),
            method="cubic",
            antialias=True,
        ).reshape((grid_height * grid_width, self.config.hidden_size))
        return jnp.concatenate((prefix, patches), axis=0).astype(dtype)

    def _block_outputs(
        self,
        pixels: Float[Array, "batch channel height width"],
        *,
        key: PRNGKeyArray | None,
    ) -> tuple[Array, Array]:
        if pixels.ndim != 4 or pixels.shape[1] != self.config.channels:
            raise ValueError("LeJEPA backbone expects [batch, channel, height, width]")
        if min(pixels.shape[-2:]) < self.config.patch_size:
            raise ValueError("LeJEPA images must contain at least one complete patch")
        compute_dtype = active_compute_dtype(pixels.dtype)
        patches = jax.lax.conv_general_dilated(
            pixels.astype(compute_dtype),
            compute_parameter(self.patch_weight).astype(compute_dtype),
            window_strides=(self.config.patch_size,) * 2,
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        patches = (
            patches
            + compute_parameter(self.patch_bias).astype(compute_dtype)[
                None, :, None, None
            ]
        )
        grid_height, grid_width = patches.shape[-2:]
        patches = jnp.moveaxis(patches, 1, -1).reshape(
            (pixels.shape[0], grid_height * grid_width, self.config.hidden_size)
        )
        class_token = jnp.broadcast_to(
            self.class_token.astype(compute_dtype)[None, None],
            (pixels.shape[0], 1, self.config.hidden_size),
        )
        hidden = jnp.concatenate((class_token, patches), axis=1)
        hidden = (
            hidden
            + self._positions(
                grid_height,
                grid_width,
                compute_dtype,
            )[None]
        )
        return self.blocks(
            hidden,
            config=self.config,
            implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
            key=key,
        )

    def last_two_cls(
        self,
        pixels: Float[Array, "batch channel height width"],
        *,
        key: PRNGKeyArray | None,
    ) -> Float[Array, "two batch hidden"]:
        """Return raw CLS states from the final two transformer blocks."""

        final_hidden, penultimate_cls = self._block_outputs(pixels, key=key)
        return jnp.stack((penultimate_cls, final_hidden[:, 0]))

    def __call__(
        self,
        pixels: Float[Array, "batch channel height width"],
        *,
        key: PRNGKeyArray | None,
    ) -> Float[Array, "batch hidden"]:
        final_hidden, _ = self._block_outputs(pixels, key=key)
        return self.final_norm(final_hidden)[:, 0]


class _ProjectionBatchNorm(eqx.Module):
    weight: Array
    bias: Array
    epsilon: float = eqx.field(static=True, default=1e-5)

    @classmethod
    def init(cls, size: int, *, dtype: jnp.dtype) -> _ProjectionBatchNorm:
        return cls(jnp.ones((size,), dtype=dtype), jnp.zeros((size,), dtype=dtype))

    def __call__(self, value: Array) -> Array:
        source_dtype = value.dtype
        value = value.astype(jnp.float32)
        mean = jnp.mean(value, axis=0, keepdims=True)
        variance = jnp.mean(jnp.square(value - mean), axis=0, keepdims=True)
        normalized = (value - mean) * jax.lax.rsqrt(variance + self.epsilon)
        return (
            normalized * self.weight.astype(jnp.float32) + self.bias.astype(jnp.float32)
        ).astype(source_dtype)


class LeJEPAProjectionMLP(eqx.Module):
    input: Linear
    hidden_one: Linear
    norm_one: _ProjectionBatchNorm
    hidden_two: Linear
    norm_two: _ProjectionBatchNorm
    output: Linear

    @classmethod
    def init(
        cls,
        config: LeJEPAViTConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> LeJEPAProjectionMLP:
        keys = jax.random.split(key, 4)
        scale = config.initializer_range
        return cls(
            input=Linear.init(
                config.hidden_size,
                config.projector_bottleneck,
                key=keys[0],
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            hidden_one=Linear.init(
                config.projector_bottleneck,
                config.projector_hidden_size,
                key=keys[1],
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            norm_one=_ProjectionBatchNorm.init(
                config.projector_hidden_size,
                dtype=dtype,
            ),
            hidden_two=Linear.init(
                config.projector_hidden_size,
                config.projector_hidden_size,
                key=keys[2],
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            norm_two=_ProjectionBatchNorm.init(
                config.projector_hidden_size,
                dtype=dtype,
            ),
            output=Linear.init(
                config.projector_hidden_size,
                config.projection_dimension,
                key=keys[3],
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(self, representations: Array) -> Array:
        hidden = self.input(representations)
        hidden = jax.nn.relu(self.norm_one(self.hidden_one(hidden)))
        hidden = jax.nn.relu(self.norm_two(self.hidden_two(hidden)))
        return self.output(hidden)


class LeJEPAModel(eqx.Module):
    """LeJEPA composition with public backbone encoding and training projection."""

    backbone: LeJEPAViTBackbone
    projector: LeJEPAProjectionMLP
    evaluation_norm: LayerNorm
    metadata: EncoderMetadata
    config: LeJEPAViTConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: LeJEPAViTConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/lejepa-vit-b-16",
        revision: str = "random-init",
    ) -> LeJEPAModel:
        backbone_key, projector_key = jax.random.split(key)
        return cls(
            backbone=LeJEPAViTBackbone.init(
                config,
                key=backbone_key,
                dtype=parameter_dtype,
                attention_implementation=attention_implementation,
                rematerialization=rematerialization,
            ),
            projector=LeJEPAProjectionMLP.init(
                config,
                key=projector_key,
                dtype=parameter_dtype,
            ),
            evaluation_norm=LayerNorm.init(
                2 * config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=parameter_dtype,
                bias=True,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=2 * config.hidden_size,
                routes=frozenset({Route.GENERIC}),
                modalities=frozenset({Modality.IMAGE}),
            ),
            config=config,
        )

    @classmethod
    def from_timm_state_dict(
        cls,
        config: LeJEPAViTConfig,
        state: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/lejepa-vit-b-16",
        revision: str = "shared-timm-initialization",
    ) -> LeJEPAModel:
        """Convert one matched timm/stable-pretraining LeJEPA state."""

        def array(name: str) -> Array:
            value = state[name]
            detach = getattr(value, "detach", None)
            if callable(detach):
                value = detach().cpu().numpy()
            return jnp.asarray(value, dtype=parameter_dtype)

        def linear(prefix: str) -> Linear:
            return Linear(
                weight=array(prefix + ".weight"),
                bias=array(prefix + ".bias"),
            )

        backbone_prefix = "backbone."
        backbone_state = {
            name.removeprefix(backbone_prefix): value
            for name, value in state.items()
            if name.startswith(backbone_prefix)
        }
        projector = LeJEPAProjectionMLP(
            input=linear("projector.0"),
            hidden_one=linear("projector.1.0"),
            norm_one=_ProjectionBatchNorm(
                weight=array("projector.1.1.weight"),
                bias=array("projector.1.1.bias"),
            ),
            hidden_two=linear("projector.1.4"),
            norm_two=_ProjectionBatchNorm(
                weight=array("projector.1.5.weight"),
                bias=array("projector.1.5.bias"),
            ),
            output=linear("projector.1.8"),
        )
        return cls(
            backbone=LeJEPAViTBackbone.from_timm_state_dict(
                config,
                backbone_state,
                dtype=parameter_dtype,
                attention_implementation=attention_implementation,
                rematerialization=rematerialization,
            ),
            projector=projector,
            evaluation_norm=LayerNorm(
                weight=array("evaluation_norm.weight"),
                bias=array("evaluation_norm.bias"),
                epsilon=config.layer_norm_epsilon,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=2 * config.hidden_size,
                routes=frozenset({Route.GENERIC}),
                modalities=frozenset({Modality.IMAGE}),
            ),
            config=config,
        )

    def _multicrop_backbone(
        self,
        inputs: LeJEPAMulticropImages,
        *,
        key: PRNGKeyArray | None,
    ) -> Array:
        rows = inputs.pixel_values.shape[0]
        batch = rows // inputs.view_count
        grouped = inputs.pixel_values.reshape(
            (batch, inputs.view_count, *inputs.pixel_values.shape[1:])
        )
        global_pixels = grouped[:, : inputs.global_views].reshape(
            (batch * inputs.global_views, *grouped.shape[2:])
        )
        local_count = inputs.view_count - inputs.global_views
        size = self.config.local_image_size
        local_pixels = grouped[:, inputs.global_views :, :, :size, :size].reshape(
            (batch * local_count, grouped.shape[2], size, size)
        )
        global_key = local_key = None
        if key is not None:
            global_key, local_key = jax.random.split(key)
        global_features = self.backbone(global_pixels, key=global_key).reshape(
            (batch, inputs.global_views, self.config.hidden_size)
        )
        local_features = self.backbone(local_pixels, key=local_key).reshape(
            (batch, local_count, self.config.hidden_size)
        )
        return jnp.concatenate((global_features, local_features), axis=1).reshape(
            (rows, self.config.hidden_size)
        )

    def project(
        self,
        inputs: LeJEPAMulticropImages,
        *,
        key: PRNGKeyArray | None,
    ) -> Array:
        """Return training projections while retaining backbone gradients."""

        return self.projector(self._multicrop_backbone(inputs, key=key))

    def training_filter(self) -> Any:
        """Train the backbone/projector while keeping probe normalization fixed."""

        selected = jax.tree.map(eqx.is_inexact_array, self)
        frozen_norm = jax.tree.map(lambda _value: False, self.evaluation_norm)
        return eqx.tree_at(
            lambda model: model.evaluation_norm,
            selected,
            frozen_norm,
        )

    def encode(
        self,
        inputs: Any,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        del route
        if isinstance(inputs, LeJEPAMulticropImages):
            raise TypeError(
                "LeJEPAMulticropImages are training inputs; call project explicitly"
            )
        pixels = jnp.asarray(inputs)
        last_two = self.backbone.last_two_cls(pixels, key=key)
        concatenated = jnp.concatenate((last_two[0], last_two[1]), axis=-1)
        return self.evaluation_norm(concatenated)


__all__ = [
    "LeJEPAAttention",
    "LeJEPABlock",
    "LeJEPAMLP",
    "LeJEPAModel",
    "LeJEPAMulticropImages",
    "LeJEPAProjectionMLP",
    "LeJEPAViTBackbone",
]
