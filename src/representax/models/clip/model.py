"""Native Equinox CLIP dual encoder with optional BGE-VL late fusion."""

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
    dot_product_attention,
    embedding_lookup,
    l2_normalize,
    rematerialize,
)
from representax.planning import RematerializationPolicy
from representax.precision import (
    active_compute_dtype,
    cast_floating_tree,
    compute_parameter,
)

from .config import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


class CLIPBatch(eqx.Module):
    """Aligned text, image, or composed text-image samples."""

    input_ids: Int[Array, "batch sequence"] | None = None
    attention_mask: (
        Bool[Array, "batch sequence"] | Int[Array, "batch sequence"] | None
    ) = None
    text_valid: Bool[Array, " batch"] | None = None
    pixel_values: Float[Array, "batch channel height width"] | None = None
    image_valid: Bool[Array, " batch"] | None = None

    @property
    def batch_size(self) -> int:
        if self.input_ids is not None:
            return self.input_ids.shape[0]
        assert self.pixel_values is not None
        return self.pixel_values.shape[0]

    def __post_init__(self) -> None:
        if self.input_ids is None:
            if self.attention_mask is not None or self.text_valid is not None:
                raise ValueError("text masks require input_ids")
        else:
            if self.input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence]")
            if (
                self.attention_mask is None
                or self.attention_mask.shape != self.input_ids.shape
            ):
                raise ValueError("attention_mask must align with input_ids")
            if (
                self.text_valid is None
                or self.text_valid.shape != self.input_ids.shape[:1]
            ):
                raise ValueError("text_valid must contain one flag per sample")
        if self.pixel_values is None:
            if self.image_valid is not None:
                raise ValueError("image_valid requires pixel_values")
        else:
            if self.pixel_values.ndim != 4:
                raise ValueError(
                    "pixel_values must have shape [batch, channel, height, width]"
                )
            if (
                self.image_valid is None
                or self.image_valid.shape != self.pixel_values.shape[:1]
            ):
                raise ValueError("image_valid must contain one flag per sample")
        if self.input_ids is None and self.pixel_values is None:
            raise ValueError("CLIP batches require text, images, or both")
        if (
            self.input_ids is not None
            and self.pixel_values is not None
            and self.input_ids.shape[0] != self.pixel_values.shape[0]
        ):
            raise ValueError("text and image inputs must share the sample dimension")


class CLIPAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        hidden_size: int,
        *,
        initializer_range: float,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> CLIPAttention:
        keys = jax.random.split(key, 4)
        return cls(
            query=Linear.init(
                hidden_size,
                hidden_size,
                key=keys[0],
                scale=initializer_range,
                dtype=dtype,
                bias=True,
            ),
            key=Linear.init(
                hidden_size,
                hidden_size,
                key=keys[1],
                scale=initializer_range,
                dtype=dtype,
                bias=True,
            ),
            value=Linear.init(
                hidden_size,
                hidden_size,
                key=keys[2],
                scale=initializer_range,
                dtype=dtype,
                bias=True,
            ),
            output=Linear.init(
                hidden_size,
                hidden_size,
                key=keys[3],
                scale=initializer_range,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        mask: Bool[Array, "#batch one sequence sequence"] | None,
        *,
        heads: int,
        attention_dropout: float,
        key: PRNGKeyArray | None,
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        batch, sequence, hidden_size = hidden.shape
        head_dimension = hidden_size // heads

        def project(linear: Linear) -> Float[Array, "batch sequence heads head"]:
            return linear(hidden).reshape((batch, sequence, heads, head_dimension))

        attended = dot_product_attention(
            project(self.query),
            project(self.key),
            project(self.value),
            attention_mask=mask,
            dropout_probability=attention_dropout,
            dropout_key=key,
            implementation=implementation,
        )
        return self.output(attended.reshape((batch, sequence, hidden_size)))


def _activate(
    hidden: Float[Array, "*batch hidden"],
    name: str,
) -> Float[Array, "*batch hidden"]:
    if name == "quick_gelu":
        return hidden * jax.nn.sigmoid(1.702 * hidden)
    if name == "gelu":
        return jax.nn.gelu(hidden, approximate=False)
    if name == "gelu_new":
        return jax.nn.gelu(hidden, approximate=True)
    raise ValueError(f"unsupported CLIP activation {name!r}")


class CLIPMLP(eqx.Module):
    up: Linear
    down: Linear

    @classmethod
    def init(
        cls,
        hidden_size: int,
        intermediate_size: int,
        *,
        initializer_range: float,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> CLIPMLP:
        up_key, down_key = jax.random.split(key)
        return cls(
            up=Linear.init(
                hidden_size,
                intermediate_size,
                key=up_key,
                scale=initializer_range,
                dtype=dtype,
                bias=True,
            ),
            down=Linear.init(
                intermediate_size,
                hidden_size,
                key=down_key,
                scale=initializer_range,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(self, hidden: jax.Array, *, activation: str) -> jax.Array:
        return self.down(_activate(self.up(hidden), activation))


class CLIPLayer(eqx.Module):
    attention_norm: LayerNorm
    attention: CLIPAttention
    mlp_norm: LayerNorm
    mlp: CLIPMLP

    @classmethod
    def init(
        cls,
        config: CLIPTextConfig | CLIPVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> CLIPLayer:
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            attention=CLIPAttention.init(
                config.hidden_size,
                initializer_range=config.initializer_range,
                key=attention_key,
                dtype=dtype,
            ),
            mlp_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            mlp=CLIPMLP.init(
                config.hidden_size,
                config.intermediate_size,
                initializer_range=config.initializer_range,
                key=mlp_key,
                dtype=dtype,
            ),
        )

    def __call__(
        self,
        hidden: jax.Array,
        mask: jax.Array | None,
        *,
        config: CLIPTextConfig | CLIPVisionConfig,
        key: PRNGKeyArray | None,
        implementation: AttentionImplementation,
    ) -> jax.Array:
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            mask,
            heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            key=key,
            implementation=implementation,
        )
        return hidden + self.mlp(
            self.mlp_norm(hidden),
            activation=config.hidden_activation,
        )


class CLIPLayerStack(eqx.Module):
    layers: CLIPLayer
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(cls, values: tuple[CLIPLayer, ...]) -> CLIPLayerStack:
        if not values:
            raise ValueError("CLIP towers require at least one layer")
        return cls(
            layers=jax.tree.map(lambda *items: jnp.stack(items), *values),
            depth=len(values),
        )

    def layer(self, index: int) -> CLIPLayer:
        if not 0 <= index < self.depth:
            raise IndexError(index)
        return jax.tree.map(lambda value: value[index], self.layers)

    def __call__(
        self,
        hidden: jax.Array,
        mask: jax.Array | None,
        *,
        config: CLIPTextConfig | CLIPVisionConfig,
        key: PRNGKeyArray | None,
        implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> jax.Array:
        keys = jax.random.split(
            jax.random.key(0) if key is None else key,
            self.depth,
        )
        training = key is not None

        def apply_layer(carry, values):
            layer, layer_key = values
            output = layer(
                carry,
                mask,
                config=config,
                key=layer_key if training else None,
                implementation=implementation,
            )
            return output, None

        hidden, _ = jax.lax.scan(
            rematerialize(apply_layer, rematerialization),
            hidden,
            (self.layers, keys),
        )
        return hidden


class CLIPTextTower(eqx.Module):
    token_embedding: Float[Array, "vocabulary hidden"]
    position_embedding: Float[Array, "position hidden"]
    layers: CLIPLayerStack
    final_norm: LayerNorm

    @classmethod
    def init(
        cls,
        config: CLIPTextConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> CLIPTextTower:
        keys = jax.random.split(key, config.num_hidden_layers + 2)
        return cls(
            token_embedding=config.initializer_range
            * jax.random.normal(
                keys[0],
                (config.vocab_size, config.hidden_size),
                dtype=dtype,
            ),
            position_embedding=config.initializer_range
            * jax.random.normal(
                keys[1],
                (config.max_position_embeddings, config.hidden_size),
                dtype=dtype,
            ),
            layers=CLIPLayerStack.from_layers(
                tuple(
                    CLIPLayer.init(config, key=layer_key, dtype=dtype)
                    for layer_key in keys[2:]
                )
            ),
            final_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch sequence"],
        attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
        *,
        config: CLIPTextConfig,
        key: PRNGKeyArray | None,
        compute_dtype: jnp.dtype,
        implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "batch hidden"]:
        sequence = input_ids.shape[1]
        if sequence > config.max_position_embeddings:
            raise ValueError("CLIP text sequence exceeds the position table")
        hidden = embedding_lookup(self.token_embedding, input_ids)
        hidden = hidden + self.position_embedding[:sequence][None]
        hidden = hidden.astype(compute_dtype)
        positions = jnp.arange(sequence)
        causal = positions[:, None] >= positions[None, :]
        mask = causal[None, None] & attention_mask.astype(bool)[:, None, None, :]
        hidden = self.layers(
            hidden,
            mask,
            config=config,
            key=key,
            implementation=implementation,
            rematerialization=rematerialization,
        )
        hidden = self.final_norm(hidden)
        if config.eos_token_id == 2:
            pooled_indices = jnp.argmax(input_ids, axis=-1)
        else:
            pooled_indices = jnp.argmax(input_ids == config.eos_token_id, axis=-1)
        return hidden[jnp.arange(hidden.shape[0]), pooled_indices]


class CLIPVisionTower(eqx.Module):
    patch_weight: Float[Array, "hidden channel patch_height patch_width"]
    class_embedding: Float[Array, " hidden"]
    position_embedding: Float[Array, "position hidden"]
    pre_norm: LayerNorm
    layers: CLIPLayerStack
    post_norm: LayerNorm

    @classmethod
    def init(
        cls,
        config: CLIPVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> CLIPVisionTower:
        keys = jax.random.split(key, config.num_hidden_layers + 3)
        return cls(
            patch_weight=config.initializer_range
            * jax.random.normal(
                keys[0],
                (
                    config.hidden_size,
                    config.num_channels,
                    config.patch_size,
                    config.patch_size,
                ),
                dtype=dtype,
            ),
            class_embedding=config.initializer_range
            * jax.random.normal(keys[1], (config.hidden_size,), dtype=dtype),
            position_embedding=config.initializer_range
            * jax.random.normal(
                keys[2],
                (config.patch_count + 1, config.hidden_size),
                dtype=dtype,
            ),
            pre_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            layers=CLIPLayerStack.from_layers(
                tuple(
                    CLIPLayer.init(config, key=layer_key, dtype=dtype)
                    for layer_key in keys[3:]
                )
            ),
            post_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        pixels: Float[Array, "batch channel height width"],
        *,
        config: CLIPVisionConfig,
        key: PRNGKeyArray | None,
        compute_dtype: jnp.dtype,
        implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "batch hidden"]:
        if pixels.shape[1:] != (
            config.num_channels,
            config.image_size,
            config.image_size,
        ):
            raise ValueError("pixel_values do not match the configured CLIP image size")
        grid_size = config.image_size // config.patch_size
        patches = pixels.astype(compute_dtype).reshape(
            (
                pixels.shape[0],
                config.num_channels,
                grid_size,
                config.patch_size,
                grid_size,
                config.patch_size,
            )
        )
        patches = patches.transpose((0, 2, 4, 1, 3, 5)).reshape(
            (pixels.shape[0], config.patch_count, -1)
        )
        patch_weight = compute_parameter(self.patch_weight).astype(compute_dtype)
        patches = patches @ patch_weight.reshape((config.hidden_size, -1)).T
        class_token = jnp.broadcast_to(
            self.class_embedding.astype(compute_dtype)[None, None],
            (patches.shape[0], 1, config.hidden_size),
        )
        hidden = jnp.concatenate((class_token, patches), axis=1)
        hidden = hidden + self.position_embedding.astype(compute_dtype)[None]
        hidden = self.pre_norm(hidden)
        hidden = self.layers(
            hidden,
            None,
            config=config,
            key=key,
            implementation=implementation,
            rematerialization=rematerialization,
        )
        return self.post_norm(hidden[:, 0])


class CLIPEncoder(eqx.Module):
    """CLIP text/image encoder with BGE-VL additive composition."""

    text: CLIPTextTower
    vision: CLIPVisionTower
    text_projection: Linear
    vision_projection: Linear
    logit_scale: Float[Array, ""]
    metadata: EncoderMetadata
    config: CLIPConfig = eqx.field(static=True)
    normalize_output: bool = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        from .loading import load_clip

        return load_clip(model_name_or_path, **options)

    @classmethod
    def init(
        cls,
        config: CLIPConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/clip",
        revision: str = "random-init",
        normalize_output: bool = False,
    ) -> CLIPEncoder:
        text_key, vision_key, projection_key = jax.random.split(key, 3)
        text_projection_key, vision_projection_key = jax.random.split(projection_key)
        return cls(
            text=CLIPTextTower.init(config.text, key=text_key, dtype=parameter_dtype),
            vision=CLIPVisionTower.init(
                config.vision,
                key=vision_key,
                dtype=parameter_dtype,
            ),
            text_projection=Linear.init(
                config.text.hidden_size,
                config.projection_dimension,
                key=text_projection_key,
                scale=config.text.initializer_range,
                dtype=parameter_dtype,
            ),
            vision_projection=Linear.init(
                config.vision.hidden_size,
                config.projection_dimension,
                key=vision_projection_key,
                scale=config.vision.initializer_range,
                dtype=parameter_dtype,
            ),
            logit_scale=jnp.asarray(
                config.logit_scale_initial_value,
                dtype=parameter_dtype,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.projection_dimension,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
            ),
            config=config,
            normalize_output=normalize_output,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )

    def text_features(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        compute_dtype = active_compute_dtype(self.compute_dtype)
        tower = cast_floating_tree(self.text, compute_dtype)
        projection = cast_floating_tree(self.text_projection, compute_dtype)
        pooled = tower(
            input_ids,
            attention_mask,
            config=self.config.text,
            key=key,
            compute_dtype=compute_dtype,
            implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )
        return projection(pooled)

    def image_features(
        self,
        pixel_values: jax.Array,
        *,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        compute_dtype = active_compute_dtype(self.compute_dtype)
        tower = cast_floating_tree(self.vision, compute_dtype)
        projection = cast_floating_tree(self.vision_projection, compute_dtype)
        pooled = tower(
            pixel_values,
            config=self.config.vision,
            key=key,
            compute_dtype=compute_dtype,
            implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )
        return projection(pooled)

    def encode(
        self,
        inputs: CLIPBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        if not isinstance(inputs, CLIPBatch):
            raise TypeError("CLIP inputs must be CLIPBatch")
        if key is None:
            text_key = vision_key = None
        else:
            text_key, vision_key = jax.random.split(key)
        combined = jnp.zeros(
            (inputs.batch_size, self.config.projection_dimension),
            dtype=jnp.float32,
        )
        if inputs.input_ids is not None:
            assert inputs.attention_mask is not None
            assert inputs.text_valid is not None
            text = self.text_features(
                inputs.input_ids,
                inputs.attention_mask,
                key=text_key,
            )
            combined = combined + jnp.where(inputs.text_valid[:, None], text, 0)
        if inputs.pixel_values is not None:
            assert inputs.image_valid is not None
            image = self.image_features(inputs.pixel_values, key=vision_key)
            combined = combined + jnp.where(inputs.image_valid[:, None], image, 0)
        return l2_normalize(combined) if self.normalize_output else combined


__all__ = [
    "CLIPBatch",
    "CLIPEncoder",
    "CLIPLayer",
    "CLIPLayerStack",
    "CLIPTextTower",
    "CLIPVisionTower",
]
