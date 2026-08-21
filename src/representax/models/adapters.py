"""Low-rank model adapters with packed frozen base parameters."""

from __future__ import annotations

import re
from typing import Any, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, UInt8, UInt16

from representax.core.sharding import (
    activation_out_sharding,
    constrain_activation,
    replicate,
)
from representax.precision import (
    activation_inputs,
    active_compute_dtype,
    compute_parameter,
    linear_matmul,
)

from .components import Linear

ModelT = TypeVar("ModelT", bound=eqx.Module)


def _bits_from_bfloat16(value: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(value.astype(jnp.bfloat16), jnp.uint16)


def _bfloat16_from_bits(value: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(value, jnp.bfloat16)


def _pack_int4_rows(
    weight: Float[Array, "*stack output input"],
) -> tuple[
    UInt8[Array, "*stack output packed"],
    UInt16[Array, "*stack output"],
]:
    numeric = weight.astype(jnp.float32)
    maximum = jnp.max(jnp.abs(numeric), axis=-1)
    scale = jnp.where(maximum > 0, maximum / 7.0, 1.0).astype(jnp.bfloat16)
    quantized = jnp.clip(
        jnp.rint(numeric / scale.astype(jnp.float32)[..., None]),
        -7,
        7,
    ).astype(jnp.int8)
    unsigned = jnp.bitwise_and(quantized, jnp.asarray(0x0F, dtype=jnp.int8)).astype(
        jnp.uint8
    )
    if unsigned.shape[-1] % 2:
        unsigned = jnp.pad(
            unsigned,
            ((0, 0),) * (unsigned.ndim - 1) + ((0, 1),),
        )
    pairs = unsigned.reshape((*unsigned.shape[:-1], -1, 2))
    packed = pairs[..., 0] | (pairs[..., 1] << jnp.uint8(4))
    return packed, _bits_from_bfloat16(scale)


def _unpack_int4_rows(
    packed: UInt8[Array, "*stack output packed"],
    scale_bits: UInt16[Array, "*stack output"],
    input_size: int,
) -> Float[Array, "*stack output input"]:
    low = packed & jnp.uint8(0x0F)
    high = packed >> jnp.uint8(4)
    unsigned = jnp.stack((low, high), axis=-1).reshape((*packed.shape[:-1], -1))
    unsigned = unsigned[..., :input_size].astype(jnp.int8)
    signed = jnp.where(unsigned < 8, unsigned, unsigned - 16)
    scale = _bfloat16_from_bits(scale_bits)
    return signed.astype(jnp.bfloat16) * scale[..., None]


class QuantizedLoRALinear(eqx.Module):
    """Packed INT4 base projection with BF16 compute and FP32 LoRA masters."""

    packed_weight: UInt8[Array, "*stack output packed"]
    scale_bits: UInt16[Array, "*stack output"]
    bias_bits: UInt16[Array, "*stack output"] | None
    lora_a: Float[Array, "*stack rank input"]
    lora_b: Float[Array, "*stack output rank"]
    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @classmethod
    def from_linear(
        cls,
        linear: Linear,
        *,
        rank: int,
        alpha: float,
        key: PRNGKeyArray,
        initialization_scale: float | None = None,
    ) -> QuantizedLoRALinear:
        """Quantize one native projection and add a zero-output adapter."""

        if rank <= 0 or rank > min(linear.weight.shape[-2:]):
            raise ValueError("adapter rank must fit both linear dimensions")
        if alpha <= 0:
            raise ValueError("adapter alpha must be positive")
        packed, scales = _pack_int4_rows(linear.weight)
        scale = (
            linear.weight.shape[-1] ** -0.5
            if initialization_scale is None
            else initialization_scale
        )
        return cls(
            packed_weight=packed,
            scale_bits=scales,
            bias_bits=(
                None if linear.bias is None else _bits_from_bfloat16(linear.bias)
            ),
            lora_a=(
                scale
                * jax.random.normal(
                    key,
                    (*linear.weight.shape[:-2], rank, linear.weight.shape[-1]),
                    dtype=jnp.float32,
                )
            ),
            lora_b=jnp.zeros(
                (*linear.weight.shape[:-2], linear.weight.shape[-2], rank),
                dtype=jnp.float32,
            ),
            input_size=linear.weight.shape[-1],
            output_size=linear.weight.shape[-2],
            rank=rank,
            alpha=alpha,
        )

    def base_weight(self) -> Float[Array, "*stack output input"]:
        """Materialize the BF16 base only at its projection use boundary."""

        return _unpack_int4_rows(
            replicate(self.packed_weight),
            replicate(self.scale_bits),
            self.input_size,
        )

    def merged_weight(self) -> Float[Array, "*stack output input"]:
        """Return the ordinary FP32 weight represented by base plus adapter."""

        adapter = self.lora_b @ self.lora_a
        return self.base_weight().astype(jnp.float32) + adapter * (
            self.alpha / self.rank
        )

    def __call__(
        self,
        value: Float[Array, "*batch input"],
    ) -> Float[Array, "*batch output"]:
        value = activation_inputs(value)
        compute_dtype = active_compute_dtype(value.dtype)
        output = linear_matmul(
            value,
            self.base_weight().astype(compute_dtype).T,
            out_sharding=activation_out_sharding(value.ndim),
        )
        adapter_hidden = linear_matmul(
            value,
            compute_parameter(self.lora_a).T,
        )
        adapter_output = linear_matmul(
            adapter_hidden,
            compute_parameter(self.lora_b).T,
        )
        output = output + adapter_output * (self.alpha / self.rank)
        if self.bias_bits is not None:
            output = output + _bfloat16_from_bits(replicate(self.bias_bits)).astype(
                compute_dtype
            )
        return constrain_activation(output)

    def merge(self) -> Linear:
        """Materialize a conventional projection for export or inference."""

        return Linear(
            weight=self.merged_weight(),
            bias=(
                None
                if self.bias_bits is None
                else _bfloat16_from_bits(self.bias_bits).astype(jnp.float32)
            ),
        )


def _path_name(path: tuple[Any, ...]) -> str:
    return jax.tree_util.keystr(path)


def apply_quantized_lora(
    model: ModelT,
    *,
    rank: int,
    alpha: float,
    key: PRNGKeyArray,
    target_pattern: str = ".*",
    initialization_scale: float | None = None,
) -> ModelT:
    """Replace matching native Linear leaves without architecture-specific code."""

    pattern = re.compile(target_pattern)
    path_leaves, structure = jax.tree_util.tree_flatten_with_path(
        model,
        is_leaf=lambda value: isinstance(value, Linear),
    )
    matched = [
        (path, leaf)
        for path, leaf in path_leaves
        if isinstance(leaf, Linear) and pattern.search(_path_name(path))
    ]
    if not matched:
        raise ValueError("adapter target_pattern matched no native Linear modules")
    keys = iter(jax.random.split(key, len(matched)))
    replacements: dict[str, QuantizedLoRALinear] = {
        _path_name(path): QuantizedLoRALinear.from_linear(
            leaf,
            rank=rank,
            alpha=alpha,
            key=next(keys),
            initialization_scale=initialization_scale,
        )
        for path, leaf in matched
    }
    leaves = [replacements.get(_path_name(path), leaf) for path, leaf in path_leaves]
    return cast(ModelT, structure.unflatten(leaves))


def merge_quantized_lora(model: ModelT) -> ModelT:
    """Replace every packed adapter projection with an ordinary merged Linear."""

    return cast(
        ModelT,
        jax.tree.map(
            lambda value: (
                value.merge() if isinstance(value, QuantizedLoRALinear) else value
            ),
            model,
            is_leaf=lambda value: isinstance(value, QuantizedLoRALinear),
        ),
    )


def lora_parameter_filter(model: eqx.Module) -> Any:
    """Select exactly trainable LoRA arrays from an adapted model tree."""

    selected_names = {"lora_a", "lora_b"}
    return jax.tree_util.tree_map_with_path(
        lambda path, value: (
            eqx.is_inexact_array(value)
            and any(
                isinstance(key, jax.tree_util.GetAttrKey) and key.name in selected_names
                for key in path
            )
        ),
        model,
    )


__all__ = [
    "QuantizedLoRALinear",
    "apply_quantized_lora",
    "lora_parameter_filter",
    "merge_quantized_lora",
]
