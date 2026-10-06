"""Exact Hugging Face checkpoint mapping for native Qwen2/Qwen2.5-VL."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.core import EncoderMetadata, Modality, Route
from representax.integrations.huggingface import load_hf_config, load_safetensor_subset
from representax.models.adapters import LoRALinear
from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    RMSNorm,
)
from representax.models.qwen2_5_omni.text import (
    Qwen2_5OmniTextLayer,
    Qwen2_5OmniTextLayerStack,
    Qwen2_5OmniTextTower,
)
from representax.planning import RematerializationPolicy

from .config import (
    BGE_VL_SCREENSHOT_MODEL_ID,
    BGE_VL_SCREENSHOT_REVISION,
    Qwen2VLConfig,
)
from .model import Qwen2VLEncoder, Qwen2VLReranker
from .vision import (
    Qwen2VLPatchMerger,
    Qwen2VLVisionAttention,
    Qwen2VLVisionBlock,
    Qwen2VLVisionBlockStack,
    Qwen2VLVisionMLP,
    Qwen2VLVisionTower,
)


def qwen2_vl_weight_names(config: Qwen2VLConfig) -> frozenset[str]:
    """Return every tensor consumed by the native backbone."""

    names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "visual.patch_embed.proj.weight",
        "visual.merger.ln_q.weight",
        "visual.merger.mlp.0.weight",
        "visual.merger.mlp.0.bias",
        "visual.merger.mlp.2.weight",
        "visual.merger.mlp.2.bias",
    }
    if config.vision.norm == "layer":
        names.add("visual.merger.ln_q.bias")
    for index in range(config.text.num_hidden_layers):
        prefix = f"model.layers.{index}."
        names.update(
            prefix + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        )
    for index in range(config.vision.depth):
        prefix = f"visual.blocks.{index}."
        suffixes = {
            "norm1.weight",
            "norm2.weight",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
        }
        if config.vision.norm == "layer":
            suffixes.update({"norm1.bias", "norm2.bias"})
            suffixes.update(
                {
                    "mlp.fc1.weight",
                    "mlp.fc1.bias",
                    "mlp.fc2.weight",
                    "mlp.fc2.bias",
                }
            )
        else:
            suffixes.update(
                {
                    "mlp.gate_proj.weight",
                    "mlp.gate_proj.bias",
                    "mlp.up_proj.weight",
                    "mlp.up_proj.bias",
                    "mlp.down_proj.weight",
                    "mlp.down_proj.bias",
                }
            )
        names.update(prefix + suffix for suffix in suffixes)
    return frozenset(names)


def _array(
    state: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    try:
        value = jnp.asarray(state[name], dtype=dtype)
    except KeyError as error:
        raise KeyError(f"Qwen2-VL checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"Qwen2-VL tensor {name} has shape {value.shape}; expected {shape}"
        )
    return value


def _linear(
    state: Mapping[str, Any],
    prefix: str,
    *,
    input_size: int,
    output_size: int,
    dtype: jnp.dtype,
    bias: bool = False,
) -> Linear:
    return Linear(
        weight=_array(state, prefix + ".weight", (output_size, input_size), dtype),
        bias=_array(state, prefix + ".bias", (output_size,), dtype) if bias else None,
    )


def _norm(
    state: Mapping[str, Any],
    prefix: str,
    size: int,
    epsilon: float,
    dtype: jnp.dtype,
    *,
    layer_norm: bool,
) -> RMSNorm | LayerNorm:
    weight = _array(state, prefix + ".weight", (size,), dtype)
    if not layer_norm:
        return RMSNorm(weight, epsilon)
    return LayerNorm(
        weight=weight,
        bias=_array(state, prefix + ".bias", (size,), dtype),
        epsilon=epsilon,
    )


def _bias(layer: Linear | LoRALinear | LayerNorm) -> jax.Array:
    if layer.bias is None:
        raise AssertionError("checkpoint-compatible projection requires a bias")
    return layer.bias


def _weight(layer: Linear | LoRALinear) -> jax.Array:
    return layer.merged_weight() if isinstance(layer, LoRALinear) else layer.weight


def _text_tower(
    config: Qwen2VLConfig,
    state: Mapping[str, Any],
    dtype: jnp.dtype,
) -> Qwen2_5OmniTextTower:
    text = config.text
    attention = text.num_attention_heads * text.head_dimension
    key_value = text.num_key_value_heads * text.head_dimension
    layers = []
    for index in range(text.num_hidden_layers):
        prefix = f"model.layers.{index}."
        layers.append(
            Qwen2_5OmniTextLayer(
                input_norm=RMSNorm(
                    _array(
                        state,
                        prefix + "input_layernorm.weight",
                        (text.hidden_size,),
                        dtype,
                    ),
                    text.norm_epsilon,
                ),
                post_attention_norm=RMSNorm(
                    _array(
                        state,
                        prefix + "post_attention_layernorm.weight",
                        (text.hidden_size,),
                        dtype,
                    ),
                    text.norm_epsilon,
                ),
                query=_linear(
                    state,
                    prefix + "self_attn.q_proj",
                    input_size=text.hidden_size,
                    output_size=attention,
                    dtype=dtype,
                    bias=True,
                ),
                key=_linear(
                    state,
                    prefix + "self_attn.k_proj",
                    input_size=text.hidden_size,
                    output_size=key_value,
                    dtype=dtype,
                    bias=True,
                ),
                value=_linear(
                    state,
                    prefix + "self_attn.v_proj",
                    input_size=text.hidden_size,
                    output_size=key_value,
                    dtype=dtype,
                    bias=True,
                ),
                output=_linear(
                    state,
                    prefix + "self_attn.o_proj",
                    input_size=attention,
                    output_size=text.hidden_size,
                    dtype=dtype,
                ),
                gate=_linear(
                    state,
                    prefix + "mlp.gate_proj",
                    input_size=text.hidden_size,
                    output_size=text.intermediate_size,
                    dtype=dtype,
                ),
                up=_linear(
                    state,
                    prefix + "mlp.up_proj",
                    input_size=text.hidden_size,
                    output_size=text.intermediate_size,
                    dtype=dtype,
                ),
                down=_linear(
                    state,
                    prefix + "mlp.down_proj",
                    input_size=text.intermediate_size,
                    output_size=text.hidden_size,
                    dtype=dtype,
                ),
            )
        )
    return Qwen2_5OmniTextTower(
        token_embedding=_array(
            state,
            "model.embed_tokens.weight",
            (text.vocab_size, text.hidden_size),
            dtype,
        ),
        layers=Qwen2_5OmniTextLayerStack.from_layers(tuple(layers), text.layer_types),
        final_norm=RMSNorm(
            _array(state, "model.norm.weight", (text.hidden_size,), dtype),
            text.norm_epsilon,
        ),
        config=text,
    )


def _vision_tower(
    config: Qwen2VLConfig,
    state: Mapping[str, Any],
    dtype: jnp.dtype,
) -> Qwen2VLVisionTower:
    vision = config.vision
    layer_norm = vision.norm == "layer"
    blocks = []
    for index in range(vision.depth):
        prefix = f"visual.blocks.{index}."
        if vision.mlp != "swiglu":
            first_prefix = prefix + "mlp.fc1"
            gate = None
            second_prefix = prefix + "mlp.fc2"
        else:
            first_prefix = prefix + "mlp.up_proj"
            gate = _linear(
                state,
                prefix + "mlp.gate_proj",
                input_size=vision.hidden_size,
                output_size=vision.intermediate_size,
                dtype=dtype,
                bias=True,
            )
            second_prefix = prefix + "mlp.down_proj"
        blocks.append(
            Qwen2VLVisionBlock(
                attention_norm=_norm(
                    state,
                    prefix + "norm1",
                    vision.hidden_size,
                    vision.norm_epsilon,
                    dtype,
                    layer_norm=layer_norm,
                ),
                attention=Qwen2VLVisionAttention(
                    qkv=_linear(
                        state,
                        prefix + "attn.qkv",
                        input_size=vision.hidden_size,
                        output_size=3 * vision.hidden_size,
                        dtype=dtype,
                        bias=True,
                    ),
                    output=_linear(
                        state,
                        prefix + "attn.proj",
                        input_size=vision.hidden_size,
                        output_size=vision.hidden_size,
                        dtype=dtype,
                        bias=True,
                    ),
                ),
                mlp_norm=_norm(
                    state,
                    prefix + "norm2",
                    vision.hidden_size,
                    vision.norm_epsilon,
                    dtype,
                    layer_norm=layer_norm,
                ),
                mlp=Qwen2VLVisionMLP(
                    first=_linear(
                        state,
                        first_prefix,
                        input_size=vision.hidden_size,
                        output_size=vision.intermediate_size,
                        dtype=dtype,
                        bias=True,
                    ),
                    gate=gate,
                    second=_linear(
                        state,
                        second_prefix,
                        input_size=vision.intermediate_size,
                        output_size=vision.hidden_size,
                        dtype=dtype,
                        bias=True,
                    ),
                    activation=vision.mlp,
                ),
            )
        )
    merged = vision.hidden_size * vision.spatial_merge_unit
    return Qwen2VLVisionTower(
        patch_embedding=Linear(
            weight=_array(
                state,
                "visual.patch_embed.proj.weight",
                (
                    vision.hidden_size,
                    vision.in_channels,
                    vision.temporal_patch_size,
                    vision.patch_size,
                    vision.patch_size,
                ),
                dtype,
            ).reshape((vision.hidden_size, vision.patch_dimension)),
            bias=None,
        ),
        blocks=Qwen2VLVisionBlockStack.from_blocks(
            tuple(blocks), vision.full_attention_layers
        ),
        merger=Qwen2VLPatchMerger(
            norm=_norm(
                state,
                "visual.merger.ln_q",
                vision.hidden_size,
                vision.norm_epsilon,
                dtype,
                layer_norm=layer_norm,
            ),
            up=_linear(
                state,
                "visual.merger.mlp.0",
                input_size=merged,
                output_size=merged,
                dtype=dtype,
                bias=True,
            ),
            down=_linear(
                state,
                "visual.merger.mlp.2",
                input_size=merged,
                output_size=vision.output_size,
                dtype=dtype,
                bias=True,
            ),
        ),
        config=vision,
    )


@dataclass(frozen=True, slots=True)
class Qwen2VLCheckpointAdapter:
    """Bidirectional Qwen2/Qwen2.5-VL Safetensors adapter."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: Qwen2VLConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.bfloat16,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        model_id: str = BGE_VL_SCREENSHOT_MODEL_ID,
        revision: str = BGE_VL_SCREENSHOT_REVISION,
        pooling: str = "last",
    ) -> Qwen2VLEncoder:
        if pooling not in {"first", "last", "mean"}:
            raise ValueError("pooling must be first, last, or mean")
        return Qwen2VLEncoder(
            text=_text_tower(config, state_dict, parameter_dtype),
            vision=_vision_tower(config, state_dict, parameter_dtype),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
            pooling=pooling,
        )

    def load(self, checkpoint: str | Path, **options: Any) -> Qwen2VLEncoder:
        config = Qwen2VLConfig.from_hf_config(load_hf_config(checkpoint))
        dtype = options.get("parameter_dtype", jnp.bfloat16)
        state = load_safetensor_subset(
            checkpoint, qwen2_vl_weight_names(config), dtype=dtype
        )
        return self.from_state_dict(config, state, **options)

    def load_lora(
        self,
        base_checkpoint: str | Path,
        adapter_checkpoint: str | Path,
        **options: Any,
    ) -> Qwen2VLEncoder:
        """Load a PEFT LoRA artifact without rounding it into the base weights."""

        config = Qwen2VLConfig.from_hf_config(load_hf_config(base_checkpoint))
        dtype = options.get("parameter_dtype", jnp.bfloat16)
        state = load_safetensor_subset(
            base_checkpoint, qwen2_vl_weight_names(config), dtype=dtype
        )
        adapter_directory = Path(adapter_checkpoint)
        adapter_config = json.loads(
            (adapter_directory / "adapter_config.json").read_text()
        )
        rank = int(adapter_config["r"])
        alpha = float(adapter_config["lora_alpha"])
        from safetensors import safe_open

        model = self.from_state_dict(config, state, **options)
        attributes = {
            "self_attn.q_proj": "query",
            "self_attn.k_proj": "key",
            "self_attn.v_proj": "value",
            "self_attn.o_proj": "output",
            "mlp.gate_proj": "gate",
            "mlp.up_proj": "up",
            "mlp.down_proj": "down",
        }
        adapters: dict[str, LoRALinear] = {}
        with safe_open(
            adapter_directory / "adapter_model.safetensors", framework="np"
        ) as handle:
            available = frozenset(handle.keys())
            for suffix, attribute in attributes.items():
                lora_a = []
                lora_b = []
                for index in range(config.text.num_hidden_layers):
                    lora = f"base_model.model.model.layers.{index}.{suffix}.lora_"
                    a_name, b_name = lora + "A.weight", lora + "B.weight"
                    if a_name not in available or b_name not in available:
                        raise KeyError(f"Nomic adapter is missing {a_name} or {b_name}")
                    # PEFT keeps adapter masters in FP32 even when the backbone
                    # runs in BF16. This also keeps inference-bundle templates
                    # identical to the updated training model.
                    with jax.default_device(jax.devices("cpu")[0]):
                        lora_a.append(
                            jnp.asarray(handle.get_tensor(a_name), dtype=jnp.float32)
                        )
                        lora_b.append(
                            jnp.asarray(handle.get_tensor(b_name), dtype=jnp.float32)
                        )
                base = getattr(model.text.layers.blocks, attribute)
                adapters[attribute] = LoRALinear(
                    weight=base.weight,
                    bias=base.bias,
                    lora_a=jnp.stack(lora_a),
                    lora_b=jnp.stack(lora_b),
                    rank=rank,
                    alpha=alpha,
                    weight_layout=base.weight_layout,
                )
        blocks = eqx.tree_at(
            lambda layer: tuple(getattr(layer, name) for name in attributes.values()),
            model.text.layers.blocks,
            tuple(adapters[name] for name in attributes.values()),
        )
        layers = eqx.tree_at(lambda stack: stack.blocks, model.text.layers, blocks)
        text = eqx.tree_at(lambda tower: tower.layers, model.text, layers)
        return eqx.tree_at(lambda candidate: candidate.text, model, text)

    def load_reranker(self, checkpoint: str | Path, **options: Any) -> Qwen2VLReranker:
        config = Qwen2VLConfig.from_hf_config(load_hf_config(checkpoint))
        dtype = options.get("parameter_dtype", jnp.bfloat16)
        names = qwen2_vl_weight_names(config) | frozenset(
            {"score.0.weight", "score.0.bias", "score.2.weight", "score.2.bias"}
        )
        state = load_safetensor_subset(checkpoint, names, dtype=dtype)
        model = self.from_state_dict(config, state, pooling="last", **options)
        return Qwen2VLReranker(
            model=model,
            hidden=_linear(
                state,
                "score.0",
                input_size=config.text.hidden_size,
                output_size=config.text.hidden_size,
                dtype=dtype,
                bias=True,
            ),
            output=_linear(
                state,
                "score.2",
                input_size=config.text.hidden_size,
                output_size=1,
                dtype=dtype,
                bias=True,
            ),
        )

    def state_dict(self, model: Qwen2VLEncoder) -> dict[str, jax.Array]:
        state: dict[str, jax.Array] = {
            "model.embed_tokens.weight": model.text.token_embedding,
            "model.norm.weight": model.text.final_norm.weight,
            "visual.patch_embed.proj.weight": (
                model.vision.patch_embedding.weight.reshape(
                    (
                        model.config.vision.hidden_size,
                        model.config.vision.in_channels,
                        model.config.vision.temporal_patch_size,
                        model.config.vision.patch_size,
                        model.config.vision.patch_size,
                    )
                )
            ),
        }
        for index in range(model.text.layers.depth):
            layer = model.text.layers.layer(index)
            prefix = f"model.layers.{index}."
            state.update(
                {
                    prefix + "input_layernorm.weight": layer.input_norm.weight,
                    prefix + "post_attention_layernorm.weight": (
                        layer.post_attention_norm.weight
                    ),
                    prefix + "self_attn.q_proj.weight": _weight(layer.query),
                    prefix + "self_attn.q_proj.bias": _bias(layer.query),
                    prefix + "self_attn.k_proj.weight": _weight(layer.key),
                    prefix + "self_attn.k_proj.bias": _bias(layer.key),
                    prefix + "self_attn.v_proj.weight": _weight(layer.value),
                    prefix + "self_attn.v_proj.bias": _bias(layer.value),
                    prefix + "self_attn.o_proj.weight": _weight(layer.output),
                    prefix + "mlp.gate_proj.weight": _weight(layer.gate),
                    prefix + "mlp.up_proj.weight": _weight(layer.up),
                    prefix + "mlp.down_proj.weight": _weight(layer.down),
                }
            )
        for index in range(model.vision.blocks.depth):
            block = jax.tree.map(
                lambda value, index=index: value[index], model.vision.blocks.blocks
            )
            prefix = f"visual.blocks.{index}."
            state.update(
                {
                    prefix + "norm1.weight": block.attention_norm.weight,
                    prefix + "norm2.weight": block.mlp_norm.weight,
                    prefix + "attn.qkv.weight": block.attention.qkv.weight,
                    prefix + "attn.qkv.bias": _bias(block.attention.qkv),
                    prefix + "attn.proj.weight": block.attention.output.weight,
                    prefix + "attn.proj.bias": _bias(block.attention.output),
                }
            )
            if model.config.vision.norm == "layer":
                state[prefix + "norm1.bias"] = _bias(block.attention_norm)
                state[prefix + "norm2.bias"] = _bias(block.mlp_norm)
                state[prefix + "mlp.fc1.weight"] = block.mlp.first.weight
                state[prefix + "mlp.fc1.bias"] = _bias(block.mlp.first)
                state[prefix + "mlp.fc2.weight"] = block.mlp.second.weight
                state[prefix + "mlp.fc2.bias"] = _bias(block.mlp.second)
            else:
                if block.mlp.gate is None:
                    raise AssertionError("Qwen2.5 vision MLP requires a gate")
                state[prefix + "mlp.gate_proj.weight"] = block.mlp.gate.weight
                state[prefix + "mlp.gate_proj.bias"] = _bias(block.mlp.gate)
                state[prefix + "mlp.up_proj.weight"] = block.mlp.first.weight
                state[prefix + "mlp.up_proj.bias"] = _bias(block.mlp.first)
                state[prefix + "mlp.down_proj.weight"] = block.mlp.second.weight
                state[prefix + "mlp.down_proj.bias"] = _bias(block.mlp.second)
        merger = model.vision.merger
        state.update(
            {
                "visual.merger.ln_q.weight": merger.norm.weight,
                "visual.merger.mlp.0.weight": merger.up.weight,
                "visual.merger.mlp.0.bias": _bias(merger.up),
                "visual.merger.mlp.2.weight": merger.down.weight,
                "visual.merger.mlp.2.bias": _bias(merger.down),
            }
        )
        if model.config.vision.norm == "layer":
            state["visual.merger.ln_q.bias"] = _bias(merger.norm)
        return state

    def save(self, model: Qwen2VLEncoder, directory: str | Path) -> Path:
        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        processor_mode = "embedding"
        if (
            model.metadata.model_id == BGE_VL_SCREENSHOT_MODEL_ID
            or (target / "modeling_bge_vl_screenshot.py").is_file()
        ):
            processor_mode = "bge_embedding"
        elif (target / "adapter_config.json").is_file() or model.metadata.model_id in {
            "nomic-ai/nomic-embed-multimodal-3b",
            "nomic-ai/nomic-embed-multimodal-7b",
        }:
            processor_mode = "nomic_embedding"
        config = model.config.to_hf_config()
        config["representax_processor_mode"] = processor_mode
        (target / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        # A merged full checkpoint is no longer a PEFT adapter. Leaving this
        # metadata behind makes Transformers attempt to resolve a missing base
        # model and ignores the newly written full weights.
        for stale in ("adapter_config.json", "adapter_model.safetensors"):
            (target / stale).unlink(missing_ok=True)
        save_file(
            {
                name: np.array(value, copy=True)
                for name, value in self.state_dict(model).items()
            },
            target / "model.safetensors",
        )
        return target

    def reranker_state_dict(self, model: Qwen2VLReranker) -> dict[str, jax.Array]:
        """Return the upstream Jina backbone and score-head layout."""

        return {
            **self.state_dict(model.model),
            "score.0.weight": model.hidden.weight,
            "score.0.bias": _bias(model.hidden),
            "score.2.weight": model.output.weight,
            "score.2.bias": _bias(model.output),
        }

    def save_reranker(self, model: Qwen2VLReranker, directory: str | Path) -> Path:
        """Export a reloadable Jina-style Qwen2-VL reranker checkpoint."""

        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        config = model.model.config.to_hf_config()
        config.update(
            {
                "architectures": ["JinaVLForRanking"],
                "auto_map": {"AutoModel": "modeling.JinaVLForRanking"},
                "representax_processor_mode": "reranking",
            }
        )
        (target / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        save_file(
            {
                name: np.array(value, copy=True)
                for name, value in self.reranker_state_dict(model).items()
            },
            target / "model.safetensors",
        )
        return target


@dataclass(frozen=True, slots=True)
class Qwen2VLRerankerCheckpointAdapter:
    """Bidirectional adapter exposing Jina's scoring head as the root model."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def _backbone(self) -> Qwen2VLCheckpointAdapter:
        return Qwen2VLCheckpointAdapter(
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def load(self, checkpoint: str | Path, **options: Any) -> Qwen2VLReranker:
        return self._backbone().load_reranker(checkpoint, **options)

    def state_dict(self, model: Qwen2VLReranker) -> dict[str, jax.Array]:
        return self._backbone().reranker_state_dict(model)

    def save(self, model: Qwen2VLReranker, directory: str | Path) -> Path:
        return self._backbone().save_reranker(model, directory)


__all__ = [
    "Qwen2VLCheckpointAdapter",
    "Qwen2VLRerankerCheckpointAdapter",
    "qwen2_vl_weight_names",
]
