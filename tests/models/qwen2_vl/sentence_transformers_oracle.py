"""Record real multimodal Sentence Transformers 5.6 reference behavior."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _image():
    from PIL import Image

    source = np.arange(224 * 224 * 3, dtype=np.uint32).reshape((224, 224, 3))
    pixels = ((source * 37 + 19) % 251).astype(np.uint8)
    return Image.fromarray(pixels), pixels


def _arrays(prefix: str, features: dict[str, Any]) -> dict[str, np.ndarray]:
    import torch

    return {
        f"{prefix}_{name}": value.detach().cpu().numpy()
        for name, value in features.items()
        if isinstance(value, torch.Tensor)
    }


def _on_device(features: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in features.items()
    }


def _bge(checkpoint: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from sentence_transformers import SentenceTransformer

    image, source_pixels = _image()
    artifact = {
        "text": "A deterministic chart about quarterly revenue.",
        "image": image,
    }
    model: Any = SentenceTransformer(
        str(checkpoint),
        trust_remote_code=True,
        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
    )
    model.eval()
    prompt = model.prompts.get("query")
    features = model.preprocess([artifact], prompt=prompt, task="query")
    arrays = {"source_pixels": source_pixels.astype(np.uint8)}
    arrays.update(_arrays("query", features))
    device_features = _on_device(features, model.device)
    upstream_error = None
    try:
        with torch.no_grad():
            embedding = model(device_features)["sentence_embedding"]
    except AttributeError as error:
        # The pinned BGE remote forward still calls ``self.model.embed_tokens``;
        # Transformers 5.6 moved this under ``language_model``. Preprocessing is
        # still canonical, and the underlying standard Qwen model is intact.
        upstream_error = str(error)
        accepted = {
            name: device_features[name]
            for name in (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "video_grid_thw",
                "mm_token_type_ids",
            )
            if name in device_features
        }
        with torch.no_grad():
            hidden = (
                model[0]
                .auto_model.model(
                    **accepted,
                    use_cache=False,
                    return_dict=True,
                )
                .last_hidden_state
            )
            embedding = torch.nn.functional.normalize(hidden[:, -1], dim=-1)
    arrays["query_embedding"] = embedding.float().cpu().numpy()
    return arrays, {"upstream_forward_error": upstream_error}


def _nomic(checkpoint: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from sentence_transformers import SentenceTransformer

    image, source_pixels = _image()
    dtype_name = os.environ.get("REPRESENTAX_ORACLE_DTYPE", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": "eager",
    }
    if dtype is torch.float32 and torch.cuda.device_count() > 1:
        model_kwargs["device_map"] = "auto"
        # Transformers' allocator warmup attempts one model-sized temporary on
        # each 24 GB device even though Accelerate has already split the FP32
        # parameters. The real weights fit; that non-semantic temporary does
        # not. Disable only the oracle loader warmup.
        import transformers.modeling_utils

        transformers.modeling_utils.caching_allocator_warmup = (  # ty: ignore[invalid-assignment]
            lambda _model, _device_map, _quantizer: None
        )
    model: Any = SentenceTransformer(
        str(checkpoint),
        model_kwargs=model_kwargs,
    )
    model.eval()
    arrays: dict[str, np.ndarray] = {"source_pixels": source_pixels.astype(np.uint8)}
    for name, artifact in (
        ("text", "A deterministic chart about quarterly revenue."),
        ("image", image),
    ):
        features = model.preprocess([artifact])
        arrays.update(_arrays(name, features))
        with torch.no_grad():
            output = model(_on_device(features, model.device))["sentence_embedding"]
        arrays[f"{name}_embedding"] = output.float().cpu().numpy()
    projection = next(
        module
        for name, module in model[0].auto_model.named_modules()
        if name.endswith("language_model.layers.0.self_attn.q_proj")
    )
    return arrays, {
        "base_dtype": str(projection.base_layer.weight.dtype),
        "adapter_a_dtype": str(projection.lora_A["default"].weight.dtype),
        "adapter_b_dtype": str(projection.lora_B["default"].weight.dtype),
        "adapter_scaling": float(projection.scaling["default"]),
        "dtype": dtype_name,
    }


def _jina(checkpoint: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from sentence_transformers import CrossEncoder

    image, source_pixels = _image()
    model: Any = CrossEncoder(
        str(checkpoint),
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "eager"},
    )
    model.model.eval()
    arrays: dict[str, np.ndarray] = {"source_pixels": source_pixels.astype(np.uint8)}
    pairs = {
        "text": (
            "Which report contains revenue growth?",
            "The annual report contains a quarterly revenue chart.",
        ),
        "image": ("Which report contains revenue growth?", image),
    }
    for name, pair in pairs.items():
        features = model.preprocess([pair])
        arrays.update(_arrays(name, features))
        with torch.no_grad():
            score = model.model(**_on_device(features, model.device))
        arrays[f"{name}_score"] = score.float().cpu().numpy()
    return arrays, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("bge", "nomic", "jina"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    import sentence_transformers
    import torch
    import transformers

    if sentence_transformers.__version__ != "5.6.1":
        raise RuntimeError("real oracle requires sentence-transformers==5.6.1")
    if transformers.__version__ != "5.6.0":
        raise RuntimeError("real oracle requires transformers==5.6.0")
    torch.backends.cudnn.enabled = False
    arrays, metadata = {
        "bge": _bge,
        "nomic": _nomic,
        "jina": _jina,
    }[arguments.variant](arguments.checkpoint)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, **arrays)  # ty: ignore[invalid-argument-type]
    (arguments.output.with_suffix(".json")).write_text(
        json.dumps(
            {
                "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__,
                "variant": arguments.variant,
                **metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
