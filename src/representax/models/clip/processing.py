"""Model-associated host preprocessing for native CLIP inputs."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from representax.core import Modality, Route
from representax.data import Artifact
from representax.models.processing import Processor

from .config import CLIPConfig
from .model import CLIPBatch


def _artifact_value(value: Any, modality: Modality) -> Any:
    if not isinstance(value, Artifact):
        return value
    if value.modality != modality:
        raise TypeError(f"expected a {modality.value} artifact, got {value.modality}")
    if value.data is not None:
        return value.data
    payload = value.read_bytes()
    if modality == Modality.TEXT:
        return payload.decode()
    if modality == Modality.IMAGE:
        image = import_module("PIL.Image")
        return image.open(io.BytesIO(payload)).copy()
    raise TypeError(f"CLIP does not accept {modality.value} artifacts")


def _components(value: Any) -> tuple[str | None, Any | None]:
    if isinstance(value, Artifact):
        if value.modality == Modality.TEXT:
            text = _artifact_value(value, Modality.TEXT)
            if not isinstance(text, str):
                raise TypeError("CLIP text artifacts must contain strings")
            return text, None
        if value.modality == Modality.IMAGE:
            return None, _artifact_value(value, Modality.IMAGE)
        raise TypeError("CLIP accepts only text and image artifacts")
    if isinstance(value, str):
        return value, None
    if not isinstance(value, Mapping):
        return None, value
    text = _artifact_value(value.get("text"), Modality.TEXT)
    if text is not None and not isinstance(text, str):
        raise TypeError("CLIP sample text must be a string")
    image = _artifact_value(value.get("image"), Modality.IMAGE)
    if text is None and image is None:
        raise ValueError("CLIP samples require text, an image, or both")
    return text, image


def make_clip_processor(
    checkpoint: str | Path,
    config: CLIPConfig,
    *,
    normalize_output: bool = False,
) -> Processor:
    """Load the checkpoint tokenizer/image assets into one generic processor."""

    try:
        transformers = import_module("transformers")
    except ImportError as error:
        raise ImportError(
            "CLIP processing requires the representax[hf] extra"
        ) from error
    upstream = transformers.CLIPProcessor.from_pretrained(checkpoint)

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> CLIPBatch:
        del route, seed
        if not artifacts:
            raise ValueError("CLIP processor batches must be non-empty")
        components = tuple(_components(value) for value in artifacts)
        text_indices = [
            index for index, (text, _) in enumerate(components) if text is not None
        ]
        image_indices = [
            index for index, (_, image) in enumerate(components) if image is not None
        ]
        values: dict[str, Any] = {}
        if text_indices:
            texts = [cast(str, components[index][0]) for index in text_indices]
            encoded = upstream.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=config.text.max_position_embeddings,
                return_tensors="np",
            )
            input_ids = np.full(
                (len(artifacts), config.text.max_position_embeddings),
                config.text.pad_token_id,
                dtype=np.int32,
            )
            attention_mask = np.zeros_like(input_ids)
            input_ids[text_indices] = np.asarray(encoded["input_ids"], dtype=np.int32)
            attention_mask[text_indices] = np.asarray(
                encoded["attention_mask"],
                dtype=np.int32,
            )
            text_valid = np.zeros((len(artifacts),), dtype=bool)
            text_valid[text_indices] = True
            values.update(
                input_ids=jnp.asarray(input_ids),
                attention_mask=jnp.asarray(attention_mask),
                text_valid=jnp.asarray(text_valid),
            )
        if image_indices:
            images = [components[index][1] for index in image_indices]
            encoded = upstream.image_processor(images=images, return_tensors="np")
            selected = np.asarray(encoded["pixel_values"], dtype=np.float32)
            pixels = np.zeros((len(artifacts), *selected.shape[1:]), dtype=np.float32)
            pixels[image_indices] = selected
            image_valid = np.zeros((len(artifacts),), dtype=bool)
            image_valid[image_indices] = True
            values.update(
                pixel_values=jnp.asarray(pixels),
                image_valid=jnp.asarray(image_valid),
            )
        return CLIPBatch(**values)

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-clip-processor-v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "max_sequence_length": config.text.max_position_embeddings,
            "image_size": config.vision.image_size,
            "patch_size": config.vision.patch_size,
            "composition": "project_then_sum",
            "normalize_output": normalize_output,
        },
    )


__all__ = ["make_clip_processor"]
