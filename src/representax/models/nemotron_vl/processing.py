"""Finite-shape native preprocessing for Llama Nemotron VL."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from representax.core import Modality, Route
from representax.data import Artifact
from representax.models.processing import Processor, select_static_shape_bucket

from .config import LlamaNemotronVLConfig
from .model import LlamaNemotronVLBatch

ProcessorMode = Literal["embedding", "reranking"]


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
    image = import_module("PIL.Image")
    return image.open(io.BytesIO(payload)).copy()


def _document(value: Any) -> tuple[str, Any | None]:
    if isinstance(value, Artifact):
        if value.modality == Modality.TEXT:
            return str(_artifact_value(value, Modality.TEXT)), None
        if value.modality == Modality.IMAGE:
            return "", _artifact_value(value, Modality.IMAGE)
        raise TypeError("Llama Nemotron VL accepts text and image artifacts")
    if isinstance(value, str):
        return value, None
    if not isinstance(value, Mapping):
        return "", value
    text = _artifact_value(value.get("text", ""), Modality.TEXT)
    image = _artifact_value(value.get("image"), Modality.IMAGE)
    if not isinstance(text, str):
        raise TypeError("sample text must be a string")
    if not text and image is None:
        raise ValueError("documents require text, an image, or both")
    return text, image


def _reranking_pair(value: Any) -> tuple[str, str, Any | None]:
    if isinstance(value, Mapping):
        query = _artifact_value(value.get("query"), Modality.TEXT)
        raw_document = value.get("document")
        if raw_document is None:
            raw_document = {"text": value.get("text", ""), "image": value.get("image")}
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        query, raw_document = value
        query = _artifact_value(query, Modality.TEXT)
    else:
        raise TypeError("reranking samples must be (query, document) pairs or mappings")
    if not isinstance(query, str) or not query:
        raise TypeError("reranking queries must be non-empty strings")
    text, image = _document(raw_document)
    return query, text, image


def _target_grid(
    image: Any,
    *,
    image_size: int,
    maximum: int,
) -> tuple[int, int]:
    width, height = image.size
    aspect = width / height
    ratios = sorted(
        {
            (columns, rows)
            for count in range(1, maximum + 1)
            for columns in range(1, count + 1)
            for rows in range(1, count + 1)
            if 1 <= columns * rows <= maximum
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    area = width * height

    def score(ratio: tuple[int, int]) -> float:
        columns, rows = ratio
        target_aspect = columns / rows
        area_ratio = columns * rows * image_size * image_size / area
        return min(area_ratio, 0.6) * min(
            target_aspect / aspect, aspect / target_aspect
        )

    return max(ratios, key=score)


def _tiles(
    image: Any,
    *,
    image_size: int,
    maximum: int,
    thumbnail: bool,
) -> tuple[Any, ...]:
    image = image.convert("RGB")
    columns, rows = _target_grid(image, image_size=image_size, maximum=maximum)
    resized = image.resize((columns * image_size, rows * image_size))
    values = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        values.append(resized.crop((left, top, left + image_size, top + image_size)))
    if thumbnail and len(values) != 1:
        values.append(image.resize((image_size, image_size)))
    return tuple(values)


def _normalize_image(image: Any, *, image_size: int) -> np.ndarray:
    pil = import_module("PIL.Image")
    image = image.convert("RGB").resize(
        (image_size, image_size), resample=pil.Resampling.BICUBIC
    )
    value = np.asarray(image, dtype=np.float32) / 255.0
    value = (value - 0.5) / 0.5
    return np.transpose(value, (2, 0, 1))


def _image_tokens(count: int, tokens_per_tile: int) -> str:
    return "<img>" + "<IMG_CONTEXT>" * (count * tokens_per_tile) + "</img>"


def _pad_left(value: np.ndarray, *, length: int, fill: int) -> np.ndarray:
    padding = length - value.shape[1]
    if padding < 0:
        raise ValueError("sequence exceeds selected static bucket")
    return np.pad(value, ((0, 0), (padding, 0)), constant_values=fill)


def make_nemotron_vl_processor(
    checkpoint: str | Path,
    config: LlamaNemotronVLConfig,
    *,
    sequence_length_buckets: Sequence[int] = (512, 1024, 2048, 4096, 8192),
    tile_count_buckets: Sequence[int] = (1, 3, 7, 14, 28, 56),
    max_input_tiles: int = 6,
    use_thumbnail: bool = True,
) -> Processor:
    """Load tokenizer assets and emit one of finitely many native array shapes."""

    try:
        transformers = import_module("transformers")
    except ImportError as error:
        raise ImportError(
            "Llama Nemotron VL processing requires the representax[hf] extra"
        ) from error
    # Loading the tokenizer never requires executing the checkpoint's custom
    # model/processor modules; the standard fast-tokenizer artifact is complete.
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(checkpoint)
    tokenizer.padding_side = "left"
    image_size = config.vision.image_size
    image_tokens = config.image_sequence_length

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> LlamaNemotronVLBatch:
        del seed
        if not artifacts:
            raise ValueError("Llama Nemotron VL batches must be non-empty")
        prompts = []
        pixels: list[np.ndarray] = []
        if config.mode == "embedding":
            for artifact in artifacts:
                text, image = _document(artifact)
                if route == Route.QUERY:
                    if image is not None:
                        raise ValueError("the accepted query contract is text-only")
                    prompts.append(f"query: {text}")
                    continue
                tile_count = 0
                if image is not None:
                    prepared = _tiles(
                        image,
                        image_size=image_size,
                        maximum=max_input_tiles,
                        thumbnail=use_thumbnail,
                    )
                    pixels.extend(
                        _normalize_image(tile, image_size=image_size)
                        for tile in prepared
                    )
                    tile_count = len(prepared)
                content = text
                if tile_count:
                    content = f"{_image_tokens(tile_count, image_tokens)} {content}"
                prompts.append(f"passage: {content}")
        else:
            for artifact in artifacts:
                query, text, image = _reranking_pair(artifact)
                prepared = ()
                if image is not None:
                    prepared = _tiles(
                        image,
                        image_size=image_size,
                        maximum=max_input_tiles,
                        thumbnail=use_thumbnail,
                    )
                    pixels.extend(
                        _normalize_image(tile, image_size=image_size)
                        for tile in prepared
                    )
                content = f"question:{query} \n \n passage:{text}"
                if prepared:
                    content = f"{_image_tokens(len(prepared), image_tokens)} {content}"
                prompts.append(content)

        maximum_length = max(int(length) for length in sequence_length_buckets)
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=maximum_length,
            return_tensors="np",
        )
        raw_ids = np.asarray(encoded["input_ids"], dtype=np.int32)
        raw_mask = np.asarray(encoded["attention_mask"], dtype=np.int32)
        sequence_bucket = select_static_shape_bucket(
            (raw_ids.shape[1],),
            tuple((int(length),) for length in sequence_length_buckets),
        )[0]
        input_ids = _pad_left(
            raw_ids,
            length=sequence_bucket,
            fill=int(tokenizer.pad_token_id),
        )
        attention_mask = _pad_left(raw_mask, length=sequence_bucket, fill=0)
        if not pixels:
            return LlamaNemotronVLBatch(
                input_ids=jnp.asarray(input_ids),
                attention_mask=jnp.asarray(attention_mask),
            )

        tile_bucket = select_static_shape_bucket(
            (len(pixels),), tuple((int(count),) for count in tile_count_buckets)
        )[0]
        pixel_values = np.zeros(
            (tile_bucket, config.vision.num_channels, image_size, image_size),
            dtype=np.float32,
        )
        pixel_values[: len(pixels)] = np.stack(pixels)
        positions = np.flatnonzero(
            input_ids.reshape(-1) == config.image_context_token_id
        ).astype(np.int32)
        expected = len(pixels) * image_tokens
        if positions.size != expected:
            raise ValueError(
                "processor image-token count does not match prepared tiles "
                f"({positions.size} != {expected})"
            )
        capacity = tile_bucket * image_tokens
        visual_indices = np.zeros((capacity,), dtype=np.int32)
        visual_valid = np.zeros((capacity,), dtype=bool)
        visual_indices[:expected] = positions
        visual_valid[:expected] = True
        return LlamaNemotronVLBatch(
            input_ids=jnp.asarray(input_ids),
            attention_mask=jnp.asarray(attention_mask),
            pixel_values=jnp.asarray(pixel_values, dtype=jnp.bfloat16),
            visual_token_indices=jnp.asarray(visual_indices),
            visual_token_valid=jnp.asarray(visual_valid),
        )

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-llama-nemotron-vl-processor-v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "mode": config.mode,
            "sequence_length_buckets": list(sequence_length_buckets),
            "tile_count_buckets": list(tile_count_buckets),
            "image_size": image_size,
            "tokens_per_tile": image_tokens,
            "max_input_tiles": max_input_tiles,
            "use_thumbnail": use_thumbnail,
        },
    )


__all__ = ["make_nemotron_vl_processor"]
