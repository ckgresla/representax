"""Model-associated finite-shape preprocessing for LLaVA-NeXT retrieval."""

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

from .config import LlavaNextConfig
from .model import LlavaNextBatch

LlavaNextProcessorMode = Literal["bge", "e5"]


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
    raise TypeError(f"LLaVA-NeXT does not accept {modality.value} artifacts")


def _components(value: Any) -> tuple[str | None, Any | None]:
    if isinstance(value, Artifact):
        if value.modality == Modality.TEXT:
            return str(_artifact_value(value, Modality.TEXT)), None
        if value.modality == Modality.IMAGE:
            return None, _artifact_value(value, Modality.IMAGE)
        raise TypeError("LLaVA-NeXT accepts text and image artifacts")
    if isinstance(value, str):
        return value, None
    if not isinstance(value, Mapping):
        return None, value
    text = _artifact_value(value.get("text"), Modality.TEXT)
    image = _artifact_value(value.get("image"), Modality.IMAGE)
    if text is not None and not isinstance(text, str):
        raise TypeError("sample text must be a string")
    if text is None and image is None:
        raise ValueError("samples require text, an image, or both")
    return text, image


def _content(
    text: str | None,
    image: Any | None,
    *,
    image_first: bool,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    ordered = (
        (("image", image), ("text", text))
        if image_first
        else (("text", text), ("image", image))
    )
    for kind, value in ordered:
        if value is None:
            continue
        values.append(
            {"type": "image"} if kind == "image" else {"type": "text", "text": value}
        )
    return values


def _prompt(
    upstream: Any,
    text: str | None,
    image: Any | None,
    *,
    mode: LlavaNextProcessorMode,
    route: Route,
    instruction: str | None,
) -> str:
    content = _content(text, image, image_first=mode == "bge")
    if mode == "bge":
        messages: list[dict[str, Any]] = []
        if instruction is not None:
            messages.append({"role": "system", "content": instruction})
        messages.append(
            {
                "role": "query" if route == Route.QUERY else "candidate",
                "content": content,
            }
        )
        return upstream.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return upstream.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _best_resolution(
    original_size: tuple[int, int],
    possibilities: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    original_height, original_width = original_size
    best = possibilities[0]
    best_effective = -1
    least_waste = float("inf")
    for height, width in possibilities:
        scale = min(width / original_width, height / original_height)
        down_width = int(original_width * scale)
        down_height = int(original_height * scale)
        effective = min(down_width * down_height, original_width * original_height)
        waste = width * height - effective
        if effective > best_effective or (
            effective == best_effective and waste < least_waste
        ):
            best = (height, width)
            best_effective = effective
            least_waste = waste
    return best


def image_pack_indices(
    image_sizes: Sequence[Sequence[int]],
    config: LlavaNextConfig,
    *,
    image_bucket: int,
    tile_bucket: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan upstream any-resolution packing as gathers plus newline sentinels."""

    tokens = config.selected_vision_tokens
    if config.vision_feature_select_strategy != "default":
        raise NotImplementedError(
            "full CLS-preserving any-resolution packing is not used by the "
            "accepted LLaVA-NeXT retrieval checkpoints"
        )
    patch_side = config.vision.image_size // config.vision.patch_size
    newline_index = image_bucket * tile_bucket * tokens
    sources: list[int] = []
    valid_tiles = np.zeros((image_bucket, tile_bucket), dtype=bool)
    for image_index, raw_size in enumerate(image_sizes):
        original = (int(raw_size[0]), int(raw_size[1]))
        best_height, best_width = _best_resolution(
            original, config.image_grid_pinpoints
        )
        grid_height = best_height // config.vision.image_size
        grid_width = best_width // config.vision.image_size
        tile_count = 1 + grid_height * grid_width
        if tile_count > tile_bucket:
            raise ValueError("image tiles exceed the selected static bucket")
        valid_tiles[image_index, :tile_count] = True
        image_offset = image_index * tile_bucket * tokens
        sources.extend(image_offset + np.arange(tokens, dtype=np.int64))

        crop_offsets = (
            image_offset
            + tokens
            + np.arange(grid_height * grid_width * tokens, dtype=np.int64)
        )
        grid = crop_offsets.reshape((grid_height, grid_width, patch_side, patch_side))
        grid = grid.transpose((0, 2, 1, 3)).reshape(
            (grid_height * patch_side, grid_width * patch_side)
        )
        current_height, current_width = grid.shape
        original_height, original_width = original
        if original_width / original_height > current_width / current_height:
            new_height = int(round(original_height * current_width / original_width, 7))
            padding = (current_height - new_height) // 2
            grid = grid[padding : current_height - padding if padding else None]
        else:
            new_width = int(round(original_width * current_height / original_height, 7))
            padding = (current_width - new_width) // 2
            grid = grid[:, padding : current_width - padding if padding else None]
        for row in grid:
            sources.extend(row.tolist())
            if config.use_image_newline:
                sources.append(newline_index)
    return np.asarray(sources, dtype=np.int32), valid_tiles


def _pad_sequence(
    values: np.ndarray,
    *,
    size: int,
    fill: int,
    side: str,
) -> np.ndarray:
    padding = size - values.shape[1]
    if padding < 0:
        raise ValueError("sequence exceeds the selected static bucket")
    widths = ((0, 0), (padding, 0) if side == "left" else (0, padding))
    return np.pad(values, widths, constant_values=fill)


def make_llava_next_processor(
    checkpoint: str | Path,
    config: LlavaNextConfig,
    *,
    mode: LlavaNextProcessorMode,
    sequence_length_buckets: Sequence[int] = (2048, 4096, 8192),
    image_count_buckets: Sequence[int] = (1, 2, 4, 8, 16, 32),
    tile_count_buckets: Sequence[int] = (1, 3, 5, 10),
) -> Processor:
    """Load tokenizer/image artifacts and admit only finite model-ready shapes."""

    try:
        transformers = import_module("transformers")
    except ImportError as error:
        raise ImportError(
            "LLaVA-NeXT processing requires the representax[hf] extra"
        ) from error
    upstream = transformers.LlavaNextProcessor.from_pretrained(checkpoint)

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
        instruction: str | None = None,
    ) -> LlavaNextBatch:
        del seed
        if not artifacts:
            raise ValueError("LLaVA-NeXT processor batches must be non-empty")
        components = tuple(_components(value) for value in artifacts)
        prompts = [
            _prompt(
                upstream,
                text,
                image,
                mode=mode,
                route=route,
                instruction=instruction,
            )
            for text, image in components
        ]
        images = [image for _, image in components if image is not None]
        encoded = upstream(
            text=prompts,
            images=images or None,
            padding=True,
            return_tensors="np",
        )
        raw_ids = np.asarray(encoded["input_ids"], dtype=np.int32)
        raw_mask = np.asarray(encoded["attention_mask"], dtype=np.int32)
        sequence_bucket = select_static_shape_bucket(
            (raw_ids.shape[1],), tuple((size,) for size in sequence_length_buckets)
        )[0]
        padding_side = str(upstream.tokenizer.padding_side)
        input_ids = _pad_sequence(
            raw_ids,
            size=sequence_bucket,
            fill=int(upstream.tokenizer.pad_token_id),
            side=padding_side,
        )
        attention_mask = _pad_sequence(
            raw_mask, size=sequence_bucket, fill=0, side=padding_side
        )
        if not images:
            return LlavaNextBatch(
                input_ids=jnp.asarray(input_ids),
                attention_mask=jnp.asarray(attention_mask),
            )

        pixels = np.asarray(encoded["pixel_values"], dtype=np.float32)
        image_sizes = np.asarray(encoded["image_sizes"], dtype=np.int32)
        image_bucket = select_static_shape_bucket(
            (pixels.shape[0],), tuple((size,) for size in image_count_buckets)
        )[0]
        tile_bucket = select_static_shape_bucket(
            (pixels.shape[1],), tuple((size,) for size in tile_count_buckets)
        )[0]
        padded_pixels = np.zeros(
            (image_bucket, tile_bucket, *pixels.shape[2:]), dtype=np.float32
        )
        padded_pixels[: pixels.shape[0], : pixels.shape[1]] = pixels
        sources, tile_valid = image_pack_indices(
            image_sizes,
            config,
            image_bucket=image_bucket,
            tile_bucket=tile_bucket,
        )
        visual_positions = np.flatnonzero(
            input_ids.reshape(-1) == config.image_token_id
        ).astype(np.int32)
        if visual_positions.size != sources.size:
            raise ValueError(
                "processor image-token count does not match native any-resolution "
                f"packing ({visual_positions.size} != {sources.size})"
            )
        capacity = sequence_bucket * len(artifacts)
        pack_indices = np.zeros((capacity,), dtype=np.int32)
        pack_valid = np.zeros((capacity,), dtype=bool)
        token_indices = np.zeros((capacity,), dtype=np.int32)
        pack_indices[: sources.size] = sources
        pack_valid[: sources.size] = True
        token_indices[: sources.size] = visual_positions
        return LlavaNextBatch(
            input_ids=jnp.asarray(input_ids),
            attention_mask=jnp.asarray(attention_mask),
            pixel_values=jnp.asarray(padded_pixels),
            tile_valid=jnp.asarray(tile_valid),
            pack_indices=jnp.asarray(pack_indices),
            pack_valid=jnp.asarray(pack_valid),
            visual_token_indices=jnp.asarray(token_indices),
        )

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-llava-next-processor-v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "mode": mode,
            "sequence_length_buckets": list(sequence_length_buckets),
            "image_count_buckets": list(image_count_buckets),
            "tile_count_buckets": list(tile_count_buckets),
            "image_grid_pinpoints": [
                list(point) for point in config.image_grid_pinpoints
            ],
            "vision_feature_layer": config.vision_feature_layer,
            "vision_feature_select_strategy": config.vision_feature_select_strategy,
        },
    )


__all__ = ["image_pack_indices", "make_llava_next_processor"]
