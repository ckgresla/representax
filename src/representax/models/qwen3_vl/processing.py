"""Host-side finite-shape layout for Qwen3-VL processor outputs."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.processing import Processor, select_static_shape_bucket

from .config import Qwen3VLConfig, Qwen3VLVisionConfig
from .model import Qwen3VLBatch

Qwen3VLProcessorMode = Literal["embedding", "reranking"]


def _components(value: Any) -> tuple[str | None, Any | None, Any | None]:
    if isinstance(value, str):
        return value, None, None
    if not isinstance(value, Mapping):
        raise TypeError("Qwen3-VL samples must be strings or mappings")
    text = value.get("text")
    if text is not None and not isinstance(text, str):
        raise TypeError("Qwen3-VL sample text must be a string")
    return text, value.get("image"), value.get("video")


def _embedding_conversation(value: Any, *, default_instruction: str) -> list[dict]:
    text, image, video = _components(value)
    instruction = (
        value.get("instruction", default_instruction)
        if isinstance(value, Mapping)
        else default_instruction
    )
    if not isinstance(instruction, str):
        raise TypeError("Qwen3-VL instructions must be strings")
    content = []
    if video is not None:
        content.append({"type": "video"})
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text or "NULL"})
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


def _reranking_conversation(value: Any, *, default_instruction: str) -> list[dict]:
    if not isinstance(value, Mapping):
        raise TypeError("reranking samples must contain query and document mappings")
    if "query" not in value or "document" not in value:
        raise KeyError("reranking samples require query and document")
    instruction = value.get("instruction", default_instruction)
    if not isinstance(instruction, str):
        raise TypeError("Qwen3-VL instructions must be strings")
    query_text, query_image, query_video = _components(value["query"])
    document_text, document_image, document_video = _components(value["document"])
    content = [{"type": "text", "text": f"<Instruct>: {instruction}"}]
    content.append({"type": "text", "text": "<Query>:"})
    if query_video is not None:
        content.append({"type": "video"})
    if query_image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": query_text or "NULL"})
    content.append({"type": "text", "text": "\n<Document>:"})
    if document_video is not None:
        content.append({"type": "video"})
    if document_image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": document_text or "NULL"})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Judge whether the Document meets the requirements based "
                        'on the Query and Instruct. Answer only "yes" or "no".'
                    ),
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def make_qwen3_vl_processor(
    checkpoint: str | Path,
    config: Qwen3VLConfig,
    *,
    mode: Qwen3VLProcessorMode = "embedding",
    sequence_length_buckets: Sequence[int] = (512, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    default_instruction: str | None = None,
) -> Processor:
    """Load HF tokenizer/media artifacts into the generic Representax processor."""

    if mode not in {"embedding", "reranking"}:
        raise ValueError("Qwen3-VL processor mode must be embedding or reranking")
    try:
        upstream_type = import_module("transformers").Qwen3VLProcessor
    except ImportError as error:
        raise ImportError(
            "Qwen3-VL artifact processing requires `pip install representax[hf]`"
        ) from error
    padding_side = "right" if mode == "embedding" else "left"
    upstream = upstream_type.from_pretrained(
        checkpoint,
        padding_side=padding_side,
    )
    instruction = default_instruction or (
        "Represent the user's input."
        if mode == "embedding"
        else "Given a search query, retrieve relevant candidates that answer the query."
    )
    maximum = max(sequence_length_buckets)

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> Qwen3VLBatch:
        del route, seed
        if not artifacts:
            raise ValueError("Qwen3-VL processor batches must be non-empty")
        conversations = []
        images = []
        videos = []
        for artifact in artifacts:
            conversation = (
                _embedding_conversation(artifact, default_instruction=instruction)
                if mode == "embedding"
                else _reranking_conversation(artifact, default_instruction=instruction)
            )
            conversations.append(conversation)
            candidates = (
                (artifact["query"], artifact["document"])
                if mode == "reranking"
                else (artifact,)
            )
            for candidate in candidates:
                _, image, video = _components(candidate)
                if video is not None:
                    videos.append(video)
                if image is not None:
                    images.append(image)
        rendered = upstream.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        features = upstream(
            text=rendered,
            images=images or None,
            videos=videos or None,
            truncation=True,
            max_length=maximum,
            padding=True,
            return_tensors="np",
        )
        return batch_from_processor_output(
            features,
            config,
            sequence_length_buckets=sequence_length_buckets,
            patch_count_buckets=patch_count_buckets,
            padding_side=padding_side,
        )

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-qwen3-vl-processor-v1",
            "mode": mode,
            "checkpoint": str(Path(checkpoint).resolve()),
            "sequence_length_buckets": list(sequence_length_buckets),
            "patch_count_buckets": list(patch_count_buckets),
            "padding_side": padding_side,
            "default_instruction": instruction,
        },
    )


def vision_layout(
    grids: Sequence[Sequence[int]],
    config: Qwen3VLVisionConfig,
    *,
    patch_bucket: int | None = None,
) -> dict[str, np.ndarray]:
    """Create segmented attention, rotary, and interpolation arrays from THW grids."""

    grid_values = tuple(tuple(int(value) for value in grid) for grid in grids)
    if any(len(grid) != 3 or any(value <= 0 for value in grid) for grid in grid_values):
        raise ValueError("vision grids must contain positive (time, height, width)")
    merge = config.spatial_merge_size
    if any(height % merge or width % merge for _, height, width in grid_values):
        raise ValueError("vision grid height and width must divide spatial_merge_size")
    patch_count = sum(time * height * width for time, height, width in grid_values)
    bucket = patch_count if patch_bucket is None else int(patch_bucket)
    if bucket < patch_count or bucket % config.spatial_merge_unit:
        raise ValueError("patch bucket must contain patches and divide the merge unit")

    coordinates = []
    segments = []
    corner_indices = [[] for _ in range(4)]
    corner_weights = [[] for _ in range(4)]
    side = math.isqrt(config.num_position_embeddings)
    segment = 0
    for time, height, width in grid_values:
        rows = np.linspace(0, side - 1, height, dtype=np.float32)
        columns = np.linspace(0, side - 1, width, dtype=np.float32)
        row_floor = rows.astype(np.int32)
        column_floor = columns.astype(np.int32)
        row_ceil = np.minimum(row_floor + 1, side - 1)
        column_ceil = np.minimum(column_floor + 1, side - 1)
        row_delta = rows - row_floor
        column_delta = columns - column_floor
        base = row_floor[:, None] * side
        base_ceil = row_ceil[:, None] * side
        indices = (
            base + column_floor[None],
            base + column_ceil[None],
            base_ceil + column_floor[None],
            base_ceil + column_ceil[None],
        )
        weights = (
            (1 - row_delta[:, None]) * (1 - column_delta[None]),
            (1 - row_delta[:, None]) * column_delta[None],
            row_delta[:, None] * (1 - column_delta[None]),
            row_delta[:, None] * column_delta[None],
        )
        merged_height = height // merge
        merged_width = width // merge
        row_ids = np.broadcast_to(
            np.arange(height).reshape(merged_height, merge)[:, None, :, None],
            (merged_height, merged_width, merge, merge),
        ).reshape(-1)
        column_ids = np.broadcast_to(
            np.arange(width).reshape(merged_width, merge)[None, :, None, :],
            (merged_height, merged_width, merge, merge),
        ).reshape(-1)
        order = (row_ids * width + column_ids).astype(np.int32)
        frame_coordinates = np.stack((row_ids, column_ids), axis=-1)
        for _ in range(time):
            coordinates.append(frame_coordinates)
            segments.append(np.full((height * width,), segment, dtype=np.int32))
            for corner, (index, weight) in enumerate(
                zip(indices, weights, strict=True)
            ):
                corner_indices[corner].append(index.reshape(-1)[order])
                corner_weights[corner].append(weight.reshape(-1)[order])
            segment += 1

    padding = bucket - patch_count
    position_ids = (
        np.concatenate(coordinates) if coordinates else np.empty((0, 2), dtype=np.int32)
    )
    segment_ids = (
        np.concatenate(segments) if segments else np.empty((0,), dtype=np.int32)
    )
    interpolation_indices = np.stack(
        [
            np.concatenate(values) if values else np.empty((0,), dtype=np.int32)
            for values in corner_indices
        ]
    )
    interpolation_weights = np.stack(
        [
            np.concatenate(values) if values else np.empty((0,), dtype=np.float32)
            for values in corner_weights
        ]
    )
    return {
        "patch_valid": np.pad(
            np.ones((patch_count,), dtype=bool),
            (0, padding),
        ),
        "vision_segment_ids": np.pad(
            segment_ids,
            (0, padding),
            constant_values=-1,
        ),
        "vision_position_ids": np.pad(position_ids, ((0, padding), (0, 0))),
        "position_interpolation_indices": np.pad(
            interpolation_indices,
            ((0, 0), (0, padding)),
        ),
        "position_interpolation_weights": np.pad(
            interpolation_weights,
            ((0, 0), (0, padding)),
        ),
    }


def multimodal_position_ids(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    modality_ids: np.ndarray,
    image_grids: Sequence[Sequence[int]],
    video_grids: Sequence[Sequence[int]],
    *,
    spatial_merge_size: int,
) -> np.ndarray:
    """Reproduce Qwen3-VL's text/temporal/height/width position assignment."""

    if input_ids.shape != attention_mask.shape or input_ids.shape != modality_ids.shape:
        raise ValueError("token, attention, and modality arrays must align")
    result = np.zeros((3, *input_ids.shape), dtype=np.int32)
    grids = {1: iter(image_grids), 2: iter(video_grids)}
    for batch_index in range(input_ids.shape[0]):
        valid = attention_mask[batch_index].astype(bool)
        types = modality_ids[batch_index, valid].tolist()
        current = 0
        chunks = []
        for modality, group in itertools.groupby(types):
            length = sum(1 for _ in group)
            if modality == 0:
                positions = np.arange(current, current + length, dtype=np.int32)
                chunks.append(np.broadcast_to(positions[None], (3, length)))
                current += length
                continue
            try:
                time, height, width = (
                    int(value) for value in next(grids[int(modality)])
                )
            except (KeyError, StopIteration) as error:
                raise ValueError(
                    "modality token runs exceed supplied vision grids"
                ) from error
            merged_height = height // spatial_merge_size
            merged_width = width // spatial_merge_size
            expected = time * merged_height * merged_width
            if expected != length:
                raise ValueError(
                    f"vision token run has length {length}; grid requires {expected}"
                )
            temporal = np.full((expected,), current, dtype=np.int32)
            row = np.repeat(
                np.arange(current, current + merged_height, dtype=np.int32),
                merged_width * time,
            )
            column = np.tile(
                np.arange(current, current + merged_width, dtype=np.int32),
                merged_height * time,
            )
            chunks.append(np.stack((temporal, row, column)))
            current += max(merged_height, merged_width)
        positions = np.concatenate(chunks, axis=1) if chunks else np.empty((3, 0))
        result[:, batch_index, valid] = positions
    for modality, iterator in grids.items():
        try:
            next(iterator)
        except StopIteration:
            continue
        raise ValueError(f"unused modality {modality} grids remain")
    return result


def batch_from_processor_output(
    features: Mapping[str, Any],
    config: Qwen3VLConfig,
    *,
    sequence_length_buckets: Sequence[int],
    patch_count_buckets: Sequence[int],
    padding_side: str = "right",
) -> Qwen3VLBatch:
    """Convert standard Qwen3VLProcessor arrays into one finite native batch."""

    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    input_ids = np.asarray(features["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(features["attention_mask"], dtype=np.int32)
    modality_ids = np.asarray(
        features.get("mm_token_type_ids", np.zeros_like(input_ids)),
        dtype=np.int32,
    )
    if input_ids.ndim != 2:
        raise ValueError("processor input_ids must have shape [batch, sequence]")
    sequence_bucket = select_static_shape_bucket(
        (input_ids.shape[1],),
        tuple((value,) for value in sequence_length_buckets),
    )[0]
    token_padding = sequence_bucket - input_ids.shape[1]
    pad_width = (token_padding, 0) if padding_side == "left" else (0, token_padding)
    input_ids = np.pad(
        input_ids,
        ((0, 0), pad_width),
        constant_values=config.text.pad_token_id,
    )
    attention_mask = np.pad(attention_mask, ((0, 0), pad_width))
    modality_ids = np.pad(modality_ids, ((0, 0), pad_width))

    image_grids = np.asarray(
        features.get("image_grid_thw", np.empty((0, 3))), dtype=np.int32
    ).reshape((-1, 3))
    video_grids = np.asarray(
        features.get("video_grid_thw", np.empty((0, 3))), dtype=np.int32
    ).reshape((-1, 3))
    position_ids = multimodal_position_ids(
        input_ids,
        attention_mask,
        modality_ids,
        image_grids,
        video_grids,
        spatial_merge_size=config.vision.spatial_merge_size,
    )
    grids = [*image_grids.tolist(), *video_grids.tolist()]
    if not grids:
        return Qwen3VLBatch(
            input_ids=jnp.asarray(input_ids),
            attention_mask=jnp.asarray(attention_mask),
            position_ids=jnp.asarray(position_ids),
        )
    image_pixels = np.asarray(
        features.get(
            "pixel_values",
            np.empty((0, config.vision.patch_dimension)),
        )
    ).reshape((-1, config.vision.patch_dimension))
    video_pixels = np.asarray(
        features.get(
            "pixel_values_videos",
            np.empty((0, config.vision.patch_dimension)),
        )
    ).reshape((-1, config.vision.patch_dimension))
    pixels = np.concatenate((image_pixels, video_pixels), axis=0)
    patch_count = sum(int(np.prod(grid)) for grid in grids)
    if pixels.shape[0] != patch_count:
        raise ValueError("pixel rows and vision grids describe different patch counts")
    patch_bucket = select_static_shape_bucket(
        (patch_count,),
        tuple((value,) for value in patch_count_buckets),
    )[0]
    layout = vision_layout(grids, config.vision, patch_bucket=patch_bucket)
    pixels = np.pad(pixels, ((0, patch_bucket - patch_count), (0, 0)))
    flattened_modalities = modality_ids.reshape(-1)
    visual_indices = np.concatenate(
        (
            np.flatnonzero(flattened_modalities == 1),
            np.flatnonzero(flattened_modalities == 2),
        )
    ).astype(np.int32)
    expected_visual = patch_count // config.vision.spatial_merge_unit
    if visual_indices.size != expected_visual:
        raise ValueError("vision placeholders and merged patch count do not match")
    visual_bucket = patch_bucket // config.vision.spatial_merge_unit
    visual_padding = visual_bucket - expected_visual
    return Qwen3VLBatch(
        input_ids=jnp.asarray(input_ids),
        attention_mask=jnp.asarray(attention_mask),
        position_ids=jnp.asarray(position_ids),
        pixel_values=jnp.asarray(pixels),
        patch_valid=jnp.asarray(layout["patch_valid"]),
        vision_segment_ids=jnp.asarray(layout["vision_segment_ids"]),
        vision_position_ids=jnp.asarray(layout["vision_position_ids"]),
        position_interpolation_indices=jnp.asarray(
            layout["position_interpolation_indices"]
        ),
        position_interpolation_weights=jnp.asarray(
            layout["position_interpolation_weights"]
        ),
        visual_token_indices=jnp.asarray(np.pad(visual_indices, (0, visual_padding))),
        visual_token_valid=jnp.asarray(
            np.pad(
                np.ones((expected_visual,), dtype=bool),
                (0, visual_padding),
            )
        ),
    )


__all__ = [
    "batch_from_processor_output",
    "make_qwen3_vl_processor",
    "multimodal_position_ids",
    "vision_layout",
]
