"""Finite-shape Qwen2/Qwen2.5-VL preprocessing for embeddings and reranking."""

from __future__ import annotations

import io
import itertools
import json
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from representax.core import Modality, Route
from representax.data import Artifact
from representax.models.processing import Processor, select_static_shape_bucket

from .config import Qwen2VLConfig, Qwen2VLVisionConfig
from .model import Qwen2VLBatch

Qwen2VLProcessorMode = Literal[
    "bge_embedding", "nomic_embedding", "reranking", "embedding"
]


def _components(value: Any) -> tuple[str | None, Any | None, Any | None]:
    if isinstance(value, Artifact):
        materialized = value.data
        if materialized is None:
            payload = value.read_bytes()
            if value.modality == Modality.TEXT:
                materialized = payload.decode()
            elif value.modality == Modality.IMAGE:
                image = import_module("PIL.Image")
                materialized = image.open(io.BytesIO(payload)).copy()
            else:
                materialized = payload
        if value.modality == Modality.TEXT:
            if not isinstance(materialized, str):
                raise TypeError("Qwen2-VL text artifacts must contain strings")
            return materialized, None, None
        if value.modality == Modality.IMAGE:
            return None, materialized, None
        if value.modality == Modality.VIDEO:
            return None, None, materialized
        raise TypeError("Qwen2-VL accepts text, image, and video artifacts")
    if isinstance(value, str):
        return value, None, None
    if not isinstance(value, Mapping):
        return None, value, None
    text = value.get("text")
    if text is not None and not isinstance(text, str):
        raise TypeError("Qwen2-VL sample text must be a string")
    return text, value.get("image"), value.get("video")


def _bge_messages(value: Any, *, instruction: str) -> list[dict[str, Any]]:
    text, image, video = _components(value)
    if video is not None:
        raise ValueError("BGE-VL-Screenshot does not define a video prompt")
    instruction = (
        value.get("instruction", instruction)
        if isinstance(value, Mapping)
        else instruction
    )
    if not isinstance(instruction, str):
        raise TypeError("BGE instructions must be strings")
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    if image is not None:
        content.append({"type": "image", "image": image})
    if not content:
        raise ValueError("BGE-VL-Screenshot samples require text or image")
    messages: list[dict[str, Any]] = []
    if instruction:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": instruction}]}
        )
    messages.append({"role": "user", "content": content})
    return messages


def _jina_prompt(value: Any) -> tuple[str, list[Any], list[Any]]:
    if isinstance(value, Mapping):
        if "query" not in value or "document" not in value:
            raise TypeError("Jina reranking samples require query and document")
        query_value, document_value = value["query"], value["document"]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise TypeError("Jina reranking pairs must contain exactly two items")
        query_value, document_value = value
    else:
        raise TypeError("Jina reranking samples require query and document")
    query_text, query_image, query_video = _components(query_value)
    document_text, document_image, document_video = _components(document_value)
    if query_video is not None or document_video is not None:
        raise ValueError("Jina reranker m0 does not define video pair formatting")
    query = (
        "<|vision_start|><|image_pad|><|vision_end|>"
        if query_image is not None
        else query_text or ""
    )
    document = (
        "<|vision_start|><|image_pad|><|vision_end|>"
        if document_image is not None
        else document_text or ""
    )
    prompt = f"**Document**:\n{document}\n**Query**:\n{query}"
    images = [item for item in (document_image, query_image) if item is not None]
    return prompt, images, []


def vision_layout(
    grids: Sequence[Sequence[int]],
    config: Qwen2VLVisionConfig,
    *,
    patch_bucket: int | None = None,
) -> dict[str, np.ndarray]:
    """Create exact packed-attention order, masks, rotary coordinates, and inverse."""

    values = tuple(tuple(int(item) for item in grid) for grid in grids)
    if any(len(grid) != 3 or any(item <= 0 for item in grid) for grid in values):
        raise ValueError("vision grids must contain positive (time, height, width)")
    merge = config.spatial_merge_size
    if any(height % merge or width % merge for _, height, width in values):
        raise ValueError("vision grid height and width must divide spatial_merge_size")
    patch_count = sum(time * height * width for time, height, width in values)
    bucket = patch_count if patch_bucket is None else int(patch_bucket)
    if bucket < patch_count or bucket % config.spatial_merge_unit:
        raise ValueError("patch bucket must contain patches and divide the merge unit")

    group_order: list[int] = []
    positions = []
    full_segments = []
    window_lengths = []
    group_offset = segment = 0
    window = config.merger_window_size
    for time, height, width in values:
        merged_height, merged_width = height // merge, width // merge
        groups = np.arange(time * merged_height * merged_width, dtype=np.int32)
        groups = groups.reshape((time, merged_height, merged_width))
        if config.generation == "qwen2_vl":
            ordered_windows = (groups.reshape(-1),)
        else:
            padded_height = ((merged_height + window - 1) // window) * window
            padded_width = ((merged_width + window - 1) // window) * window
            padded = np.full((time, padded_height, padded_width), -1, dtype=np.int32)
            padded[:, :merged_height, :merged_width] = groups
            ordered_windows = (
                padded.reshape(
                    time,
                    padded_height // window,
                    window,
                    padded_width // window,
                    window,
                )
                .transpose(0, 1, 3, 2, 4)
                .reshape((-1, window * window))
            )
        for candidate in ordered_windows:
            valid = np.asarray(candidate).reshape(-1)
            valid = valid[valid >= 0]
            if valid.size:
                group_order.extend((valid + group_offset).tolist())
                window_lengths.append(int(valid.size * config.spatial_merge_unit))

        rows = np.broadcast_to(
            np.arange(height).reshape(merged_height, merge)[:, None, :, None],
            (merged_height, merged_width, merge, merge),
        ).reshape(-1)
        columns = np.broadcast_to(
            np.arange(width).reshape(merged_width, merge)[None, :, None, :],
            (merged_height, merged_width, merge, merge),
        ).reshape(-1)
        coordinates = np.stack((rows, columns), axis=-1)
        for _ in range(time):
            positions.append(coordinates)
            full_segments.append(np.full((height * width,), segment, np.int32))
            segment += 1
        group_offset += time * merged_height * merged_width

    unit = config.spatial_merge_unit
    valid_groups, bucket_groups = patch_count // unit, bucket // unit
    group_order_array = np.asarray(group_order, dtype=np.int32)
    if group_order_array.shape != (valid_groups,):
        raise AssertionError("vision order did not cover every merged patch group")
    complete_order = np.concatenate(
        (group_order_array, np.arange(valid_groups, bucket_groups, dtype=np.int32))
    )
    patch_order = (complete_order[:, None] * unit + np.arange(unit)[None]).reshape(-1)
    original_positions = np.pad(
        np.concatenate(positions), ((0, bucket - patch_count), (0, 0))
    ).astype(np.int32)
    original_segments = np.pad(
        np.concatenate(full_segments), (0, bucket - patch_count), constant_values=-1
    )
    window_segments = np.full((bucket,), -1, dtype=np.int32)
    cursor = 0
    for window_segment, length in enumerate(window_lengths):
        window_segments[cursor : cursor + length] = window_segment
        cursor += length
    return {
        "patch_order": patch_order,
        "patch_valid": np.pad(np.ones((patch_count,), bool), (0, bucket - patch_count)),
        "full_segment_ids": original_segments[patch_order],
        "window_segment_ids": window_segments,
        "position_ids": original_positions[patch_order],
        "reverse_merged_indices": np.argsort(complete_order).astype(np.int32),
    }


def multimodal_position_ids(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    token_types: np.ndarray,
    config: Qwen2VLConfig,
    *,
    image_grids: Sequence[Sequence[int]],
    video_grids: Sequence[Sequence[int]],
    video_seconds_per_grid: Sequence[float] | None = None,
) -> np.ndarray:
    """Reproduce upstream grouped text/image/video MRoPE positions."""

    tokens = np.asarray(input_ids, dtype=np.int32)
    mask = np.asarray(attention_mask, dtype=bool)
    types = np.asarray(token_types, dtype=np.int32)
    if tokens.shape != mask.shape or types.shape != tokens.shape:
        raise ValueError("token, mask, and modality-type arrays must align")
    result = np.zeros((3, *tokens.shape), dtype=np.int32)
    grids = {1: iter(image_grids), 2: iter(video_grids)}
    seconds = iter(video_seconds_per_grid or [1.0] * len(video_grids))
    merge = config.vision.spatial_merge_size
    for row_index in range(tokens.shape[0]):
        valid = np.flatnonzero(mask[row_index])
        row_types = types[row_index, valid]
        groups = []
        for modality, group in itertools.groupby(
            enumerate(row_types.tolist()), lambda item: item[1]
        ):
            members = list(group)
            groups.append((modality, members[0][0], members[-1][0] + 1))
        current = 0
        chunks = []
        for modality, start, stop in groups:
            if modality == 0:
                length = stop - start
                chunks.append(
                    np.broadcast_to(np.arange(length)[None] + current, (3, length))
                )
                current += length
                continue
            if modality not in grids:
                raise ValueError(f"unsupported Qwen2-VL modality type {modality}")
            try:
                time, height, width = (int(item) for item in next(grids[modality]))
            except StopIteration as error:
                raise ValueError("vision token groups exceed supplied grids") from error
            time, height, width = time, height // merge, width // merge
            interval = (
                int(config.vision.tokens_per_second * int(float(next(seconds))))
                if modality == 2 and config.generation == "qwen2_5_vl"
                else 1
            )
            temporal = np.repeat(np.arange(time) * interval + current, height * width)
            rows = np.tile(np.repeat(np.arange(height) + current, width), time)
            columns = np.tile(np.arange(width) + current, time * height)
            chunk = np.stack((temporal, rows, columns)).astype(np.int32)
            if chunk.shape[1] != stop - start:
                raise ValueError("vision type run and grid token count differ")
            chunks.append(chunk)
            current += max(height, width)
        positions = np.concatenate(chunks, axis=1)
        result[:, row_index, valid] = positions
    for modality, iterator in grids.items():
        if next(iterator, None) is not None:
            name = "image" if modality == 1 else "video"
            raise ValueError(f"supplied {name} grids exceed vision token groups")
    return result


def batch_from_processor_output(
    features: Mapping[str, Any],
    config: Qwen2VLConfig,
    *,
    sequence_length_buckets: Sequence[int],
    patch_count_buckets: Sequence[int],
    padding_side: Literal["left", "right"],
) -> Qwen2VLBatch:
    """Convert standard Qwen processor arrays into one finite native batch."""

    input_ids = np.asarray(features["input_ids"], dtype=np.int32)
    attention = np.asarray(features["attention_mask"], dtype=np.int32)
    token_types = np.asarray(
        features.get("mm_token_type_ids", np.zeros_like(input_ids)), dtype=np.int32
    )
    sequence_bucket = select_static_shape_bucket(
        (input_ids.shape[1],), tuple((value,) for value in sequence_length_buckets)
    )[0]
    padding = sequence_bucket - input_ids.shape[1]
    widths = ((0, 0), (padding, 0) if padding_side == "left" else (0, padding))
    input_ids = np.pad(input_ids, widths, constant_values=config.pad_token_id)
    attention = np.pad(attention, widths)
    token_types = np.pad(token_types, widths)
    image_grids = np.asarray(
        features.get("image_grid_thw", np.empty((0, 3))), dtype=np.int32
    ).reshape((-1, 3))
    video_grids = np.asarray(
        features.get("video_grid_thw", np.empty((0, 3))), dtype=np.int32
    ).reshape((-1, 3))
    position_ids = multimodal_position_ids(
        input_ids,
        attention,
        token_types,
        config,
        image_grids=image_grids,
        video_grids=video_grids,
        video_seconds_per_grid=features.get("second_per_grid_ts"),
    )
    values: dict[str, Any] = {
        "input_ids": jnp.asarray(input_ids),
        "attention_mask": jnp.asarray(attention),
        "position_ids": jnp.asarray(position_ids),
    }
    image_pixels = np.asarray(
        features.get("pixel_values", np.empty((0, config.vision.patch_dimension))),
        dtype=np.float32,
    ).reshape((-1, config.vision.patch_dimension))
    video_pixels = np.asarray(
        features.get(
            "pixel_values_videos", np.empty((0, config.vision.patch_dimension))
        ),
        dtype=np.float32,
    ).reshape((-1, config.vision.patch_dimension))
    pixels = np.concatenate((image_pixels, video_pixels))
    grids = [*image_grids.tolist(), *video_grids.tolist()]
    if grids:
        patch_count = sum(int(np.prod(grid)) for grid in grids)
        if pixels.shape[0] != patch_count:
            raise ValueError(
                "pixel rows and vision grids describe different patch counts"
            )
        patch_bucket = select_static_shape_bucket(
            (patch_count,), tuple((value,) for value in patch_count_buckets)
        )[0]
        layout = vision_layout(grids, config.vision, patch_bucket=patch_bucket)
        padded = np.pad(pixels, ((0, patch_bucket - patch_count), (0, 0)))
        flattened_types = token_types.reshape(-1)
        indices = np.flatnonzero(
            (flattened_types == 1) | (flattened_types == 2)
        ).astype(np.int32)
        visual_count = patch_count // config.vision.spatial_merge_unit
        visual_bucket = len(layout["reverse_merged_indices"])
        if indices.size != visual_count:
            raise ValueError("vision placeholders and merged patches do not match")
        values.update(
            pixel_values=jnp.asarray(padded[layout["patch_order"]]),
            patch_valid=jnp.asarray(layout["patch_valid"]),
            vision_full_segment_ids=jnp.asarray(layout["full_segment_ids"]),
            vision_window_segment_ids=jnp.asarray(layout["window_segment_ids"]),
            vision_position_ids=jnp.asarray(layout["position_ids"]),
            reverse_merged_indices=jnp.asarray(layout["reverse_merged_indices"]),
            visual_token_indices=jnp.asarray(
                np.pad(indices, (0, visual_bucket - visual_count))
            ),
            visual_token_valid=jnp.asarray(
                np.pad(
                    np.ones((visual_count,), bool), (0, visual_bucket - visual_count)
                )
            ),
        )
    return Qwen2VLBatch(**values)


def make_qwen2_vl_processor(
    checkpoint: str | Path,
    config: Qwen2VLConfig,
    *,
    mode: Qwen2VLProcessorMode = "embedding",
    sequence_length_buckets: Sequence[int] = (512, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    default_instruction: str = "Represent the given input.",
) -> Processor:
    """Load tokenizer/media artifacts and expose the generic Processor contract."""

    transformers = import_module("transformers")
    processor_type = (
        transformers.Qwen2VLProcessor
        if config.generation == "qwen2_vl"
        else transformers.Qwen2_5_VLProcessor
    )
    padding_side: Literal["left", "right"] = "left"
    upstream: Any = processor_type.from_pretrained(
        checkpoint, padding_side=padding_side
    )
    maximum = max(sequence_length_buckets)
    sentence_path = Path(checkpoint) / "config_sentence_transformers.json"
    sentence = json.loads(sentence_path.read_text()) if sentence_path.is_file() else {}
    prompts = sentence.get("prompts", {}) if isinstance(sentence, Mapping) else {}
    default_prompt_name = (
        sentence.get("default_prompt_name") if isinstance(sentence, Mapping) else None
    )

    def process(
        artifacts: Sequence[Any], *, route: Route, seed: int | None
    ) -> Qwen2VLBatch:
        del seed
        rendered = []
        images = []
        videos = []
        for artifact in artifacts:
            text, image, video = (
                _components(artifact) if mode != "reranking" else (None, None, None)
            )
            if mode == "bge_embedding":
                prompt_name = (
                    "query"
                    if route == Route.QUERY
                    else "document"
                    if route == Route.DOCUMENT
                    else default_prompt_name
                )
                instruction = (
                    str(prompts.get(prompt_name, default_instruction))
                    if isinstance(prompts, Mapping) and prompt_name is not None
                    else default_instruction
                )
                messages = _bge_messages(artifact, instruction=instruction)
                rendered.append(
                    upstream.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )
            elif mode == "nomic_embedding":
                content = []
                if text is not None:
                    content.append({"type": "text", "text": text})
                if image is not None:
                    content.append({"type": "image", "image": image})
                rendered.append(
                    upstream.apply_chat_template(
                        [{"role": "user", "content": content}],
                        add_generation_prompt=False,
                        tokenize=False,
                    )
                )
            elif mode == "reranking":
                prompt, pair_images, pair_videos = _jina_prompt(artifact)
                rendered.append(prompt)
                images.extend(pair_images)
                videos.extend(pair_videos)
                continue
            else:
                content = []
                if image is not None:
                    content.append({"type": "image"})
                if video is not None:
                    content.append({"type": "video"})
                if text is not None:
                    content.append({"type": "text", "text": text})
                rendered.append(
                    upstream.apply_chat_template(
                        [{"role": "user", "content": content}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )
            if image is not None:
                images.append(image)
            if video is not None:
                videos.append(video)
        features = dict(
            upstream(
                text=rendered,
                images=images or None,
                videos=videos or None,
                truncation=True,
                max_length=maximum - (1 if mode == "reranking" else 0),
                padding=True,
                return_tensors="np",
            )
        )
        if mode == "reranking":
            batch = len(rendered)
            features["input_ids"] = np.concatenate(
                (features["input_ids"], np.full((batch, 1), 100, np.int32)), axis=1
            )
            features["attention_mask"] = np.concatenate(
                (features["attention_mask"], np.ones((batch, 1), np.int32)), axis=1
            )
            if "mm_token_type_ids" in features:
                features["mm_token_type_ids"] = np.concatenate(
                    (features["mm_token_type_ids"], np.zeros((batch, 1), np.int32)),
                    axis=1,
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
            "schema_version": "representax-qwen2-vl-processor-v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "mode": mode,
            "sequence_length_buckets": list(sequence_length_buckets),
            "patch_count_buckets": list(patch_count_buckets),
            "padding_side": padding_side,
        },
    )


__all__ = [
    "Qwen2VLProcessorMode",
    "batch_from_processor_output",
    "make_qwen2_vl_processor",
    "multimodal_position_ids",
    "vision_layout",
]
