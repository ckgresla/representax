"""Host-side finite-shape preprocessing for Qwen2.5-Omni artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.processing import Processor, select_static_shape_bucket

from .config import Qwen2_5OmniConfig, Qwen2_5OmniVisionConfig
from .model import Qwen2_5OmniBatch


def _components(
    value: Any,
) -> tuple[str | None, Any | None, Any | None, Any | None]:
    if isinstance(value, str):
        return value, None, None, None
    if not isinstance(value, Mapping):
        raise TypeError("Qwen2.5-Omni samples must be strings or mappings")
    text = value.get("text")
    if text is not None and not isinstance(text, str):
        raise TypeError("Qwen2.5-Omni sample text must be a string")
    return text, value.get("image"), value.get("audio"), value.get("video")


def _conversation(value: Any, *, text_prefix: str = "") -> list[dict[str, Any]]:
    text, image, audio, video = _components(value)
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image"})
    if audio is not None:
        content.append({"type": "audio"})
    if video is not None:
        content.append({"type": "video"})
    if text is not None:
        content.append({"type": "text", "text": text_prefix + text})
    if not content:
        raise ValueError("Qwen2.5-Omni samples must contain at least one modality")
    return [{"role": "user", "content": content}]


def _expand_placeholders(
    rendered: Sequence[str],
    *,
    image_grids: Sequence[Sequence[int]],
    video_grids: Sequence[Sequence[int]],
    audio_token_lengths: Sequence[int],
    spatial_merge_size: int,
) -> list[str]:
    """Expand one multimodal marker into the exact model-token population."""

    image_iterator = iter(image_grids)
    video_iterator = iter(video_grids)
    audio_iterator = iter(audio_token_lengths)
    markers = {
        "<|IMAGE|>": (image_iterator, "<|image_placeholder|>", "image"),
        "<|VIDEO|>": (video_iterator, "<|video_placeholder|>", "video"),
        "<|AUDIO|>": (audio_iterator, "<|audio_placeholder|>", "audio"),
    }
    expanded = []
    for sample in rendered:
        positions = sorted(
            (sample.index(marker), marker) for marker in markers if marker in sample
        )
        while positions:
            _, marker = positions[0]
            iterator, placeholder, modality = markers[marker]
            try:
                value = next(iterator)
            except StopIteration as error:
                raise ValueError(
                    f"{modality} markers exceed processed artifacts"
                ) from error
            if modality == "audio":
                count = int(cast(int, value))
            else:
                count = int(np.prod(value)) // spatial_merge_size**2
            sample = sample.replace(marker, placeholder * count, 1)
            positions = sorted(
                (sample.index(candidate), candidate)
                for candidate in markers
                if candidate in sample
            )
        sample = sample.replace("<|image_placeholder|>", "<|IMAGE|>")
        sample = sample.replace("<|video_placeholder|>", "<|VIDEO|>")
        sample = sample.replace("<|audio_placeholder|>", "<|AUDIO|>")
        expanded.append(sample)
    for iterator, _, modality in markers.values():
        try:
            next(iterator)
        except StopIteration:
            continue
        raise ValueError(f"unused processed {modality} artifacts remain")
    return expanded


def _video_frames(value: Any) -> tuple[np.ndarray, float]:
    fps = 2.0
    if isinstance(value, Mapping):
        fps = float(value.get("fps", fps))
        value = value.get("frames")
    if value is None:
        raise ValueError("video artifacts require decoded frames")
    if isinstance(value, np.ndarray):
        frames = value
    else:
        frames = np.stack([np.asarray(frame) for frame in value])
    if frames.ndim != 4:
        raise ValueError(
            "decoded video must have shape [frame, height, width, channel]"
        )
    if frames.shape[-1] not in {1, 3, 4} and frames.shape[1] in {1, 3, 4}:
        frames = np.moveaxis(frames, 1, -1)
    if frames.shape[-1] not in {1, 3, 4}:
        raise ValueError("video frames must expose one, three, or four channels")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("video fps must be finite and positive")
    return frames, fps


def _process_videos(
    videos: Sequence[Any],
    config: Qwen2_5OmniConfig,
    *,
    min_pixels: int,
    max_pixels: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> dict[str, np.ndarray]:
    """Prepare already-decoded frames without introducing a Torch dependency."""

    Image = import_module("PIL.Image")

    patch = config.vision.patch_size
    temporal = config.vision.temporal_patch_size
    merge = config.vision.spatial_merge_size
    factor = patch * merge
    mean = np.asarray(image_mean, dtype=np.float32)
    std = np.asarray(image_std, dtype=np.float32)
    all_patches = []
    all_grids = []
    seconds_per_grid = []
    for video in videos:
        frames, fps = _video_frames(video)
        height, width = frames.shape[1:3]
        aspect = max(height, width) / min(height, width)
        if aspect > 200:
            raise ValueError("video frame aspect ratio must be smaller than 200")
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        if resized_height * resized_width > max_pixels:
            beta = np.sqrt((height * width) / max_pixels)
            resized_height = max(factor, int(np.floor(height / beta / factor)) * factor)
            resized_width = max(factor, int(np.floor(width / beta / factor)) * factor)
        elif resized_height * resized_width < min_pixels:
            beta = np.sqrt(min_pixels / (height * width))
            resized_height = int(np.ceil(height * beta / factor)) * factor
            resized_width = int(np.ceil(width * beta / factor)) * factor
        normalized = []
        for frame in frames:
            if frame.dtype != np.uint8:
                scale = 255.0 if np.max(frame) <= 1.0 else 1.0
                frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
            resized = Image.fromarray(frame).resize(
                (resized_width, resized_height),
                resample=Image.Resampling.BICUBIC,
            )
            values = np.asarray(resized, dtype=np.float32) / 255.0
            values = (values - mean) / std
            normalized.append(np.moveaxis(values, -1, 0))
        values = np.stack(normalized)
        if padding := -len(values) % temporal:
            values = np.concatenate((values, np.repeat(values[-1:], padding, axis=0)))
        grid_time = len(values) // temporal
        grid_height = resized_height // patch
        grid_width = resized_width // patch
        values = values.reshape(
            grid_time,
            temporal,
            3,
            grid_height // merge,
            merge,
            patch,
            grid_width // merge,
            merge,
            patch,
        ).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        all_patches.append(values.reshape((-1, config.vision.patch_dimension)))
        all_grids.append((grid_time, grid_height, grid_width))
        seconds_per_grid.append(temporal / fps)
    return {
        "pixel_values_videos": np.concatenate(all_patches),
        "video_grid_thw": np.asarray(all_grids, dtype=np.int32),
        "video_second_per_grid": np.asarray(seconds_per_grid, dtype=np.float32),
    }


def vision_layout(
    grids: Sequence[Sequence[int]],
    config: Qwen2_5OmniVisionConfig,
    *,
    patch_bucket: int | None = None,
) -> dict[str, np.ndarray]:
    """Create Qwen2.5 window order, segmented masks, RoPE, and inverse order."""

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

    group_order = []
    position_ids = []
    full_segment_ids = []
    window_lengths = []
    group_offset = 0
    full_segment = 0
    window = config.merger_window_size
    for time, height, width in values:
        merged_height = height // merge
        merged_width = width // merge
        groups = np.arange(time * merged_height * merged_width, dtype=np.int32)
        groups = groups.reshape((time, merged_height, merged_width))
        padded_height = ((merged_height + window - 1) // window) * window
        padded_width = ((merged_width + window - 1) // window) * window
        padded = np.full((time, padded_height, padded_width), -1, dtype=np.int32)
        padded[:, :merged_height, :merged_width] = groups
        windowed = padded.reshape(
            time,
            padded_height // window,
            window,
            padded_width // window,
            window,
        ).transpose(0, 1, 3, 2, 4)
        for candidate in windowed.reshape((-1, window * window)):
            valid = candidate[candidate >= 0]
            if valid.size:
                group_order.extend((valid + group_offset).tolist())
                window_lengths.append(int(valid.size * config.spatial_merge_unit))

        row_ids = np.broadcast_to(
            np.arange(height).reshape(merged_height, merge)[:, None, :, None],
            (merged_height, merged_width, merge, merge),
        ).reshape(-1)
        column_ids = np.broadcast_to(
            np.arange(width).reshape(merged_width, merge)[None, :, None, :],
            (merged_height, merged_width, merge, merge),
        ).reshape(-1)
        coordinates = np.stack((row_ids, column_ids), axis=-1)
        for _ in range(time):
            position_ids.append(coordinates)
            full_segment_ids.append(
                np.full((height * width,), full_segment, dtype=np.int32)
            )
            full_segment += 1
        group_offset += time * merged_height * merged_width

    spatial_unit = config.spatial_merge_unit
    valid_groups = patch_count // spatial_unit
    bucket_groups = bucket // spatial_unit
    group_order_array = np.asarray(group_order, dtype=np.int32)
    if group_order_array.shape != (valid_groups,):
        raise AssertionError("window order did not cover every merged patch group")
    padded_groups = np.arange(valid_groups, bucket_groups, dtype=np.int32)
    complete_group_order = np.concatenate((group_order_array, padded_groups))
    patch_order = (
        complete_group_order[:, None] * spatial_unit
        + np.arange(spatial_unit, dtype=np.int32)[None]
    ).reshape(-1)

    original_positions = np.pad(
        np.concatenate(position_ids) if position_ids else np.empty((0, 2)),
        ((0, bucket - patch_count), (0, 0)),
    ).astype(np.int32)
    original_full_segments = np.pad(
        np.concatenate(full_segment_ids)
        if full_segment_ids
        else np.empty((0,), dtype=np.int32),
        (0, bucket - patch_count),
        constant_values=-1,
    )
    window_segments = np.full((bucket,), -1, dtype=np.int32)
    cursor = 0
    for segment, length in enumerate(window_lengths):
        window_segments[cursor : cursor + length] = segment
        cursor += length
    if cursor != patch_count:
        raise AssertionError("window lengths did not cover every valid patch")

    return {
        "patch_order": patch_order,
        "patch_valid": np.pad(
            np.ones((patch_count,), dtype=bool), (0, bucket - patch_count)
        ),
        "full_segment_ids": original_full_segments[patch_order],
        "window_segment_ids": window_segments,
        "position_ids": original_positions[patch_order],
        "reverse_merged_indices": np.argsort(complete_group_order).astype(np.int32),
    }


def audio_layout(
    input_features: np.ndarray,
    feature_attention_mask: np.ndarray,
    config: Qwen2_5OmniConfig,
    *,
    chunk_count_buckets: Sequence[int],
    token_count_buckets: Sequence[int],
) -> dict[str, np.ndarray]:
    """Pack valid mel frames into bounded chunks and exact AvgPool index pairs."""

    features = np.asarray(input_features, dtype=np.float32)
    mask = np.asarray(feature_attention_mask, dtype=bool)
    if features.ndim != 3 or features.shape[:2] != (
        mask.shape[0],
        config.audio.num_mel_bins,
    ):
        raise ValueError("audio features must have shape [audio, mel, feature]")
    if mask.shape != (features.shape[0], features.shape[2]):
        raise ValueError("feature_attention_mask must align with audio features")
    chunk_size = 2 * config.audio.window_size
    cnn_size = config.audio.window_size
    chunks = []
    chunk_lengths = []
    audio_cnn_indices = []
    for audio_index in range(features.shape[0]):
        length = int(mask[audio_index].sum())
        values = features[audio_index, :, :length]
        packed_indices = []
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            chunk = np.zeros((config.audio.num_mel_bins, chunk_size), np.float32)
            chunk[:, : stop - start] = values[:, start:stop]
            chunk_index = len(chunks)
            chunks.append(chunk)
            chunk_lengths.append(stop - start)
            after_cnn = (stop - start - 1) // 2 + 1
            packed_indices.extend(
                (chunk_index * cnn_size + np.arange(after_cnn)).tolist()
            )
        audio_cnn_indices.append(np.asarray(packed_indices, dtype=np.int32))
    required_chunks = len(chunks)
    if required_chunks == 0:
        raise ValueError("audio features contain no valid frames")
    chunk_bucket = select_static_shape_bucket(
        (required_chunks,), tuple((value,) for value in chunk_count_buckets)
    )[0]
    padded_chunks = np.zeros(
        (chunk_bucket, config.audio.num_mel_bins, chunk_size), dtype=np.float32
    )
    padded_chunks[:required_chunks] = np.stack(chunks)
    feature_valid = np.zeros((chunk_bucket, chunk_size), dtype=bool)
    after_cnn_valid = np.zeros((chunk_bucket, cnn_size), dtype=bool)
    for index, length in enumerate(chunk_lengths):
        feature_valid[index, :length] = True
        after_cnn_valid[index, : (length - 1) // 2 + 1] = True

    pairs = []
    for indices in audio_cnn_indices:
        pair_count = max((len(indices) - 2) // 2 + 1, 0)
        pairs.extend(indices[: 2 * pair_count].reshape((-1, 2)).tolist())
    token_count = len(pairs)
    if token_count == 0:
        raise ValueError("audio is too short to produce one pooled token")
    token_bucket = select_static_shape_bucket(
        (token_count,), tuple((value,) for value in token_count_buckets)
    )[0]
    pool_indices = np.zeros((token_bucket, 2), dtype=np.int32)
    pool_indices[:token_count] = np.asarray(pairs, dtype=np.int32)
    return {
        "input_features": padded_chunks,
        "feature_valid": feature_valid,
        "after_cnn_valid": after_cnn_valid,
        "pool_indices": pool_indices,
        "token_valid": np.pad(
            np.ones((token_count,), dtype=bool), (0, token_bucket - token_count)
        ),
        "feature_lengths": mask.sum(axis=1).astype(np.int32),
    }


def multimodal_position_ids(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    config: Qwen2_5OmniConfig,
    *,
    image_grids: Sequence[Sequence[int]],
    video_grids: Sequence[Sequence[int]],
    audio_feature_lengths: Sequence[int],
    video_seconds_per_grid: Sequence[float] | None = None,
) -> np.ndarray:
    """Reproduce the standard non-interleaved Qwen2.5-Omni MRoPE layout."""

    tokens = np.asarray(input_ids, dtype=np.int32)
    mask = np.asarray(attention_mask, dtype=bool)
    if tokens.shape != mask.shape or tokens.ndim != 2:
        raise ValueError("input_ids and attention_mask must be aligned matrices")
    result = np.ones((3, *tokens.shape), dtype=np.int32)
    images = [tuple(int(value) for value in grid) for grid in image_grids]
    videos = [tuple(int(value) for value in grid) for grid in video_grids]
    audio_lengths = [int(value) for value in audio_feature_lengths]
    video_seconds = (
        [1.0] * len(videos)
        if video_seconds_per_grid is None
        else [float(value) for value in video_seconds_per_grid]
    )
    image_index = video_index = audio_index = 0
    modality_ids = {
        config.image_token_id: "image",
        config.video_token_id: "video",
        config.audio_token_id: "audio",
    }
    merge = config.vision.spatial_merge_size

    for batch_index, row in enumerate(tokens):
        valid_positions = np.flatnonzero(mask[batch_index])
        row_tokens = row[valid_positions].tolist()
        chunks = []
        cursor = 0
        current = 0
        while cursor < len(row_tokens):
            candidates = [
                (row_tokens.index(token_id, cursor), name)
                for token_id, name in modality_ids.items()
                if token_id in row_tokens[cursor:]
            ]
            if not candidates:
                length = len(row_tokens) - cursor
                positions = np.arange(current, current + length, dtype=np.int32)
                chunks.append(np.broadcast_to(positions[None], (3, length)))
                current += length
                break
            placeholder, modality = min(candidates)
            bos = placeholder - 1
            if bos < cursor:
                raise ValueError("modality placeholder is missing its boundary token")
            text_length = bos - cursor
            if text_length:
                positions = np.arange(current, current + text_length, dtype=np.int32)
                chunks.append(np.broadcast_to(positions[None], (3, text_length)))
                current += text_length
            chunks.append(np.full((3, 1), current, dtype=np.int32))
            current += 1

            if modality == "audio":
                if audio_index >= len(audio_lengths):
                    raise ValueError("audio placeholders exceed supplied features")
                feature_length = audio_lengths[audio_index]
                after_cnn = (feature_length - 1) // 2 + 1
                modality_length = max((after_cnn - 2) // 2 + 1, 0)
                positions = np.arange(
                    current, current + modality_length, dtype=np.int32
                )
                modality_positions = np.broadcast_to(
                    positions[None], (3, modality_length)
                )
                audio_index += 1
            else:
                grids = images if modality == "image" else videos
                index = image_index if modality == "image" else video_index
                if index >= len(grids):
                    raise ValueError(f"{modality} placeholders exceed supplied grids")
                time, height, width = grids[index]
                merged_height = height // merge
                merged_width = width // merge
                seconds = 1.0 if modality == "image" else video_seconds[index]
                temporal = np.repeat(
                    (
                        np.arange(time, dtype=np.float32)
                        * seconds
                        * config.position_ids_per_second
                    ).astype(np.int32),
                    merged_height * merged_width,
                )
                rows = np.tile(np.repeat(np.arange(merged_height), merged_width), time)
                columns = np.tile(np.arange(merged_width), time * merged_height)
                modality_positions = (
                    np.stack((temporal, rows, columns)).astype(np.int32) + current
                )
                modality_length = modality_positions.shape[1]
                if modality == "image":
                    image_index += 1
                else:
                    video_index += 1
            chunks.append(modality_positions)
            current = int(np.max(modality_positions).item()) + 1
            chunks.append(np.full((3, 1), current, dtype=np.int32))
            current += 1
            cursor = placeholder + modality_length + 1

        positions = np.concatenate(chunks, axis=1)
        if positions.shape[1] != len(valid_positions):
            raise ValueError("multimodal positions and valid tokens do not align")
        result[:, batch_index, valid_positions] = positions

    if image_index != len(images) or video_index != len(videos):
        raise ValueError("unused vision grids remain after position assignment")
    if audio_index != len(audio_lengths):
        raise ValueError("unused audio features remain after position assignment")
    return result


def batch_from_processor_output(
    features: Mapping[str, Any],
    config: Qwen2_5OmniConfig,
    *,
    sequence_length_buckets: Sequence[int],
    patch_count_buckets: Sequence[int],
    audio_chunk_count_buckets: Sequence[int],
    audio_token_count_buckets: Sequence[int],
) -> Qwen2_5OmniBatch:
    """Convert Qwen2_5OmniProcessor arrays into one finite native batch."""

    input_ids = np.asarray(features["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(features["attention_mask"], dtype=np.int32)
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("processor token arrays must be aligned matrices")
    sequence_bucket = select_static_shape_bucket(
        (input_ids.shape[1],),
        tuple((value,) for value in sequence_length_buckets),
    )[0]
    token_padding = sequence_bucket - input_ids.shape[1]
    input_ids = np.pad(
        input_ids,
        ((0, 0), (0, token_padding)),
        constant_values=config.pad_token_id,
    )
    attention_mask = np.pad(attention_mask, ((0, 0), (0, token_padding)))

    image_grids = np.asarray(
        features.get("image_grid_thw", np.empty((0, 3))), dtype=np.int32
    ).reshape((-1, 3))
    video_grids = np.asarray(
        features.get("video_grid_thw", np.empty((0, 3))), dtype=np.int32
    ).reshape((-1, 3))
    has_audio = "input_features" in features
    audio = None
    if has_audio:
        audio = audio_layout(
            np.asarray(features["input_features"]),
            np.asarray(features["feature_attention_mask"]),
            config,
            chunk_count_buckets=audio_chunk_count_buckets,
            token_count_buckets=audio_token_count_buckets,
        )
    position_ids = multimodal_position_ids(
        input_ids,
        attention_mask,
        config,
        image_grids=image_grids,
        video_grids=video_grids,
        audio_feature_lengths=(
            () if audio is None else audio["feature_lengths"].tolist()
        ),
        video_seconds_per_grid=features.get("video_second_per_grid"),
    )

    vision = None
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
    pixels = np.concatenate((image_pixels, video_pixels), axis=0)
    grids = [*image_grids.tolist(), *video_grids.tolist()]
    if grids:
        patch_count = sum(int(np.prod(grid)) for grid in grids)
        if pixels.shape[0] != patch_count:
            raise ValueError("pixel rows and grids describe different patch counts")
        patch_bucket = select_static_shape_bucket(
            (patch_count,), tuple((value,) for value in patch_count_buckets)
        )[0]
        layout = vision_layout(grids, config.vision, patch_bucket=patch_bucket)
        padded_pixels = np.pad(pixels, ((0, patch_bucket - patch_count), (0, 0)))
        vision = {
            **layout,
            "pixel_values": padded_pixels[layout["patch_order"]],
        }

    flattened = input_ids.reshape(-1)
    values: dict[str, Any] = {
        "input_ids": jnp.asarray(input_ids),
        "attention_mask": jnp.asarray(attention_mask),
        "position_ids": jnp.asarray(position_ids),
    }
    if vision is not None:
        visual_indices = np.concatenate(
            (
                np.flatnonzero(flattened == config.image_token_id),
                np.flatnonzero(flattened == config.video_token_id),
            )
        ).astype(np.int32)
        visual_count = (
            int(vision["patch_valid"].sum()) // config.vision.spatial_merge_unit
        )
        visual_bucket = len(vision["reverse_merged_indices"])
        if visual_indices.size != visual_count:
            raise ValueError("vision placeholders and merged patches do not match")
        values.update(
            pixel_values=jnp.asarray(vision["pixel_values"]),
            patch_valid=jnp.asarray(vision["patch_valid"]),
            vision_full_segment_ids=jnp.asarray(vision["full_segment_ids"]),
            vision_window_segment_ids=jnp.asarray(vision["window_segment_ids"]),
            vision_position_ids=jnp.asarray(vision["position_ids"]),
            reverse_merged_indices=jnp.asarray(vision["reverse_merged_indices"]),
            visual_token_indices=jnp.asarray(
                np.pad(visual_indices, (0, visual_bucket - visual_count))
            ),
            visual_token_valid=jnp.asarray(
                np.pad(
                    np.ones((visual_count,), dtype=bool),
                    (0, visual_bucket - visual_count),
                )
            ),
        )
    if audio is not None:
        audio_indices = np.flatnonzero(flattened == config.audio_token_id).astype(
            np.int32
        )
        audio_count = int(audio["token_valid"].sum())
        audio_bucket = len(audio["token_valid"])
        if audio_indices.size != audio_count:
            raise ValueError("audio placeholders and pooled tokens do not match")
        values.update(
            input_features=jnp.asarray(audio["input_features"]),
            audio_feature_valid=jnp.asarray(audio["feature_valid"]),
            audio_after_cnn_valid=jnp.asarray(audio["after_cnn_valid"]),
            audio_pool_indices=jnp.asarray(audio["pool_indices"]),
            audio_token_indices=jnp.asarray(
                np.pad(audio_indices, (0, audio_bucket - audio_count))
            ),
            audio_token_valid=jnp.asarray(audio["token_valid"]),
        )
    return Qwen2_5OmniBatch(**values)


def make_qwen2_5_omni_processor(
    checkpoint: str | Path,
    config: Qwen2_5OmniConfig,
    *,
    sequence_length_buckets: Sequence[int] = (128, 512, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    audio_chunk_count_buckets: Sequence[int] = (1, 4, 16, 64, 256),
    audio_token_count_buckets: Sequence[int] = (64, 256, 1024, 4096),
    chat_template: str = "sentence_transformers",
) -> Processor:
    """Load tokenizer/media artifacts into the generic Representax processor."""

    try:
        transformers = import_module("transformers")
        image_module = import_module(
            "transformers.models.qwen2_vl.image_processing_pil_qwen2_vl"
        )
    except ImportError as error:
        raise ImportError(
            "Qwen2.5-Omni processing requires Transformers 5.6 or newer"
        ) from error
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint,
        padding_side="right",
    )
    if tokenizer is None:
        raise TypeError("Qwen2.5-Omni checkpoint did not produce a tokenizer")
    image_processor = image_module.Qwen2VLImageProcessorPil.from_pretrained(checkpoint)
    audio_processor = transformers.WhisperFeatureExtractor.from_pretrained(checkpoint)
    maximum = max(sequence_length_buckets)
    sentence_config_path = Path(checkpoint) / "config_sentence_transformers.json"
    sentence_config = (
        json.loads(sentence_config_path.read_text())
        if sentence_config_path.is_file()
        else {}
    )
    raw_prompts = sentence_config.get("prompts", {})
    prompts = (
        {
            Route.QUERY: str(raw_prompts.get("query", "")),
            Route.DOCUMENT: str(raw_prompts.get("document", "")),
        }
        if isinstance(raw_prompts, Mapping)
        else {Route.QUERY: "", Route.DOCUMENT: ""}
    )

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> Qwen2_5OmniBatch:
        del seed
        if not artifacts:
            raise ValueError("Qwen2.5-Omni processor batches must be non-empty")
        conversations = []
        images = []
        audios = []
        videos = []
        for artifact in artifacts:
            conversations.append(
                _conversation(artifact, text_prefix=prompts.get(route, ""))
            )
            _, image, audio, video = _components(artifact)
            if image is not None:
                images.append(image)
            if audio is not None:
                audios.append(audio)
            if video is not None:
                videos.append(video)
        rendered = cast(
            list[str],
            tokenizer.apply_chat_template(
                conversations,
                chat_template=chat_template,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
        image_features = (
            {}
            if not images
            else dict(image_processor(images=images, return_tensors="np"))
        )
        audio_features = {}
        audio_lengths: np.ndarray = np.empty((0,), dtype=np.int32)
        if audios:
            extracted = dict(
                audio_processor(
                    audios,
                    sampling_rate=audio_processor.sampling_rate,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="np",
                )
            )
            extracted["feature_attention_mask"] = extracted.pop("attention_mask")
            audio_features = extracted
            feature_lengths = np.asarray(extracted["feature_attention_mask"]).sum(
                axis=-1
            )
            after_cnn = (feature_lengths - 1) // 2 + 1
            audio_lengths = ((after_cnn - 2) // 2 + 1).astype(np.int32)
        video_features = (
            {}
            if not videos
            else _process_videos(
                videos,
                config,
                min_pixels=128 * 28 * 28,
                max_pixels=768 * 28 * 28,
                image_mean=image_processor.image_mean,
                image_std=image_processor.image_std,
            )
        )
        image_grids = np.asarray(
            image_features.get("image_grid_thw", np.empty((0, 3)))
        ).reshape((-1, 3))
        video_grids = np.asarray(
            video_features.get("video_grid_thw", np.empty((0, 3)))
        ).reshape((-1, 3))
        expanded = _expand_placeholders(
            rendered,
            image_grids=image_grids,
            video_grids=video_grids.tolist(),
            audio_token_lengths=audio_lengths.tolist(),
            spatial_merge_size=config.vision.spatial_merge_size,
        )
        token_features = dict(
            tokenizer(
                expanded,
                truncation=True,
                max_length=maximum,
                padding=True,
                return_tensors="np",
            )
        )
        features = {
            **token_features,
            **image_features,
            **video_features,
            **audio_features,
        }
        return batch_from_processor_output(
            features,
            config,
            sequence_length_buckets=sequence_length_buckets,
            patch_count_buckets=patch_count_buckets,
            audio_chunk_count_buckets=audio_chunk_count_buckets,
            audio_token_count_buckets=audio_token_count_buckets,
        )

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-qwen2.5-omni-processor-v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "sequence_length_buckets": list(sequence_length_buckets),
            "patch_count_buckets": list(patch_count_buckets),
            "audio_chunk_count_buckets": list(audio_chunk_count_buckets),
            "audio_token_count_buckets": list(audio_token_count_buckets),
            "chat_template": chat_template,
            "padding_side": "right",
            "audio_in_video": False,
            "route_prompts": {
                route.value: prompt for route, prompt in prompts.items() if prompt
            },
        },
    )


__all__ = [
    "audio_layout",
    "batch_from_processor_output",
    "make_qwen2_5_omni_processor",
    "multimodal_position_ids",
    "vision_layout",
]
