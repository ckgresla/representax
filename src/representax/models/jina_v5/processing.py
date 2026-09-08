"""Finite-shape preprocessing for both Jina Embeddings v5 Omni sizes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.processing import Processor, select_static_shape_bucket
from representax.models.qwen2_5_omni.processing import (
    audio_layout,
    process_video_frames,
)
from representax.models.qwen3_vl.processing import vision_layout

from .config import JinaV5OmniConfig
from .omni import JinaV5OmniBatch


def _components(value: Any) -> tuple[str | None, Any | None, Any | None, Any | None]:
    if isinstance(value, str):
        return value, None, None, None
    if not isinstance(value, Mapping):
        raise TypeError("Jina v5 Omni samples must be strings or mappings")
    text = value.get("text")
    if text is not None and not isinstance(text, str):
        raise TypeError("Jina v5 Omni sample text must be a string")
    return text, value.get("image"), value.get("audio"), value.get("video")


def _route_prompts(checkpoint: Path) -> dict[Route, str]:
    path = checkpoint / "config_sentence_transformers.json"
    value = json.loads(path.read_text()) if path.is_file() else {}
    prompts = value.get("prompts", {})
    if not isinstance(prompts, Mapping):
        return {Route.QUERY: "", Route.DOCUMENT: ""}
    return {
        Route.QUERY: str(prompts.get("query", "")),
        Route.DOCUMENT: str(prompts.get("document", "")),
    }


def _audio_array(value: Any) -> tuple[np.ndarray, int]:
    sampling_rate = 16_000
    if isinstance(value, Mapping):
        sampling_rate = int(value.get("sampling_rate", sampling_rate))
        value = value.get("array")
    if not isinstance(value, np.ndarray):
        raise TypeError(
            "Jina v5 Omni audio must be a waveform array or an array/rate mapping"
        )
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0 if audio.shape[0] <= 8 else 1)
    if audio.ndim != 1:
        raise ValueError("Jina v5 Omni audio waveforms must be one-dimensional")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio, sampling_rate


def _audio_features(
    values: Sequence[Any],
    config: JinaV5OmniConfig,
) -> tuple[dict[str, np.ndarray], list[int]]:
    if not values:
        return {}, []
    feature_extractor = import_module("transformers").WhisperFeatureExtractor(
        feature_size=config.audio.num_mel_bins
    )
    arrays = []
    for value in values:
        audio, sampling_rate = _audio_array(value)
        if sampling_rate != feature_extractor.sampling_rate:
            raise ValueError(
                f"audio sampling rate must be {feature_extractor.sampling_rate} Hz"
            )
        arrays.append(audio)
    features = dict(
        feature_extractor(
            arrays,
            sampling_rate=feature_extractor.sampling_rate,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="np",
        )
    )
    mask = np.asarray(features.pop("attention_mask"), dtype=np.int32)
    features["feature_attention_mask"] = mask
    feature_lengths = mask.sum(axis=-1)
    after_cnn = (feature_lengths - 1) // 2 + 1
    token_lengths = ((after_cnn - 2) // 2 + 1).astype(np.int32)
    return features, token_lengths.tolist()


def _prompt(
    value: Any,
    *,
    prefix: str,
    config: JinaV5OmniConfig,
    tokenizer: Any,
    image_token_length: int | None,
    video_token_length: int | None,
    audio_token_length: int | None,
) -> str:
    text, image, audio, video = _components(value)
    parts = [prefix]
    if image is not None:
        if image_token_length is None:
            raise AssertionError("image token length was not computed")
        token = tokenizer.convert_ids_to_tokens(config.image_token_id)
        value = token * image_token_length
        if config.vision_start_token_id is not None:
            start = tokenizer.convert_ids_to_tokens(config.vision_start_token_id)
            end = tokenizer.convert_ids_to_tokens(config.vision_end_token_id)
            value = start + value + end
        parts.append(value)
    if video is not None:
        if video_token_length is None:
            raise AssertionError("video token length was not computed")
        token = tokenizer.convert_ids_to_tokens(config.video_token_id)
        value = token * video_token_length
        if config.vision_start_token_id is not None:
            start = tokenizer.convert_ids_to_tokens(config.vision_start_token_id)
            end = tokenizer.convert_ids_to_tokens(config.vision_end_token_id)
            value = start + value + end
        parts.append(value)
    if audio is not None:
        if audio_token_length is None:
            raise AssertionError("audio token length was not computed")
        start = tokenizer.convert_ids_to_tokens(config.audio_start_token_id)
        token = tokenizer.convert_ids_to_tokens(config.audio_token_id)
        end = tokenizer.convert_ids_to_tokens(config.audio_end_token_id)
        parts.append(start + token * audio_token_length + end)
    if text is not None:
        parts.append(text)
    if len(parts) == 1 and not prefix:
        raise ValueError("Jina v5 Omni samples must contain at least one modality")
    return "".join(parts)


def _token_positions(
    input_ids: np.ndarray,
    rows: Sequence[int],
    token_id: int,
    lengths: Sequence[int],
) -> list[tuple[int, int]]:
    positions = []
    for row, expected in zip(rows, lengths, strict=True):
        columns = np.flatnonzero(input_ids[row] == token_id)
        if columns.size != expected:
            raise ValueError("modality placeholders and encoded tokens do not match")
        positions.extend((row, int(column)) for column in columns)
    return positions


def _shared_visual_positions(
    input_ids: np.ndarray,
    *,
    token_id: int,
    image_rows: Sequence[int],
    image_lengths: Sequence[int],
    video_rows: Sequence[int],
    video_lengths: Sequence[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    image_counts = dict(zip(image_rows, image_lengths, strict=True))
    video_counts = dict(zip(video_rows, video_lengths, strict=True))
    by_row = {}
    for row in {*image_rows, *video_rows}:
        columns = np.flatnonzero(input_ids[row] == token_id)
        expected = image_counts.get(row, 0) + video_counts.get(row, 0)
        if columns.size != expected:
            raise ValueError("shared visual placeholders do not match encoded tokens")
        by_row[row] = columns
    image_positions = [
        (row, int(column))
        for row in image_rows
        for column in by_row[row][: image_counts[row]]
    ]
    video_positions = [
        (row, int(column))
        for row in video_rows
        for column in by_row[row][
            image_counts.get(row, 0) : image_counts.get(row, 0) + video_counts[row]
        ]
    ]
    return image_positions, video_positions


def prepare_jina_v5_omni_features(
    artifacts: Sequence[Any],
    *,
    route: Route,
    checkpoint: str | Path,
    config: JinaV5OmniConfig,
    tokenizer: Any,
    image_processor: Any,
    maximum_sequence_length: int,
    video_min_pixels: int,
    video_max_pixels: int,
) -> dict[str, np.ndarray]:
    """Create the exact unpadded arrays consumed by both reference and native models."""

    path = Path(checkpoint)
    prompts = _route_prompts(path)
    images, videos, audios = [], [], []
    image_rows: list[int] = []
    video_rows: list[int] = []
    audio_rows: list[int] = []
    for row, artifact in enumerate(artifacts):
        _, image, audio, video = _components(artifact)
        if image is not None:
            images.append(image)
            image_rows.append(row)
        if video is not None:
            videos.append(video)
            video_rows.append(row)
        if audio is not None:
            audios.append(audio)
            audio_rows.append(row)

    image_features = (
        {} if not images else dict(image_processor(images=images, return_tensors="np"))
    )
    video_features = (
        {}
        if not videos
        else process_video_frames(
            videos,
            config.vision,
            min_pixels=video_min_pixels,
            max_pixels=video_max_pixels,
            image_mean=image_processor.image_mean,
            image_std=image_processor.image_std,
        )
    )
    audio_features, audio_lengths = _audio_features(audios, config)
    merge = config.vision.spatial_merge_unit
    image_lengths = [
        int(np.prod(grid)) // merge
        for grid in np.asarray(
            image_features.get("image_grid_thw", np.empty((0, 3)))
        ).reshape((-1, 3))
    ]
    video_lengths = [
        int(np.prod(grid)) // merge
        for grid in np.asarray(
            video_features.get("video_grid_thw", np.empty((0, 3)))
        ).reshape((-1, 3))
    ]
    image_lengths_by_row = dict(zip(image_rows, image_lengths, strict=True))
    video_lengths_by_row = dict(zip(video_rows, video_lengths, strict=True))
    audio_lengths_by_row = dict(zip(audio_rows, audio_lengths, strict=True))
    rendered = [
        _prompt(
            artifact,
            prefix=prompts.get(route, ""),
            config=config,
            tokenizer=tokenizer,
            image_token_length=image_lengths_by_row.get(row),
            video_token_length=video_lengths_by_row.get(row),
            audio_token_length=audio_lengths_by_row.get(row),
        )
        for row, artifact in enumerate(artifacts)
    ]
    token_features = dict(
        tokenizer(
            rendered,
            truncation=True,
            max_length=maximum_sequence_length,
            padding=True,
            return_tensors="np",
        )
    )
    input_ids = np.asarray(token_features["input_ids"], dtype=np.int32)
    if config.image_token_id == config.video_token_id:
        image_positions, video_positions = _shared_visual_positions(
            input_ids,
            token_id=config.image_token_id,
            image_rows=image_rows,
            image_lengths=image_lengths,
            video_rows=video_rows,
            video_lengths=video_lengths,
        )
    else:
        image_positions = _token_positions(
            input_ids, image_rows, config.image_token_id, image_lengths
        )
        video_positions = _token_positions(
            input_ids, video_rows, config.video_token_id, video_lengths
        )
    return {
        **token_features,
        **image_features,
        **video_features,
        **audio_features,
        "_visual_token_positions": np.asarray(
            [*image_positions, *video_positions], dtype=np.int32
        ).reshape((-1, 2)),
    }


def batch_from_processor_output(
    features: Mapping[str, Any],
    config: JinaV5OmniConfig,
    *,
    sequence_length_buckets: Sequence[int],
    patch_count_buckets: Sequence[int],
    audio_chunk_count_buckets: Sequence[int],
    audio_token_count_buckets: Sequence[int],
) -> JinaV5OmniBatch:
    """Convert released processor arrays into one native finite-shape batch."""

    input_ids = np.asarray(features["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(features["attention_mask"], dtype=np.int32)
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("processor token arrays must be aligned matrices")
    sequence_bucket = select_static_shape_bucket(
        (input_ids.shape[1],),
        tuple((value,) for value in sequence_length_buckets),
    )[0]
    padding = sequence_bucket - input_ids.shape[1]
    input_ids = np.pad(
        input_ids,
        ((0, 0), (0, padding)),
        constant_values=config.text.pad_token_id,
    )
    attention_mask = np.pad(attention_mask, ((0, 0), (0, padding)))
    positions = np.maximum(np.cumsum(attention_mask, axis=-1) - 1, 0).astype(np.int32)

    values: dict[str, Any] = {
        "input_ids": jnp.asarray(input_ids),
        "attention_mask": jnp.asarray(attention_mask),
        "position_ids": jnp.asarray(np.broadcast_to(positions, (3, *positions.shape))),
    }

    image_grids = np.asarray(
        features.get("image_grid_thw", np.empty((0, 3))), dtype=np.int32
    ).reshape((-1, 3))
    video_grids = np.asarray(
        features.get("video_grid_thw", np.empty((0, 3))), dtype=np.int32
    ).reshape((-1, 3))
    grids = [*image_grids.tolist(), *video_grids.tolist()]
    if grids:
        image_pixels = np.asarray(
            features.get("pixel_values", np.empty((0, config.vision.patch_dimension))),
            dtype=np.float32,
        ).reshape((-1, config.vision.patch_dimension))
        video_pixels = np.asarray(
            features.get(
                "pixel_values_videos",
                np.empty((0, config.vision.patch_dimension)),
            ),
            dtype=np.float32,
        ).reshape((-1, config.vision.patch_dimension))
        pixels = np.concatenate((image_pixels, video_pixels), axis=0)
        patch_count = sum(int(np.prod(grid)) for grid in grids)
        if pixels.shape[0] != patch_count:
            raise ValueError("pixel rows and vision grids describe different patches")
        patch_bucket = select_static_shape_bucket(
            (patch_count,), tuple((value,) for value in patch_count_buckets)
        )[0]
        layout = vision_layout(grids, config.vision, patch_bucket=patch_bucket)
        pixels = np.pad(pixels, ((0, patch_bucket - patch_count), (0, 0)))
        token_positions = features.get("_visual_token_positions")
        if token_positions is not None:
            token_positions = np.asarray(token_positions, dtype=np.int32).reshape(
                (-1, 2)
            )
            visual_indices = (
                token_positions[:, 0] * sequence_bucket + token_positions[:, 1]
            )
        else:
            flat = input_ids.reshape(-1)
            if config.image_token_id == config.video_token_id:
                visual_indices = np.flatnonzero(flat == config.image_token_id)
            else:
                visual_indices = np.concatenate(
                    (
                        np.flatnonzero(flat == config.image_token_id),
                        np.flatnonzero(flat == config.video_token_id),
                    )
                )
        visual_indices = visual_indices.astype(np.int32)
        visual_count = patch_count // config.vision.spatial_merge_unit
        if visual_indices.size != visual_count:
            raise ValueError("vision placeholders and merged patches do not match")
        visual_bucket = patch_bucket // config.vision.spatial_merge_unit
        owners = visual_indices // sequence_bucket
        counts = np.bincount(owners, minlength=input_ids.shape[0])
        sample_patch_bucket = select_static_shape_bucket(
            (int(counts.max()) * config.vision.spatial_merge_unit,),
            tuple((value,) for value in patch_count_buckets),
        )[0]
        group_table = np.full(
            (
                input_ids.shape[0],
                sample_patch_bucket // config.vision.spatial_merge_unit,
            ),
            -1,
            dtype=np.int32,
        )
        for row in range(input_ids.shape[0]):
            groups = np.flatnonzero(owners == row)
            group_table[row, : groups.size] = groups
        values.update(
            visual_group_indices=jnp.asarray(group_table),
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

    if "input_features" in features:
        audio = audio_layout(
            np.asarray(features["input_features"]),
            np.asarray(features["feature_attention_mask"]),
            config.audio,
            chunk_count_buckets=audio_chunk_count_buckets,
            token_count_buckets=audio_token_count_buckets,
        )
        audio_rows = np.flatnonzero(np.any(input_ids == config.audio_token_id, axis=1))
        if len(audio_rows) != audio["input_features"].shape[0]:
            raise ValueError("audio features and sample rows do not align")
        audio_bucket = audio["token_valid"].shape[1]
        audio_indices = np.zeros((input_ids.shape[0], audio_bucket), dtype=np.int32)
        audio_valid = np.zeros((input_ids.shape[0], audio_bucket), dtype=bool)
        for audio_index, row_index in enumerate(audio_rows):
            indices = np.flatnonzero(input_ids[row_index] == config.audio_token_id)
            token_count = int(audio["token_valid"][audio_index].sum())
            if indices.size != token_count:
                raise ValueError("audio placeholders and pooled tokens do not match")
            audio_indices[row_index, :token_count] = indices
            audio_valid[row_index, :token_count] = True

        def align_rows(array: np.ndarray) -> np.ndarray:
            aligned = np.zeros(
                (input_ids.shape[0], *array.shape[1:]), dtype=array.dtype
            )
            aligned[audio_rows] = array
            return aligned

        values.update(
            input_features=jnp.asarray(align_rows(audio["input_features"])),
            audio_feature_valid=jnp.asarray(align_rows(audio["feature_valid"])),
            audio_after_cnn_valid=jnp.asarray(align_rows(audio["after_cnn_valid"])),
            audio_pool_indices=jnp.asarray(align_rows(audio["pool_indices"])),
            audio_token_indices=jnp.asarray(audio_indices),
            audio_token_valid=jnp.asarray(audio_valid),
        )
    return JinaV5OmniBatch(**values)


def make_jina_v5_omni_processor(
    checkpoint: str | Path,
    config: JinaV5OmniConfig,
    *,
    sequence_length_buckets: Sequence[int] = (128, 512, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    audio_chunk_count_buckets: Sequence[int] = (1, 4, 16, 64, 256),
    audio_token_count_buckets: Sequence[int] = (64, 256, 1024, 4096),
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
) -> Processor:
    """Load Jina's tokenizer/media assets and retain a finite shape policy."""

    path = Path(checkpoint)
    try:
        transformers = import_module("transformers")
        tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(
            path,
            padding_side="right",
        )
        image_options = {}
        if image_min_pixels is not None:
            image_options["min_pixels"] = image_min_pixels
        if image_max_pixels is not None:
            image_options["max_pixels"] = image_max_pixels
        image_processor = transformers.Qwen2VLImageProcessor.from_pretrained(
            path, **image_options
        )
    except ImportError as error:
        raise ImportError(
            "Jina v5 Omni processing requires `pip install representax[hf]`"
        ) from error
    prompts = _route_prompts(path)
    maximum = min(max(sequence_length_buckets), config.text.max_position_embeddings)
    if max(sequence_length_buckets) > config.text.max_position_embeddings:
        raise ValueError("sequence buckets exceed the model position limit")
    resolved_image_min = int(image_processor.size.shortest_edge)
    resolved_image_max = int(image_processor.size.longest_edge)
    video_config_path = path / "video_preprocessor_config.json"
    video_config = (
        json.loads(video_config_path.read_text()) if video_config_path.is_file() else {}
    )
    video_size = video_config.get("size", {})
    resolved_video_min = int(
        video_min_pixels
        if video_min_pixels is not None
        else video_size.get("shortest_edge", resolved_image_min)
    )
    resolved_video_max = int(
        video_max_pixels
        if video_max_pixels is not None
        else video_size.get("longest_edge", resolved_image_max)
    )

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> JinaV5OmniBatch:
        del seed
        if not artifacts:
            raise ValueError("Jina v5 Omni processor batches must be non-empty")
        features = prepare_jina_v5_omni_features(
            artifacts,
            route=route,
            checkpoint=path,
            config=config,
            tokenizer=tokenizer,
            image_processor=image_processor,
            maximum_sequence_length=maximum,
            video_min_pixels=resolved_video_min,
            video_max_pixels=resolved_video_max,
        )
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
            "checkpoint": str(path.resolve()),
            "variant": config.variant,
            "sequence_length_buckets": list(sequence_length_buckets),
            "patch_count_buckets": list(patch_count_buckets),
            "audio_chunk_count_buckets": list(audio_chunk_count_buckets),
            "audio_token_count_buckets": list(audio_token_count_buckets),
            "padding_side": "right",
            "image_min_pixels": resolved_image_min,
            "image_max_pixels": resolved_image_max,
            "video_min_pixels": resolved_video_min,
            "video_max_pixels": resolved_video_max,
            "route_prompts": {
                route.value: prompt for route, prompt in prompts.items() if prompt
            },
        },
    )


__all__ = [
    "batch_from_processor_output",
    "make_jina_v5_omni_processor",
    "prepare_jina_v5_omni_features",
]
