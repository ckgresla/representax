"""Model-associated finite preprocessing for BidirLM Omni."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.processing import Processor, select_static_shape_bucket
from representax.models.qwen3_vl.processing import (
    multimodal_position_ids,
    vision_layout,
)

from .config import BidirLMOmniConfig
from .model import BidirLMOmniBatch


def convolution_output_length(length: int | np.ndarray) -> int | np.ndarray:
    """Apply the three kernel-3, stride-2, padding-1 length transforms."""

    value = length
    for _ in range(3):
        value = (value - 1) // 2 + 1
    return value


def _video_timestamps(
    frame_indices: Sequence[int] | np.ndarray,
    *,
    frames_per_second: float,
    merge_size: int,
) -> tuple[float, ...]:
    """Return one timestamp for each temporally merged vision patch group."""

    indices = [int(index) for index in frame_indices]
    if not indices:
        raise ValueError("video metadata must contain at least one frame index")
    indices.extend(indices[-1] for _ in range((-len(indices)) % merge_size))
    timestamps = [index / frames_per_second for index in indices]
    return tuple(
        (timestamps[start] + timestamps[start + merge_size - 1]) / 2
        for start in range(0, len(timestamps), merge_size)
    )


def _normalize_audio(value: Any, *, target_rate: int) -> np.ndarray:
    if isinstance(value, Mapping):
        samples = np.asarray(value["array"], dtype=np.float32)
        source_rate = int(value.get("sampling_rate", target_rate))
    else:
        samples = np.asarray(value, dtype=np.float32)
        source_rate = target_rate
    if samples.ndim != 1:
        raise ValueError("BidirLM Omni audio must be one-dimensional mono samples")
    if source_rate != target_rate:
        try:
            librosa = import_module("librosa")
        except ImportError as error:
            raise ImportError(
                "audio resampling requires `pip install librosa`"
            ) from error
        samples = librosa.resample(
            samples, orig_sr=source_rate, target_sr=target_rate
        ).astype(np.float32)
    return samples


def _sample_to_conversation(
    value: Any,
) -> tuple[list[dict[str, Any]], list, list, list]:
    if isinstance(value, str):
        content = [{"type": "text", "text": value}]
    elif isinstance(value, Mapping) and "role" in value:
        raise TypeError("one conversation must be a sequence of role mappings")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if value and isinstance(value[0], Mapping) and "role" in value[0]:
            conversation = [dict(message) for message in value]
            content_items = [
                item
                for message in conversation
                for item in (
                    message.get("content", [])
                    if isinstance(message.get("content"), Sequence)
                    and not isinstance(message.get("content"), str)
                    else ()
                )
                if isinstance(item, Mapping)
            ]
            return (
                conversation,
                [item["image"] for item in content_items if "image" in item],
                [item["video"] for item in content_items if "video" in item],
                [item["audio"] for item in content_items if "audio" in item],
            )
        content = [{"type": "audio", "audio": value}]
    elif isinstance(value, Mapping):
        content = []
        if value.get("image") is not None:
            content.append({"type": "image", "image": value["image"]})
        if value.get("video") is not None:
            content.append({"type": "video", "video": value["video"]})
        if value.get("audio") is not None:
            content.append({"type": "audio", "audio": value["audio"]})
        if value.get("text") is not None:
            content.append({"type": "text", "text": str(value["text"])})
        if not content:
            # Hugging Face audio dictionaries are themselves one artifact.
            if "array" in value:
                content.append({"type": "audio", "audio": value})
            else:
                raise ValueError("BidirLM Omni samples require a supported modality")
    else:
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - image extra is installed in CI
            Image = None
        if (Image is not None and isinstance(value, Image.Image)) or (
            isinstance(value, np.ndarray) and value.ndim >= 2
        ):
            content = [{"type": "image", "image": value}]
        else:
            content = [{"type": "audio", "audio": value}]
    conversation = [{"role": "user", "content": content}]
    return (
        conversation,
        [item["image"] for item in content if "image" in item],
        [item["video"] for item in content if "video" in item],
        [item["audio"] for item in content if "audio" in item],
    )


def _audio_layout(
    input_features: np.ndarray,
    feature_lengths: np.ndarray,
    *,
    chunk_bucket: int,
    chunk_size: int,
    output_window: int,
) -> dict[str, np.ndarray]:
    chunks = []
    chunk_lengths = []
    audio_indexes = []
    for audio_index, length_value in enumerate(feature_lengths.tolist()):
        length = int(length_value)
        feature = input_features[audio_index, :, :length]
        for start in range(0, length, chunk_size):
            part = feature[:, start : start + chunk_size]
            padded = np.zeros((feature.shape[0], chunk_size), dtype=np.float32)
            padded[:, : part.shape[1]] = part
            chunks.append(padded)
            chunk_lengths.append(part.shape[1])
            audio_indexes.append(audio_index)
    if len(chunks) > chunk_bucket:
        raise ValueError("audio chunks exceed the selected static bucket")
    while len(chunks) < chunk_bucket:
        chunks.append(np.zeros((input_features.shape[1], chunk_size), np.float32))
        chunk_lengths.append(0)
        audio_indexes.append(-1)
    output_per_chunk = int(convolution_output_length(chunk_size))
    valid = np.zeros((chunk_bucket, output_per_chunk), dtype=bool)
    segments = np.full((chunk_bucket, output_per_chunk), -1, dtype=np.int32)
    feature_indices = []
    next_segment = 0
    per_audio_position = [0] * len(feature_lengths)
    segment_offsets = [0] * len(feature_lengths)
    for chunk_index, (length, audio_index) in enumerate(
        zip(chunk_lengths, audio_indexes, strict=True)
    ):
        if audio_index < 0:
            continue
        count = int(convolution_output_length(length))
        valid[chunk_index, :count] = True
        for local in range(count):
            position = per_audio_position[audio_index]
            segments[chunk_index, local] = segment_offsets[audio_index] + (
                position // output_window
            )
            per_audio_position[audio_index] += 1
            feature_indices.append(chunk_index * output_per_chunk + local)
        if (
            chunk_index + 1 == len(audio_indexes)
            or audio_indexes[chunk_index + 1] != audio_index
        ):
            used = max(
                1,
                (per_audio_position[audio_index] + output_window - 1) // output_window,
            )
            next_segment += used
            for later in range(audio_index + 1, len(segment_offsets)):
                segment_offsets[later] = next_segment
    capacity = chunk_bucket * output_per_chunk
    return {
        "input_features": np.stack(chunks),
        "audio_chunk_lengths": np.asarray(chunk_lengths, dtype=np.int32),
        "audio_output_valid": valid.reshape(-1),
        "audio_segment_ids": segments.reshape(-1),
        "audio_feature_indices": np.pad(
            np.asarray(feature_indices, dtype=np.int32),
            (0, capacity - len(feature_indices)),
        ),
        "audio_feature_valid": np.pad(
            np.ones((len(feature_indices),), dtype=bool),
            (0, capacity - len(feature_indices)),
        ),
    }


def batch_from_processor_output(
    features: Mapping[str, Any],
    config: BidirLMOmniConfig,
    *,
    sequence_length_buckets: Sequence[int],
    patch_count_buckets: Sequence[int],
    audio_chunk_buckets: Sequence[int],
) -> BidirLMOmniBatch:
    """Convert model processor arrays into one finite native batch."""

    input_ids = np.asarray(features["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(features["attention_mask"], dtype=np.int32)
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("processor token arrays must have shape [batch, sequence]")
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
    modality_ids = np.zeros_like(input_ids)
    modality_ids[input_ids == config.image_token_id] = 1
    modality_ids[input_ids == config.video_token_id] = 2
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
    values: dict[str, Any] = {
        "input_ids": jnp.asarray(input_ids),
        "attention_mask": jnp.asarray(attention_mask),
        "position_ids": jnp.asarray(position_ids),
    }
    grids = [*image_grids.tolist(), *video_grids.tolist()]
    if grids:
        image_pixels = np.asarray(
            features.get("pixel_values", np.empty((0, config.vision.patch_dimension)))
        ).reshape((-1, config.vision.patch_dimension))
        video_pixels = np.asarray(
            features.get(
                "pixel_values_videos", np.empty((0, config.vision.patch_dimension))
            )
        ).reshape((-1, config.vision.patch_dimension))
        pixels = np.concatenate((image_pixels, video_pixels), axis=0)
        patch_count = sum(int(np.prod(grid)) for grid in grids)
        if pixels.shape[0] != patch_count:
            raise ValueError("pixel rows and vision grids disagree")
        patch_bucket = select_static_shape_bucket(
            (patch_count,), tuple((value,) for value in patch_count_buckets)
        )[0]
        layout = vision_layout(grids, config.vision, patch_bucket=patch_bucket)
        pixels = np.pad(pixels, ((0, patch_bucket - patch_count), (0, 0)))
        flattened = modality_ids.reshape(-1)
        visual_indices = np.concatenate(
            (np.flatnonzero(flattened == 1), np.flatnonzero(flattened == 2))
        ).astype(np.int32)
        expected = patch_count // config.vision.spatial_merge_unit
        if visual_indices.size != expected:
            raise ValueError("vision placeholders and merged patches disagree")
        capacity = patch_bucket // config.vision.spatial_merge_unit
        values.update(
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
                np.pad(visual_indices, (0, capacity - expected))
            ),
            visual_token_valid=jnp.asarray(
                np.pad(np.ones((expected,), bool), (0, capacity - expected))
            ),
        )
    if "input_features" in features:
        input_features = np.asarray(features["input_features"], dtype=np.float32)
        feature_mask = np.asarray(features["feature_attention_mask"], dtype=bool)
        feature_lengths = feature_mask.sum(axis=-1).astype(np.int32)
        chunk_count = sum(
            (int(length) + 2 * config.audio.window_size - 1)
            // (2 * config.audio.window_size)
            for length in feature_lengths
        )
        chunk_bucket = select_static_shape_bucket(
            (chunk_count,), tuple((value,) for value in audio_chunk_buckets)
        )[0]
        audio = _audio_layout(
            input_features,
            feature_lengths,
            chunk_bucket=chunk_bucket,
            chunk_size=2 * config.audio.window_size,
            output_window=(
                int(convolution_output_length(2 * config.audio.window_size))
                * config.audio.inference_window_size
                // (2 * config.audio.window_size)
            ),
        )
        token_indices = np.flatnonzero(
            input_ids.reshape(-1) == config.audio_token_id
        ).astype(np.int32)
        valid_count = int(audio["audio_feature_valid"].sum())
        if token_indices.size != valid_count:
            raise ValueError("audio placeholders and convolved features disagree")
        capacity = audio["audio_feature_indices"].shape[0]
        values.update(
            input_features=jnp.asarray(audio["input_features"]),
            audio_chunk_lengths=jnp.asarray(audio["audio_chunk_lengths"]),
            audio_output_valid=jnp.asarray(audio["audio_output_valid"]),
            audio_segment_ids=jnp.asarray(audio["audio_segment_ids"]),
            audio_feature_indices=jnp.asarray(audio["audio_feature_indices"]),
            audio_token_indices=jnp.asarray(
                np.pad(token_indices, (0, capacity - valid_count))
            ),
            audio_token_valid=jnp.asarray(audio["audio_feature_valid"]),
        )
    return BidirLMOmniBatch(**values)


def make_bidirlm_omni_processor(
    checkpoint: str | Path,
    config: BidirLMOmniConfig,
    *,
    sequence_length_buckets: Sequence[int] = (512, 1024, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    audio_chunk_buckets: Sequence[int] = (1, 4, 8, 16),
) -> Processor:
    """Load standard HF artifacts without executing checkpoint Python code."""

    try:
        transformers = import_module("transformers")
    except ImportError as error:
        raise ImportError(
            "BidirLM Omni processing requires `pip install representax[hf]`"
        ) from error
    checkpoint = Path(checkpoint)
    tokenizer = cast(
        Any,
        transformers.AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False),
    )
    image_processor = transformers.Qwen2VLImageProcessor.from_pretrained(checkpoint)
    video_processor_type = import_module(
        "transformers.models.qwen3_vl.video_processing_qwen3_vl"
    ).Qwen3VLVideoProcessor
    video_processor = video_processor_type.from_pretrained(checkpoint)
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(checkpoint)
    sampling_rate = int(feature_extractor.sampling_rate)
    maximum = max(sequence_length_buckets)

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> BidirLMOmniBatch:
        del route, seed
        if not artifacts:
            raise ValueError("BidirLM Omni processor batches must be non-empty")
        conversations = []
        images = []
        videos = []
        audios = []
        for artifact in artifacts:
            conversation, sample_images, sample_videos, sample_audios = (
                _sample_to_conversation(artifact)
            )
            conversations.append(conversation)
            images.extend(sample_images)
            videos.extend(sample_videos)
            audios.extend(
                _normalize_audio(audio, target_rate=sampling_rate)
                for audio in sample_audios
            )
        rendered = cast(
            list[str],
            tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            ),
        )
        image_inputs = (
            image_processor(images=images, return_tensors="np") if images else {}
        )
        video_inputs = (
            video_processor(
                videos=videos,
                return_metadata=True,
                return_tensors="np",
            )
            if videos
            else {}
        )
        if images:
            grids = np.asarray(image_inputs["image_grid_thw"])
            for row in range(len(rendered)):
                while "<|image_pad|>" in rendered[row]:
                    grid = grids[0]
                    grids = grids[1:]
                    count = int(np.prod(grid)) // config.vision.spatial_merge_unit
                    rendered[row] = rendered[row].replace(
                        "<|image_pad|>", "<|image_placeholder|>" * count, 1
                    )
                rendered[row] = rendered[row].replace(
                    "<|image_placeholder|>", "<|image_pad|>"
                )
        if videos:
            grids = np.asarray(video_inputs["video_grid_thw"])
            metadata = video_inputs.pop("video_metadata")
            video_index = 0
            for row in range(len(rendered)):
                while "<|video_pad|>" in rendered[row]:
                    grid = grids[video_index]
                    video_metadata = metadata[video_index]
                    frames_per_second = video_metadata.fps
                    if frames_per_second is None:
                        frames_per_second = 24.0
                    timestamps = _video_timestamps(
                        video_metadata.frames_indices,
                        frames_per_second=float(frames_per_second),
                        merge_size=config.vision.spatial_merge_size,
                    )
                    frame_tokens = (
                        int(grid[1] * grid[2]) // config.vision.spatial_merge_unit
                    )
                    placeholder = "".join(
                        f"<{timestamp:.1f} seconds>"
                        "<|vision_start|>"
                        + "<|video_placeholder|>" * frame_tokens
                        + "<|vision_end|>"
                        for timestamp in timestamps
                    )
                    wrapped = "<|vision_start|><|video_pad|><|vision_end|>"
                    if wrapped in rendered[row]:
                        rendered[row] = rendered[row].replace(wrapped, placeholder, 1)
                    else:
                        rendered[row] = rendered[row].replace(
                            "<|video_pad|>", placeholder, 1
                        )
                    video_index += 1
                rendered[row] = rendered[row].replace(
                    "<|video_placeholder|>", "<|video_pad|>"
                )
        audio_inputs: dict[str, Any] = {}
        if audios:
            extracted = feature_extractor(
                audios,
                sampling_rate=sampling_rate,
                padding="longest",
                truncation=False,
                return_attention_mask=True,
                return_tensors="np",
            )
            audio_inputs = {
                "input_features": extracted["input_features"],
                "feature_attention_mask": extracted["attention_mask"],
            }
            lengths_array = np.asarray(
                convolution_output_length(
                    np.asarray(extracted["attention_mask"]).sum(axis=-1)
                )
            )
            lengths = iter(lengths_array.tolist())
            for row in range(len(rendered)):
                while "<|audio_pad|>" in rendered[row]:
                    rendered[row] = rendered[row].replace(
                        "<|audio_pad|>",
                        "<|audio_placeholder|>" * int(next(lengths)),
                        1,
                    )
                rendered[row] = rendered[row].replace(
                    "<|audio_placeholder|>", "<|audio_pad|>"
                )
        tokenized = tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=maximum,
            return_attention_mask=True,
            return_tensors="np",
        )
        return batch_from_processor_output(
            {**tokenized, **image_inputs, **video_inputs, **audio_inputs},
            config,
            sequence_length_buckets=sequence_length_buckets,
            patch_count_buckets=patch_count_buckets,
            audio_chunk_buckets=audio_chunk_buckets,
        )

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-bidirlm-omni-processor-v1",
            "checkpoint": str(checkpoint.resolve()),
            "sequence_length_buckets": list(sequence_length_buckets),
            "patch_count_buckets": list(patch_count_buckets),
            "audio_chunk_buckets": list(audio_chunk_buckets),
            "sampling_rate": sampling_rate,
        },
    )


__all__ = [
    "batch_from_processor_output",
    "convolution_output_length",
    "make_bidirlm_omni_processor",
]
