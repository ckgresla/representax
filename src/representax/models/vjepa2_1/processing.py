"""Finite-shape image/video preprocessing and multiblock masking for V-JEPA."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from pydantic import Field, PositiveInt, model_validator

from representax._config import FrozenConfig
from representax.core import Modality, Route
from representax.data import Artifact
from representax.models.processing import Processor
from representax.tasks.jepa.vjepa2_1 import VJEPA2_1Batch

from .config import VJEPA2_1Config


class VJEPA2_1Pixels(eqx.Module):
    """One model-ready image or video tensor batch."""

    pixels: jnp.ndarray


class VJEPAMaskConfig(FrozenConfig):
    """One official multi-block context/target mask distribution."""

    spatial_scale: tuple[float, float]
    temporal_scale: tuple[float, float] = (1.0, 1.0)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    num_blocks: PositiveInt = 1
    max_temporal_keep: float = Field(default=1.0, gt=0.0, le=1.0)
    max_context_tokens: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        for name in ("spatial_scale", "temporal_scale", "aspect_ratio"):
            low, high = getattr(self, name)
            if low <= 0 or high < low:
                raise ValueError(f"{name} must be a positive increasing range")
        if self.spatial_scale[1] > 1 or self.temporal_scale[1] > 1:
            raise ValueError("mask scales cannot exceed one")
        return self


def _mask_pattern(
    config: VJEPAMaskConfig,
    *,
    batch_size: int,
    grid: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    depth, height, width = grid
    temporal_scale = rng.uniform(*config.temporal_scale)
    spatial_scale = rng.uniform(*config.spatial_scale)
    aspect_ratio = rng.uniform(*config.aspect_ratio)
    block_depth = max(1, int(depth * temporal_scale))
    spatial_tokens = int(height * width * spatial_scale)
    block_height = min(height, int(round(math.sqrt(spatial_tokens * aspect_ratio))))
    block_width = min(width, int(round(math.sqrt(spatial_tokens / aspect_ratio))))
    block_height = max(block_height, 1)
    block_width = max(block_width, 1)
    maximum_context_depth = max(1, int(depth * config.max_temporal_keep))
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(batch_size):
        for _attempt in range(100):
            keep = np.ones(grid, dtype=bool)
            for _ in range(config.num_blocks):
                start = rng.integers(0, depth - block_depth + 1)
                top = rng.integers(0, height - block_height + 1)
                left = rng.integers(0, width - block_width + 1)
                keep[
                    start : start + block_depth,
                    top : top + block_height,
                    left : left + block_width,
                ] = False
            keep[maximum_context_depth:] = False
            context = np.flatnonzero(keep.reshape(-1))
            if len(context):
                break
        else:
            raise ValueError(
                f"mask distribution cannot produce non-empty context for grid {grid!r}"
            )
        target = np.flatnonzero(~keep.reshape(-1))
        contexts.append(context.astype(np.int32))
        targets.append(target.astype(np.int32))
    context_count = min(len(value) for value in contexts)
    target_count = min(len(value) for value in targets)
    if config.max_context_tokens is not None:
        context_count = min(context_count, config.max_context_tokens)
    return (
        [value[:context_count] for value in contexts],
        [value[:target_count] for value in targets],
    )


def sample_vjepa_masks(
    patterns: Sequence[VJEPAMaskConfig],
    *,
    batch_size: int,
    grid: tuple[int, int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample official-style multiblock masks into finite padded tensors."""

    if not patterns:
        raise ValueError("V-JEPA requires at least one mask pattern")
    if batch_size <= 0 or any(size <= 0 for size in grid):
        raise ValueError("batch size and tubelet grid must be positive")
    children = np.random.SeedSequence(seed).spawn(len(patterns))
    sampled = tuple(
        _mask_pattern(
            pattern,
            batch_size=batch_size,
            grid=grid,
            rng=np.random.default_rng(child),
        )
        for pattern, child in zip(patterns, children, strict=True)
    )
    context_size = max(len(contexts[0]) for contexts, _ in sampled)
    target_size = max(len(targets[0]) for _, targets in sampled)
    context_ids = np.zeros((batch_size, len(patterns), context_size), dtype=np.int32)
    target_ids = np.zeros((batch_size, len(patterns), target_size), dtype=np.int32)
    context_valid = np.zeros_like(context_ids, dtype=bool)
    target_valid = np.zeros_like(target_ids, dtype=bool)
    for mask_index, (contexts, targets) in enumerate(sampled):
        for row, (context, target) in enumerate(zip(contexts, targets, strict=True)):
            context_ids[row, mask_index, : len(context)] = context
            target_ids[row, mask_index, : len(target)] = target
            context_valid[row, mask_index, : len(context)] = True
            target_valid[row, mask_index, : len(target)] = True
    return context_ids, target_ids, context_valid, target_valid


def _raw_media(value: Any, modality: Modality) -> Any:
    if not isinstance(value, Artifact):
        return value
    if value.modality != modality:
        raise TypeError(f"expected {modality.value}, received {value.modality.value}")
    if value.data is not None:
        return value.data
    payload = value.read_bytes()
    if modality == Modality.IMAGE:
        image = import_module("PIL.Image")
        return image.open(io.BytesIO(payload)).convert("RGB")
    raise TypeError(
        "encoded video artifacts require a source mapper or resolver that returns "
        "decoded [frame,height,width,channel] arrays"
    )


def _resize(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match Torch bilinear interpolation with ``align_corners=False``."""

    source = np.asarray(frames, dtype=np.float32)
    source_height, source_width = source.shape[1:3]
    rows = np.maximum(
        (np.arange(height, dtype=np.float32) + 0.5) * (source_height / height) - 0.5,
        0.0,
    )
    columns = np.maximum(
        (np.arange(width, dtype=np.float32) + 0.5) * (source_width / width) - 0.5,
        0.0,
    )
    row_low = np.floor(rows).astype(np.int32)
    column_low = np.floor(columns).astype(np.int32)
    row_high = np.minimum(row_low + 1, source_height - 1)
    column_high = np.minimum(column_low + 1, source_width - 1)
    row_weight = (rows - row_low)[:, None, None]
    column_weight = (columns - column_low)[None, :, None]
    top = (
        source[:, row_low[:, None], column_low[None, :]] * (1.0 - column_weight)
        + source[:, row_low[:, None], column_high[None, :]] * column_weight
    )
    bottom = (
        source[:, row_high[:, None], column_low[None, :]] * (1.0 - column_weight)
        + source[:, row_high[:, None], column_high[None, :]] * column_weight
    )
    return top * (1.0 - row_weight) + bottom * row_weight


def _random_resized_crop(
    frames: np.ndarray,
    *,
    size: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = frames.shape[1:3]
    area = height * width
    for _ in range(10):
        target_area = area * rng.uniform(*scale)
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        crop_width = int(round(math.sqrt(target_area * aspect)))
        crop_height = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_height <= height and 0 < crop_width <= width:
            top = int(rng.integers(0, height - crop_height + 1))
            left = int(rng.integers(0, width - crop_width + 1))
            return _resize(
                frames[:, top : top + crop_height, left : left + crop_width],
                size,
                size,
            )
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return _resize(frames[:, top : top + side, left : left + side], size, size)


def make_vjepa2_1_processor(
    config: VJEPA2_1Config,
    *,
    modality: Modality | str,
    training: bool = True,
    resize_scale: tuple[float, float] = (0.3, 1.0),
    resize_aspect_ratio: tuple[float, float] = (0.75, 4 / 3),
    horizontal_flip: bool = True,
) -> Processor:
    """Construct the paper's shared crop/normalize image or video processor."""

    resolved = Modality(modality)
    if resolved not in (Modality.IMAGE, Modality.VIDEO):
        raise ValueError("V-JEPA 2.1 processing accepts image or video")
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> VJEPA2_1Pixels:
        del route
        if not artifacts:
            raise ValueError("V-JEPA processor batches must be non-empty")
        rng = np.random.default_rng(seed)
        rows = []
        for artifact in artifacts:
            value = _raw_media(artifact, resolved)
            if hasattr(value, "convert"):
                value = np.asarray(value.convert("RGB"))
            frames = np.asarray(value)
            if resolved == Modality.IMAGE:
                if frames.ndim != 3:
                    raise ValueError("image samples must have shape [H,W,C]")
                frames = frames[None]
            elif frames.ndim != 4:
                raise ValueError("video samples must have shape [T,H,W,C]")
            if frames.shape[-1] != config.channels:
                raise ValueError("media channel count differs from V-JEPA config")
            if resolved == Modality.VIDEO:
                if frames.shape[0] < config.video_frames:
                    raise ValueError("video contains fewer frames than configured")
                maximum = frames.shape[0] - config.video_frames
                start = int(rng.integers(0, maximum + 1)) if training else maximum // 2
                frames = frames[start : start + config.video_frames]
            if training:
                frames = _random_resized_crop(
                    frames,
                    size=config.image_size,
                    scale=resize_scale,
                    ratio=resize_aspect_ratio,
                    rng=rng,
                )
                if horizontal_flip and rng.random() < 0.5:
                    frames = frames[:, :, ::-1]
            else:
                frames = _resize(frames, config.image_size, config.image_size)
            normalized = (frames.astype(np.float32) / 255.0 - mean) / std
            if resolved == Modality.IMAGE:
                rows.append(np.moveaxis(normalized[0], -1, 0))
            else:
                rows.append(np.moveaxis(normalized, -1, 0))
        return VJEPA2_1Pixels(pixels=jnp.asarray(np.stack(rows)))

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-vjepa2-1-processor-v1",
            "reference_commit": "204698b45b3712590f06245fbfba32d3be539812",
            "modality": resolved.value,
            "training": training,
            "image_size": config.image_size,
            "video_frames": config.video_frames,
            "tubelet_size": config.tubelet_size,
            "resize_scale": list(resize_scale),
            "resize_aspect_ratio": list(resize_aspect_ratio),
            "horizontal_flip": horizontal_flip,
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    )


@dataclass(frozen=True, slots=True)
class VJEPA2_1Collator:
    """Apply model preprocessing and task-owned multiblock masking."""

    processor: Processor
    config: VJEPA2_1Config
    patterns: tuple[VJEPAMaskConfig, ...]
    artifact_field: str = "artifact"
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "config",
            VJEPA2_1Config.model_validate(self.config),
        )
        object.__setattr__(
            self,
            "patterns",
            tuple(VJEPAMaskConfig.model_validate(value) for value in self.patterns),
        )
        if not self.patterns:
            raise ValueError("V-JEPA collation requires at least one mask pattern")
        if self.seed < 0:
            raise ValueError("V-JEPA collation seed must be non-negative")

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-vjepa2-1-collator-v1",
            "processor": self.processor.data_contract(),
            "patterns": [pattern.model_dump(mode="json") for pattern in self.patterns],
            "artifact_field": self.artifact_field,
            "seed": self.seed,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> VJEPA2_1Batch:
        pixels = self.processor(
            tuple(example[self.artifact_field] for example in examples),
            seed=self.seed,
        ).pixels
        temporal = (
            1 if pixels.ndim == 4 else pixels.shape[2] // self.config.tubelet_size
        )
        masks = sample_vjepa_masks(
            self.patterns,
            batch_size=len(examples),
            grid=(temporal, self.config.spatial_grid, self.config.spatial_grid),
            seed=self.seed,
        )
        return VJEPA2_1Batch(
            pixels=pixels,
            context_ids=jnp.asarray(masks[0]),
            target_ids=jnp.asarray(masks[1]),
            context_valid=jnp.asarray(masks[2]),
            target_valid=jnp.asarray(masks[3]),
        )


__all__ = [
    "VJEPA2_1Collator",
    "VJEPA2_1Pixels",
    "VJEPAMaskConfig",
    "make_vjepa2_1_processor",
    "sample_vjepa_masks",
]
