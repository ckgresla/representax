"""Canonical ImageNet preprocessing and collation for LeJEPA."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from representax.evaluation import labeled_evaluation_batch
from representax.tasks.jepa import JEPABatch

from .model import LeJEPAMulticropImages

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
GLOBAL_VIEWS = 2
LOCAL_VIEWS = 6


def _random_resized_crop(
    image: Image.Image,
    *,
    size: int,
    scale: tuple[float, float],
    rng: np.random.Generator,
) -> Image.Image:
    width, height = image.size
    area = width * height
    ratio = (3.0 / 4.0, 4.0 / 3.0)
    for _ in range(10):
        target = area * rng.uniform(*scale)
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        crop_width = int(round(math.sqrt(target * aspect)))
        crop_height = int(round(math.sqrt(target / aspect)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = int(rng.integers(0, width - crop_width + 1))
            top = int(rng.integers(0, height - crop_height + 1))
            return image.crop((left, top, left + crop_width, top + crop_height)).resize(
                (size, size), Image.Resampling.BICUBIC
            )
    input_ratio = width / height
    if input_ratio < ratio[0]:
        crop_width, crop_height = width, int(round(width / ratio[0]))
    elif input_ratio > ratio[1]:
        crop_width, crop_height = int(round(height * ratio[1])), height
    else:
        crop_width, crop_height = width, height
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(
        (size, size),
        Image.Resampling.BICUBIC,
    )


def _color_jitter(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    operations = ["brightness", "contrast", "saturation", "hue"]
    rng.shuffle(operations)
    for operation in operations:
        if operation == "brightness":
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.6, 1.4))
        elif operation == "contrast":
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.6, 1.4))
        elif operation == "saturation":
            image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
        else:
            hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
            shift = int(round(rng.uniform(-0.1, 0.1) * 255.0))
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")
    return image


def _photometric(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.8:
        image = _color_jitter(image, rng)
    if rng.random() < 0.2:
        image = ImageOps.grayscale(image).convert("RGB")
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 2.0)))
    if rng.random() < 0.2:
        image = ImageOps.solarize(image, threshold=128)
    return image


def _normalize(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.moveaxis(pixels, -1, 0), dtype=np.float32)


def canonical_multicrop_views(
    image: Image.Image,
    *,
    image_size: int = 224,
    local_image_size: int = 98,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two global and six local views in one padded static tensor."""

    rng = np.random.default_rng(seed)
    values = []
    sizes = []
    recipes = (
        (GLOBAL_VIEWS, image_size, (0.3, 1.0)),
        (LOCAL_VIEWS, local_image_size, (0.05, 0.3)),
    )
    for count, size, scale in recipes:
        for _ in range(count):
            view = _random_resized_crop(image, size=size, scale=scale, rng=rng)
            view = _photometric(view, rng)
            normalized = _normalize(view)
            padded = np.zeros((3, image_size, image_size), dtype=np.float32)
            padded[:, :size, :size] = normalized
            values.append(padded)
            sizes.append(size)
    return np.stack(values), np.asarray(sizes, dtype=np.int32)


def evaluation_image(image: Image.Image, *, image_size: int = 224) -> np.ndarray:
    """Apply the standard resize-256/center-crop-224 ImageNet transform."""

    width, height = image.size
    resized_short = int(round(image_size / 0.875))
    scale = resized_short / min(width, height)
    resized = image.resize(
        (int(round(width * scale)), int(round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    return _normalize(resized.crop((left, top, left + image_size, top + image_size)))


class LeJEPATrainCollator:
    def __init__(
        self,
        *,
        image_size: int = 224,
        local_image_size: int = 98,
        seed: int = 0,
    ) -> None:
        self.image_size = image_size
        self.local_image_size = local_image_size
        self.seed = seed

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> JEPABatch:
        pixels = []
        crop_sizes = []
        for example in examples:
            with Image.open(Path(str(example["image"]))) as source:
                views, sizes = canonical_multicrop_views(
                    source.convert("RGB"),
                    image_size=self.image_size,
                    local_image_size=self.local_image_size,
                    seed=self.seed + int(example["view_seed"]),
                )
            pixels.append(views)
            crop_sizes.append(sizes)
        values = np.stack(pixels)
        sizes = np.stack(crop_sizes)
        return JEPABatch(
            views=LeJEPAMulticropImages(
                pixel_values=jnp.asarray(values),
                crop_sizes=jnp.asarray(sizes),
            ),
            valid=jnp.ones(values.shape[:2], dtype=jnp.bool_),
        )


class LeJEPAEvaluationCollator:
    def __init__(self, *, image_size: int = 224) -> None:
        self.image_size = image_size

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Any:
        pixels = []
        labels = []
        splits = []
        for example in examples:
            with Image.open(Path(str(example["image"]))) as source:
                pixels.append(
                    evaluation_image(source.convert("RGB"), image_size=self.image_size)
                )
            labels.append(int(example["label"]))
            splits.append(int(example["split"]))
        return labeled_evaluation_batch(
            examples=jnp.asarray(np.stack(pixels)),
            labels=jnp.asarray(labels, dtype=jnp.int32),
            split=jnp.asarray(splits, dtype=jnp.int32),
        )


__all__ = [
    "GLOBAL_VIEWS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LOCAL_VIEWS",
    "LeJEPAEvaluationCollator",
    "LeJEPATrainCollator",
    "canonical_multicrop_views",
    "evaluation_image",
]
