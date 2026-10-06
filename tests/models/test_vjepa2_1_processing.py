from __future__ import annotations

import jax
import numpy as np

from representax.config import ComponentConfig, DataConfig, ModelConfig
from representax.core import Modality
from representax.data import mix, source
from representax.models.vjepa2_1 import (
    VJEPA2_1Collator,
    VJEPA2_1Config,
    VJEPA2_1Model,
    VJEPAMaskConfig,
    make_vjepa2_1_processor,
    sample_vjepa_masks,
)
from representax.train import build_collate, load_model


def tiny_config() -> VJEPA2_1Config:
    return VJEPA2_1Config(
        image_size=8,
        patch_size=4,
        video_frames=4,
        tubelet_size=2,
        hidden_size=12,
        depth=2,
        heads=2,
        predictor_hidden_size=12,
        predictor_depth=2,
        predictor_heads=2,
        supervision_layers=(0, 1),
    )


def test_image_and_video_processors_emit_finite_model_shapes() -> None:
    config = tiny_config()
    image = np.arange(12 * 10 * 3, dtype=np.uint8).reshape((12, 10, 3))
    video = np.stack(tuple(np.roll(image, frame, axis=0) for frame in range(7)))
    image_processor = make_vjepa2_1_processor(
        config,
        modality=Modality.IMAGE,
        training=False,
    )
    video_processor = make_vjepa2_1_processor(
        config,
        modality=Modality.VIDEO,
        training=False,
    )
    images = image_processor((image, image), seed=3)
    videos = video_processor((video, video), seed=3)
    assert images.pixels.shape == (2, 3, 8, 8)
    assert videos.pixels.shape == (2, 3, 4, 8, 8)
    assert np.isfinite(images.pixels).all()
    assert np.isfinite(videos.pixels).all()


def test_multiblock_masks_are_reproducible_and_preserve_validity() -> None:
    patterns = (
        VJEPAMaskConfig(spatial_scale=(0.25, 0.25), num_blocks=1),
        VJEPAMaskConfig(spatial_scale=(0.75, 0.75), num_blocks=1),
    )
    first = sample_vjepa_masks(patterns, batch_size=3, grid=(2, 4, 4), seed=17)
    second = sample_vjepa_masks(patterns, batch_size=3, grid=(2, 4, 4), seed=17)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    context, target, context_valid, target_valid = first
    assert context.shape[:2] == target.shape[:2] == (3, 2)
    for row in range(3):
        for pattern in range(2):
            selected_context = set(context[row, pattern][context_valid[row, pattern]])
            selected_target = set(target[row, pattern][target_valid[row, pattern]])
            assert selected_context
            assert selected_target
            assert selected_context.isdisjoint(selected_target)


def test_collator_combines_processor_and_task_masks() -> None:
    config = tiny_config()
    processor = make_vjepa2_1_processor(
        config,
        modality=Modality.IMAGE,
        training=False,
    )
    collator = VJEPA2_1Collator(
        processor=processor,
        config=config,
        patterns=(VJEPAMaskConfig(spatial_scale=(0.5, 0.5), num_blocks=1),),
        seed=11,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    batch = collator(({"artifact": image}, {"artifact": image}))
    assert batch.pixels.shape == (2, 3, 8, 8)
    assert batch.context_ids.shape[:2] == (2, 1)
    assert batch.target_ids.shape[:2] == (2, 1)
    assert batch.context_valid.all()
    assert batch.target_valid.all()


def test_job_config_builds_model_processor_and_collator_once() -> None:
    config = tiny_config()
    model, processor = load_model(
        ModelConfig(
            target="representax.models.vjepa2_1.load_vjepa2_1",
            parameters={
                "config": config.model_dump(mode="json"),
                "modality": "image",
                "training": False,
            },
        ),
        key=jax.random.key(17),
        activation_rematerialization="none",
    )
    assert isinstance(model, VJEPA2_1Model)
    collate = build_collate(
        DataConfig(
            distribution=mix(
                source("memory://unused", map="representax.data.identity")
            ),
            collate=ComponentConfig(
                target="representax.models.vjepa2_1.VJEPA2_1Collator",
                parameters={
                    "config": config.model_dump(mode="json"),
                    "patterns": [
                        VJEPAMaskConfig(
                            spatial_scale=(0.5, 0.5),
                        ).model_dump(mode="json")
                    ],
                    "seed": 19,
                },
            ),
        ),
        processor=processor,
    )
    assert model.metadata.modalities == frozenset({Modality.IMAGE, Modality.VIDEO})
    assert collate is not None
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    batch = collate(({"artifact": image}, {"artifact": image}))
    assert batch.pixels.shape == (2, 3, 8, 8)
