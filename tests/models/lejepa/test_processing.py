"""Paper multicrop and evaluation preprocessing contracts."""

import numpy as np
from PIL import Image

from representax.evaluation import LabeledEvaluationBatch
from representax.models.lejepa import (
    LeJEPAEvaluationCollator,
    LeJEPATrainCollator,
    canonical_multicrop_views,
)
from representax.tasks.jepa import JEPABatch


def test_paper_multicrop_uses_two_globals_and_six_98px_locals() -> None:
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    )

    pixels, sizes = canonical_multicrop_views(image, seed=3)

    assert pixels.shape == (8, 3, 224, 224)
    assert tuple(sizes) == (224, 224, 98, 98, 98, 98, 98, 98)
    assert np.all(pixels[2:, :, 98:, :] == 0.0)
    assert np.all(pixels[2:, :, :, 98:] == 0.0)


def test_collators_preserve_static_multicrop_and_labeled_splits(tmp_path) -> None:
    image_path = tmp_path / "image.JPEG"
    Image.new("RGB", (48, 40), (40, 80, 120)).save(image_path)
    rows = (
        {"image": str(image_path), "view_seed": 1, "label": 0, "split": 0},
        {"image": str(image_path), "view_seed": 2, "label": 1, "split": 2},
    )

    training = LeJEPATrainCollator(seed=5)(rows)
    evaluation = LeJEPAEvaluationCollator()(rows)

    assert isinstance(training, JEPABatch)
    assert training.views.pixel_values.shape == (2, 8, 3, 224, 224)
    assert training.views.view_count == 8
    assert training.views.global_views == 2
    assert isinstance(evaluation, LabeledEvaluationBatch)
    assert evaluation.examples.shape == (2, 3, 224, 224)
    assert tuple(evaluation.splits) == (0, 2)
