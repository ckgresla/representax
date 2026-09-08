from __future__ import annotations

import pytest
from experiments.preflights.audio_text import (
    _validate_training_manifest as validate_audio,
)
from experiments.preflights.image_text import (
    _validate_training_manifest as validate_image,
)
from experiments.preflights.video_text import (
    _validate_training_manifest as validate_video,
)


@pytest.mark.parametrize(
    ("validate", "manifest"),
    (
        (validate_image, {"unique_image_ids": True, "unique_captions": True}),
        (validate_audio, {"unique_audio_ids": True, "unique_captions": True}),
        (validate_video, {"unique_video_ids": True, "unique_captions": True}),
    ),
)
def test_training_manifests_require_duplicate_free_data(validate, manifest) -> None:
    validate(manifest)

    with pytest.raises(ValueError, match="predates duplicate-free preparation"):
        validate({})
