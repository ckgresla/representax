from __future__ import annotations

import numpy as np

from representax.models.vjepa2_1.reference import read_reference_checkpoint


def test_numpy_reference_checkpoint_requires_no_torch(tmp_path) -> None:
    path = tmp_path / "checkpoint.npz"
    np.savez(
        path,
        **{
            "encoder::layer.weight": np.ones((2, 3), dtype=np.float32),
            "predictor::layer.bias": np.zeros((3,), dtype=np.float32),
            "target_encoder::layer.weight": np.full((2, 3), 2.0, dtype=np.float32),
        },
    )

    checkpoint = read_reference_checkpoint(path)

    np.testing.assert_array_equal(
        checkpoint["encoder"]["layer.weight"], np.ones((2, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        checkpoint["predictor"]["layer.bias"], np.zeros((3,), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        checkpoint["target_encoder"]["layer.weight"],
        np.full((2, 3), 2.0, dtype=np.float32),
    )
