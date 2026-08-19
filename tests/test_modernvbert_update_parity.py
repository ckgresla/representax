"""Full-tensor ModernVBERT optimizer-update comparison contracts."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.modernvbert_update_parity import _aggregate_metrics, _tensor_metrics


def test_identical_optimizer_deltas_have_zero_relative_difference():
    original = np.asarray([1.0, 2.0], dtype=np.float32)
    updated = np.asarray([0.9, 2.1], dtype=np.float32)

    metrics = _tensor_metrics(original, updated, updated)

    assert metrics["delta_relative_difference"] == 0.0
    assert metrics["delta_cosine"] == pytest.approx(1.0)
    assert metrics["maximum_updated_absolute_difference"] == 0.0


def test_optimizer_delta_comparison_is_relative_to_the_upstream_update():
    original = np.asarray([1.0, 2.0], dtype=np.float32)
    native = np.asarray([0.8, 2.2], dtype=np.float32)
    upstream = np.asarray([0.9, 2.1], dtype=np.float32)

    metrics = _tensor_metrics(original, native, upstream)

    assert metrics["delta_relative_difference"] == pytest.approx(1.0)
    assert metrics["delta_cosine"] == pytest.approx(1.0)
    assert metrics["maximum_updated_absolute_difference"] == pytest.approx(
        0.1, abs=1e-6
    )
    assert metrics["maximum_delta_absolute_difference"] == pytest.approx(0.1, abs=1e-6)


def test_aggregate_metrics_compare_the_concatenated_tensor_vector():
    first = _tensor_metrics(
        np.zeros(2, dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
    )
    second = _tensor_metrics(
        np.zeros(2, dtype=np.float32),
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    )

    aggregate = _aggregate_metrics({"first": first, "second": second})

    assert aggregate["native_l2"] == pytest.approx(np.sqrt(5.0))
    assert aggregate["sentence_transformers_l2"] == pytest.approx(np.sqrt(2.0))
    assert aggregate["difference_l2"] == pytest.approx(1.0)
    assert aggregate["relative_difference"] == pytest.approx(1.0 / np.sqrt(2.0))
