"""Retrieval collation preserves payloads; processors own modality handling."""

import numpy as np
import pytest

from representax.core import Route
from representax.models.processing import Processor
from representax.tasks.retrieval import RetrievalCollator


@pytest.mark.parametrize("modality", ["text", "image", "audio", "video", "custom"])
@pytest.mark.parametrize("media_side", ["query", "positive"])
def test_payloads_routes_and_pair_alignment(modality, media_side):
    payloads = [
        "first text" if modality == "text" else {modality: np.zeros((2, 3))},
        "second text" if modality == "text" else {modality: np.ones((2, 3))},
    ]
    rows = [{"query": f"query {i}", "positive": f"document {i}"} for i in range(2)]
    for row, payload in zip(rows, payloads, strict=True):
        row[media_side] = payload
    calls = []

    def process(values, *, route, seed):
        calls.append((values, route))
        return np.full((len(values), 4), len(calls), dtype=np.float32)

    collator = RetrievalCollator(processor=Processor(process, {"test": True}))
    batch = collator(rows)
    for (values, route), field, expected_route in zip(
        calls, ("query", "positive"), (Route.QUERY, Route.DOCUMENT), strict=True
    ):
        assert route == expected_route
        assert all(value is row[field] for value, row in zip(values, rows, strict=True))
    np.testing.assert_array_equal(batch.positive_mask, np.eye(2, dtype=bool))
    np.testing.assert_array_equal(batch.query, np.ones((2, 4)))
    np.testing.assert_array_equal(batch.document, np.full((2, 4), 2))


def test_custom_fields_and_missing_field():
    calls = []

    def process(values, *, route, seed):
        calls.append(values)
        return np.zeros((len(values), 4), dtype=np.float32)

    collator = RetrievalCollator(
        processor=Processor(process, {}), query_field="caption", document_field="media"
    )
    media = {"image": np.zeros((2, 3))}
    collator([{"caption": "caption", "media": media}])
    assert calls[0] == ("caption",)
    assert calls[1][0] is media
    calls.clear()
    with pytest.raises(KeyError, match="missing field 'media'"):
        collator([{"caption": "caption"}])
    assert calls == []


def test_processor_rejects_unsupported_payload_without_coercion():
    def process(values, *, route, seed):
        if any(not isinstance(value, str) for value in values):
            raise TypeError("this processor requires text")
        return np.zeros((len(values), 4), dtype=np.float32)

    collator = RetrievalCollator(processor=Processor(process, {}))
    with pytest.raises(TypeError, match="requires text"):
        collator([{"query": 123, "positive": "text"}])
