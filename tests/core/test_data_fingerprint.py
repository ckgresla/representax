"""Data-contract fingerprint regression tests."""

from __future__ import annotations

from functools import partial

import grain

from representax.data import build_data_loader
from representax.models import Processor


def processor_aware_batch(values, *, processor):
    return processor(values)


def test_processor_contract_fingerprints_injected_partial_batch_mapper():
    def processor(shape):
        return Processor(
            process=lambda values, *, route, seed: tuple(values),
            contract={"kind": "identity", "shape": shape},
        )

    def loader(shape):
        return build_data_loader(
            grain.MapDataset.source((1, 2)),
            batch_size=2,
            batch_fn=partial(processor_aware_batch, processor=processor(shape)),
            num_threads=0,
            prefetch_buffer_size=0,
            data_contract={"name": "processor-partial"},
        )

    first = loader([2])
    same = loader([2])
    different = loader([4])

    assert first.data_fingerprint == same.data_fingerprint
    assert first.data_fingerprint != different.data_fingerprint
