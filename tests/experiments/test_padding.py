"""Paper packing geometry and processor contracts."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.paper.padding import (
    PackShape,
    admitted_pack_shapes,
    make_packed_mpnet_processor,
    pack_mpnet_token_sequences,
    select_pack_shape,
)

from representax.core import Route


class _Tokenizer:
    pad_token_id = 1

    def __call__(self, texts, **_options):
        return {
            "input_ids": [
                [0, *[ord(character) % 17 + 3 for character in text], 2]
                for text in texts
            ]
        }


def test_select_pack_shape_minimizes_physical_token_capacity():
    shape, assignments = select_pack_shape(
        (10, 9, 8, 7),
        (
            PackShape(rows=4, sequence_length=16),
            PackShape(rows=2, sequence_length=32),
            PackShape(rows=1, sequence_length=64),
        ),
    )
    assert shape == PackShape(rows=4, sequence_length=16)
    assert sorted(index for values in assignments for index in values) == [0, 1, 2, 3]


def test_packed_tokens_preserve_sequences_and_reset_positions():
    sequences = ([0, 4, 5, 2], [0, 6, 2], [0, 7, 8, 9, 2])
    batch = pack_mpnet_token_sequences(
        sequences,
        admitted_shapes=(PackShape(rows=1, sequence_length=16),),
    )
    assert batch.batch_size == 3
    assert batch.input_ids is not None
    assert batch.segment_ids is not None
    assert batch.position_ids is not None
    for index, expected in enumerate(sequences):
        mask = np.asarray(batch.segment_ids) == index
        np.testing.assert_array_equal(np.asarray(batch.input_ids)[mask], expected)
        np.testing.assert_array_equal(
            np.asarray(batch.position_ids)[mask],
            np.arange(2, 2 + len(expected)),
        )


def test_packed_processor_has_a_finite_shape_contract():
    processor = make_packed_mpnet_processor(
        _Tokenizer(),
        maximum_batch_size=4,
        sequence_lengths=(8, 16),
    )
    batch = processor(("ab", "c"))
    assert batch.batch_size == 2
    contract = processor.data_contract()
    assert contract["attention_isolation"] == "block-diagonal"
    assert contract["position_ids_reset_per_segment"] is True
    assert contract["admitted_shapes"] == [
        [1, 8],
        [2, 8],
        [4, 8],
        [1, 16],
        [2, 16],
        [4, 16],
    ]


def test_fixed_route_shapes_keep_the_compiled_training_abi_constant():
    processor = make_packed_mpnet_processor(
        _Tokenizer(),
        maximum_batch_size=4,
        sequence_lengths=(8, 16),
        fixed_batch_size=2,
        fixed_query_shape=(2, 8),
        fixed_document_shape=(1, 16),
    )
    queries = processor(("ab", "c"), route=Route.QUERY)
    documents = processor(("ab", "c"), route=Route.DOCUMENT)
    assert queries.attention_mask.shape == (2, 8)
    assert documents.attention_mask.shape == (1, 16)


def test_packing_rejects_sequences_outside_the_finite_shape_set():
    with pytest.raises(ValueError, match="exceeds the admitted"):
        pack_mpnet_token_sequences(
            ([0] * 17,),
            admitted_shapes=admitted_pack_shapes(1, sequence_lengths=(16,)),
        )
