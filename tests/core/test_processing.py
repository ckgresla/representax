"""Host-side processor primitive tests."""

import pytest

from representax.models import Processor, select_static_shape_bucket


def test_processor_is_a_serializable_host_boundary():
    processor = Processor(
        process=lambda values, *, route, seed: (tuple(values), route, seed),
        contract={"kind": "identity", "shape": [2]},
    )

    values, route, seed = processor(("a", "b"), seed=3)

    assert values == ("a", "b")
    assert route.value == "generic"
    assert seed == 3
    assert processor.data_contract() == {"kind": "identity", "shape": [2]}


def test_static_shape_bucket_selects_the_smallest_containing_shape():
    assert select_static_shape_bucket(
        (7, 200, 200),
        ((8, 224, 224), (16, 224, 224), (8, 336, 336)),
    ) == (8, 224, 224)


def test_static_shape_bucket_selection_is_order_independent():
    shapes = ((16, 256), (32, 512))

    assert select_static_shape_bucket((7, 240), shapes) == (16, 256)
    assert select_static_shape_bucket((7, 240), tuple(reversed(shapes))) == (
        16,
        256,
    )


def test_static_shape_bucket_rejects_incomparable_media_tradeoffs():
    with pytest.raises(ValueError, match="model processor must choose"):
        select_static_shape_bucket((7, 240), ((16, 256), (8, 512)))


@pytest.mark.parametrize(
    ("required", "buckets", "message"),
    [
        ((), ((8,),), "required_shape"),
        ((4,), (), "at least one"),
        ((4, 4), ((8,),), "match required_shape rank"),
        ((9,), ((4,), (8,)), "exceeds admitted buckets"),
        ((4.0,), ((8,),), "positive integer dimensions"),
    ],
)
def test_static_shape_bucket_rejects_invalid_or_unadmitted_shapes(
    required,
    buckets,
    message,
):
    with pytest.raises(ValueError, match=message):
        select_static_shape_bucket(required, buckets)
