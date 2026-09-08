"""Source-local weighted batches, bounded execution, and resumable cursors."""

import hashlib
import json
from contextlib import ExitStack, closing
from urllib.parse import urlparse

import grain
import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest

from representax import data
from representax._config import ParameterRole, project_parameters
from representax.config import DataConfig


def resolve_records(config):
    uri = urlparse(config.uri)
    return [(int(uri.netloc), row) for row in range(int(uri.path[1:]))]


def collate_records(records):
    return np.asarray(records, dtype=np.int32)


def distribution(*, sizes=(11, 23), seed=31, shuffle=True, sampling_unit="batch"):
    return data.mix(
        *(
            data.source(f"memory://{index}/{size}", map=data.identity)
            for index, size in enumerate(sizes)
        ),
        weights=tuple(index + 1 for index in range(len(sizes))),
        seed=seed,
        shuffle=shuffle,
        sampling_unit=sampling_unit,
    )


def loader(config=None, **kwargs):
    options = {
        "batch_size": 3,
        "batch_fn": collate_records,
        "num_threads": 0,
        "prefetch_buffer_size": 0,
        "repeat": True,
        **kwargs,
    }
    return data.build_data_loader(
        distribution() if config is None else config,
        resolvers={"memory": resolve_records},
        **options,
    )


def take(iterator, count):
    return [next(iterator).tolist() for _ in range(count)]


def test_default_serialization_and_fingerprints_remain_compatible():
    config = distribution(sampling_unit="example")
    legacy = {
        "sources": [source.model_dump(mode="json") for source in config.sources],
        "weights": [1.0, 2.0],
        "seed": 31,
        "shuffle": True,
    }
    implicit = data.DataDistributionConfig.model_validate(legacy)
    assert implicit.sampling_unit == "example"
    assert config.model_dump(mode="json") == legacy
    assert json.loads(config.model_dump_json()) == legacy
    digest = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert implicit.fingerprint() == config.fingerprint() == f"sha256:{digest}"
    assert loader(implicit).data_fingerprint == loader(config).data_fingerprint
    assert loader(config).data_contract["source"]["distribution"] == legacy
    assert (
        DataConfig(distribution=implicit).model_dump(mode="json")["distribution"]
        == legacy
    )
    assert (
        project_parameters(DataConfig(distribution=implicit), ParameterRole.SCIENTIFIC)[
            "distribution"
        ]
        == legacy
    )

    batches = config.model_copy(update={"sampling_unit": "batch"})
    assert batches.model_dump(mode="json") == {**legacy, "sampling_unit": "batch"}
    assert (
        data.DataDistributionConfig.model_validate_json(batches.model_dump_json())
        == batches
    )
    assert batches.fingerprint() != config.fingerprint()
    assert loader(batches).data_fingerprint != loader(config).data_fingerprint
    assert loader(batches).data_contract["schema_version"] == (
        "representax-data-loader-v3"
    )


def test_sampling_unit_validation_and_unbatched_dataset_guard():
    with pytest.raises(ValueError, match="sampling_unit"):
        distribution(sampling_unit="epoch")
    with pytest.raises(ValueError, match="build_data_loader"):
        data.build_dataset(distribution(), resolvers={"memory": resolve_records})


def test_default_example_mixing_is_unchanged():
    config = distribution(sampling_unit="example", shuffle=False)
    reference = data.build_dataset(config, resolvers={"memory": resolve_records}).batch(
        3, drop_remainder=True, batch_fn=collate_records
    )
    actual = [batch.tolist() for batch in loader(config, repeat=False)]
    assert actual == [reference[index].tolist() for index in range(len(reference))]
    assert any(len({row[0] for row in batch}) > 1 for batch in actual)


@pytest.mark.parametrize("shuffle", (False, True))
def test_one_seeded_weighted_draw_per_batch_and_independent_source_order(shuffle):
    config = distribution(shuffle=shuffle)
    expected_sources = np.random.Generator(np.random.PCG64(config.seed)).choice(
        2, size=100, p=config.normalized_weights
    )
    references = [
        grain.MapDataset.source(resolve_records(source))
        .seed(config.seed + index)
        .map(data.identity)
        for index, source in enumerate(config.sources)
    ]
    if shuffle:
        references = [source.shuffle() for source in references]
    references = [
        source.batch(3, drop_remainder=True, batch_fn=collate_records).repeat()
        for source in references
    ]
    counts = [0, 0]
    with closing(iter(loader(config))) as iterator:
        for selected in expected_sources:
            batch = next(iterator)
            assert batch.shape == (3, 2)
            assert np.all(batch[:, 0] == selected)
            np.testing.assert_array_equal(batch, references[selected][counts[selected]])
            counts[selected] += 1
    assert all(count > 10 for count in counts)


def test_weighted_frequencies_and_seed_change():
    with (
        closing(iter(loader())) as first,
        closing(iter(loader(distribution(seed=32)))) as second,
    ):
        selected = [int(next(first)[0, 0]) for _ in range(3000)]
        other = [int(next(second)[0, 0]) for _ in range(40)]
    assert np.mean(selected) == pytest.approx(2 / 3, abs=0.035)
    assert selected[:40] != other


def test_four_source_seed_seven_canary_resumes_after_four_steps():
    config = data.mix(
        *distribution(sizes=(8, 110, 8, 8)).sources,
        sampling_unit="batch",
        seed=7,
    )
    batches = loader(config, batch_size=2, num_threads=2, prefetch_buffer_size=2)
    with closing(iter(batches)) as first, closing(iter(batches)) as resumed:
        prefix = take(first, 4)
        resumed.set_state(json.loads(json.dumps(first.get_state())))
        suffix = take(resumed, 4)
        assert [batch[0][0] for batch in prefix + suffix] == [2, 3, 3, 0, 1, 3, 0, 3]
        assert take(first, 4) == suffix


@pytest.mark.parametrize("before,after", ((0, 2), (2, 0), (2, 2)))
def test_orbax_checkpoint_round_trip_preserves_batch_cursor(tmp_path, before, after):
    config = data.mix(
        *distribution(sizes=(8, 110, 8, 8)).sources,
        sampling_unit="batch",
        seed=7,
    )
    original = loader(
        config, batch_size=2, num_threads=before, prefetch_buffer_size=before
    )
    restored = loader(
        config, batch_size=2, num_threads=after, prefetch_buffer_size=after
    )
    with (
        closing(iter(original)) as first,
        closing(iter(restored)) as resumed,
        ocp.PyTreeCheckpointer() as checkpointer,
    ):
        prefix = take(first, 4)
        state = first.get_state()
        assert isinstance(state["rng"], str)
        rng_state = json.loads(state["rng"])
        assert np.asarray(rng_state["state"]["state"]).dtype == np.dtype("object")
        assert np.asarray(rng_state["state"]["inc"]).dtype == np.dtype("object")
        assert all(
            np.asarray(leaf).dtype != np.dtype("object")
            for leaf in jax.tree_util.tree_leaves(state)
        )
        path = tmp_path / "cursor"
        checkpointer.save(path, {"data_state": state})
        expected = take(first, 40)
        recovered = checkpointer.restore(path)["data_state"]
        assert recovered == state
        resumed.set_state(recovered)
        assert take(resumed, 40) == expected
        assert [batch[0][0] for batch in prefix + expected[:4]] == (
            [2, 3, 3, 0, 1, 3, 0, 3]
        )


@pytest.mark.parametrize("before,after", ((0, 4), (4, 0), (2, 4)))
@pytest.mark.parametrize("consumed", (0, 1, 31))
def test_json_checkpoint_resume_across_prefetch_and_source_repetitions(
    before, after, consumed
):
    original = loader(num_threads=before, prefetch_buffer_size=before)
    restored = loader(num_threads=after, prefetch_buffer_size=after)
    assert original.data_fingerprint == restored.data_fingerprint
    with ExitStack() as stack:
        reference = stack.enter_context(closing(iter(loader())))
        first = stack.enter_context(closing(iter(original)))
        resumed = stack.enter_context(closing(iter(restored)))
        expected = take(reference, consumed + 50)
        prefix = take(first, consumed)
        state = json.loads(json.dumps(first.get_state()))
        # Advancing the producer must not mutate an earlier checkpoint.
        take(first, 12)
        resumed.set_state(state)
        assert prefix + take(resumed, 50) == expected
        resumed.set_state(state)
        assert take(resumed, 50) == expected[consumed:]


@pytest.mark.parametrize("drop_remainder", (False, True))
def test_finite_stream_stops_on_selected_exhaustion_and_restores(drop_remainder):
    config = distribution(sizes=(5, 8), shuffle=False)
    batches = loader(config, repeat=False, drop_remainder=drop_remainder)
    counts = [0, 0]
    rng = np.random.Generator(np.random.PCG64(config.seed))
    expected = []
    while True:
        selected = int(rng.choice(2, p=config.normalized_weights))
        records = resolve_records(config.sources[selected])
        batch = records[counts[selected] * 3 : (counts[selected] + 1) * 3]
        if not batch or (drop_remainder and len(batch) < 3):
            break
        expected.append(np.asarray(batch).tolist())
        counts[selected] += 1
    with closing(iter(batches)) as iterator, closing(iter(batches)) as restored:
        assert [batch.tolist() for batch in iterator] == expected
        restored.set_state(json.loads(json.dumps(iterator.get_state())))
        with pytest.raises(StopIteration):
            next(restored)
        with pytest.raises(StopIteration):
            next(iterator)
    assert batches.global_batch_size == (3 if drop_remainder else None)


@pytest.mark.parametrize("size", (0, 2))
@pytest.mark.parametrize("repeat", (False, True))
def test_sources_without_complete_batches_are_rejected(size, repeat):
    with pytest.raises(ValueError, match="source 0 cannot produce a batch"):
        loader(distribution(sizes=(size, 11)), repeat=repeat)


def test_single_small_source_can_repeat_partial_batches_when_requested():
    with closing(
        iter(loader(distribution(sizes=(2,), shuffle=False), drop_remainder=False))
    ) as iterator:
        assert take(iterator, 4) == [[[0, 0], [0, 1]]] * 4


def test_repeat_is_lazy_for_large_sources_and_checkpoint_size_is_bounded():
    reads = []

    class LargeSource:
        def __len__(self):
            return 10**12

        def __getitem__(self, index):
            reads.append(index)
            return (0, index)

    def resolve_large(config):
        return LargeSource()

    batches = data.build_data_loader(
        distribution(sizes=(11,), shuffle=False),
        batch_size=3,
        batch_fn=collate_records,
        resolvers={"memory": resolve_large},
        num_threads=0,
        prefetch_buffer_size=0,
        repeat=True,
    )
    assert not reads
    with closing(iter(batches)) as iterator:
        initial_size = len(json.dumps(iterator.get_state()))
        assert not reads
        take(iterator, 200)
        assert reads == list(range(600))
        assert len(json.dumps(iterator.get_state())) < initial_size + 100


def test_prefetch_uses_one_shared_host_memory_budget_and_reports_telemetry():
    with closing(
        iter(loader(num_threads=4, prefetch_buffer_size=2, host_memory_budget_bytes=96))
    ) as iterator:
        next(iterator)
        telemetry = iterator.last_telemetry
        assert telemetry.host_batch_bytes == 24
        assert telemetry.preprocess_seconds is not None
        assert telemetry.prefetch_capacity == 2

    with (
        closing(
            iter(
                loader(
                    num_threads=4,
                    prefetch_buffer_size=2,
                    host_memory_budget_bytes=95,
                )
            )
        ) as iterator,
        pytest.raises(MemoryError, match="per-slot limit"),
    ):
        next(iterator)


def test_serial_batch_sampling_only_reserves_one_memory_slot():
    with closing(
        iter(loader(prefetch_buffer_size=8, host_memory_budget_bytes=24))
    ) as iterator:
        next(iterator)
        assert iterator.last_telemetry.host_batch_bytes == 24
        iterator.close()
        with pytest.raises(ValueError, match="closed"):
            next(iterator)


def test_batch_sampling_supports_native_grain_collation():
    with closing(iter(loader(batch_fn=None))) as iterator:
        source_ids, row_ids = next(iterator)
        assert source_ids.shape == row_ids.shape == (3,)
        assert len(set(source_ids)) == 1
