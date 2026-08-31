"""Original-schema adapters for the canonical evaluator panel."""

import jax.numpy as jnp

from representax.evaluation import (
    BANKING77_TEST,
    BANKING77_TRAIN,
    CIFAR100_TEST,
    CIFAR100_TRAIN,
    SPRINT_DUPLICATE_QUESTIONS,
    TWENTY_NEWSGROUPS,
    EvaluationSplit,
    clustering_evaluation_batches,
    clustering_samples,
    labeled_evaluation_batches,
    pair_classification_batches,
)
from representax.models.processing import Processor


def _processor(values, **options):
    del options
    return jnp.arange(len(values) * 3, dtype=jnp.float32).reshape(len(values), 3)


PROCESSOR = Processor(process=_processor, contract={"kind": "test"})


def test_canonical_sources_pin_original_revisions_and_splits() -> None:
    sources = (
        SPRINT_DUPLICATE_QUESTIONS,
        BANKING77_TRAIN,
        BANKING77_TEST,
        TWENTY_NEWSGROUPS,
        CIFAR100_TRAIN,
        CIFAR100_TEST,
    )
    assert all(value.source.uri.startswith("hf://") for value in sources)
    assert all(len(value.source.revision or "") == 40 for value in sources)
    assert [BANKING77_TRAIN.source.split, BANKING77_TEST.source.split] == [
        "train",
        "test",
    ]
    assert [CIFAR100_TRAIN.source.split, CIFAR100_TEST.source.split] == [
        "train",
        "test",
    ]


def test_sprint_packed_columns_stream_fixed_batches() -> None:
    records = (
        {
            "sent1": ["a", "b", "c"],
            "sent2": ["A", "B", "C"],
            "labels": [1, 0, 1],
        },
    )
    batches = tuple(
        pair_classification_batches(records, processor=PROCESSOR, batch_size=2)
    )
    assert len(batches) == 2
    assert batches[0].labels.tolist() == [1, 0]
    assert batches[1].labels.tolist() == [1, 1]
    assert batches[1].valid.tolist() == [True, False]


def test_labeled_adapter_makes_deterministic_stratified_splits() -> None:
    train = tuple({"text": f"train-{index}", "label": index % 2} for index in range(20))
    test = tuple({"text": f"test-{index}", "label": index % 2} for index in range(4))
    batches = tuple(
        labeled_evaluation_batches(
            train=train,
            test=test,
            processor=PROCESSOR,
            batch_size=4,
            validation_fraction=0.2,
            seed=7,
        )
    )
    split_rows = {
        split: sum(
            int(jnp.sum(batch.valid))
            for batch in batches
            if int(batch.splits[0]) == int(split)
        )
        for split in EvaluationSplit
    }
    assert split_rows == {
        EvaluationSplit.TRAIN: 16,
        EvaluationSplit.VALIDATION: 4,
        EvaluationSplit.TEST: 4,
    }


def test_twenty_newsgroups_samples_remain_independent() -> None:
    records = (
        {"sentences": ["a", "b"], "labels": [0, 1]},
        {"sentences": ["c", "d"], "labels": [1, 1]},
    )
    samples = tuple(clustering_samples(records))
    assert len(samples) == 2
    assert len(samples[0]) == 2
    assert samples[0][1] == {"sentences": "b", "labels": 1}
    assert samples[1][0] == {"sentences": "c", "labels": 1}
    batches = tuple(
        clustering_evaluation_batches(samples[0], processor=PROCESSOR, batch_size=3)
    )
    assert len(batches) == 1
    assert batches[0].labels.tolist() == [0, 1, 1]
    assert batches[0].valid.tolist() == [True, True, False]
