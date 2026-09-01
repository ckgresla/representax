"""Contracts for bounded NQ and MIRACL real-data acceptance."""

from __future__ import annotations

import json

import pytest
from experiments.paper.nq_miracl_acceptance import (
    EVALUATION_BATCH_SIZE,
    LANGUAGES,
    MAXIMUM_LENGTH,
    MIRACL_CORPUS_PREFIX_BYTES,
    NQ_CORPUS_PREFIX_BYTES,
    STEPS,
    TRAINING_BATCH_SIZE,
    acceptance_job,
    frozen_contract,
    select_nq_evaluation,
)


def _data(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "train.jsonl").write_text(
        '{"query":"q","positive":"d"}\n' * (TRAINING_BATCH_SIZE * STEPS)
    )
    (directory / "evaluation.json").write_text(
        json.dumps(
            {
                "queries": [
                    {"id": 0, "text": "q", "source_id": "q0", "language": "en"}
                ],
                "documents": [
                    {
                        "id": 10_000,
                        "text": "d",
                        "source_id": "d0",
                        "language": "en",
                    }
                ],
                "relevant_documents": {"0": [10_000]},
            }
        )
    )
    return directory


def test_contracts_match_the_frozen_nq_and_miracl_paper_rows() -> None:
    nq = frozen_contract("nq")
    assert nq.workload == "dense-natural-questions"
    assert nq.model_id == "sentence-transformers/all-mpnet-base-v2"
    assert nq.model_revision == "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    assert nq.training_datasets == (
        (
            "sentence-transformers/natural-questions",
            "f9e894e1081e206e577b4eaa9ee6de2b06ae6f17",
        ),
    )
    assert nq.evaluation_datasets == (
        ("mteb/nq", "b84726e65fd226125cf7c0cbeeb5c214d49e8187"),
    )

    miracl = frozen_contract("miracl")
    assert miracl.workload == "dense-multilingual"
    assert miracl.model_id == "jinaai/jina-embeddings-v5-omni-small-retrieval"
    assert miracl.model_revision == "12949877f0092093f366c6450340011320152a05"
    assert miracl.languages == LANGUAGES == ("ar", "en", "hi", "ja")


@pytest.mark.parametrize("row", ["nq", "miracl"])
def test_job_uses_the_canonical_lifecycle(row, tmp_path) -> None:
    job = acceptance_job(
        row,
        checkpoint=tmp_path / frozen_contract(row).model_revision,
        data_directory=_data(tmp_path),
    )
    assert job.training.max_steps == STEPS == 2
    assert job.training.global_batch_size == TRAINING_BATCH_SIZE == 4
    assert job.training.batch.micro_batch_size == TRAINING_BATCH_SIZE
    assert job.model.parameters["sequence_length_buckets"] == [MAXIMUM_LENGTH]
    assert job.task.kind == "retrieval"
    assert job.loss.kind == "mnr"
    assert job.checkpointing is not None
    assert job.checkpointing.every == 1
    assert job.checkpointing.save_final
    assert job.evaluation is not None
    assert job.evaluation.batch_size == EVALUATION_BATCH_SIZE
    assert job.evaluation.on_start and job.evaluation.on_end
    assert job.evaluation.evaluators[0].kind == "information_retrieval"
    assert job.evaluation.primary_metric.endswith("/cosine_ndcg@10")
    assert job.export.enabled and job.export.selection == "final"


def test_real_data_fetches_are_explicitly_bounded() -> None:
    assert NQ_CORPUS_PREFIX_BYTES == 4 * 1024 * 1024
    assert MIRACL_CORPUS_PREFIX_BYTES == 16 * 1024 * 1024


def test_nq_subset_requires_real_query_qrel_and_corpus_alignment() -> None:
    selected = select_nq_evaluation(
        {"q0": "zero", "q1": "one"},
        {"d0": "document zero", "d1": "document one"},
        (("q0", "missing"), ("q0", "d0"), ("q1", "d1")),
        count=2,
    )
    assert selected.queries == ((0, "zero"), (1, "one"))
    assert selected.documents == ((10_000, "document zero"), (10_001, "document one"))
    assert selected.relevant_documents == {
        0: frozenset((10_000,)),
        1: frozenset((10_001,)),
    }

    with pytest.raises(ValueError, match="yielded 0"):
        select_nq_evaluation(
            {"q0": "zero"},
            {"d0": "document zero"},
            (("q0", "missing"),),
            count=1,
        )
