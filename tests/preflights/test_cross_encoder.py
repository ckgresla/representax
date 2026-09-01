"""Contracts for the paper cross-encoder reranking preflight."""

import json

from experiments.preflights.cross_encoder import (
    _reference_training_dataset,
    frozen_contract,
    pointwise_training_rows,
    representax_steady_state,
    select_judged_candidates,
)


def test_frozen_contract_names_minilm_msmarco_and_trec_dl() -> None:
    contract = frozen_contract()
    assert contract.model_id == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert contract.model_revision == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    assert contract.training_dataset["repo_id"] == (
        "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
    )
    assert contract.evaluation_dataset["dataset_id"] == (
        "msmarco-passage/trec-dl-2019/judged"
    )
    assert contract.batch_size == 128
    assert contract.maximum_length == 512
    assert contract.reference_version == "5.6.1"


def test_training_rows_use_distinct_queries_and_balanced_labels() -> None:
    triples = (
        {"query": "q1", "positive": "p1", "negative": "n1"},
        {"query": "q1", "positive": "p2", "negative": "n2"},
        {"query": "q2", "positive": "p3", "negative": "n3"},
    )
    rows = pointwise_training_rows(triples, query_count=2)
    assert [row["query"] for row in rows] == ["q1", "q1", "q2", "q2"]
    assert [row["label"] for row in rows] == [1.0, 0.0, 1.0, 0.0]


def test_judged_candidates_retain_both_relevance_classes() -> None:
    selected = select_judged_candidates(
        (("4", 0), ("2", 2), ("1", 0), ("3", 1), ("5", 0)),
        count=4,
    )
    assert len(selected) == 4
    assert any(label > 0 for _, label in selected)
    assert any(label == 0 for _, label in selected)
    assert list(selected) == sorted(selected)


def test_steady_state_excludes_compilation_rows() -> None:
    result = representax_steady_state(
        (
            {
                "metrics": {
                    "perf/compilation_and_first_step_seconds": 3.0,
                    "perf/step_seconds": 3.0,
                }
            },
            {"metrics": {"perf/step_seconds": 2.0}},
            {"metrics": {"perf/step_seconds": 4.0}},
        ),
        batch_size=12,
    )
    assert result == {
        "measured_steps": 2.0,
        "median_step_seconds": 3.0,
        "aggregate_examples_per_second": 4.0,
    }


def test_reference_training_dataset_keeps_query_before_document(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "document": "passage",
                "label": 1.0,
                "query": "question",
                "source_index": 0,
            }
        )
        + "\n"
    )
    dataset = _reference_training_dataset(path)
    assert dataset.column_names == ["query", "document", "label"]
