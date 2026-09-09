"""Full-source recipe and independent held-out corpus contracts."""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

DATA = importlib.import_module("experiments.14-text-to-any-modality.data")
CONFIG = importlib.import_module("experiments.14-text-to-any-modality.config")
EVAL = importlib.import_module("experiments.14-text-to-any-modality.evaluation")


def manifest():
    return {
        "seeds": {
            str(seed): {
                name: f"/data/{name}-{seed}.jsonl"
                for name in ("image", "audio", "video", "text")
            }
            for seed in CONFIG.SEEDS
        },
        "evaluation": {
            name: {"path": f"/data/{name}.jsonl", "relevant_documents": {0: [0]}}
            for name in ("audiocaps", "msrvtt", "flickr30k", "nanomsmarco")
        },
    }


@pytest.mark.parametrize("strategy", CONFIG.LEARNING_RATES)
@pytest.mark.parametrize("seed", CONFIG.SEEDS)
def test_frozen_recipe_passes_all_science_to_job(strategy, seed):
    from representax.config import JobConfig
    from representax.train.job import build_component

    job = CONFIG.training_job(
        manifest(), strategy=strategy, seed=seed, assets=Path("/assets")
    )
    assert JobConfig.model_validate_json(job.model_dump_json()) == job
    assert job.training.max_steps == 2000 and job.training.global_batch_size == 32
    assert job.training.seed == job.data.distribution.seed == seed
    assert job.training.grad_cache.micro_batch_size == 2
    assert job.training.grad_cache.implementation == "custom_vjp"
    assert (
        not job.data.distribution.shuffle
    )  # Files already shuffled without batch duplicates.
    count = 3 if strategy == "connectors" else 4
    assert job.data.distribution.normalized_weights == (1 / count,) * count
    assert all(f"-{seed}.jsonl" in s.uri for s in job.data.distribution.sources)
    schedule = build_component(job.optimization.schedule)
    np.testing.assert_allclose(schedule(120), CONFIG.LEARNING_RATES[strategy])
    assert float(schedule(0)) == float(schedule(2000)) == 0
    assert job.evaluation.every_steps == 500 and job.evaluation.on_start
    assert job.evaluation.batch_size == 2 and job.evaluation.max_batches is None
    assert job.checkpointing.every == 1000 and job.export.enabled
    assert [e.name for e in EVAL.make_evaluator(job.evaluation).evaluators] == list(
        manifest()["evaluation"]
    )


def test_batch_order_preserves_every_pair_and_is_seeded():
    rows = [
        {"media_id": str(i % 11), "caption": f"caption {i % 13}", "id": i}
        for i in range(132)
    ]
    first = DATA.unique_batch_order(rows, seed=7, batch_size=4)
    assert first == DATA.unique_batch_order(rows, seed=7, batch_size=4)
    assert first != DATA.unique_batch_order(rows, seed=42, batch_size=4)
    assert sorted(r["id"] for r in first) == list(range(132))
    for start in range(0, len(first), 4):
        assert len({r["media_id"] for r in first[start : start + 4]}) == 4
        assert len({r["caption"] for r in first[start : start + 4]}) == 4


def test_panel_routing_keeps_reused_ids_and_corpora_independent():
    from representax.evaluation import (
        InformationRetrievalEvaluator,
        retrieval_evaluation_batch,
    )
    from representax.models import DenseEncoder
    from representax.train import EvaluationRunner

    model = DenseEncoder(2, 2, key=jax.random.key(4))
    evaluators = tuple(
        InformationRetrievalEvaluator(name=name, relevant_documents={0: {0}})
        for name in ("audio", "image")
    )
    batches = []
    for name, vector in (("audio", [1.0, 0.0]), ("image", [0.0, 1.0])):
        for kind in ("query", "document"):
            batch = retrieval_evaluation_batch(
                jnp.asarray([vector, vector]),
                jnp.asarray([0, 0]),
                kind=kind,
                valid=jnp.asarray([True, False]),
            )
            batches.append(EVAL.PanelBatch(batch, name))
    runner = EvaluationRunner(EVAL.PanelEvaluator(evaluators))
    result = runner.run(model, batches)
    for name in ("audio", "image"):
        assert result.metrics[f"valid/{name}/cosine_ndcg@10"] == 1.0
