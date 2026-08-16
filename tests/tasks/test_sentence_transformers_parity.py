"""Same-tensor loss and representation-gradient parity with Sentence Transformers.

This is the canonical inventory for every Sentence Transformers loss class that
Representax claims as native. The optional upstream runtime is a repository-only
oracle; production code never imports it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib.metadata import version
from statistics import median
from time import perf_counter
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.tasks.distillation import (
    distribution_kl_loss_terms,
    embedding_distillation_loss_terms,
    margin_mse_loss_terms,
)
from representax.tasks.pairwise import (
    contrastive_loss_terms,
    cosine_regression_loss_terms,
    online_contrastive_loss_terms,
    pair_ranking_loss_terms,
)
from representax.tasks.retrieval import MNRTask, mnr_loss_terms, retrieval_batch
from representax.tasks.triplet import (
    batch_all_triplet_loss_terms,
    batch_hard_triplet_loss_terms,
    batch_semi_hard_triplet_loss_terms,
    explicit_triplet_loss_terms,
)

_ORACLE_VERSION = "5.6.1"
_NATIVE_UPSTREAM_LOSSES = frozenset(
    {
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "MultipleNegativesSymmetricRankingLoss",
        "CachedMultipleNegativesSymmetricRankingLoss",
        "CosineSimilarityLoss",
        "ContrastiveLoss",
        "OnlineContrastiveLoss",
        "CoSENTLoss",
        "AnglELoss",
        "TripletLoss",
        "BatchAllTripletLoss",
        "BatchHardTripletLoss",
        "BatchHardSoftMarginTripletLoss",
        "BatchSemiHardTripletLoss",
        "MSELoss",
        "EmbedDistillLoss",
        "MarginMSELoss",
        "DistillKLDivLoss",
    }
)


def _oracle():
    torch = pytest.importorskip("torch")
    upstream = pytest.importorskip("sentence_transformers.sentence_transformer.losses")
    upstream_util = pytest.importorskip("sentence_transformers.util")
    assert version("sentence-transformers") == _ORACLE_VERSION

    class IdentityModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": features["embedding"]}

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self

    return torch, upstream, upstream_util, IdentityModel()


def _assert_value_and_gradients(
    actual,
    actual_gradients,
    expected,
    expected_gradients,
    *,
    gradient_rtol: float = 8e-5,
    gradient_atol: float = 3e-5,
) -> None:
    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().cpu().numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
    for native, reference in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(native),
            reference.detach().cpu().numpy(),
            rtol=gradient_rtol,
            atol=gradient_atol,
        )


@dataclass(frozen=True, slots=True)
class _MNRCase:
    upstream_class: str
    cached: bool
    symmetric: bool


_MNR_CASES = (
    _MNRCase("MultipleNegativesRankingLoss", cached=False, symmetric=False),
    _MNRCase("CachedMultipleNegativesRankingLoss", cached=True, symmetric=False),
    _MNRCase("MultipleNegativesSymmetricRankingLoss", cached=False, symmetric=True),
    _MNRCase(
        "CachedMultipleNegativesSymmetricRankingLoss",
        cached=True,
        symmetric=True,
    ),
)


@pytest.mark.parametrize(
    "case",
    _MNR_CASES,
    ids=lambda case: case.upstream_class,
)
@pytest.mark.parity
def test_mnr_value_and_representation_gradient_parity(case):
    torch, upstream, _, model = _oracle()
    rng = np.random.default_rng(7)
    queries = rng.normal(size=(6, 8)).astype(np.float32)
    documents = rng.normal(size=(6, 8)).astype(np.float32)
    scale = 13.0

    loss_type = getattr(upstream, case.upstream_class)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        oracle_loss = loss_type(
            model, scale=scale, **({"mini_batch_size": 2} if case.cached else {})
        )

    if case.cached:
        query_chunks = tuple(
            torch.tensor(chunk, requires_grad=True) for chunk in np.split(queries, 3)
        )
        document_chunks = tuple(
            torch.tensor(chunk, requires_grad=True) for chunk in np.split(documents, 3)
        )
        # calculate_loss is the cached class's exact scientific objective over
        # already materialized representation chunks. Its replay mechanics have
        # separate native GradCache acceptance tests.
        expected = oracle_loss.calculate_loss(
            [list(query_chunks), list(document_chunks)]
        )
        chunk_gradients = torch.autograd.grad(
            expected,
            (*query_chunks, *document_chunks),
        )
        expected_gradients = (
            torch.cat(chunk_gradients[:3]),
            torch.cat(chunk_gradients[3:]),
        )
    else:
        torch_queries = torch.tensor(queries, requires_grad=True)
        torch_documents = torch.tensor(documents, requires_grad=True)
        expected = oracle_loss(
            [
                {"embedding": torch_queries},
                {"embedding": torch_documents},
            ],
            torch.empty((queries.shape[0],)),
        )
        expected_gradients = torch.autograd.grad(
            expected,
            (torch_queries, torch_documents),
        )

    positives = jnp.eye(queries.shape[0], dtype=jnp.bool_)
    native_batch = retrieval_batch(
        query=jnp.asarray(queries),
        document=jnp.asarray(documents),
        positive_mask=positives,
    )
    cached_task = MNRTask(scale=scale, symmetric=case.symmetric)

    def native_loss(query, document):
        if case.cached:
            return cached_task.loss_from_embeddings(
                query,
                document,
                native_batch,
                row_chunk_size=2,
            ).loss
        return mnr_loss_terms(
            query,
            document,
            positives,
            scale=scale,
            symmetric=case.symmetric,
        ).loss

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            native_loss,
            argnums=(0, 1),
        )(jnp.asarray(queries), jnp.asarray(documents))

    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
    )


@dataclass(frozen=True, slots=True)
class _PairCase:
    upstream_class: str
    objective: Literal["cosine", "contrastive", "online", "cosent", "angle"]
    metric: Literal["cosine", "euclidean", "manhattan"] = "cosine"


_PAIR_CASES = (
    _PairCase("CosineSimilarityLoss", "cosine"),
    _PairCase("ContrastiveLoss", "contrastive", "cosine"),
    _PairCase("ContrastiveLoss", "contrastive", "euclidean"),
    _PairCase("ContrastiveLoss", "contrastive", "manhattan"),
    _PairCase("OnlineContrastiveLoss", "online"),
    _PairCase("CoSENTLoss", "cosent"),
    _PairCase("AnglELoss", "angle"),
)


def _native_pair_loss(case: _PairCase, left, right, labels):
    if case.objective == "cosine":
        return cosine_regression_loss_terms(left, right, labels).loss
    if case.objective == "contrastive":
        return contrastive_loss_terms(
            left,
            right,
            labels,
            metric=case.metric,
            margin=0.5,
        ).loss
    if case.objective == "online":
        return online_contrastive_loss_terms(
            left,
            right,
            labels,
            metric=case.metric,
            margin=0.5,
        ).loss
    return pair_ranking_loss_terms(
        left,
        right,
        labels,
        scale=20.0,
        similarity="angle" if case.objective == "angle" else "cosine",
    ).loss


@pytest.mark.parametrize(
    "case",
    _PAIR_CASES,
    ids=lambda case: f"{case.upstream_class}-{case.metric}",
)
@pytest.mark.parity
def test_pairwise_value_and_representation_gradient_parity(case):
    torch, upstream, _, model = _oracle()
    rng = np.random.default_rng(17)
    left = rng.normal(size=(6, 7)).astype(np.float32)
    right = rng.normal(size=(6, 7)).astype(np.float32)
    labels = np.asarray([1.0, 0.0, 0.8, 0.2, 0.6, 0.4], dtype=np.float32)
    if case.objective in {"contrastive", "online"}:
        labels = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    torch_left = torch.tensor(left, requires_grad=True)
    torch_right = torch.tensor(right, requires_grad=True)
    torch_labels = torch.tensor(labels)
    if case.objective == "cosine":
        oracle_loss = upstream.CosineSimilarityLoss(model)
    elif case.objective in {"contrastive", "online"}:
        distance = {
            "cosine": upstream.SiameseDistanceMetric.COSINE_DISTANCE,
            "euclidean": upstream.SiameseDistanceMetric.EUCLIDEAN,
            "manhattan": upstream.SiameseDistanceMetric.MANHATTAN,
        }[case.metric]
        loss_type = (
            upstream.OnlineContrastiveLoss
            if case.objective == "online"
            else upstream.ContrastiveLoss
        )
        oracle_loss = loss_type(model, distance_metric=distance, margin=0.5)
    elif case.objective == "cosent":
        oracle_loss = upstream.CoSENTLoss(model, scale=20.0)
    else:
        oracle_loss = upstream.AnglELoss(model, scale=20.0)
    expected = oracle_loss(
        [{"embedding": torch_left}, {"embedding": torch_right}],
        torch_labels,
    )
    expected_gradients = torch.autograd.grad(expected, (torch_left, torch_right))

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            lambda first, second: _native_pair_loss(
                case,
                first,
                second,
                jnp.asarray(labels),
            ),
            argnums=(0, 1),
        )(jnp.asarray(left), jnp.asarray(right))

    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
        gradient_rtol=5e-5,
        gradient_atol=2e-5,
    )


@dataclass(frozen=True, slots=True)
class _TripletCase:
    upstream_class: str
    objective: Literal["explicit", "all", "hard", "hard_soft_margin", "semi_hard"]


_TRIPLET_CASES = (
    _TripletCase("TripletLoss", "explicit"),
    _TripletCase("BatchAllTripletLoss", "all"),
    _TripletCase("BatchHardTripletLoss", "hard"),
    _TripletCase("BatchHardSoftMarginTripletLoss", "hard_soft_margin"),
    _TripletCase("BatchSemiHardTripletLoss", "semi_hard"),
)


def _native_triplet_loss(case, embeddings, positive, negative, labels):
    if case.objective == "explicit":
        return explicit_triplet_loss_terms(
            embeddings,
            positive,
            negative,
            metric="cosine",
            margin=0.5,
        ).loss
    if case.objective == "all":
        return batch_all_triplet_loss_terms(embeddings, labels, margin=0.5).loss
    if case.objective == "semi_hard":
        return batch_semi_hard_triplet_loss_terms(
            embeddings,
            labels,
            margin=0.5,
        ).loss
    return batch_hard_triplet_loss_terms(
        embeddings,
        labels,
        margin=0.5,
        soft_margin=case.objective == "hard_soft_margin",
    ).loss


@pytest.mark.parametrize(
    "case",
    _TRIPLET_CASES,
    ids=lambda case: case.upstream_class,
)
@pytest.mark.parity
def test_triplet_value_and_representation_gradient_parity(case):
    torch, upstream, _, model = _oracle()
    rng = np.random.default_rng(29)
    embeddings = rng.normal(size=(6, 7)).astype(np.float32)
    positive = rng.normal(size=(6, 7)).astype(np.float32)
    negative = rng.normal(size=(6, 7)).astype(np.float32)
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    torch_embeddings = torch.tensor(embeddings, requires_grad=True)
    torch_positive = torch.tensor(positive, requires_grad=True)
    torch_negative = torch.tensor(negative, requires_grad=True)
    torch_labels = torch.tensor(labels)
    if case.objective == "explicit":
        oracle_loss = upstream.TripletLoss(
            model,
            distance_metric=upstream.TripletDistanceMetric.COSINE,
            triplet_margin=0.5,
        )
        features = [
            {"embedding": torch_embeddings},
            {"embedding": torch_positive},
            {"embedding": torch_negative},
        ]
        gradient_inputs = (torch_embeddings, torch_positive, torch_negative)
    else:
        loss_type = {
            "all": upstream.BatchAllTripletLoss,
            "hard": upstream.BatchHardTripletLoss,
            "hard_soft_margin": upstream.BatchHardSoftMarginTripletLoss,
            "semi_hard": upstream.BatchSemiHardTripletLoss,
        }[case.objective]
        kwargs = {} if case.objective == "hard_soft_margin" else {"margin": 0.5}
        oracle_loss = loss_type(model, **kwargs)
        features = [{"embedding": torch_embeddings}]
        gradient_inputs = (torch_embeddings,)
    expected = oracle_loss(features, torch_labels)
    expected_gradients = torch.autograd.grad(expected, gradient_inputs)

    with jax.default_matmul_precision("highest"):
        if case.objective == "explicit":
            actual, actual_gradients = jax.value_and_grad(
                lambda anchor, pos, neg: _native_triplet_loss(
                    case,
                    anchor,
                    pos,
                    neg,
                    jnp.asarray(labels),
                ),
                argnums=(0, 1, 2),
            )(
                jnp.asarray(embeddings),
                jnp.asarray(positive),
                jnp.asarray(negative),
            )
        else:
            actual, gradient = jax.value_and_grad(
                lambda value: _native_triplet_loss(
                    case,
                    value,
                    jnp.asarray(positive),
                    jnp.asarray(negative),
                    jnp.asarray(labels),
                )
            )(jnp.asarray(embeddings))
            actual_gradients = (gradient,)

    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
    )


@dataclass(frozen=True, slots=True)
class _EmbeddingDistillationCase:
    upstream_class: str
    distance: Literal["mse", "l2", "cosine"]
    broadcast_teacher: bool


_EMBEDDING_DISTILLATION_CASES = (
    _EmbeddingDistillationCase("MSELoss", "mse", broadcast_teacher=True),
    _EmbeddingDistillationCase("EmbedDistillLoss", "mse", broadcast_teacher=False),
    _EmbeddingDistillationCase("EmbedDistillLoss", "l2", broadcast_teacher=False),
    _EmbeddingDistillationCase("EmbedDistillLoss", "cosine", broadcast_teacher=False),
)


@pytest.mark.parametrize(
    "case",
    _EMBEDDING_DISTILLATION_CASES,
    ids=lambda case: f"{case.upstream_class}-{case.distance}",
)
@pytest.mark.parity
def test_embedding_distillation_value_and_representation_gradient_parity(case):
    torch, upstream, _, model = _oracle()
    rng = np.random.default_rng(43)
    first = rng.normal(size=(6, 7)).astype(np.float32)
    second = rng.normal(size=(6, 7)).astype(np.float32)
    per_column_teacher = rng.normal(size=(6, 2, 7)).astype(np.float32)
    teacher = per_column_teacher[:, 0] if case.broadcast_teacher else per_column_teacher

    torch_first = torch.tensor(first, requires_grad=True)
    torch_second = torch.tensor(second, requires_grad=True)
    loss_type = getattr(upstream, case.upstream_class)
    kwargs = (
        {} if case.upstream_class == "MSELoss" else {"distance_metric": case.distance}
    )
    oracle_loss = loss_type(model, **kwargs)
    expected = oracle_loss(
        [{"embedding": torch_first}, {"embedding": torch_second}],
        torch.tensor(teacher),
    )
    expected_gradients = torch.autograd.grad(expected, (torch_first, torch_second))

    native_teacher = (
        np.broadcast_to(teacher[None, :, :], (2, *teacher.shape))
        if case.broadcast_teacher
        else np.transpose(teacher, (1, 0, 2))
    )
    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            lambda left, right: (
                embedding_distillation_loss_terms(
                    jnp.stack((left, right)),
                    jnp.asarray(native_teacher),
                    distance=case.distance,
                ).loss
            ),
            argnums=(0, 1),
        )(jnp.asarray(first), jnp.asarray(second))

    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
    )


@pytest.mark.parametrize("similarity", ("dot", "cosine"))
@pytest.mark.parity
def test_margin_mse_value_and_representation_gradient_parity(similarity):
    torch, upstream, upstream_util, model = _oracle()
    rng = np.random.default_rng(47)
    query = rng.normal(size=(6, 7)).astype(np.float32)
    positive = rng.normal(size=(6, 7)).astype(np.float32)
    first_negative = rng.normal(size=(6, 7)).astype(np.float32)
    second_negative = rng.normal(size=(6, 7)).astype(np.float32)
    teacher_scores = rng.normal(size=(6, 3)).astype(np.float32)
    similarity_fct = {
        "dot": upstream_util.pairwise_dot_score,
        "cosine": upstream_util.pairwise_cos_sim,
    }[similarity]

    torch_values = tuple(
        torch.tensor(value, requires_grad=True)
        for value in (query, positive, first_negative, second_negative)
    )
    oracle_loss = upstream.MarginMSELoss(model, similarity_fct=similarity_fct)
    expected = oracle_loss(
        [{"embedding": value} for value in torch_values],
        torch.tensor(teacher_scores),
    )
    expected_gradients = torch.autograd.grad(expected, torch_values)
    teacher_margins = teacher_scores[:, :1] - teacher_scores[:, 1:]

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            lambda q, pos, neg_a, neg_b: (
                margin_mse_loss_terms(
                    q,
                    pos,
                    jnp.stack((neg_a, neg_b)),
                    jnp.asarray(teacher_margins),
                    similarity=similarity,
                ).loss
            ),
            argnums=(0, 1, 2, 3),
        )(
            jnp.asarray(query),
            jnp.asarray(positive),
            jnp.asarray(first_negative),
            jnp.asarray(second_negative),
        )

    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
    )


@pytest.mark.parametrize("similarity", ("dot", "cosine"))
@pytest.mark.parity
def test_distribution_kl_value_and_representation_gradient_parity(similarity):
    torch, upstream, upstream_util, model = _oracle()
    rng = np.random.default_rng(53)
    query = rng.normal(size=(6, 7)).astype(np.float32)
    candidates = tuple(rng.normal(size=(6, 7)).astype(np.float32) for _ in range(3))
    teacher_scores = rng.normal(size=(6, 3)).astype(np.float32)
    temperature = 2.0
    similarity_fct = {
        "dot": upstream_util.pairwise_dot_score,
        "cosine": upstream_util.pairwise_cos_sim,
    }[similarity]

    torch_values = tuple(
        torch.tensor(value, requires_grad=True) for value in (query, *candidates)
    )
    oracle_loss = upstream.DistillKLDivLoss(
        model,
        similarity_fct=similarity_fct,
        temperature=temperature,
    )
    expected = oracle_loss(
        [{"embedding": value} for value in torch_values],
        torch.tensor(teacher_scores),
    )
    expected_gradients = torch.autograd.grad(expected, torch_values)

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            lambda q, first, second, third: (
                distribution_kl_loss_terms(
                    q,
                    jnp.stack((first, second, third)),
                    jnp.asarray(teacher_scores),
                    similarity=similarity,
                    temperature=temperature,
                ).loss
            ),
            argnums=(0, 1, 2, 3),
        )(
            jnp.asarray(query),
            *(jnp.asarray(candidate) for candidate in candidates),
        )

    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
    )


@pytest.mark.parity
def test_every_native_sentence_transformers_loss_has_a_paired_oracle_case():
    paired = {
        *(case.upstream_class for case in _MNR_CASES),
        *(case.upstream_class for case in _PAIR_CASES),
        *(case.upstream_class for case in _TRIPLET_CASES),
        *(case.upstream_class for case in _EMBEDDING_DISTILLATION_CASES),
        "MarginMSELoss",
        "DistillKLDivLoss",
    }

    assert paired == _NATIVE_UPSTREAM_LOSSES


def _timed_samples(invoke, synchronize, *, warmups: int, iterations: int):
    for _ in range(warmups):
        synchronize(invoke())
    samples = []
    for _ in range(iterations):
        started = perf_counter()
        synchronize(invoke())
        samples.append(perf_counter() - started)
    return samples


@pytest.mark.performance
def test_complete_native_loss_suite_forward_backward_performance():
    """Measure each paired class without making timing a correctness contract."""

    torch, upstream, _, model = _oracle()
    if jax.devices()[0].platform != "gpu" or not torch.cuda.is_available():
        pytest.skip("matched loss performance requires a JAX and Torch CUDA device")

    rng = np.random.default_rng(61)
    batch_size = 48
    dimension = 128
    values = tuple(
        rng.normal(size=(batch_size, dimension)).astype(np.float32) for _ in range(3)
    )
    teacher_embeddings = rng.normal(size=(2, batch_size, dimension)).astype(np.float32)
    teacher_scores = rng.normal(size=(batch_size, 2)).astype(np.float32)
    pair_labels = np.linspace(0.0, 1.0, batch_size, dtype=np.float32)
    binary_labels = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), batch_size // 2)
    class_labels = np.repeat(np.arange(batch_size // 2, dtype=np.int32), 2)

    native_inputs = tuple(jnp.asarray(value) for value in values)
    native_teacher_embeddings = jnp.asarray(teacher_embeddings)
    native_teacher_scores = jnp.asarray(teacher_scores)
    native_pair_labels = jnp.asarray(pair_labels)
    native_binary_labels = jnp.asarray(binary_labels)
    native_class_labels = jnp.asarray(class_labels)
    native_positive_mask = jnp.eye(batch_size, dtype=jnp.bool_)
    native_retrieval_batch = retrieval_batch(
        query=native_inputs[0],
        document=native_inputs[1],
        positive_mask=native_positive_mask,
    )
    native_cached_mnr = MNRTask(scale=13.0)
    native_cached_symmetric_mnr = MNRTask(scale=13.0, symmetric=True)
    native_objectives = {
        "MultipleNegativesRankingLoss": lambda query, positive, _negative: (
            mnr_loss_terms(
                query,
                positive,
                native_positive_mask,
                scale=13.0,
            ).loss
        ),
        "CachedMultipleNegativesRankingLoss": lambda query, positive, _negative: (
            native_cached_mnr.loss_from_embeddings(
                query,
                positive,
                native_retrieval_batch,
                row_chunk_size=16,
            ).loss
        ),
        "MultipleNegativesSymmetricRankingLoss": lambda query, positive, _negative: (
            mnr_loss_terms(
                query,
                positive,
                native_positive_mask,
                scale=13.0,
                symmetric=True,
            ).loss
        ),
        "CachedMultipleNegativesSymmetricRankingLoss": (
            lambda query, positive, _negative: (
                native_cached_symmetric_mnr.loss_from_embeddings(
                    query,
                    positive,
                    native_retrieval_batch,
                    row_chunk_size=16,
                ).loss
            )
        ),
        "CosineSimilarityLoss": lambda query, positive, _negative: (
            cosine_regression_loss_terms(
                query,
                positive,
                native_pair_labels,
            ).loss
        ),
        "ContrastiveLoss": lambda query, positive, _negative: (
            contrastive_loss_terms(
                query,
                positive,
                native_binary_labels,
                margin=0.5,
            ).loss
        ),
        "OnlineContrastiveLoss": lambda query, positive, _negative: (
            online_contrastive_loss_terms(
                query,
                positive,
                native_binary_labels,
                margin=0.5,
            ).loss
        ),
        "CoSENTLoss": lambda query, positive, _negative: (
            pair_ranking_loss_terms(
                query,
                positive,
                native_pair_labels,
                scale=20.0,
                similarity="cosine",
            ).loss
        ),
        "AnglELoss": lambda query, positive, _negative: (
            pair_ranking_loss_terms(
                query,
                positive,
                native_pair_labels,
                scale=20.0,
                similarity="angle",
            ).loss
        ),
        "TripletLoss": lambda query, positive, negative: (
            explicit_triplet_loss_terms(
                query,
                positive,
                negative,
                metric="cosine",
                margin=0.5,
            ).loss
        ),
        "BatchAllTripletLoss": lambda query, _positive, _negative: (
            batch_all_triplet_loss_terms(
                query,
                native_class_labels,
                margin=0.5,
            ).loss
        ),
        "BatchHardTripletLoss": lambda query, _positive, _negative: (
            batch_hard_triplet_loss_terms(
                query,
                native_class_labels,
                margin=0.5,
            ).loss
        ),
        "BatchHardSoftMarginTripletLoss": lambda query, _positive, _negative: (
            batch_hard_triplet_loss_terms(
                query,
                native_class_labels,
                soft_margin=True,
            ).loss
        ),
        "BatchSemiHardTripletLoss": lambda query, _positive, _negative: (
            batch_semi_hard_triplet_loss_terms(
                query,
                native_class_labels,
                margin=0.5,
            ).loss
        ),
        "MSELoss": lambda query, positive, _negative: (
            embedding_distillation_loss_terms(
                jnp.stack((query, positive)),
                native_teacher_embeddings,
                distance="mse",
            ).loss
        ),
        "EmbedDistillLoss": lambda query, positive, _negative: (
            embedding_distillation_loss_terms(
                jnp.stack((query, positive)),
                native_teacher_embeddings,
                distance="cosine",
            ).loss
        ),
        "MarginMSELoss": lambda query, positive, negative: (
            margin_mse_loss_terms(
                query,
                positive,
                negative[None, :, :],
                native_teacher_scores[:, :1] - native_teacher_scores[:, 1:],
            ).loss
        ),
        "DistillKLDivLoss": lambda query, positive, negative: (
            distribution_kl_loss_terms(
                query,
                jnp.stack((positive, negative)),
                native_teacher_scores,
                temperature=2.0,
            ).loss
        ),
    }
    assert set(native_objectives) == _NATIVE_UPSTREAM_LOSSES

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch_inputs = tuple(
        torch.tensor(value, device="cuda", requires_grad=True) for value in values
    )
    torch_teacher_embeddings = torch.tensor(teacher_embeddings, device="cuda")
    torch_teacher_scores = torch.tensor(teacher_scores, device="cuda")
    torch_pair_labels = torch.tensor(pair_labels, device="cuda")
    torch_binary_labels = torch.tensor(binary_labels, device="cuda")
    torch_class_labels = torch.tensor(class_labels, device="cuda")
    torch_empty_labels = torch.empty((batch_size,), device="cuda")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        oracle_losses = {
            "mnr": upstream.MultipleNegativesRankingLoss(model, scale=13.0),
            "cached_mnr": upstream.CachedMultipleNegativesRankingLoss(
                model,
                scale=13.0,
                mini_batch_size=16,
            ),
            "symmetric": upstream.MultipleNegativesSymmetricRankingLoss(
                model,
                scale=13.0,
            ),
            "cached_symmetric": upstream.CachedMultipleNegativesSymmetricRankingLoss(
                model,
                scale=13.0,
                mini_batch_size=16,
            ),
            "cosine": upstream.CosineSimilarityLoss(model),
            "contrastive": upstream.ContrastiveLoss(model, margin=0.5),
            "online": upstream.OnlineContrastiveLoss(model, margin=0.5),
            "cosent": upstream.CoSENTLoss(model, scale=20.0),
            "angle": upstream.AnglELoss(model, scale=20.0),
            "triplet": upstream.TripletLoss(
                model,
                distance_metric=upstream.TripletDistanceMetric.COSINE,
                triplet_margin=0.5,
            ),
            "all": upstream.BatchAllTripletLoss(model, margin=0.5),
            "hard": upstream.BatchHardTripletLoss(model, margin=0.5),
            "hard_soft": upstream.BatchHardSoftMarginTripletLoss(model),
            "semi_hard": upstream.BatchSemiHardTripletLoss(model, margin=0.5),
            "mse": upstream.MSELoss(model),
            "embed": upstream.EmbedDistillLoss(model),
            "margin": upstream.MarginMSELoss(model),
            "distribution": upstream.DistillKLDivLoss(model, temperature=2.0),
        }

    query, positive, negative = torch_inputs
    pair_features = [{"embedding": query}, {"embedding": positive}]
    triplet_features = [*pair_features, {"embedding": negative}]
    query_chunks = list(torch.split(query, 16))
    positive_chunks = list(torch.split(positive, 16))
    label_features = [{"embedding": query}]
    labels_by_column = torch_teacher_embeddings.permute(1, 0, 2)
    upstream_objectives = {
        "MultipleNegativesRankingLoss": lambda: oracle_losses["mnr"](
            pair_features,
            torch_empty_labels,
        ),
        "CachedMultipleNegativesRankingLoss": lambda: oracle_losses[
            "cached_mnr"
        ].calculate_loss([query_chunks, positive_chunks]),
        "MultipleNegativesSymmetricRankingLoss": lambda: oracle_losses["symmetric"](
            pair_features,
            torch_empty_labels,
        ),
        "CachedMultipleNegativesSymmetricRankingLoss": lambda: oracle_losses[
            "cached_symmetric"
        ].calculate_loss([query_chunks, positive_chunks]),
        "CosineSimilarityLoss": lambda: oracle_losses["cosine"](
            pair_features,
            torch_pair_labels,
        ),
        "ContrastiveLoss": lambda: oracle_losses["contrastive"](
            pair_features,
            torch_binary_labels,
        ),
        "OnlineContrastiveLoss": lambda: oracle_losses["online"](
            pair_features,
            torch_binary_labels,
        ),
        "CoSENTLoss": lambda: oracle_losses["cosent"](
            pair_features,
            torch_pair_labels,
        ),
        "AnglELoss": lambda: oracle_losses["angle"](
            pair_features,
            torch_pair_labels,
        ),
        "TripletLoss": lambda: oracle_losses["triplet"](
            triplet_features,
            torch_empty_labels,
        ),
        "BatchAllTripletLoss": lambda: oracle_losses["all"](
            label_features,
            torch_class_labels,
        ),
        "BatchHardTripletLoss": lambda: oracle_losses["hard"](
            label_features,
            torch_class_labels,
        ),
        "BatchHardSoftMarginTripletLoss": lambda: oracle_losses["hard_soft"](
            label_features,
            torch_class_labels,
        ),
        "BatchSemiHardTripletLoss": lambda: oracle_losses["semi_hard"](
            label_features,
            torch_class_labels,
        ),
        "MSELoss": lambda: oracle_losses["mse"](
            pair_features,
            labels_by_column,
        ),
        "EmbedDistillLoss": lambda: oracle_losses["embed"](
            pair_features,
            labels_by_column,
        ),
        "MarginMSELoss": lambda: oracle_losses["margin"](
            triplet_features,
            torch_teacher_scores,
        ),
        "DistillKLDivLoss": lambda: oracle_losses["distribution"](
            triplet_features,
            torch_teacher_scores,
        ),
    }
    assert set(upstream_objectives) == _NATIVE_UPSTREAM_LOSSES

    results = []
    shortfalls = []
    for name in sorted(_NATIVE_UPSTREAM_LOSSES):
        native_program = jax.jit(
            jax.value_and_grad(native_objectives[name], argnums=(0, 1, 2))
        )
        with jax.default_matmul_precision("highest"):
            started = perf_counter()
            native_compiled = native_program.lower(*native_inputs).compile()
            native_compile_seconds = perf_counter() - started

            def native_invoke(compiled=native_compiled):
                return compiled(*native_inputs)

            native_samples = _timed_samples(
                native_invoke,
                jax.block_until_ready,
                warmups=5,
                iterations=20,
            )

        def upstream_invoke(objective=upstream_objectives[name]):
            value = objective()
            gradients = torch.autograd.grad(
                value,
                torch_inputs,
                allow_unused=True,
            )
            return value, tuple(
                torch.zeros_like(parameter) if gradient is None else gradient
                for parameter, gradient in zip(
                    torch_inputs,
                    gradients,
                    strict=True,
                )
            )

        upstream_samples = _timed_samples(
            upstream_invoke,
            lambda _value: torch.cuda.synchronize(),
            warmups=5,
            iterations=20,
        )
        native_seconds = median(native_samples)
        upstream_seconds = median(upstream_samples)
        ratio = upstream_seconds / native_seconds
        results.append(
            {
                "class": name,
                "native_compile_seconds": native_compile_seconds,
                "native_median_seconds": native_seconds,
                "sentence_transformers_median_seconds": upstream_seconds,
                "native_speedup": ratio,
            }
        )
        if ratio < 1.0:
            shortfalls.append((name, ratio))

    print(results)
    if shortfalls:
        details = ", ".join(
            f"{name}={1.0 / ratio:.3f}x slower" for name, ratio in shortfalls
        )
        warnings.warn(
            f"Representax loss performance shortfalls on this uncontrolled device: "
            f"{details}",
            RuntimeWarning,
            stacklevel=2,
        )
