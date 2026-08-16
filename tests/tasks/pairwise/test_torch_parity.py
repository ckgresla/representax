"""Numerical and representation-gradient parity with pinned upstream losses."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.tasks.pairwise import (
    contrastive_loss_terms,
    cosine_regression_loss_terms,
    online_contrastive_loss_terms,
    pair_ranking_loss_terms,
)

pytestmark = pytest.mark.parity


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    objective: Literal["cosine", "contrastive", "online", "cosent", "angle"]
    metric: Literal["cosine", "euclidean", "manhattan"] = "cosine"


_CASES = (
    _Case("cosine-regression", "cosine"),
    _Case("contrastive-cosine", "contrastive", "cosine"),
    _Case("contrastive-euclidean", "contrastive", "euclidean"),
    _Case("contrastive-manhattan", "contrastive", "manhattan"),
    _Case("online-contrastive", "online"),
    _Case("cosent", "cosent"),
    _Case("angle", "angle"),
)


def _native(case: _Case, left, right, labels):
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


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_pinned_sentence_transformers_value_and_gradient_parity(case):
    torch = pytest.importorskip("torch")
    upstream = pytest.importorskip("sentence_transformers.sentence_transformer.losses")
    assert version("sentence-transformers") == "5.6.1"

    class IdentityModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": features["embedding"]}

    rng = np.random.default_rng(17)
    left = rng.normal(size=(6, 7)).astype(np.float32)
    right = rng.normal(size=(6, 7)).astype(np.float32)
    labels = np.asarray([1.0, 0.0, 0.8, 0.2, 0.6, 0.4], dtype=np.float32)
    if case.objective in {"contrastive", "online"}:
        labels = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    torch_left = torch.tensor(left, requires_grad=True)
    torch_right = torch.tensor(right, requires_grad=True)
    torch_labels = torch.tensor(labels)
    model = IdentityModel()
    if case.objective == "cosine":
        loss = upstream.CosineSimilarityLoss(model)
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
        loss = loss_type(model, distance_metric=distance, margin=0.5)
    elif case.objective == "cosent":
        loss = upstream.CoSENTLoss(model, scale=20.0)
    else:
        loss = upstream.AnglELoss(model, scale=20.0)
    expected = loss(
        [{"embedding": torch_left}, {"embedding": torch_right}],
        torch_labels,
    )
    expected_gradients = torch.autograd.grad(expected, (torch_left, torch_right))

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            lambda first, second: _native(case, first, second, jnp.asarray(labels)),
            argnums=(0, 1),
        )(jnp.asarray(left), jnp.asarray(right))

    np.testing.assert_allclose(
        actual,
        expected.detach().numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
    for native, reference in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            native,
            reference.detach().numpy(),
            rtol=5e-5,
            atol=2e-5,
        )
