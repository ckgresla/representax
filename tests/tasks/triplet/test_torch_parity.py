"""Triplet value and representation-gradient parity with the pinned oracle."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.tasks.triplet import (
    batch_all_triplet_loss_terms,
    batch_hard_triplet_loss_terms,
    batch_semi_hard_triplet_loss_terms,
    explicit_triplet_loss_terms,
)

pytestmark = pytest.mark.parity


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    objective: Literal["explicit", "all", "hard", "hard_soft_margin", "semi_hard"]


_CASES = (
    _Case("explicit-triplet", "explicit"),
    _Case("batch-all", "all"),
    _Case("batch-hard", "hard"),
    _Case("batch-hard-soft-margin", "hard_soft_margin"),
    _Case("batch-semi-hard", "semi_hard"),
)


def _native(case: _Case, embeddings, positive, negative, labels):
    if case.objective == "explicit":
        return explicit_triplet_loss_terms(
            embeddings,
            positive,
            negative,
            metric="cosine",
            margin=0.5,
        ).loss
    if case.objective == "all":
        return batch_all_triplet_loss_terms(
            embeddings,
            labels,
            margin=0.5,
        ).loss
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


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_pinned_sentence_transformers_triplet_value_and_gradient_parity(case):
    torch = pytest.importorskip("torch")
    upstream = pytest.importorskip("sentence_transformers.sentence_transformer.losses")
    assert version("sentence-transformers") == "5.6.1"

    class IdentityModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": features["embedding"]}

    rng = np.random.default_rng(29)
    embeddings = rng.normal(size=(6, 7)).astype(np.float32)
    positive = rng.normal(size=(6, 7)).astype(np.float32)
    negative = rng.normal(size=(6, 7)).astype(np.float32)
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    torch_embeddings = torch.tensor(embeddings, requires_grad=True)
    torch_positive = torch.tensor(positive, requires_grad=True)
    torch_negative = torch.tensor(negative, requires_grad=True)
    torch_labels = torch.tensor(labels)
    model = IdentityModel()
    if case.objective == "explicit":
        loss = upstream.TripletLoss(
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
        loss = loss_type(model, **kwargs)
        features = [{"embedding": torch_embeddings}]
        gradient_inputs = (torch_embeddings,)
    expected = loss(features, torch_labels)
    expected_gradients = torch.autograd.grad(expected, gradient_inputs)

    with jax.default_matmul_precision("highest"):
        if case.objective == "explicit":
            actual, actual_gradients = jax.value_and_grad(
                lambda anchor, pos, neg: _native(
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
                lambda value: _native(
                    case,
                    value,
                    jnp.asarray(positive),
                    jnp.asarray(negative),
                    jnp.asarray(labels),
                )
            )(jnp.asarray(embeddings))
            actual_gradients = (gradient,)

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
            rtol=8e-5,
            atol=3e-5,
        )
