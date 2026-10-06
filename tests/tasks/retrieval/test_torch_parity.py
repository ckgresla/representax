"""Numerical parity against a Torch MNR reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from representax.tasks.retrieval import mnr_loss_terms


@pytest.mark.parity
def test_single_positive_mnr_matches_torch_cross_entropy():
    torch = pytest.importorskip("torch")
    torch_functional = pytest.importorskip("torch.nn.functional")
    rng = np.random.default_rng(7)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    documents = rng.normal(size=(5, 8)).astype(np.float32)
    positives = np.eye(5, dtype=np.bool_)
    scale = 13.0

    actual = mnr_loss_terms(
        jnp.asarray(queries),
        jnp.asarray(documents),
        jnp.asarray(positives),
        scale=scale,
    ).loss
    torch_queries = torch_functional.normalize(torch.from_numpy(queries), dim=1)
    torch_documents = torch_functional.normalize(torch.from_numpy(documents), dim=1)
    logits = scale * torch_queries @ torch_documents.T
    expected = torch_functional.cross_entropy(logits, torch.arange(5))

    assert np.asarray(actual) == pytest.approx(expected.numpy(), abs=1e-5)
