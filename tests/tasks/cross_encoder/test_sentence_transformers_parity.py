"""Same-score loss and gradient parity with Sentence Transformers 5.6.1."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.tasks.cross_encoder import (
    binary_cross_entropy,
    cross_mnr_loss,
    lambda_loss,
    list_mle_loss,
    listnet_loss,
    multiclass_cross_entropy,
    ranknet_loss,
    score_mse,
)


def _upstream():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    if sentence_transformers.__version__ != "5.6.1":
        pytest.fail("cross-encoder parity requires sentence-transformers==5.6.1")
    return (
        pytest.importorskip("torch"),
        pytest.importorskip("sentence_transformers.cross_encoder.losses"),
        sentence_transformers.CrossEncoder,
    )


def _oracle(values: np.ndarray, keys: list[tuple[str, str]], *, outputs: int = 1):
    torch, _, cross_encoder = _upstream()

    class Oracle(cross_encoder):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.values = torch.nn.Parameter(torch.from_numpy(values.copy()))
            self.lookup = {pair: index for index, pair in enumerate(keys)}

        @property
        def num_labels(self):
            return outputs

        def preprocess(self, pairs, prompt=None, task=None):
            del prompt, task
            return {
                "indices": torch.tensor([self.lookup[tuple(pair)] for pair in pairs])
            }

        def forward(self, features):
            return {"scores": self.values[features["indices"]]}

    return Oracle()


def _value_and_gradient(loss, model):
    value = loss()
    value.backward()
    return float(value.detach()), model.values.grad.detach().numpy()


@pytest.mark.parity
@pytest.mark.parametrize("objective", ("bce", "mse"))
def test_pointwise_scalar_losses_match_value_and_gradient(objective: str) -> None:
    torch, losses, _ = _upstream()
    scores = np.asarray((0.7, -1.2, 0.1, 1.8), dtype=np.float32)
    labels = np.asarray((1.0, 0.0, 0.35, 1.0), dtype=np.float32)
    keys = [(f"q{i}", f"d{i}") for i in range(len(scores))]
    model = _oracle(scores, keys)
    columns = [[pair[index] for pair in keys] for index in (0, 1)]
    target = torch.from_numpy(labels)
    upstream = (
        losses.BinaryCrossEntropyLoss(model)
        if objective == "bce"
        else losses.MSELoss(model)
    )
    expected_value, expected_gradient = _value_and_gradient(
        lambda: upstream(columns, target), model
    )

    valid = jnp.ones(labels.shape, dtype=jnp.bool_)

    def native(values):
        if objective == "bce":
            return binary_cross_entropy(values, jnp.asarray(labels), valid)
        return score_mse(values, jnp.asarray(labels), valid)

    actual_value, actual_gradient = jax.value_and_grad(native)(jnp.asarray(scores))
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parity
def test_multiclass_cross_entropy_matches_value_and_gradient() -> None:
    torch, losses, _ = _upstream()
    logits = np.asarray(
        ((0.2, -0.1, 0.8), (1.1, 0.2, -0.5), (-0.4, 0.7, 0.1)),
        dtype=np.float32,
    )
    labels = np.asarray((2, 0, 1), dtype=np.int64)
    keys = [(f"q{i}", f"d{i}") for i in range(len(logits))]
    model = _oracle(logits, keys, outputs=3)
    columns = [[pair[index] for pair in keys] for index in (0, 1)]
    upstream = losses.CrossEntropyLoss(model)
    expected_value, expected_gradient = _value_and_gradient(
        lambda: upstream(columns, torch.from_numpy(labels)), model
    )

    def native(values):
        return multiclass_cross_entropy(
            values,
            jnp.asarray(labels),
            jnp.ones(labels.shape, dtype=jnp.bool_),
        )

    actual_value, actual_gradient = jax.value_and_grad(native)(jnp.asarray(logits))
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parity
def test_margin_mse_matches_value_and_gradient() -> None:
    torch, losses, _ = _upstream()
    positive = np.asarray((1.2, 0.4, -0.2), dtype=np.float32)
    negative = np.asarray((-0.3, 0.1, 0.7), dtype=np.float32)
    values = np.concatenate((positive, negative))
    margins = np.asarray((1.0, 0.2, -0.5), dtype=np.float32)
    keys = [
        *((f"q{i}", f"p{i}") for i in range(3)),
        *((f"q{i}", f"n{i}") for i in range(3)),
    ]
    model = _oracle(values, keys)
    columns = [
        [f"q{i}" for i in range(3)],
        [f"p{i}" for i in range(3)],
        [f"n{i}" for i in range(3)],
    ]
    upstream = losses.MarginMSELoss(model)
    expected_value, expected_gradient = _value_and_gradient(
        lambda: upstream(columns, torch.from_numpy(margins)), model
    )

    def native(candidate):
        predicted = candidate[:3] - candidate[3:]
        return score_mse(
            predicted,
            jnp.asarray(margins),
            jnp.ones(margins.shape, dtype=jnp.bool_),
        )

    actual_value, actual_gradient = jax.value_and_grad(native)(jnp.asarray(values))
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parity
@pytest.mark.parametrize("cached", (False, True))
def test_cross_mnr_matches_value_and_gradient(cached: bool) -> None:
    torch, losses, _ = _upstream()
    logits = np.asarray(
        ((1.1, -0.4, 0.2), (0.3, 0.9, -0.7), (-0.2, 0.4, 1.3)),
        dtype=np.float32,
    )
    queries = [f"q{i}" for i in range(3)]
    positives = [f"p{i}" for i in range(3)]
    keys = [(query, positive) for query in queries for positive in positives]
    model = _oracle(logits.reshape(-1, 1), keys)
    upstream = (
        losses.CachedMultipleNegativesRankingLoss(
            model, num_negatives=None, mini_batch_size=2
        )
        if cached
        else losses.MultipleNegativesRankingLoss(model, num_negatives=None)
    )
    expected_value, flat_gradient = _value_and_gradient(
        lambda: upstream([queries, positives], torch.empty(3)), model
    )
    expected_gradient = flat_gradient.reshape(logits.shape)

    def native(values):
        return cross_mnr_loss(
            values,
            jnp.arange(3, dtype=jnp.int32),
            jnp.ones(values.shape, dtype=jnp.bool_),
        )

    actual_value, actual_gradient = jax.value_and_grad(native)(jnp.asarray(logits))
    np.testing.assert_allclose(actual_value, expected_value, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-6, atol=2e-7)


def _listwise_case():
    scores = np.asarray(
        ((1.1, -0.2, 0.5, 1.7), (0.3, 1.2, -0.4, 0.0)), dtype=np.float32
    )
    labels = np.asarray(((3.0, 0.0, 1.0, 2.0), (0.0, 2.0, 1.0, 0.0)), dtype=np.float32)
    valid = np.asarray(((True, True, True, True), (True, True, True, False)))
    queries = ["q0", "q1"]
    documents = [[f"d0-{i}" for i in range(4)], [f"d1-{i}" for i in range(3)]]
    keys = [
        (query, document)
        for query, docs in zip(queries, documents, strict=True)
        for document in docs
    ]
    flat_scores = scores[valid]
    label_list = [
        torch_row[: len(docs)]
        for torch_row, docs in zip(labels, documents, strict=True)
    ]
    return scores, labels, valid, queries, documents, keys, flat_scores, label_list


@pytest.mark.parity
@pytest.mark.parametrize(
    ("upstream_factory", "native"),
    (
        (lambda losses, model: losses.RankNetLoss(model), ranknet_loss),
        (lambda losses, model: losses.LambdaLoss(model), lambda_loss),
        (lambda losses, model: losses.ListNetLoss(model), listnet_loss),
        (lambda losses, model: losses.ListMLELoss(model), list_mle_loss),
        (
            lambda losses, model: losses.PListMLELoss(model),
            lambda scores, labels, valid: list_mle_loss(
                scores, labels, valid, position_aware=True
            ),
        ),
    ),
)
def test_listwise_losses_match_value_and_gradient(
    upstream_factory: Callable,
    native: Callable,
) -> None:
    torch, losses, _ = _upstream()
    scores, labels, valid, queries, documents, keys, flat_scores, label_list = (
        _listwise_case()
    )
    model = _oracle(flat_scores, keys)
    upstream = upstream_factory(losses, model)
    expected_value, flat_gradient = _value_and_gradient(
        lambda: upstream(
            [queries, documents],
            [torch.from_numpy(value) for value in label_list],
        ),
        model,
    )
    expected_gradient = np.zeros_like(scores)
    expected_gradient[valid] = flat_gradient

    actual_value, actual_gradient = jax.value_and_grad(native)(
        jnp.asarray(scores), jnp.asarray(labels), jnp.asarray(valid)
    )
    np.testing.assert_allclose(actual_value, expected_value, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-6)
