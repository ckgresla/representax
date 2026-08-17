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
from types import SimpleNamespace
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import EncoderMetadata, Modality, Route
from representax.tasks.classification import (
    pair_features as classification_pair_features,
)
from representax.tasks.classification import (
    softmax_classification_loss_terms,
)
from representax.tasks.contrastive_tension import (
    contrastive_tension_in_batch_loss_terms,
    contrastive_tension_loss_terms,
)
from representax.tasks.distillation import (
    distribution_kl_loss_terms,
    embedding_distillation_loss_terms,
    margin_mse_loss_terms,
)
from representax.tasks.guided import gist_loss_terms
from representax.tasks.mega_batch import mega_batch_margin_loss_terms
from representax.tasks.modifiers import (
    AdaptiveLayerTask,
    Matryoshka2dTask,
    MatryoshkaTask,
)
from representax.tasks.pairwise import (
    contrastive_loss_terms,
    cosine_regression_loss_terms,
    online_contrastive_loss_terms,
    pair_ranking_loss_terms,
)
from representax.tasks.reconstruction import denoising_autoencoder_loss_terms
from representax.tasks.regularization import global_orthogonal_regularization_terms
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
        "MatryoshkaLoss",
        "AdaptiveLayerLoss",
        "Matryoshka2dLoss",
        "GISTEmbedLoss",
        "CachedGISTEmbedLoss",
        "SoftmaxLoss",
        "ContrastiveTensionLoss",
        "ContrastiveTensionLossInBatchNegatives",
        "GlobalOrthogonalRegularizationLoss",
        "DenoisingAutoEncoderLoss",
        "MegaBatchMarginLoss",
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


def _modifier_oracle(dimension: int):
    torch, upstream, _, _ = _oracle()
    base_model = pytest.importorskip("sentence_transformers.base.model")
    base_modules = pytest.importorskip("sentence_transformers.base.modules")

    class TensorTransformer(base_modules.Transformer):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.model = SimpleNamespace(
                config=SimpleNamespace(output_hidden_states=False)
            )

        def forward(self, features):
            layers = features["layers"]
            features["token_embeddings"] = layers[-1]
            features["all_layer_embeddings"] = tuple(layers)
            return features

    class Normalize(torch.nn.Module):
        def forward(self, features):
            features["sentence_embedding"] = torch.nn.functional.normalize(
                features["token_embeddings"],
                dim=-1,
            )
            return features

    class TensorModel(base_model.BaseModel):
        def __init__(self):
            super().__init__(modules=[TensorTransformer(), Normalize()])

        def _load_default_modules(self, *args, **kwargs):
            raise NotImplementedError

        def get_embedding_dimension(self):
            return dimension

    return torch, upstream, TensorModel()


class _LayerIdentityEncoder(eqx.Module):
    metadata: EncoderMetadata

    def encode(self, inputs, *, route, key=None):
        del route, key
        value = inputs[:, -1]
        return value / jnp.linalg.norm(value, axis=-1, keepdims=True)

    def encode_layers(self, inputs, *, route, key=None):
        del route, key
        value = jnp.transpose(inputs, (1, 0, 2))
        return value / jnp.linalg.norm(value, axis=-1, keepdims=True)


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


@dataclass(frozen=True, slots=True)
class _ModifierCase:
    upstream_class: str


_MODIFIER_CASES = tuple(
    _ModifierCase(name)
    for name in ("MatryoshkaLoss", "AdaptiveLayerLoss", "Matryoshka2dLoss")
)


@pytest.mark.parametrize(
    "case",
    _MODIFIER_CASES,
    ids=lambda case: case.upstream_class,
)
@pytest.mark.parity
def test_modifier_value_and_representation_gradient_parity(case):
    dimension = 7
    torch, upstream, oracle_model = _modifier_oracle(dimension)
    rng = np.random.default_rng(5)
    query_layers = rng.normal(size=(4, 5, dimension)).astype(np.float32)
    document_layers = rng.normal(size=(4, 5, dimension)).astype(np.float32)
    torch_query = torch.tensor(query_layers, requires_grad=True)
    torch_document = torch.tensor(document_layers, requires_grad=True)
    oracle_base = upstream.MultipleNegativesRankingLoss(oracle_model, scale=11.0)
    if case.upstream_class == "MatryoshkaLoss":
        oracle_loss = upstream.MatryoshkaLoss(
            oracle_model,
            oracle_base,
            [7, 4],
            [1.0, 0.3],
            n_dims_per_step=-1,
        )
    elif case.upstream_class == "AdaptiveLayerLoss":
        oracle_loss = upstream.AdaptiveLayerLoss(
            oracle_model,
            oracle_base,
            n_layers_per_step=-1,
        )
    else:
        oracle_loss = upstream.Matryoshka2dLoss(
            oracle_model,
            oracle_base,
            [7, 4],
            [1.0, 0.3],
            n_dims_per_step=-1,
            n_layers_per_step=-1,
        )
    expected = oracle_loss(
        [{"layers": torch_query}, {"layers": torch_document}],
        torch.empty((query_layers.shape[1],)),
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (torch_query, torch_document),
    )

    encoder = _LayerIdentityEncoder(
        metadata=EncoderMetadata(
            model_id="tests/layer-identity",
            revision="1",
            output_dimension=dimension,
            routes=frozenset(Route),
            modalities=frozenset({Modality.TEXT}),
        )
    )
    base_task = MNRTask(scale=11.0)
    if case.upstream_class == "MatryoshkaLoss":
        native_task = MatryoshkaTask(
            base_task,
            (7, 4),
            weights=(1.0, 0.3),
        )
    elif case.upstream_class == "AdaptiveLayerLoss":
        native_task = AdaptiveLayerTask(base_task, layers_per_step=-1)
    else:
        native_task = Matryoshka2dTask(
            base_task,
            (7, 4),
            weights=(1.0, 0.3),
            dimensions_per_step=-1,
            layers_per_step=-1,
        )
    positive_mask = jnp.eye(query_layers.shape[1], dtype=jnp.bool_)

    def native_loss(query, document):
        batch = retrieval_batch(
            query=jnp.transpose(query, (1, 0, 2)),
            document=jnp.transpose(document, (1, 0, 2)),
            positive_mask=positive_mask,
        )
        return native_task.loss(encoder, batch).loss

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            native_loss,
            argnums=(0, 1),
        )(jnp.asarray(query_layers), jnp.asarray(document_layers))

    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
        gradient_rtol=2e-4,
        gradient_atol=4e-5,
    )


@pytest.mark.parametrize(
    "cached",
    (False, True),
    ids=("GISTEmbedLoss", "CachedGISTEmbedLoss"),
)
@pytest.mark.parametrize("margin_strategy", ("absolute", "relative"))
@pytest.mark.parity
def test_gist_value_and_representation_gradient_parity(cached, margin_strategy):
    torch, upstream, _, identity_model = _oracle()

    class GuideModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": features["guide_embedding"]}

    loss_type = upstream.CachedGISTEmbedLoss if cached else upstream.GISTEmbedLoss
    oracle_loss = loss_type.__new__(loss_type)
    torch.nn.Module.__init__(oracle_loss)
    oracle_loss.model = identity_model
    oracle_loss.guide = GuideModel()
    oracle_loss.temperature = 0.07
    oracle_loss.similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    oracle_loss.must_retokenize = False
    oracle_loss.margin_strategy = margin_strategy
    oracle_loss.margin = 0.1
    oracle_loss.contrast_anchors = True
    oracle_loss.contrast_positives = True
    oracle_loss.gather_across_devices = False
    oracle_loss.cross_entropy_loss = torch.nn.CrossEntropyLoss()
    if cached:
        oracle_loss.mini_batch_size = 2
        oracle_loss.show_progress_bar = False

    rng = np.random.default_rng(29)
    student = tuple(rng.normal(size=(6, 8)).astype(np.float32) for _ in range(3))
    guide = tuple(rng.normal(size=(6, 5)).astype(np.float32) for _ in range(3))
    torch_student = tuple(torch.tensor(value, requires_grad=True) for value in student)
    torch_guide = tuple(torch.tensor(value) for value in guide)
    if cached:
        representation_chunks = [list(torch.split(value, 2)) for value in torch_student]
        guide_chunks = [list(torch.split(value, 2)) for value in torch_guide]
        expected = oracle_loss.calculate_loss(representation_chunks, guide_chunks)
    else:
        expected = oracle_loss(
            [
                {"embedding": value, "guide_embedding": guide_value}
                for value, guide_value in zip(
                    torch_student,
                    torch_guide,
                    strict=True,
                )
            ],
            torch.empty((student[0].shape[0],)),
        )
    expected_gradients = torch.autograd.grad(expected, torch_student)

    def native_loss(*values):
        return gist_loss_terms(
            values,
            tuple(jnp.asarray(value) for value in guide),
            temperature=0.07,
            margin_strategy=margin_strategy,
            margin=0.1,
            row_chunk_size=2 if cached else None,
        ).loss

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            native_loss,
            argnums=(0, 1, 2),
        )(*(jnp.asarray(value) for value in student))

    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
        gradient_rtol=2e-4,
        gradient_atol=4e-5,
    )


@pytest.mark.parity
def test_softmax_value_and_representation_gradient_parity():
    torch, upstream, _, identity_model = _oracle()
    dimension = 6
    class_count = 3
    oracle_loss = upstream.SoftmaxLoss.__new__(upstream.SoftmaxLoss)
    torch.nn.Module.__init__(oracle_loss)
    oracle_loss.model = identity_model
    oracle_loss.num_labels = class_count
    oracle_loss.concatenation_sent_rep = True
    oracle_loss.concatenation_sent_difference = True
    oracle_loss.concatenation_sent_multiplication = True
    oracle_loss.classifier = torch.nn.Linear(4 * dimension, class_count)
    oracle_loss.loss_fct = torch.nn.CrossEntropyLoss()

    rng = np.random.default_rng(31)
    left = rng.normal(size=(7, dimension)).astype(np.float32)
    right = rng.normal(size=(7, dimension)).astype(np.float32)
    weight = rng.normal(size=(class_count, 4 * dimension)).astype(np.float32)
    bias = rng.normal(size=(class_count,)).astype(np.float32)
    labels = np.asarray([0, 1, 2, 1, 0, 2, 1], dtype=np.int64)
    torch_left = torch.tensor(left, requires_grad=True)
    torch_right = torch.tensor(right, requires_grad=True)
    with torch.no_grad():
        oracle_loss.classifier.weight.copy_(torch.tensor(weight))
        oracle_loss.classifier.bias.copy_(torch.tensor(bias))
    expected = oracle_loss(
        [{"embedding": torch_left}, {"embedding": torch_right}],
        torch.tensor(labels),
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (
            torch_left,
            torch_right,
            oracle_loss.classifier.weight,
            oracle_loss.classifier.bias,
        ),
    )

    def native_loss(first, second, classifier_weight, classifier_bias):
        features = classification_pair_features(
            first,
            second,
            concatenate_product=True,
        )
        logits = features @ classifier_weight.T + classifier_bias
        return softmax_classification_loss_terms(
            logits,
            jnp.asarray(labels),
        ).loss

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            native_loss,
            argnums=(0, 1, 2, 3),
        )(
            jnp.asarray(left),
            jnp.asarray(right),
            jnp.asarray(weight),
            jnp.asarray(bias),
        )
    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
    )


@pytest.mark.parity
def test_contrastive_tension_value_and_representation_gradient_parity():
    torch, upstream, _, identity_model = _oracle()
    rng = np.random.default_rng(37)
    first = rng.normal(size=(6, 7)).astype(np.float32)
    second = rng.normal(size=(6, 7)).astype(np.float32)
    labels = np.asarray([1, 0, 0, 1, 0, 1], dtype=np.float32)
    torch_first = torch.tensor(first, requires_grad=True)
    torch_second = torch.tensor(second, requires_grad=True)
    oracle_loss = upstream.ContrastiveTensionLoss(identity_model)
    expected = oracle_loss(
        [{"embedding": torch_first}, {"embedding": torch_second}],
        torch.tensor(labels),
    )
    expected_gradients = torch.autograd.grad(expected, (torch_first, torch_second))

    actual, actual_gradients = jax.value_and_grad(
        lambda left, right: (
            contrastive_tension_loss_terms(
                left,
                right,
                jnp.asarray(labels),
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


@pytest.mark.parametrize("similarity", ("cosine", "dot"))
@pytest.mark.parity
def test_contrastive_tension_in_batch_value_and_gradient_parity(similarity):
    torch, upstream, upstream_util, _ = _oracle()

    class FirstModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": features["first"]}

    class SecondModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": features["second"]}

    similarity_fct = (
        upstream_util.cos_sim if similarity == "cosine" else upstream_util.dot_score
    )
    oracle_loss = upstream.ContrastiveTensionLossInBatchNegatives(
        SecondModel(),
        scale=13.0,
        similarity_fct=similarity_fct,
    )
    oracle_loss.model1 = FirstModel()
    oracle_loss.model2 = SecondModel()
    rng = np.random.default_rng(41)
    first = rng.normal(size=(6, 7)).astype(np.float32)
    second = rng.normal(size=(6, 7)).astype(np.float32)
    torch_first = torch.tensor(first, requires_grad=True)
    torch_second = torch.tensor(second, requires_grad=True)
    expected = oracle_loss(
        [{"first": torch_first, "second": torch_second}],
        torch.empty((6,)),
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (torch_first, torch_second, oracle_loss.logit_scale),
    )

    native_scale = jnp.asarray(np.log(13.0), dtype=jnp.float32)

    def native_loss(left, right, logit_scale):
        return contrastive_tension_in_batch_loss_terms(
            left,
            right,
            logit_scale,
            similarity=similarity,
        ).loss

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            native_loss,
            argnums=(0, 1, 2),
        )(jnp.asarray(first), jnp.asarray(second), native_scale)
    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
        gradient_rtol=2e-4,
        gradient_atol=4e-5,
    )


@pytest.mark.parametrize("aggregation", ("mean", "sum"))
@pytest.mark.parity
def test_global_orthogonal_regularization_value_and_gradient_parity(aggregation):
    torch, upstream, _, identity_model = _oracle()
    rng = np.random.default_rng(43)
    embeddings = tuple(rng.normal(size=(7, 9)).astype(np.float32) for _ in range(2))
    torch_embeddings = tuple(
        torch.tensor(value, requires_grad=True) for value in embeddings
    )
    oracle_loss = upstream.GlobalOrthogonalRegularizationLoss(
        identity_model,
        mean_weight=0.7,
        second_moment_weight=1.3,
        aggregation=aggregation,
    )
    expected_terms = oracle_loss.compute_loss_from_embeddings(list(torch_embeddings))
    expected = sum(expected_terms.values())
    expected_gradients = torch.autograd.grad(expected, torch_embeddings)

    def native_loss(*values):
        return global_orthogonal_regularization_terms(
            values,
            mean_weight=0.7,
            second_moment_weight=1.3,
            aggregation=aggregation,
        ).loss

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            native_loss,
            argnums=(0, 1),
        )(*(jnp.asarray(value) for value in embeddings))
    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
        gradient_rtol=2e-4,
        gradient_atol=4e-5,
    )


@pytest.mark.parity
def test_denoising_autoencoder_value_and_gradient_parity():
    torch, upstream, _, identity_model = _oracle()
    vocabulary_size = 11
    dimension = 6

    class Decoder(torch.nn.Module):
        def __init__(self, weight, token_bias):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)
            self.token_bias = torch.nn.Parameter(token_bias)

        def forward(self, *, input_ids, encoder_hidden_states, **kwargs):
            del kwargs
            memory_logits = encoder_hidden_states[:, 0] @ self.weight.T
            logits = memory_logits[:, None, :] + self.token_bias[input_ids]
            return (logits,)

    rng = np.random.default_rng(47)
    representations = rng.normal(size=(5, dimension)).astype(np.float32)
    weight = rng.normal(size=(vocabulary_size, dimension)).astype(np.float32)
    token_bias = rng.normal(size=(vocabulary_size, vocabulary_size)).astype(np.float32)
    target_ids = np.asarray(
        [
            [1, 2, 3, 4, 0],
            [1, 3, 2, 0, 0],
            [1, 4, 5, 2, 0],
            [1, 5, 3, 2, 0],
            [1, 2, 4, 3, 0],
        ],
        dtype=np.int64,
    )
    torch_representations = torch.tensor(representations, requires_grad=True)
    oracle_loss = upstream.DenoisingAutoEncoderLoss.__new__(
        upstream.DenoisingAutoEncoderLoss
    )
    torch.nn.Module.__init__(oracle_loss)
    oracle_loss.encoder = identity_model
    oracle_loss.decoder = Decoder(torch.tensor(weight), torch.tensor(token_bias))
    oracle_loss.need_retokenization = False
    oracle_loss.tokenizer_decoder = SimpleNamespace(pad_token_id=0)
    expected = oracle_loss(
        [
            {
                "embedding": torch_representations,
                "attention_mask": torch.ones((5, 1)),
            },
            {"input_ids": torch.tensor(target_ids)},
        ],
        torch.empty((5,)),
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (
            torch_representations,
            oracle_loss.decoder.weight,
            oracle_loss.decoder.token_bias,
        ),
    )

    def native_loss(embedding, decoder_weight, decoder_token_bias):
        decoder_inputs = jnp.asarray(target_ids[:, :-1])
        logits = embedding @ decoder_weight.T
        logits = logits[:, None, :] + decoder_token_bias[decoder_inputs]
        return denoising_autoencoder_loss_terms(
            logits,
            jnp.asarray(target_ids[:, 1:]),
            pad_token_id=0,
        ).loss

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            native_loss,
            argnums=(0, 1, 2),
        )(
            jnp.asarray(representations),
            jnp.asarray(weight),
            jnp.asarray(token_bias),
        )
    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
        gradient_rtol=2e-4,
        gradient_atol=4e-5,
    )


@pytest.mark.parity
def test_mega_batch_margin_value_and_representation_gradient_parity():
    torch, upstream, _, identity_model = _oracle()
    rng = np.random.default_rng(53)
    anchor = rng.normal(size=(7, 9)).astype(np.float32)
    positive = rng.normal(size=(7, 9)).astype(np.float32)
    torch_anchor = torch.tensor(anchor, requires_grad=True)
    torch_positive = torch.tensor(positive, requires_grad=True)
    oracle_loss = upstream.MegaBatchMarginLoss(
        identity_model,
        positive_margin=0.7,
        negative_margin=0.2,
        use_mini_batched_version=False,
    )
    expected = oracle_loss(
        [{"embedding": torch_anchor}, {"embedding": torch_positive}],
        torch.empty((anchor.shape[0],)),
    )
    expected_gradients = torch.autograd.grad(expected, (torch_anchor, torch_positive))

    with jax.default_matmul_precision("highest"):
        actual, actual_gradients = jax.value_and_grad(
            lambda first, second: (
                mega_batch_margin_loss_terms(
                    first,
                    second,
                    positive_margin=0.7,
                    negative_margin=0.2,
                ).loss
            ),
            argnums=(0, 1),
        )(jnp.asarray(anchor), jnp.asarray(positive))
    _assert_value_and_gradients(
        actual,
        actual_gradients,
        expected,
        expected_gradients,
        gradient_rtol=2e-4,
        gradient_atol=4e-5,
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
        *(case.upstream_class for case in _MODIFIER_CASES),
        *(case.upstream_class for case in _PAIR_CASES),
        *(case.upstream_class for case in _TRIPLET_CASES),
        *(case.upstream_class for case in _EMBEDDING_DISTILLATION_CASES),
        "MarginMSELoss",
        "DistillKLDivLoss",
        "GISTEmbedLoss",
        "CachedGISTEmbedLoss",
        "SoftmaxLoss",
        "ContrastiveTensionLoss",
        "ContrastiveTensionLossInBatchNegatives",
        "GlobalOrthogonalRegularizationLoss",
        "DenoisingAutoEncoderLoss",
        "MegaBatchMarginLoss",
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
    guide_values = tuple(
        rng.normal(size=(batch_size, dimension)).astype(np.float32) for _ in range(3)
    )
    classifier_weight = rng.normal(size=(3, 4 * dimension)).astype(np.float32)
    classifier_bias = rng.normal(size=(3,)).astype(np.float32)
    decoder_weight = rng.normal(size=(17, dimension)).astype(np.float32)
    decoder_token_bias = rng.normal(size=(17, 17)).astype(np.float32)
    decoder_targets = rng.integers(1, 17, size=(batch_size, 12), dtype=np.int32)
    decoder_targets[:, -1] = 0
    pair_labels = np.linspace(0.0, 1.0, batch_size, dtype=np.float32)
    binary_labels = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), batch_size // 2)
    class_labels = np.repeat(np.arange(batch_size // 2, dtype=np.int32), 2)

    native_inputs = tuple(jnp.asarray(value) for value in values)
    native_teacher_embeddings = jnp.asarray(teacher_embeddings)
    native_teacher_scores = jnp.asarray(teacher_scores)
    native_guide_values = tuple(jnp.asarray(value) for value in guide_values)
    native_classifier_weight = jnp.asarray(classifier_weight)
    native_classifier_bias = jnp.asarray(classifier_bias)
    native_decoder_weight = jnp.asarray(decoder_weight)
    native_decoder_token_bias = jnp.asarray(decoder_token_bias)
    native_decoder_targets = jnp.asarray(decoder_targets)
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
    native_modifier_encoder = _LayerIdentityEncoder(
        metadata=EncoderMetadata(
            model_id="tests/performance-layers",
            revision="1",
            output_dimension=dimension,
            routes=frozenset(Route),
            modalities=frozenset({Modality.TEXT}),
        )
    )
    native_matryoshka = MatryoshkaTask(
        MNRTask(scale=13.0),
        (dimension, dimension // 2),
        weights=(1.0, 0.3),
    )
    native_adaptive = AdaptiveLayerTask(MNRTask(scale=13.0), layers_per_step=-1)
    native_matryoshka_2d = Matryoshka2dTask(
        MNRTask(scale=13.0),
        (dimension, dimension // 2),
        weights=(1.0, 0.3),
        dimensions_per_step=-1,
        layers_per_step=-1,
    )

    def native_modifier_loss(task, query, positive):
        batch = retrieval_batch(
            query=jnp.stack((query * 0.9, query * 1.1, query), axis=1),
            document=jnp.stack((positive * 0.9, positive * 1.1, positive), axis=1),
            positive_mask=native_positive_mask,
        )
        return task.loss(native_modifier_encoder, batch).loss

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
        "MatryoshkaLoss": lambda query, positive, _negative: native_modifier_loss(
            native_matryoshka,
            query,
            positive,
        ),
        "AdaptiveLayerLoss": lambda query, positive, _negative: native_modifier_loss(
            native_adaptive, query, positive
        ),
        "Matryoshka2dLoss": lambda query, positive, _negative: native_modifier_loss(
            native_matryoshka_2d, query, positive
        ),
        "GISTEmbedLoss": lambda query, positive, negative: (
            gist_loss_terms(
                (query, positive, negative),
                native_guide_values,
                temperature=0.07,
                margin=0.1,
            ).loss
        ),
        "CachedGISTEmbedLoss": lambda query, positive, negative: (
            gist_loss_terms(
                (query, positive, negative),
                native_guide_values,
                temperature=0.07,
                margin=0.1,
                row_chunk_size=16,
            ).loss
        ),
        "SoftmaxLoss": lambda query, positive, _negative: (
            softmax_classification_loss_terms(
                classification_pair_features(
                    query,
                    positive,
                    concatenate_product=True,
                )
                @ native_classifier_weight.T
                + native_classifier_bias,
                native_class_labels % 3,
            ).loss
        ),
        "ContrastiveTensionLoss": lambda query, positive, _negative: (
            contrastive_tension_loss_terms(
                query,
                positive,
                native_binary_labels,
            ).loss
        ),
        "ContrastiveTensionLossInBatchNegatives": (
            lambda query, positive, _negative: (
                contrastive_tension_in_batch_loss_terms(
                    query,
                    positive,
                    jnp.asarray(np.log(13.0), dtype=jnp.float32),
                ).loss
            )
        ),
        "GlobalOrthogonalRegularizationLoss": (
            lambda query, positive, _negative: (
                global_orthogonal_regularization_terms((query, positive)).loss
            )
        ),
        "DenoisingAutoEncoderLoss": lambda query, _positive, _negative: (
            denoising_autoencoder_loss_terms(
                (query @ native_decoder_weight.T)[:, None, :]
                + native_decoder_token_bias[native_decoder_targets[:, :-1]],
                native_decoder_targets[:, 1:],
                pad_token_id=0,
            ).loss
        ),
        "MegaBatchMarginLoss": lambda query, positive, _negative: (
            mega_batch_margin_loss_terms(query, positive).loss
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
    torch_guide_values = tuple(
        torch.tensor(value, device="cuda") for value in guide_values
    )
    torch_decoder_targets = torch.tensor(
        decoder_targets,
        device="cuda",
        dtype=torch.long,
    )
    torch_pair_labels = torch.tensor(pair_labels, device="cuda")
    torch_binary_labels = torch.tensor(binary_labels, device="cuda")
    torch_class_labels = torch.tensor(
        class_labels,
        device="cuda",
        dtype=torch.long,
    )
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

    _, _, modifier_model = _modifier_oracle(dimension)
    modifier_base = upstream.MultipleNegativesRankingLoss(
        modifier_model,
        scale=13.0,
    )
    oracle_losses["matryoshka"] = upstream.MatryoshkaLoss(
        modifier_model,
        modifier_base,
        [dimension, dimension // 2],
        [1.0, 0.3],
        n_dims_per_step=-1,
    )
    oracle_losses["adaptive"] = upstream.AdaptiveLayerLoss(
        modifier_model,
        modifier_base,
        n_layers_per_step=-1,
    )
    oracle_losses["matryoshka_2d"] = upstream.Matryoshka2dLoss(
        modifier_model,
        modifier_base,
        [dimension, dimension // 2],
        [1.0, 0.3],
        n_dims_per_step=-1,
        n_layers_per_step=-1,
    )

    class GuideModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": features["guide_embedding"]}

    def make_gist(loss_type, *, cached):
        loss = loss_type.__new__(loss_type)
        torch.nn.Module.__init__(loss)
        loss.model = model
        loss.guide = GuideModel()
        loss.temperature = 0.07
        loss.similarity_fct = torch.nn.CosineSimilarity(dim=-1)
        loss.must_retokenize = False
        loss.margin_strategy = "absolute"
        loss.margin = 0.1
        loss.contrast_anchors = True
        loss.contrast_positives = True
        loss.gather_across_devices = False
        loss.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        if cached:
            loss.mini_batch_size = 16
            loss.show_progress_bar = False
        return loss

    oracle_losses["gist"] = make_gist(upstream.GISTEmbedLoss, cached=False)
    oracle_losses["cached_gist"] = make_gist(
        upstream.CachedGISTEmbedLoss,
        cached=True,
    )
    softmax = upstream.SoftmaxLoss.__new__(upstream.SoftmaxLoss)
    torch.nn.Module.__init__(softmax)
    softmax.model = model
    softmax.num_labels = 3
    softmax.concatenation_sent_rep = True
    softmax.concatenation_sent_difference = True
    softmax.concatenation_sent_multiplication = True
    softmax.classifier = torch.nn.Linear(4 * dimension, 3).cuda()
    softmax.loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax.classifier.weight.copy_(torch.tensor(classifier_weight, device="cuda"))
        softmax.classifier.bias.copy_(torch.tensor(classifier_bias, device="cuda"))
    oracle_losses["softmax"] = softmax
    oracle_losses["contrastive_tension"] = upstream.ContrastiveTensionLoss(model)

    class FirstModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": features["first"]}

    class SecondModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": features["second"]}

    in_batch_tension = upstream.ContrastiveTensionLossInBatchNegatives(
        SecondModel(),
        scale=13.0,
    )
    in_batch_tension.model1 = FirstModel()
    in_batch_tension.model2 = SecondModel()
    in_batch_tension = in_batch_tension.cuda()
    oracle_losses["contrastive_tension_in_batch"] = in_batch_tension
    oracle_losses["gor"] = upstream.GlobalOrthogonalRegularizationLoss(model)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor(decoder_weight, device="cuda")
            )
            self.token_bias = torch.nn.Parameter(
                torch.tensor(decoder_token_bias, device="cuda")
            )

        def forward(self, *, input_ids, encoder_hidden_states, **kwargs):
            del kwargs
            memory_logits = encoder_hidden_states[:, 0] @ self.weight.T
            return (memory_logits[:, None, :] + self.token_bias[input_ids],)

    denoising = upstream.DenoisingAutoEncoderLoss.__new__(
        upstream.DenoisingAutoEncoderLoss
    )
    torch.nn.Module.__init__(denoising)
    denoising.encoder = model
    denoising.decoder = Decoder()
    denoising.need_retokenization = False
    denoising.tokenizer_decoder = SimpleNamespace(pad_token_id=0)
    oracle_losses["denoising"] = denoising
    oracle_losses["mega_batch"] = upstream.MegaBatchMarginLoss(
        model,
        use_mini_batched_version=False,
    )

    query, positive, negative = torch_inputs
    pair_features = [{"embedding": query}, {"embedding": positive}]
    triplet_features = [*pair_features, {"embedding": negative}]
    query_chunks = list(torch.split(query, 16))
    positive_chunks = list(torch.split(positive, 16))
    label_features = [{"embedding": query}]
    labels_by_column = torch_teacher_embeddings.permute(1, 0, 2)

    def modifier_features():
        return [
            {"layers": torch.stack((query * 0.9, query * 1.1, query))},
            {"layers": torch.stack((positive * 0.9, positive * 1.1, positive))},
        ]

    gist_features = [
        {"embedding": value, "guide_embedding": guide}
        for value, guide in zip(torch_inputs, torch_guide_values, strict=True)
    ]
    gist_representation_chunks = [
        list(torch.split(value, 16)) for value in torch_inputs
    ]
    gist_guide_chunks = [list(torch.split(value, 16)) for value in torch_guide_values]
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
        "MatryoshkaLoss": lambda: oracle_losses["matryoshka"](
            modifier_features(),
            torch_empty_labels,
        ),
        "AdaptiveLayerLoss": lambda: oracle_losses["adaptive"](
            modifier_features(),
            torch_empty_labels,
        ),
        "Matryoshka2dLoss": lambda: oracle_losses["matryoshka_2d"](
            modifier_features(),
            torch_empty_labels,
        ),
        "GISTEmbedLoss": lambda: oracle_losses["gist"](
            gist_features,
            torch_empty_labels,
        ),
        "CachedGISTEmbedLoss": lambda: oracle_losses["cached_gist"].calculate_loss(
            gist_representation_chunks, gist_guide_chunks
        ),
        "SoftmaxLoss": lambda: oracle_losses["softmax"](
            pair_features,
            torch_class_labels % 3,
        ),
        "ContrastiveTensionLoss": lambda: oracle_losses["contrastive_tension"](
            pair_features,
            torch_binary_labels,
        ),
        "ContrastiveTensionLossInBatchNegatives": lambda: oracle_losses[
            "contrastive_tension_in_batch"
        ](
            [{"first": query, "second": positive}],
            torch_empty_labels,
        ),
        "GlobalOrthogonalRegularizationLoss": lambda: sum(
            oracle_losses["gor"]
            .compute_loss_from_embeddings([query, positive])
            .values()
        ),
        "DenoisingAutoEncoderLoss": lambda: oracle_losses["denoising"](
            [
                {
                    "embedding": query,
                    "attention_mask": torch.ones(
                        (batch_size, 1),
                        device="cuda",
                    ),
                },
                {"input_ids": torch_decoder_targets},
            ],
            torch_empty_labels,
        ),
        "MegaBatchMarginLoss": lambda: oracle_losses["mega_batch"](
            pair_features,
            torch_empty_labels,
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
