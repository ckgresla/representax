"""Fixed-shape pointwise, pairwise, and listwise scorer batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from representax.tasks._batch import asarray, payload_row_count

if TYPE_CHECKING:
    from representax.models.processing import Processor

PointwiseLabelType = Literal["float", "integer"]


def _reshape_candidates(payload: Any, query_count: int, document_count: int) -> Any:
    def reshape(value: Any) -> Any:
        if isinstance(value, (jax.Array, np.ndarray)):
            if value.shape[0] != query_count * document_count:
                raise ValueError("processed listwise leaves must contain every pair")
            return value.reshape((query_count, document_count, *value.shape[1:]))
        return value

    return jax.tree.map(reshape, payload)


class PointwiseCollator:
    """Process labeled query/document pairs with the loaded scorer processor."""

    def __init__(
        self,
        *,
        processor: Processor,
        query_field: str = "query",
        document_field: str = "document",
        label_field: str = "label",
        label_type: PointwiseLabelType = "float",
    ) -> None:
        self.processor = processor
        self.query_field = query_field
        self.document_field = document_field
        self.label_field = label_field
        self.label_type = label_type

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-pointwise-scorer-collator-v1",
            "processor": self.processor.data_contract(),
            "query_field": self.query_field,
            "document_field": self.document_field,
            "label_field": self.label_field,
            "label_type": self.label_type,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> PointwiseBatch:
        pairs = [
            (example[self.query_field], example[self.document_field])
            for example in examples
        ]
        dtype = np.float32 if self.label_type == "float" else np.int32
        labels = np.asarray(
            [example[self.label_field] for example in examples], dtype=dtype
        )
        return PointwiseBatch(
            inputs=self.processor(pairs),
            labels=labels,
            valid=np.ones(labels.shape, dtype=np.bool_),
        )


class PairwiseRankingCollator:
    """Process query/positive/negative rows and target teacher margins."""

    def __init__(
        self,
        *,
        processor: Processor,
        query_field: str = "query",
        positive_field: str = "positive",
        negative_field: str = "negative",
        margin_field: str = "margin",
    ) -> None:
        self.processor = processor
        self.query_field = query_field
        self.positive_field = positive_field
        self.negative_field = negative_field
        self.margin_field = margin_field

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-pairwise-ranker-collator-v1",
            "processor": self.processor.data_contract(),
            "query_field": self.query_field,
            "positive_field": self.positive_field,
            "negative_field": self.negative_field,
            "margin_field": self.margin_field,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> PairwiseRankingBatch:
        queries = [example[self.query_field] for example in examples]
        positive = [
            (query, example[self.positive_field])
            for query, example in zip(queries, examples, strict=True)
        ]
        negative = [
            (query, example[self.negative_field])
            for query, example in zip(queries, examples, strict=True)
        ]
        margins = np.asarray(
            [example[self.margin_field] for example in examples], dtype=np.float32
        )
        return PairwiseRankingBatch(
            positive=self.processor(positive),
            negative=self.processor(negative),
            margins=margins,
            valid=np.ones(margins.shape, dtype=np.bool_),
        )


class ListwiseRankingCollator:
    """Process finite candidate lists while retaining query as the batch axis."""

    def __init__(
        self,
        *,
        processor: Processor,
        documents_per_query: int,
        query_field: str = "query",
        documents_field: str = "documents",
        labels_field: str = "labels",
    ) -> None:
        if documents_per_query < 2:
            raise ValueError("listwise batches require at least two document slots")
        self.processor = processor
        self.documents_per_query = documents_per_query
        self.query_field = query_field
        self.documents_field = documents_field
        self.labels_field = labels_field

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-listwise-ranker-collator-v1",
            "processor": self.processor.data_contract(),
            "documents_per_query": self.documents_per_query,
            "query_field": self.query_field,
            "documents_field": self.documents_field,
            "labels_field": self.labels_field,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> ListwiseRankingBatch:
        pairs: list[tuple[Any, Any]] = []
        labels = np.zeros((len(examples), self.documents_per_query), dtype=np.float32)
        valid = np.zeros(labels.shape, dtype=np.bool_)
        for row, example in enumerate(examples):
            documents = list(example[self.documents_field])
            relevance = list(example[self.labels_field])
            if len(documents) != len(relevance):
                raise ValueError("listwise documents and labels must align")
            if not 2 <= len(documents) <= self.documents_per_query:
                raise ValueError("listwise row is outside the configured finite bucket")
            query = example[self.query_field]
            labels[row, : len(relevance)] = relevance
            valid[row, : len(relevance)] = True
            padded = [
                *documents,
                *("" for _ in range(self.documents_per_query - len(documents))),
            ]
            pairs.extend((query, document) for document in padded)
        processed = self.processor(pairs)
        return listwise_ranking_batch(
            inputs=_reshape_candidates(
                processed, len(examples), self.documents_per_query
            ),
            labels=labels,
            valid=valid,
        )


class CrossMNRBatch(eqx.Module):
    """All query/candidate pairs for exact cross-encoder in-batch negatives."""

    inputs: Any
    positive_indices: Int[Array, " query"]
    valid: Bool[Array, "query candidate"]

    def __post_init__(self) -> None:
        if self.positive_indices.ndim != 1 or not jnp.issubdtype(
            self.positive_indices.dtype, jnp.integer
        ):
            raise TypeError("positive_indices must be an integer vector")
        if (
            self.valid.ndim != 2
            or self.valid.shape[0] != self.positive_indices.shape[0]
        ):
            raise ValueError("cross MNR validity must have [query, candidate] shape")
        leaves = [
            leaf
            for leaf in jax.tree.leaves(self.inputs)
            if isinstance(leaf, (jax.Array, np.ndarray))
        ]
        if not leaves or any(leaf.shape[:2] != self.valid.shape for leaf in leaves):
            raise ValueError("cross MNR inputs must begin with [query, candidate]")


def cross_mnr_batch(
    *,
    inputs: Any,
    positive_indices: Int[ArrayLike, " query"],
    valid: Bool[ArrayLike, "query candidate"],
) -> CrossMNRBatch:
    """Construct an exact cross-MNR batch in the caller's array domain."""

    return CrossMNRBatch(
        inputs=inputs,
        positive_indices=asarray(positive_indices, dtype=jnp.int32),
        valid=asarray(valid, dtype=jnp.bool_),
    )


class CrossMNRCollator:
    """Create the exact query by global-candidate pair grid for one batch."""

    def __init__(
        self,
        *,
        processor: Processor,
        hard_negatives_per_query: int = 0,
        query_field: str = "query",
        positive_field: str = "positive",
        negatives_field: str = "negatives",
    ) -> None:
        if hard_negatives_per_query < 0:
            raise ValueError("hard_negatives_per_query must be non-negative")
        self.processor = processor
        self.hard_negatives_per_query = hard_negatives_per_query
        self.query_field = query_field
        self.positive_field = positive_field
        self.negatives_field = negatives_field

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-cross-mnr-collator-v1",
            "processor": self.processor.data_contract(),
            "hard_negatives_per_query": self.hard_negatives_per_query,
            "query_field": self.query_field,
            "positive_field": self.positive_field,
            "negatives_field": self.negatives_field,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> CrossMNRBatch:
        if not examples:
            raise ValueError("cross MNR batches must be non-empty")
        columns = 1 + self.hard_negatives_per_query
        documents: list[Any] = []
        for example in examples:
            negatives = list(example.get(self.negatives_field, ()))
            if len(negatives) != self.hard_negatives_per_query:
                raise ValueError("cross MNR row has the wrong hard-negative count")
            documents.extend((example[self.positive_field], *negatives))
        pairs = [
            (example[self.query_field], document)
            for example in examples
            for document in documents
        ]
        query_count = len(examples)
        candidate_count = len(documents)
        return cross_mnr_batch(
            inputs=_reshape_candidates(
                self.processor(pairs), query_count, candidate_count
            ),
            positive_indices=np.arange(query_count, dtype=np.int32) * columns,
            valid=np.ones((query_count, candidate_count), dtype=np.bool_),
        )


class PointwiseBatch(eqx.Module):
    """One jointly processed input and target per row."""

    inputs: Any
    labels: Float[Array, " batch"] | Int[Array, " batch"]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 1:
            raise ValueError("pointwise labels must be a vector")
        if self.valid.shape != self.labels.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("pointwise valid must be a matching boolean vector")
        if payload_row_count(self.inputs, name="inputs") != self.labels.shape[0]:
            raise ValueError("pointwise inputs must contain one row per label")


class PairwiseRankingBatch(eqx.Module):
    """Aligned positive and negative pairs with a target score margin."""

    positive: Any
    negative: Any
    margins: Float[Array, " batch"]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if self.margins.ndim != 1 or not jnp.issubdtype(
            self.margins.dtype, jnp.floating
        ):
            raise TypeError("pairwise margins must be a floating-point vector")
        if self.valid.shape != self.margins.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("pairwise valid must be a matching boolean vector")
        for name, payload in (
            ("positive", self.positive),
            ("negative", self.negative),
        ):
            if payload_row_count(payload, name=name) != self.margins.shape[0]:
                raise ValueError(f"{name} must contain one row per margin")


class ListwiseRankingBatch(eqx.Module):
    """Joint inputs grouped into finite padded candidate lists."""

    inputs: Any
    labels: Float[Array, "query document"]
    valid: Bool[Array, "query document"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 2 or not jnp.issubdtype(self.labels.dtype, jnp.floating):
            raise TypeError("listwise labels must be a floating-point matrix")
        if self.valid.shape != self.labels.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("listwise valid must be a matching boolean matrix")
        leaves = [
            leaf
            for leaf in jax.tree.leaves(self.inputs)
            if isinstance(leaf, (jax.Array, np.ndarray))
        ]
        if not leaves:
            raise TypeError("listwise inputs must contain array leaves")
        if any(leaf.shape[:2] != self.labels.shape for leaf in leaves):
            raise ValueError("listwise input leaves must begin with [query, document]")


def listwise_ranking_batch(
    *,
    inputs: Any,
    labels: Float[ArrayLike, "query document"],
    valid: Bool[ArrayLike, "query document"],
) -> ListwiseRankingBatch:
    """Construct a listwise batch while preserving host-side NumPy arrays."""

    return ListwiseRankingBatch(
        inputs=inputs,
        labels=asarray(labels, dtype=jnp.float32),
        valid=asarray(valid, dtype=jnp.bool_),
    )


__all__ = [
    "CrossMNRBatch",
    "CrossMNRCollator",
    "cross_mnr_batch",
    "ListwiseRankingBatch",
    "ListwiseRankingCollator",
    "listwise_ranking_batch",
    "PairwiseRankingBatch",
    "PairwiseRankingCollator",
    "PointwiseBatch",
    "PointwiseCollator",
]
