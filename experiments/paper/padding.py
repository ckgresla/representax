"""Scientifically isolated MPNet packing for the paper preflight."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from representax.core import Modality, Route
from representax.data import Artifact
from representax.models.mpnet import MPNetBatch, MPNetEncoder
from representax.models.processing import Processor
from representax.models.sentence import SentenceEncoder


@dataclass(frozen=True, order=True, slots=True)
class PackShape:
    """One admitted physical row count and sequence length."""

    rows: int
    sequence_length: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.sequence_length <= 0:
            raise ValueError("pack shapes must be positive")

    @property
    def token_capacity(self) -> int:
        return self.rows * self.sequence_length


def admitted_pack_shapes(
    maximum_batch_size: int,
    *,
    sequence_lengths: Sequence[int] = (16, 32, 64, 128, 256),
) -> tuple[PackShape, ...]:
    """Build a finite physical-shape set up to one logical batch size."""

    if maximum_batch_size <= 0:
        raise ValueError("maximum_batch_size must be positive")
    row_buckets = tuple(
        value
        for value in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
        if value <= maximum_batch_size
    )
    if not row_buckets or row_buckets[-1] != maximum_batch_size:
        row_buckets = (*row_buckets, maximum_batch_size)
    lengths = tuple(sorted(set(sequence_lengths)))
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("sequence lengths must be positive")
    return tuple(
        PackShape(rows=rows, sequence_length=length)
        for length in lengths
        for rows in row_buckets
    )


def _first_fit_decreasing(
    lengths: Sequence[int],
    capacity: int,
) -> tuple[tuple[int, ...], ...]:
    if any(length <= 0 or length > capacity for length in lengths):
        raise ValueError("every sequence must fit the pack capacity")
    bins: list[list[int]] = []
    remaining: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda item: (-lengths[item], item)):
        length = lengths[index]
        target = next(
            (position for position, value in enumerate(remaining) if value >= length),
            None,
        )
        if target is None:
            bins.append([index])
            remaining.append(capacity - length)
        else:
            bins[target].append(index)
            remaining[target] -= length
    return tuple(tuple(indices) for indices in bins)


def select_pack_shape(
    lengths: Sequence[int],
    admitted_shapes: Sequence[PackShape],
) -> tuple[PackShape, tuple[tuple[int, ...], ...]]:
    """Select the lowest-token-capacity shape that can hold every sequence."""

    if not lengths:
        raise ValueError("packing requires at least one sequence")
    candidates = []
    for shape in admitted_shapes:
        if max(lengths) > shape.sequence_length:
            continue
        assignments = _first_fit_decreasing(lengths, shape.sequence_length)
        if len(assignments) <= shape.rows:
            candidates.append((shape, assignments))
    if not candidates:
        raise ValueError("logical batch exceeds the admitted pack shapes")
    return min(
        candidates,
        key=lambda item: (
            item[0].token_capacity,
            item[0].rows * item[0].sequence_length ** 2,
            item[0].sequence_length,
        ),
    )


def pack_mpnet_token_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    admitted_shapes: Sequence[PackShape],
    pad_token_id: int = 1,
) -> MPNetBatch:
    """Pack independent MPNet sequences while retaining logical segment IDs."""

    normalized = tuple(np.asarray(value, dtype=np.int32) for value in sequences)
    if any(value.ndim != 1 or not value.size for value in normalized):
        raise ValueError("packed token sequences must be non-empty vectors")
    shape, assignments = select_pack_shape(
        tuple(int(value.size) for value in normalized),
        admitted_shapes,
    )
    input_ids = np.full(
        (shape.rows, shape.sequence_length),
        pad_token_id,
        dtype=np.int32,
    )
    attention_mask = np.zeros_like(input_ids, dtype=np.int32)
    position_ids = np.full_like(input_ids, pad_token_id, dtype=np.int32)
    segment_ids = np.full_like(input_ids, -1, dtype=np.int32)
    for row, indices in enumerate(assignments):
        offset = 0
        for index in indices:
            tokens = normalized[index]
            end = offset + tokens.size
            input_ids[row, offset:end] = tokens
            attention_mask[row, offset:end] = 1
            position_ids[row, offset:end] = np.arange(
                pad_token_id + 1,
                pad_token_id + 1 + tokens.size,
                dtype=np.int32,
            )
            segment_ids[row, offset:end] = index
            offset = end
    return MPNetBatch(
        input_ids=cast(Any, input_ids),
        attention_mask=cast(Any, attention_mask),
        position_ids=cast(Any, position_ids),
        segment_ids=cast(Any, segment_ids),
        logical_batch_size=len(normalized),
    )


def make_packed_mpnet_processor(
    tokenizer: Any,
    *,
    maximum_batch_size: int,
    sequence_lengths: Sequence[int] = (16, 32, 64, 128, 256),
    fixed_batch_size: int | None = None,
    fixed_query_shape: Sequence[int] | None = None,
    fixed_document_shape: Sequence[int] | None = None,
) -> Processor:
    """Construct the finite-shape host processor used only by this preflight."""

    shapes = admitted_pack_shapes(
        maximum_batch_size,
        sequence_lengths=sequence_lengths,
    )
    maximum_length = max(shape.sequence_length for shape in shapes)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("packed MPNet tokenization requires a pad token")
    if (fixed_query_shape is None) != (fixed_document_shape is None):
        raise ValueError("fixed query and document shapes must be specified together")
    if fixed_query_shape is None:
        fixed_shapes = None
    else:
        assert fixed_document_shape is not None
        if fixed_batch_size is None or fixed_batch_size <= 0:
            raise ValueError("fixed packed shapes require fixed_batch_size")
        if len(fixed_query_shape) != 2 or len(fixed_document_shape) != 2:
            raise ValueError("fixed packed shapes must be [rows, sequence_length]")
        fixed_shapes = {
            Route.QUERY: PackShape(fixed_query_shape[0], fixed_query_shape[1]),
            Route.DOCUMENT: PackShape(fixed_document_shape[0], fixed_document_shape[1]),
        }

    def process(
        artifacts: Sequence[str | Artifact],
        *,
        route: Route,
        seed: int | None,
    ) -> MPNetBatch:
        del seed
        texts = []
        for artifact in artifacts:
            if isinstance(artifact, Artifact):
                if artifact.modality is not Modality.TEXT or not isinstance(
                    artifact.data, str
                ):
                    raise TypeError("packed MPNet processing requires inline text")
                texts.append(artifact.data)
            elif isinstance(artifact, str):
                texts.append(artifact)
            else:
                raise TypeError("packed MPNet processing requires text")
        encoded = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=maximum_length,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise TypeError("tokenizer must return an input_ids mapping")
        selected_shapes = shapes
        if fixed_shapes is not None and len(texts) == fixed_batch_size:
            try:
                selected_shapes = (fixed_shapes[route],)
            except KeyError as error:
                raise ValueError(
                    "fixed packed shapes require query or document routes"
                ) from error
        return pack_mpnet_token_sequences(
            encoded["input_ids"],
            admitted_shapes=selected_shapes,
            pad_token_id=int(pad_token_id),
        )

    tokenizer_type = type(tokenizer)
    return Processor(
        process=process,
        contract={
            "schema_version": "representax-paper-mpnet-packing-v1",
            "tokenizer_type": (
                f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}"
            ),
            "maximum_batch_size": maximum_batch_size,
            "admitted_shapes": [
                [shape.rows, shape.sequence_length] for shape in shapes
            ],
            "position_ids_reset_per_segment": True,
            "attention_isolation": "block-diagonal",
            "fixed_batch_size": fixed_batch_size,
            "fixed_query_shape": (
                None
                if fixed_shapes is None
                else [
                    fixed_shapes[Route.QUERY].rows,
                    fixed_shapes[Route.QUERY].sequence_length,
                ]
            ),
            "fixed_document_shape": (
                None
                if fixed_shapes is None
                else [
                    fixed_shapes[Route.DOCUMENT].rows,
                    fixed_shapes[Route.DOCUMENT].sequence_length,
                ]
            ),
        },
    )


def load_packed_mpnet_sentence_encoder(
    model_name_or_path: str | Path,
    *,
    maximum_batch_size: int = 128,
    sequence_lengths: Sequence[int] = (16, 32, 64, 128, 256),
    fixed_batch_size: int | None = None,
    fixed_query_shape: Sequence[int] | None = None,
    fixed_document_shape: Sequence[int] | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    **options: Any,
) -> tuple[SentenceEncoder, Processor]:
    """Load the ordinary sentence model with the paper's packed processor."""

    model, _ = SentenceEncoder.load_from_hf(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        sequence_length_buckets=(max(sequence_lengths),),
        **options,
    )
    if not isinstance(model.backbone, MPNetEncoder):
        raise TypeError("the packing preflight currently supports MPNet only")
    if model.pooling.modes != ("mean",):
        raise TypeError("the packing preflight requires mean sentence pooling")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - paper environment invariant
        raise ImportError("the packing preflight requires Transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
    )
    processor = make_packed_mpnet_processor(
        tokenizer,
        maximum_batch_size=maximum_batch_size,
        sequence_lengths=sequence_lengths,
        fixed_batch_size=fixed_batch_size,
        fixed_query_shape=fixed_query_shape,
        fixed_document_shape=fixed_document_shape,
    )
    return model, processor


__all__ = [
    "PackShape",
    "admitted_pack_shapes",
    "load_packed_mpnet_sentence_encoder",
    "make_packed_mpnet_processor",
    "pack_mpnet_token_sequences",
    "select_pack_shape",
]
