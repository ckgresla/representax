"""One-shot loading for native BERT sequence classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor, select_static_shape_bucket

from .model import BertEncoder
from .scoring import BertScorer
from .scoring_checkpoint import BertScorerCheckpointAdapter


def _pair_processor(
    tokenizer: Any,
    *,
    maximum_length: int,
    sequence_length_buckets: Sequence[int],
) -> Processor:
    lengths = tuple(sorted(set(int(value) for value in sequence_length_buckets)))
    if not lengths or lengths[0] <= 0 or lengths[-1] > maximum_length:
        raise ValueError("invalid BERT scorer sequence-length buckets")

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> Any:
        del route, seed
        if not artifacts:
            raise ValueError("BERT scorer batches must be non-empty")
        pairs = []
        for value in artifacts:
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError("BERT scorer inputs must be query/document pairs")
            query, document = value
            if not isinstance(query, str) or not isinstance(document, str):
                raise TypeError("BERT scorer pair members must be strings")
            pairs.append((query, document))
        encoded = tokenizer(
            [query for query, _ in pairs],
            [document for _, document in pairs],
            padding=True,
            truncation=True,
            max_length=maximum_length,
            return_tensors="np",
        )
        arrays = {name: np.asarray(value) for name, value in encoded.items()}
        input_ids = arrays["input_ids"]
        bucket = select_static_shape_bucket(
            (input_ids.shape[1],),
            tuple((length,) for length in lengths),
        )[0]
        padded = {}
        for name, value in arrays.items():
            if value.ndim != 2 or value.shape[0] != len(pairs):
                continue
            fill = int(tokenizer.pad_token_id) if name == "input_ids" else 0
            padded[name] = np.pad(
                value,
                ((0, 0), (0, bucket - value.shape[1])),
                constant_values=fill,
            )
        return BertEncoder.make_batch(
            input_ids=padded["input_ids"],
            attention_mask=padded["attention_mask"],
            token_type_ids=padded.get("token_type_ids"),
        )

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-bert-scorer-processor-v1",
            "tokenizer": type(tokenizer).__name__,
            "maximum_length": maximum_length,
            "sequence_length_buckets": list(lengths),
        },
    )


def load_bert_scorer(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.float32,
    compute_dtype: jnp.dtype = jnp.float32,
    sequence_length_buckets: Sequence[int] = (64, 128, 256, 512),
    **adapter_options: Any,
) -> tuple[BertScorer, Processor]:
    """Resolve one immutable scalar BERT artifact and paired-text processor."""

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
        allow_patterns=(
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ),
    )
    model = BertScorerCheckpointAdapter(**adapter_options).load(
        resolved.path,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        model_id=resolved.model_id,
        revision=resolved.revision,
    )
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "BERT scorer preprocessing requires representax[hf]"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(resolved.path, local_files_only=True)
    processor = _pair_processor(
        tokenizer,
        maximum_length=model.backbone.tower.config.max_position_embeddings,
        sequence_length_buckets=sequence_length_buckets,
    )
    return model, processor


__all__ = ["load_bert_scorer"]
