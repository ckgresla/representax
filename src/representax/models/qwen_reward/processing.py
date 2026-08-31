"""Checkpoint-associated text preprocessing for Qwen reward models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.processing import Processor, select_static_shape_bucket
from representax.models.qwen_reranker import QwenRerankerBatch


def _render(tokenizer: Any, value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and "messages" in value:
        value = value["messages"]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return str(
            tokenizer.apply_chat_template(
                list(value), tokenize=False, add_generation_prompt=False
            )
        )
    raise TypeError("reward artifacts must be text or chat-message sequences")


def make_qwen_reward_processor(
    checkpoint: str | Path,
    pad_token_id: int,
    *,
    sequence_length_buckets: Sequence[int] = (512, 2048, 8192),
) -> Processor:
    try:
        transformers = import_module("transformers")
    except ImportError as error:
        raise ImportError("Qwen reward processing requires representax[hf]") from error
    tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = pad_token_id
    maximum = max(sequence_length_buckets)

    def process(artifacts: Sequence[Any], *, route: Route, seed: int | None):
        del route, seed
        if not artifacts:
            raise ValueError("reward batches must be non-empty")
        features = tokenizer(
            [_render(tokenizer, value) for value in artifacts],
            padding=True,
            truncation=True,
            max_length=maximum,
            return_attention_mask=True,
            return_tensors="np",
        )
        input_ids = np.asarray(features["input_ids"], dtype=np.int32)
        attention_mask = np.asarray(features["attention_mask"], dtype=np.int32)
        bucket = select_static_shape_bucket(
            (input_ids.shape[1],), tuple((size,) for size in sequence_length_buckets)
        )[0]
        padding = bucket - input_ids.shape[1]
        return QwenRerankerBatch(
            input_ids=jnp.asarray(
                np.pad(input_ids, ((0, 0), (0, padding)), constant_values=pad_token_id)
            ),
            attention_mask=jnp.asarray(np.pad(attention_mask, ((0, 0), (0, padding)))),
        )

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-qwen-reward-processor-v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "padding_side": "right",
            "sequence_length_buckets": list(sequence_length_buckets),
        },
    )


__all__ = ["make_qwen_reward_processor"]
