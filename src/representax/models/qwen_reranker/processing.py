"""Checkpoint-authored finite preprocessing for Qwen text rerankers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.processing import Processor, select_static_shape_bucket

from .config import QwenRerankerConfig
from .model import QwenRerankerBatch


def _messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, Mapping):
        try:
            query = str(value["query"])
            document = str(value["document"])
        except KeyError as error:
            raise KeyError(
                "Qwen reranker samples require query and document fields"
            ) from error
        messages = []
        instruction = value.get("instruction", value.get("prompt"))
        if instruction is not None:
            messages.append({"role": "system", "content": str(instruction)})
        messages.extend(
            (
                {"role": "query", "content": query},
                {"role": "document", "content": document},
            )
        )
        return messages
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
    ):
        return [
            {"role": "query", "content": str(value[0])},
            {"role": "document", "content": str(value[1])},
        ]
    raise TypeError("Qwen reranker samples must be (query, document) pairs or mappings")


def make_qwen_reranker_processor(
    checkpoint: str | Path,
    config: QwenRerankerConfig,
    *,
    sequence_length_buckets: Sequence[int] = (512, 2048, 8192, 32768),
) -> Processor:
    """Load the tokenizer and serialized chat template for one reranker."""

    try:
        transformers = import_module("transformers")
    except ImportError as error:
        raise ImportError(
            "Qwen reranking requires `pip install representax[hf]`"
        ) from error
    tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
        checkpoint,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    maximum = max(sequence_length_buckets)

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> QwenRerankerBatch:
        del route, seed
        if not artifacts:
            raise ValueError("Qwen reranker batches must be non-empty")
        rendered = [
            tokenizer.apply_chat_template(
                _messages(value),
                tokenize=False,
                add_generation_prompt=False,
            )
            for value in artifacts
        ]
        features = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=maximum,
            return_attention_mask=True,
            return_tensors="np",
        )
        input_ids = np.asarray(features["input_ids"], dtype=np.int32)
        attention_mask = np.asarray(features["attention_mask"], dtype=np.int32)
        sequence_bucket = select_static_shape_bucket(
            (input_ids.shape[1],),
            tuple((value,) for value in sequence_length_buckets),
        )[0]
        padding = sequence_bucket - input_ids.shape[1]
        input_ids = np.pad(
            input_ids,
            ((0, 0), (padding, 0)),
            constant_values=config.pad_token_id,
        )
        attention_mask = np.pad(attention_mask, ((0, 0), (padding, 0)))
        if not np.all(attention_mask[:, -1]):
            raise ValueError("reranker formatting must leave the final token valid")
        return QwenRerankerBatch(
            input_ids=jnp.asarray(input_ids),
            attention_mask=jnp.asarray(attention_mask),
        )

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-qwen-reranker-processor-v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "generation": config.generation,
            "sequence_length_buckets": list(sequence_length_buckets),
            "padding_side": "left",
            "true_token_id": config.true_token_id,
            "false_token_id": config.false_token_id,
        },
    )


__all__ = ["make_qwen_reranker_processor"]
