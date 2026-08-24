"""Torch-free import and export of PyLate-compatible ColBERT artifacts."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import AttentionImplementation, Linear
from representax.models.late_interaction import LateInteractionTextEncoder
from representax.models.modernvbert import (
    ModernVBERTTextCheckpointAdapter,
    ModernVBERTTextEncoder,
)
from representax.models.processing import Processor, select_static_shape_bucket
from representax.planning import RematerializationPolicy

from .huggingface import (
    load_hf_config,
    load_safetensor_subset,
    resolve_hf_checkpoint,
)
from .sentence_transformers import load_sentence_transformer_modules

GTE_MODERN_COLBERT_MODEL_ID = "lightonai/GTE-ModernColBERT-v1"
GTE_MODERN_COLBERT_REVISION = "cbbe53366e564450558f5e639dd499171f127538"

_ALLOW_PATTERNS = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.*",
    "merges.txt",
    "*/config.json",
    "*/model.safetensors",
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _token_ids(checkpoint: Path, tokens: Sequence[str]) -> tuple[int, ...]:
    """Resolve literal checkpoint tokens without importing a tokenizer runtime."""

    serialized = _json_object(checkpoint / "tokenizer.json")
    model = serialized.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise TypeError("tokenizer.json must contain model.vocab")
    vocabulary = {str(token): int(index) for token, index in model["vocab"].items()}
    added = serialized.get("added_tokens", ())
    if not isinstance(added, list):
        raise TypeError("tokenizer.json added_tokens must be a list")
    for entry in added:
        if not isinstance(entry, dict) or "content" not in entry or "id" not in entry:
            raise TypeError("tokenizer.json contains an invalid added token")
        vocabulary[str(entry["content"])] = int(entry["id"])
    try:
        return tuple(vocabulary[token] for token in tokens)
    except KeyError as error:
        raise KeyError(f"tokenizer vocabulary is missing {error.args[0]!r}") from error


@dataclass(frozen=True)
class LateInteractionCheckpointAdapter:
    """Bidirectional mapping for a ModernBERT plus PyLate Dense graph."""

    model_id: str = GTE_MODERN_COLBERT_MODEL_ID
    revision: str = GTE_MODERN_COLBERT_REVISION

    def load(
        self,
        checkpoint: str | Path,
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
    ) -> LateInteractionTextEncoder:
        root = Path(checkpoint)
        modules = load_sentence_transformer_modules(root)
        if len(modules) != 2 or modules[0].path or modules[1].path != "1_Dense":
            raise ValueError(
                "native late interaction currently requires Transformer -> 1_Dense"
            )
        if modules[1].type.rsplit(".", 1)[-1] != "Dense":
            raise ValueError("late-interaction output module must be Dense")
        config = load_hf_config(root)
        if config.get("model_type") != "modernbert":
            raise ValueError("the first native late-interaction backbone is ModernBERT")
        backbone = ModernVBERTTextCheckpointAdapter(
            model_id=self.model_id,
            revision=self.revision,
            weight_prefix="",
        ).load(
            root,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )
        dense_config = _json_object(root / "1_Dense" / "config.json")
        input_size = int(dense_config["in_features"])
        output_size = int(dense_config["out_features"])
        if input_size != backbone.metadata.output_dimension:
            raise ValueError(
                "late-interaction projection input does not match backbone"
            )
        if bool(dense_config.get("bias", False)):
            raise ValueError("biased late-interaction projection is not yet supported")
        dense = load_safetensor_subset(
            root / "1_Dense",
            {"linear.weight"},
            dtype=parameter_dtype,
        )
        weight = dense["linear.weight"]
        if weight.shape != (output_size, input_size):
            raise ValueError("late-interaction projection has an invalid shape")
        metadata = _json_object(root / "config_sentence_transformers.json")
        skip_words = metadata.get("skiplist_words", [])
        if not isinstance(skip_words, list) or not all(
            isinstance(word, str) for word in skip_words
        ):
            raise TypeError("skiplist_words must be a list of strings")
        skip_ids = _token_ids(root, tuple(skip_words))
        return LateInteractionTextEncoder(
            backbone=backbone,
            projection=Linear(weight=weight),
            metadata=EncoderMetadata(
                model_id=self.model_id,
                revision=self.revision,
                output_dimension=output_size,
                routes=frozenset({Route.QUERY, Route.DOCUMENT}),
                modalities=frozenset({Modality.TEXT}),
            ),
            skip_token_ids=skip_ids,
            query_expansion=bool(metadata.get("do_query_expansion", True)),
        )

    def state_dict(self, model: LateInteractionTextEncoder) -> dict[str, jax.Array]:
        if not isinstance(model.backbone, ModernVBERTTextEncoder):
            raise TypeError("this checkpoint adapter requires a ModernVBERT backbone")
        backbone = ModernVBERTTextCheckpointAdapter(
            model_id=self.model_id,
            revision=self.revision,
            weight_prefix="",
        ).state_dict(model.backbone)
        return {
            **backbone,
            "1_Dense/linear.weight": model.projection.weight,
        }

    def save(self, model: LateInteractionTextEncoder, directory: str | Path) -> Path:
        from safetensors.numpy import save_file

        target = Path(directory)
        state = self.state_dict(model)
        save_file(
            {
                name: np.asarray(jax.device_get(value))
                for name, value in state.items()
                if not name.startswith("1_Dense/")
            },
            target / "model.safetensors",
        )
        dense = target / "1_Dense"
        dense.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "linear.weight": np.asarray(
                    jax.device_get(state["1_Dense/linear.weight"])
                )
            },
            dense / "model.safetensors",
        )
        return target


def _insert_prefix(values: np.ndarray, prefix: int) -> np.ndarray:
    prefix_values = np.full((values.shape[0], 1), prefix, dtype=values.dtype)
    return np.concatenate((values[:, :1], prefix_values, values[:, 1:]), axis=1)


def _make_processor(
    checkpoint: Path,
    *,
    model: LateInteractionTextEncoder,
    tokenizer: Any | None,
    query_sequence_length_buckets: Sequence[int] | None,
    document_sequence_length_buckets: Sequence[int] | None,
) -> Processor:
    metadata = _json_object(checkpoint / "config_sentence_transformers.json")
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "text preprocessing requires `pip install representax[hf]`"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = cast(Any, tokenizer)
    if tokenizer.mask_token_id is not None:
        tokenizer.pad_token_id = tokenizer.mask_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("late-interaction tokenizer requires a pad or mask token")

    query_length = int(metadata.get("query_length", 32))
    document_length = int(metadata.get("document_length", 180))
    query_buckets = tuple(query_sequence_length_buckets or (query_length,))
    document_buckets = tuple(document_sequence_length_buckets or (document_length,))
    if max(query_buckets) > query_length or max(document_buckets) > document_length:
        raise ValueError("late-interaction buckets cannot exceed checkpoint limits")
    prefixes = {
        Route.QUERY: str(metadata.get("query_prefix", "[Q] ")),
        Route.DOCUMENT: str(metadata.get("document_prefix", "[D] ")),
    }
    prefix_ids = {
        route: (None if not prefix else int(tokenizer.convert_tokens_to_ids(prefix)))
        for route, prefix in prefixes.items()
    }
    if any(
        prefix and prefix_id == tokenizer.unk_token_id
        for (prefix, prefix_id) in zip(
            prefixes.values(), prefix_ids.values(), strict=True
        )
    ):
        raise ValueError("late-interaction prefix is not an atomic tokenizer token")
    skip_words = tuple(metadata.get("skiplist_words", ()))
    skip_ids = tuple(int(tokenizer.convert_tokens_to_ids(word)) for word in skip_words)
    if skip_ids != model.skip_token_ids:
        raise ValueError("tokenizer skip-list IDs disagree with the loaded model")

    def process(
        artifacts: Sequence[str],
        *,
        route: Route,
        seed: int | None,
    ) -> Any:
        del seed
        if route not in (Route.QUERY, Route.DOCUMENT):
            raise ValueError("late-interaction text requires query or document route")
        texts = tuple(artifacts)
        if not texts or any(not isinstance(text, str) for text in texts):
            raise TypeError("late-interaction text processor requires strings")
        maximum = query_length if route is Route.QUERY else document_length
        buckets = query_buckets if route is Route.QUERY else document_buckets
        prefix_id = prefix_ids[route]
        encoded = tokenizer(
            list(texts),
            padding=(
                "max_length" if route is Route.QUERY and model.query_expansion else True
            ),
            truncation=True,
            max_length=maximum - (prefix_id is not None),
            return_tensors="np",
        )
        arrays = {name: np.asarray(value) for name, value in encoded.items()}
        if prefix_id is not None:
            arrays["input_ids"] = _insert_prefix(arrays["input_ids"], prefix_id)
            arrays["attention_mask"] = _insert_prefix(arrays["attention_mask"], 1)
            if "token_type_ids" in arrays:
                arrays["token_type_ids"] = _insert_prefix(arrays["token_type_ids"], 0)
        if route is Route.QUERY and bool(
            metadata.get("attend_to_expansion_tokens", False)
        ):
            arrays["attention_mask"].fill(1)
        length = arrays["input_ids"].shape[1]
        bucket = select_static_shape_bucket(
            (length,), tuple((value,) for value in buckets)
        )[0]
        for name, value in tuple(arrays.items()):
            if value.ndim == 2:
                fill = int(tokenizer.pad_token_id) if name == "input_ids" else 0
                arrays[name] = np.pad(
                    value,
                    ((0, 0), (0, bucket - length)),
                    constant_values=fill,
                )
        builder = getattr(type(model.backbone), "make_batch", None)
        if not callable(builder):
            raise TypeError("late-interaction backbone must provide make_batch")
        return builder(
            input_ids=arrays["input_ids"],
            attention_mask=arrays["attention_mask"],
            token_type_ids=arrays.get("token_type_ids"),
        )

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-late-interaction-text-v1",
            "model_id": model.metadata.model_id,
            "revision": model.metadata.revision,
            "query_length_buckets": list(query_buckets),
            "document_length_buckets": list(document_buckets),
            "query_prefix": prefixes[Route.QUERY],
            "document_prefix": prefixes[Route.DOCUMENT],
            "do_query_expansion": model.query_expansion,
            "attend_to_expansion_tokens": bool(
                metadata.get("attend_to_expansion_tokens", False)
            ),
            "skiplist_words": list(skip_words),
        },
    )


def load_late_interaction_text_model(
    model_name_or_path: str | Path = GTE_MODERN_COLBERT_MODEL_ID,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    token: bool | str | None = None,
    parameter_dtype: str | jnp.dtype = jnp.float32,
    compute_dtype: str | jnp.dtype = jnp.float32,
    attention_implementation: AttentionImplementation = "xla",
    rematerialization: RematerializationPolicy = "full",
    tokenizer: Any | None = None,
    query_sequence_length_buckets: Sequence[int] | None = None,
    document_sequence_length_buckets: Sequence[int] | None = None,
) -> tuple[LateInteractionTextEncoder, Processor]:
    """Load one real PyLate artifact into a native model and processor."""

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
        token=token,
        allow_patterns=_ALLOW_PATTERNS,
    )
    adapter = LateInteractionCheckpointAdapter(
        model_id=resolved.model_id,
        revision=resolved.revision,
    )
    # Token IDs are tokenizer-owned. Build the graph first and then attach the
    # immutable IDs while constructing its associated host processor.
    model = adapter.load(
        resolved.path,
        parameter_dtype=jnp.dtype(parameter_dtype),
        compute_dtype=jnp.dtype(compute_dtype),
        attention_implementation=attention_implementation,
        rematerialization=rematerialization,
    )
    processor = _make_processor(
        resolved.path,
        model=model,
        tokenizer=tokenizer,
        query_sequence_length_buckets=query_sequence_length_buckets,
        document_sequence_length_buckets=document_sequence_length_buckets,
    )
    return model, processor


def save_late_interaction_text_model(
    model: LateInteractionTextEncoder,
    directory: str | Path,
    *,
    source_checkpoint: str | Path,
) -> Path:
    """Write and exactly reload one PyLate-compatible artifact."""

    source = Path(source_checkpoint).resolve()
    target = Path(directory).resolve()
    if source == target:
        raise ValueError("late-interaction export must not overwrite its source")
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("*.safetensors", "*.bin"),
    )
    adapter = LateInteractionCheckpointAdapter(
        model_id=model.metadata.model_id,
        revision=model.metadata.revision,
    )
    adapter.save(model, target)
    restored = adapter.load(target)
    expected = adapter.state_dict(model)
    actual = adapter.state_dict(restored)
    for name in expected:
        if not np.array_equal(
            np.asarray(jax.device_get(expected[name])),
            np.asarray(jax.device_get(actual[name])),
        ):
            raise ValueError(f"late-interaction export failed exact reload: {name}")
    return target


__all__ = [
    "GTE_MODERN_COLBERT_MODEL_ID",
    "GTE_MODERN_COLBERT_REVISION",
    "LateInteractionCheckpointAdapter",
    "load_late_interaction_text_model",
    "save_late_interaction_text_model",
]
