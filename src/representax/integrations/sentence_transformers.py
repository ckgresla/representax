"""Torch-free loading of standard Sentence Transformers dense artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import jax.numpy as jnp

from representax.core import EncoderMetadata, Modality, Route
from representax.inference import TextEmbeddingModel
from representax.models.bert import BertCheckpointAdapter
from representax.models.components import AttentionImplementation, Linear
from representax.models.mpnet import MPNetCheckpointAdapter
from representax.models.sentence import (
    POOLING_MODES,
    DenseActivation,
    PoolingMode,
    SentenceDense,
    SentenceEncoder,
    SentenceNormalize,
    SentencePooling,
    SentencePostprocessor,
)
from representax.planning import RematerializationPolicy

from .huggingface import (
    ResolvedHuggingFaceCheckpoint,
    load_hf_config,
    load_safetensor_subset,
    resolve_hf_checkpoint,
)

SENTENCE_TRANSFORMERS_ORACLE_VERSION = "5.6.1"
SimilarityFunction = Literal["cosine", "dot", "euclidean", "manhattan"]

_MODEL_ALLOW_PATTERNS = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.*",
    "merges.txt",
    "*.model",
    "*/config.json",
    "*/model.safetensors",
    "*/model.safetensors.index.json",
    "*/model-*.safetensors",
)

_LEGACY_POOLING_KEYS: tuple[tuple[str, PoolingMode], ...] = (
    ("pooling_mode_cls_token", "cls"),
    ("pooling_mode_max_tokens", "max"),
    ("pooling_mode_mean_tokens", "mean"),
    ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len_tokens"),
    ("pooling_mode_weightedmean_tokens", "weightedmean"),
    ("pooling_mode_lasttoken", "lasttoken"),
)

_DENSE_ACTIVATIONS: Mapping[str, DenseActivation] = MappingProxyType(
    {
        "torch.nn.Identity": "identity",
        "torch.nn.modules.linear.Identity": "identity",
        "torch.nn.Tanh": "tanh",
        "torch.nn.modules.activation.Tanh": "tanh",
        "torch.nn.ReLU": "relu",
        "torch.nn.modules.activation.ReLU": "relu",
        "torch.nn.GELU": "gelu",
        "torch.nn.modules.activation.GELU": "gelu",
        "torch.nn.SiLU": "silu",
        "torch.nn.modules.activation.SiLU": "silu",
    }
)


@dataclass(frozen=True, slots=True)
class SentenceTransformerModuleSpec:
    """One reviewed entry from an upstream ``modules.json`` graph."""

    index: int
    name: str
    path: str
    type: str

    @property
    def kind(self) -> str:
        return self.type.rsplit(".", 1)[-1]


@dataclass(frozen=True, slots=True)
class LoadedSentenceTransformer:
    """Native dense encoder plus host preprocessing metadata."""

    encoder: SentenceEncoder
    checkpoint: ResolvedHuggingFaceCheckpoint
    processor_path: Path
    max_sequence_length: int
    do_lower_case: bool
    prompts: Mapping[str, str]
    default_prompt_name: str | None
    similarity_function: SimilarityFunction


def _json_object(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if not required:
            return {}
        raise FileNotFoundError(f"Sentence Transformers metadata not found: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _module_directory(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"module path escapes checkpoint root: {relative!r}")
    if not path.is_dir():
        raise FileNotFoundError(
            f"Sentence Transformers module directory not found: {path}"
        )
    return path


def load_sentence_transformer_modules(
    checkpoint: str | Path,
) -> tuple[SentenceTransformerModuleSpec, ...]:
    """Parse and validate the static module graph without importing upstream."""

    root = Path(checkpoint)
    path = root / "modules.json"
    if not path.is_file():
        raise FileNotFoundError(f"Sentence Transformers modules not found: {path}")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise TypeError("modules.json must contain a JSON array")
    modules: list[SentenceTransformerModuleSpec] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("each modules.json entry must be an object")
        try:
            module = SentenceTransformerModuleSpec(
                index=int(row["idx"]),
                name=str(row["name"]),
                path=str(row["path"]),
                type=str(row["type"]),
            )
        except KeyError as error:
            raise ValueError(
                f"modules.json entry is missing {error.args[0]!r}"
            ) from error
        modules.append(module)
    modules.sort(key=lambda module: module.index)
    if tuple(module.index for module in modules) != tuple(range(len(modules))):
        raise ValueError("modules.json indices must be unique and contiguous")
    if len({module.name for module in modules}) != len(modules):
        raise ValueError("modules.json names must be unique")
    if not modules or modules[0].kind != "Transformer":
        raise ValueError("dense sentence models must begin with a Transformer module")
    return tuple(modules)


def _pooling_modes(config: Mapping[str, Any]) -> tuple[PoolingMode, ...]:
    configured = config.get("pooling_mode")
    if configured is None:
        modes = tuple(
            mode for key, mode in _LEGACY_POOLING_KEYS if bool(config.get(key, False))
        )
        return modes or ("mean",)
    values = (configured,) if isinstance(configured, str) else tuple(configured)
    modes = tuple(str(value) for value in values)
    invalid = tuple(mode for mode in modes if mode not in POOLING_MODES)
    if invalid:
        raise ValueError(f"unsupported pooling modes: {invalid}")
    return cast(tuple[PoolingMode, ...], modes)


def _load_pooling(
    directory: Path,
    *,
    input_dimension: int,
) -> SentencePooling:
    config = _json_object(directory / "config.json")
    configured_dimension = int(
        config.get("embedding_dimension", config.get("word_embedding_dimension", 0))
    )
    if configured_dimension != input_dimension:
        raise ValueError(
            "pooling input dimension does not match the native token backbone: "
            f"{configured_dimension} != {input_dimension}"
        )
    return SentencePooling(
        input_dimension=input_dimension,
        modes=_pooling_modes(config),
        include_prompt=bool(config.get("include_prompt", True)),
    )


def _dense_activation(config: Mapping[str, Any]) -> DenseActivation:
    configured = config.get(
        "activation_function",
        "torch.nn.modules.activation.Tanh",
    )
    if configured is None:
        return "identity"
    try:
        return _DENSE_ACTIVATIONS[str(configured)]
    except KeyError as error:
        raise ValueError(
            f"untrusted or unsupported Sentence Transformers Dense activation: "
            f"{configured!r}"
        ) from error


def _load_dense(
    directory: Path,
    *,
    input_dimension: int,
    parameter_dtype: jnp.dtype,
) -> SentenceDense:
    config = _json_object(directory / "config.json")
    configured_input = int(config["in_features"])
    output_dimension = int(config["out_features"])
    if configured_input != input_dimension:
        raise ValueError(
            "Dense input dimension does not match the preceding module: "
            f"{configured_input} != {input_dimension}"
        )
    if output_dimension <= 0:
        raise ValueError("Dense output dimension must be positive")
    input_name = str(config.get("module_input_name", "sentence_embedding"))
    output_name = str(config.get("module_output_name", input_name))
    if input_name != "sentence_embedding" or output_name != "sentence_embedding":
        raise ValueError("named Dense feature routing is not yet supported")
    has_bias = bool(config.get("bias", True))
    names = {"linear.weight"}
    if has_bias:
        names.add("linear.bias")
    state = load_safetensor_subset(directory, names, dtype=parameter_dtype)
    weight = state["linear.weight"]
    if weight.shape != (output_dimension, input_dimension):
        raise ValueError(
            f"Dense weight has shape {weight.shape}; expected "
            f"{(output_dimension, input_dimension)}"
        )
    bias = state.get("linear.bias")
    if bias is not None and bias.shape != (output_dimension,):
        raise ValueError(
            f"Dense bias has shape {bias.shape}; expected {(output_dimension,)}"
        )
    return SentenceDense(
        linear=Linear(weight=weight, bias=bias),
        activation=_dense_activation(config),
    )


def _text_backbone(
    checkpoint: Path,
    *,
    parameter_dtype: jnp.dtype,
    compute_dtype: jnp.dtype,
    attention_implementation: AttentionImplementation,
    rematerialization: RematerializationPolicy,
) -> Any:
    config = load_hf_config(checkpoint)
    model_type = str(config.get("model_type", ""))
    if model_type == "bert":
        return BertCheckpointAdapter(
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        ).load(
            checkpoint,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
        )
    if model_type == "mpnet":
        return MPNetCheckpointAdapter(
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        ).load(
            checkpoint,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
        )
    raise ValueError(
        f"Sentence Transformers backbone {model_type!r} is catalogued but not native"
    )


def _model_metadata(
    checkpoint: Path,
) -> tuple[
    Mapping[str, str],
    str | None,
    SimilarityFunction,
    int | None,
]:
    config = _json_object(
        checkpoint / "config_sentence_transformers.json",
        required=False,
    )
    model_type = str(config.get("model_type", "SentenceTransformer"))
    if model_type != "SentenceTransformer":
        raise ValueError(
            f"expected a dense SentenceTransformer artifact; found {model_type!r}"
        )
    raw_prompts = config.get("prompts", {})
    if not isinstance(raw_prompts, dict):
        raise TypeError("Sentence Transformers prompts must be an object")
    prompts = MappingProxyType(
        {str(name): str(prompt) for name, prompt in raw_prompts.items()}
    )
    default_prompt = config.get("default_prompt_name")
    if default_prompt is not None:
        default_prompt = str(default_prompt)
        if default_prompt not in prompts:
            raise ValueError("default_prompt_name does not name a saved prompt")
    similarity = str(config.get("similarity_fn_name") or "cosine")
    if similarity not in ("cosine", "dot", "euclidean", "manhattan"):
        raise ValueError(f"unsupported similarity function: {similarity!r}")
    truncate = config.get("truncate_dim")
    truncate_dimension = None if truncate is None else int(truncate)
    return (
        prompts,
        default_prompt,
        similarity,
        truncate_dimension,
    )


def _transformer_metadata(
    checkpoint: Path,
    hf_config: Mapping[str, Any],
) -> tuple[int, bool]:
    config = _json_object(
        checkpoint / "sentence_bert_config.json",
        required=False,
    )
    maximum = int(
        config.get(
            "max_seq_length",
            hf_config.get("max_position_embeddings", 0),
        )
    )
    if maximum <= 0:
        raise ValueError("unable to determine a positive maximum sequence length")
    return maximum, bool(config.get("do_lower_case", False))


def load_sentence_transformer_artifact(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    token: bool | str | None = None,
    parameter_dtype: jnp.dtype = jnp.float32,
    compute_dtype: jnp.dtype = jnp.float32,
    attention_implementation: AttentionImplementation = "xla",
    rematerialization: RematerializationPolicy = "none",
) -> LoadedSentenceTransformer:
    """Load a standard dense Sentence Transformers artifact into native modules."""

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
        token=token,
        allow_patterns=_MODEL_ALLOW_PATTERNS,
    )
    modules = load_sentence_transformer_modules(resolved.path)
    transformer = modules[0]
    transformer_directory = _module_directory(resolved.path, transformer.path)
    hf_config = load_hf_config(transformer_directory)
    backbone = _text_backbone(
        transformer_directory,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        attention_implementation=attention_implementation,
        rematerialization=rematerialization,
    )
    input_dimension = int(backbone.metadata.output_dimension)
    pooling: SentencePooling | None = None
    postprocessors: list[SentencePostprocessor] = []
    current_dimension = input_dimension
    for module in modules[1:]:
        if module.kind == "Pooling":
            directory = _module_directory(resolved.path, module.path)
            if pooling is not None or postprocessors:
                raise ValueError("Pooling must occur exactly once before dense modules")
            pooling = _load_pooling(directory, input_dimension=input_dimension)
            current_dimension = pooling.output_dimension
        elif module.kind == "Dense":
            directory = _module_directory(resolved.path, module.path)
            if pooling is None:
                raise ValueError("Dense cannot precede Pooling")
            dense = _load_dense(
                directory,
                input_dimension=current_dimension,
                parameter_dtype=parameter_dtype,
            )
            postprocessors.append(dense)
            current_dimension = dense.output_dimension
        elif module.kind == "Normalize":
            if pooling is None:
                raise ValueError("Normalize cannot precede Pooling")
            postprocessors.append(SentenceNormalize())
        else:
            raise ValueError(
                "unsupported or executable Sentence Transformers module: "
                f"{module.type!r}"
            )
    if pooling is None:
        raise ValueError("dense sentence models require exactly one Pooling module")

    prompts, default_prompt, similarity, truncate_dimension = _model_metadata(
        resolved.path
    )
    if truncate_dimension is not None:
        if not 0 < truncate_dimension <= current_dimension:
            raise ValueError("truncate_dim must index the final embedding dimension")
        output_dimension = truncate_dimension
    else:
        output_dimension = current_dimension
    encoder = SentenceEncoder(
        backbone=backbone,
        pooling=pooling,
        postprocessors=tuple(postprocessors),
        metadata=EncoderMetadata(
            model_id=resolved.model_id,
            revision=resolved.revision,
            output_dimension=output_dimension,
            routes=frozenset(Route),
            modalities=frozenset({Modality.TEXT}),
        ),
        truncate_dimension=truncate_dimension,
    )
    maximum, do_lower_case = _transformer_metadata(
        transformer_directory,
        hf_config,
    )
    return LoadedSentenceTransformer(
        encoder=encoder,
        checkpoint=resolved,
        processor_path=transformer_directory,
        max_sequence_length=maximum,
        do_lower_case=do_lower_case,
        prompts=prompts,
        default_prompt_name=default_prompt,
        similarity_function=similarity,
    )


def load_sentence_transformer(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    token: bool | str | None = None,
    parameter_dtype: jnp.dtype = jnp.float32,
    compute_dtype: jnp.dtype = jnp.float32,
    attention_implementation: AttentionImplementation = "xla",
    rematerialization: RematerializationPolicy = "none",
    processor: Any | None = None,
) -> TextEmbeddingModel:
    """Load a dense Sentence Transformers artifact and its host tokenizer."""

    loaded = load_sentence_transformer_artifact(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
        token=token,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        attention_implementation=attention_implementation,
        rematerialization=rematerialization,
    )
    if processor is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "text preprocessing requires `pip install representax[hf]`"
            ) from error
        processor_kwargs: dict[str, Any] = {"local_files_only": True}
        if loaded.do_lower_case:
            processor_kwargs["do_lower_case"] = True
        processor = AutoTokenizer.from_pretrained(
            loaded.processor_path,
            **processor_kwargs,
        )
    return TextEmbeddingModel(
        encoder=loaded.encoder,
        processor=processor,
        max_sequence_length=loaded.max_sequence_length,
        prompts=loaded.prompts,
        default_prompt_name=loaded.default_prompt_name,
        similarity_function=loaded.similarity_function,
    )


def load_sentence_transformer_encoder(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    parameter_dtype: str = "float32",
    compute_dtype: str = "float32",
    attention_implementation: AttentionImplementation = "xla",
    rematerialization: RematerializationPolicy = "none",
) -> SentenceEncoder:
    """Construct a native encoder from fully serializable job parameters."""

    return load_sentence_transformer_artifact(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        parameter_dtype=jnp.dtype(parameter_dtype),
        compute_dtype=jnp.dtype(compute_dtype),
        attention_implementation=attention_implementation,
        rematerialization=rematerialization,
    ).encoder


class SentenceTextCollator:
    """Tokenize text into one native, fixed-shape sentence-model batch."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        maximum_length: int,
    ) -> None:
        if maximum_length <= 0:
            raise ValueError("maximum_length must be positive")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        config = load_hf_config(self.checkpoint)
        self.model_type = str(config.get("model_type", ""))
        if self.model_type not in {"bert", "mpnet", "qwen3_vl_audio"}:
            raise ValueError(
                f"native text collation does not support {self.model_type!r}"
            )
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "sentence-pair preprocessing requires `pip install representax[hf]`"
            ) from error
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            local_files_only=True,
            trust_remote_code=self.model_type == "qwen3_vl_audio",
        )
        self.maximum_length = maximum_length

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-sentence-text-collator-v1",
            "checkpoint": str(self.checkpoint),
            "model_type": self.model_type,
            "maximum_length": self.maximum_length,
        }

    def __call__(self, texts: Sequence[str]) -> Any:
        tokenizer = cast(Callable[..., Any], self.tokenizer)
        encoded = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.maximum_length,
            return_tensors="np",
        )
        input_ids = jnp.asarray(encoded["input_ids"])
        attention_mask = jnp.asarray(encoded["attention_mask"])
        if self.model_type == "bert":
            from representax.models.bert import BertBatch

            token_type_ids = encoded.get("token_type_ids")
            return BertBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=(
                    None if token_type_ids is None else jnp.asarray(token_type_ids)
                ),
            )
        if self.model_type == "qwen3_vl_audio":
            from representax.models.jina_v5 import JinaV5TextBatch

            return JinaV5TextBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        from representax.models.mpnet import MPNetBatch

        return MPNetBatch(input_ids=input_ids, attention_mask=attention_mask)


class SentencePairCollator:
    """Tokenize raw labeled sentence pairs into one native static-shape batch."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        maximum_length: int,
        left_field: str = "sentence1",
        right_field: str = "sentence2",
        label_field: str = "score",
        pad_to_size: int | None = None,
    ) -> None:
        self._text = SentenceTextCollator(
            checkpoint,
            maximum_length=maximum_length,
        )
        self.left_field = left_field
        self.right_field = right_field
        self.label_field = label_field
        if pad_to_size is not None and pad_to_size <= 0:
            raise ValueError("pad_to_size must be positive or None")
        self.pad_to_size = pad_to_size

    def data_contract(self) -> Mapping[str, Any]:
        """Return stable state incorporated into Grain resume fingerprints."""

        return {
            **self._text.data_contract(),
            "schema_version": "representax-sentence-pair-collator-v1",
            "left_field": self.left_field,
            "right_field": self.right_field,
            "label_field": self.label_field,
            "pad_to_size": self.pad_to_size,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Any:
        from representax.tasks.pairwise import pairwise_batch

        try:
            left = tuple(str(example[self.left_field]) for example in examples)
            right = tuple(str(example[self.right_field]) for example in examples)
            labels = tuple(float(example[self.label_field]) for example in examples)
        except KeyError as error:
            raise KeyError(
                f"sentence-pair record is missing field {error.args[0]!r}"
            ) from error
        valid = [True] * len(labels)
        if self.pad_to_size is not None:
            if len(labels) > self.pad_to_size:
                raise ValueError("sentence-pair batch exceeds pad_to_size")
            padding = self.pad_to_size - len(labels)
            left = (*left, *("" for _ in range(padding)))
            right = (*right, *("" for _ in range(padding)))
            labels = (*labels, *(0.0 for _ in range(padding)))
            valid.extend(False for _ in range(padding))
        return pairwise_batch(
            left=self._text(left),
            right=self._text(right),
            labels=jnp.asarray(labels, dtype=jnp.float32),
            valid=jnp.asarray(valid),
        )


class RetrievalPairCollator:
    """Tokenize aligned query/positive rows for exact in-batch-negative MNR."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        maximum_length: int,
        query_field: str = "query",
        document_field: str = "positive",
    ) -> None:
        self._text = SentenceTextCollator(
            checkpoint,
            maximum_length=maximum_length,
        )
        self.query_field = query_field
        self.document_field = document_field

    def data_contract(self) -> Mapping[str, Any]:
        """Return stable state incorporated into Grain resume fingerprints."""

        return {
            **self._text.data_contract(),
            "schema_version": "representax-retrieval-pair-collator-v1",
            "query_field": self.query_field,
            "document_field": self.document_field,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Any:
        from representax.tasks.retrieval import retrieval_batch

        try:
            queries = tuple(str(example[self.query_field]) for example in examples)
            documents = tuple(str(example[self.document_field]) for example in examples)
        except KeyError as error:
            raise KeyError(
                f"retrieval-pair record is missing field {error.args[0]!r}"
            ) from error
        size = len(examples)
        return retrieval_batch(
            query=self._text(queries),
            document=self._text(documents),
            positive_mask=jnp.eye(size, dtype=jnp.bool_),
        )


__all__ = [
    "LoadedSentenceTransformer",
    "RetrievalPairCollator",
    "SENTENCE_TRANSFORMERS_ORACLE_VERSION",
    "SentencePairCollator",
    "SentenceTextCollator",
    "SentenceTransformerModuleSpec",
    "SimilarityFunction",
    "load_sentence_transformer",
    "load_sentence_transformer_artifact",
    "load_sentence_transformer_encoder",
    "load_sentence_transformer_modules",
]
