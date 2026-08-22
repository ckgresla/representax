"""Import standard Sentence Transformers dense artifacts into native modules.

This module understands the upstream serialization format. It does not provide
an alternate training runtime: models, processors, task collation, and compiled
execution remain ordinary Representax components after import.
"""

from __future__ import annotations

import json
from ast import literal_eval
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import jax.numpy as jnp

from representax.core import EncoderMetadata, Modality, Route
from representax.inference import TextEmbeddingModel
from representax.models.bert import BertCheckpointAdapter
from representax.models.components import AttentionImplementation, Linear
from representax.models.mpnet import MPNetCheckpointAdapter
from representax.models.processing import make_text_processor
from representax.models.sentence import (
    POOLING_MODES,
    DenseActivation,
    PoolingMode,
    SentenceBatch,
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

    @property
    def role(self) -> SentenceTransformerModuleRole:
        """Return the static role without importing the serialized class."""

        if not self.type.startswith("sentence_transformers."):
            return SentenceTransformerModuleRole.CUSTOM
        return _STANDARD_MODULE_ROLES.get(
            self.kind,
            SentenceTransformerModuleRole.CUSTOM,
        )


class SentenceTransformerModuleRole(StrEnum):
    """Non-executable roles understood in an upstream module graph."""

    TRANSFORMER = "transformer"
    POOLING = "pooling"
    DENSE = "dense"
    NORMALIZE = "normalize"
    LOGIT_SCORE = "logit_score"
    LEGACY_CLIP = "legacy_clip"
    ROUTER = "router"
    CUSTOM = "custom"


class SentenceTransformerGraphKind(StrEnum):
    """Reviewed Sentence Transformers composition forms."""

    DENSE_EMBEDDING = "dense_embedding"
    DIRECT_EMBEDDING = "direct_embedding"
    GENERATIVE_RERANKER = "generative_reranker"
    FEATURE_RERANKER = "feature_reranker"
    LEGACY_CLIP = "legacy_clip"
    ROUTED_ENCODER = "routed_encoder"


_STANDARD_MODULE_ROLES: Mapping[str, SentenceTransformerModuleRole] = MappingProxyType(
    {
        "Transformer": SentenceTransformerModuleRole.TRANSFORMER,
        "Pooling": SentenceTransformerModuleRole.POOLING,
        "Dense": SentenceTransformerModuleRole.DENSE,
        "Normalize": SentenceTransformerModuleRole.NORMALIZE,
        "LogitScore": SentenceTransformerModuleRole.LOGIT_SCORE,
        "CLIPModel": SentenceTransformerModuleRole.LEGACY_CLIP,
        "Router": SentenceTransformerModuleRole.ROUTER,
    }
)


@dataclass(frozen=True, slots=True)
class SentenceTransformerInputSpec:
    """One serialized input form and the atomic modalities it composes."""

    name: str
    modalities: tuple[Modality, ...]
    method: str
    output_name: str
    format: str | None = None


@dataclass(frozen=True, slots=True)
class SentenceTransformerRouteMapping:
    """One task/modality selector targeting a serialized router branch."""

    task: str | None
    modalities: tuple[Modality, ...]
    route: str


@dataclass(frozen=True, slots=True)
class SentenceTransformerRouteSpec:
    """One ordered, statically inspected branch of an upstream Router."""

    name: str
    modules: tuple[SentenceTransformerModuleSpec, ...]


@dataclass(frozen=True, slots=True)
class SentenceTransformerGraphSpec:
    """Torch-free description of an upstream Sentence Transformers graph.

    This records serialization structure only. It never imports checkpoint
    Python and therefore does not imply that a native backbone is implemented.
    """

    kind: SentenceTransformerGraphKind
    model_type: str
    transformer_task: str | None
    module_output_name: str | None
    modules: tuple[SentenceTransformerModuleSpec, ...]
    inputs: tuple[SentenceTransformerInputSpec, ...]
    routes: tuple[SentenceTransformerRouteSpec, ...] = ()
    route_mappings: tuple[SentenceTransformerRouteMapping, ...] = ()
    default_route: str | None = None
    allow_empty_key: bool | None = None

    @property
    def modalities(self) -> frozenset[Modality]:
        """Return every atomic modality named by inputs or route selectors."""

        return frozenset(
            modality
            for spec in (*self.inputs, *self.route_mappings)
            for modality in spec.modalities
        )


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
    if not modules:
        raise ValueError("modules.json must contain at least one module")
    return tuple(modules)


def _atomic_modalities(name: str) -> tuple[Modality, ...]:
    if name == "message":
        # Messages are structured containers, not an additional modality.
        return ()
    values = tuple(part.strip() for part in name.split("+") if part.strip())
    if not values:
        raise ValueError("modality names must be non-empty")
    return tuple(Modality(value) for value in values)


def _input_specs(
    config: Mapping[str, Any],
    *,
    legacy_clip: bool,
) -> tuple[SentenceTransformerInputSpec, ...]:
    raw = config.get("modality_config")
    if raw is None:
        names = ("text", "image") if legacy_clip else ("text",)
        return tuple(
            SentenceTransformerInputSpec(
                name=name,
                modalities=(Modality(name),),
                method="forward",
                output_name=str(config.get("module_output_name", "token_embeddings")),
            )
            for name in names
        )
    if not isinstance(raw, dict):
        raise TypeError("modality_config must be a JSON object")
    inputs: list[SentenceTransformerInputSpec] = []
    for raw_name, raw_spec in raw.items():
        name = str(raw_name)
        if not isinstance(raw_spec, dict):
            raise TypeError(f"modality_config[{name!r}] must be a JSON object")
        inputs.append(
            SentenceTransformerInputSpec(
                name=name,
                modalities=_atomic_modalities(name),
                method=str(raw_spec.get("method", "forward")),
                output_name=str(
                    raw_spec.get(
                        "method_output_name",
                        raw_spec.get(
                            "output_name",
                            config.get("module_output_name", "token_embeddings"),
                        ),
                    )
                ),
                format=(
                    None if raw_spec.get("format") is None else str(raw_spec["format"])
                ),
            )
        )
    if not inputs:
        raise ValueError("modality_config must contain at least one input form")
    return tuple(inputs)


def _classify_graph(
    modules: Sequence[SentenceTransformerModuleSpec],
    *,
    model_type: str,
    module_output_name: str | None,
) -> SentenceTransformerGraphKind:
    roles = tuple(module.role for module in modules)
    first, *suffix = roles
    if first is SentenceTransformerModuleRole.ROUTER:
        if any(
            role
            not in {
                SentenceTransformerModuleRole.POOLING,
                SentenceTransformerModuleRole.DENSE,
                SentenceTransformerModuleRole.NORMALIZE,
            }
            for role in suffix
        ):
            raise ValueError(f"unsupported or executable routed graph roles: {roles}")
        return SentenceTransformerGraphKind.ROUTED_ENCODER
    if first is SentenceTransformerModuleRole.LEGACY_CLIP:
        if any(role is not SentenceTransformerModuleRole.NORMALIZE for role in suffix):
            raise ValueError(
                f"unsupported or executable legacy CLIP graph roles: {roles}"
            )
        return SentenceTransformerGraphKind.LEGACY_CLIP
    if first not in {
        SentenceTransformerModuleRole.TRANSFORMER,
        SentenceTransformerModuleRole.CUSTOM,
    }:
        raise ValueError(
            f"unsupported or executable Sentence Transformers graph roles: {roles}"
        )
    if suffix == [SentenceTransformerModuleRole.LOGIT_SCORE]:
        return SentenceTransformerGraphKind.GENERATIVE_RERANKER
    if SentenceTransformerModuleRole.POOLING in suffix:
        if suffix[0] is not SentenceTransformerModuleRole.POOLING or any(
            role
            not in {
                SentenceTransformerModuleRole.DENSE,
                SentenceTransformerModuleRole.NORMALIZE,
            }
            for role in suffix[1:]
        ):
            raise ValueError(f"unsupported or executable pooled graph roles: {roles}")
        if model_type == "CrossEncoder":
            return SentenceTransformerGraphKind.FEATURE_RERANKER
        if model_type != "SentenceTransformer":
            raise ValueError(f"unsupported pooled model type: {model_type!r}")
        return SentenceTransformerGraphKind.DENSE_EMBEDDING
    if module_output_name == "sentence_embedding" and all(
        role
        in {
            SentenceTransformerModuleRole.DENSE,
            SentenceTransformerModuleRole.NORMALIZE,
        }
        for role in suffix
    ):
        return SentenceTransformerGraphKind.DIRECT_EMBEDDING
    raise ValueError(
        f"unsupported or executable Sentence Transformers graph roles: {roles}"
    )


def _router_mapping(
    key: str,
    route: str,
    *,
    route_names: frozenset[str],
) -> SentenceTransformerRouteMapping:
    try:
        selector = literal_eval(key)
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"invalid Router selector: {key!r}") from error
    if not isinstance(selector, tuple) or len(selector) != 2:
        raise ValueError(f"Router selector must be a (task, modality) pair: {key!r}")
    task, modality = selector
    if task is not None and not isinstance(task, str):
        raise TypeError("Router task selectors must be strings or null")
    if modality is None:
        modalities: tuple[Modality, ...] = ()
    elif isinstance(modality, str):
        modalities = _atomic_modalities(modality)
    elif isinstance(modality, tuple) and all(
        isinstance(value, str) for value in modality
    ):
        modalities = tuple(Modality(value) for value in modality)
    else:
        raise TypeError("Router modality selectors must be strings, tuples, or null")
    if route not in route_names:
        raise ValueError(f"Router selector targets unknown route {route!r}")
    return SentenceTransformerRouteMapping(
        task=task,
        modalities=modalities,
        route=route,
    )


def _router_spec(
    root: Path,
    module: SentenceTransformerModuleSpec,
) -> tuple[
    tuple[SentenceTransformerRouteSpec, ...],
    tuple[SentenceTransformerRouteMapping, ...],
    str | None,
    bool,
]:
    directory = _module_directory(root, module.path)
    config = _json_object(directory / "router_config.json")
    types = config.get("types")
    structure = config.get("structure")
    parameters = config.get("parameters", {})
    if not isinstance(types, dict) or not isinstance(structure, dict):
        raise TypeError("Router config requires types and structure objects")
    if not isinstance(parameters, dict):
        raise TypeError("Router parameters must be a JSON object")
    routes: list[SentenceTransformerRouteSpec] = []
    used: set[str] = set()
    for raw_name, raw_module_ids in structure.items():
        name = str(raw_name)
        if not isinstance(raw_module_ids, list) or not raw_module_ids:
            raise ValueError(f"Router route {name!r} must contain modules")
        route_modules: list[SentenceTransformerModuleSpec] = []
        for index, raw_module_id in enumerate(raw_module_ids):
            module_id = str(raw_module_id)
            try:
                module_type = str(types[module_id])
            except KeyError as error:
                raise ValueError(
                    f"Router route {name!r} references unknown module {module_id!r}"
                ) from error
            used.add(module_id)
            module_directory = _module_directory(directory, module_id)
            route_modules.append(
                SentenceTransformerModuleSpec(
                    index=index,
                    name=module_id,
                    path=str(module_directory.relative_to(root.resolve())),
                    type=module_type,
                )
            )
        if route_modules[0].role not in {
            SentenceTransformerModuleRole.TRANSFORMER,
            SentenceTransformerModuleRole.CUSTOM,
            SentenceTransformerModuleRole.LEGACY_CLIP,
        }:
            raise ValueError(f"Router route {name!r} lacks an input module")
        routes.append(
            SentenceTransformerRouteSpec(name=name, modules=tuple(route_modules))
        )
    unused = set(map(str, types)) - used
    if unused:
        raise ValueError(f"Router config contains unused modules: {sorted(unused)}")
    route_names = frozenset(route.name for route in routes)
    default_route = parameters.get("default_route")
    if default_route is not None:
        default_route = str(default_route)
        if default_route not in route_names:
            raise ValueError(f"Router default route {default_route!r} is unknown")
    raw_mappings = parameters.get("route_mappings", {})
    if not isinstance(raw_mappings, dict):
        raise TypeError("Router route_mappings must be a JSON object")
    mappings = tuple(
        _router_mapping(str(key), str(route), route_names=route_names)
        for key, route in raw_mappings.items()
    )
    allow_empty_key = parameters.get("allow_empty_key", True)
    if not isinstance(allow_empty_key, bool):
        raise TypeError("Router allow_empty_key must be a boolean")
    return tuple(routes), mappings, default_route, allow_empty_key


def load_sentence_transformer_graph(
    checkpoint: str | Path,
) -> SentenceTransformerGraphSpec:
    """Inspect a local Sentence Transformers graph without executing its code."""

    root = Path(checkpoint)
    modules = load_sentence_transformer_modules(root)
    model_config = _json_object(
        root / "config_sentence_transformers.json",
        required=False,
    )
    model_type = str(model_config.get("model_type", "SentenceTransformer"))
    first = modules[0]
    first_directory = _module_directory(root, first.path)
    transformer_config = _json_object(
        first_directory / "sentence_bert_config.json",
        required=False,
    )
    module_output_name = transformer_config.get("module_output_name")
    if module_output_name is not None:
        module_output_name = str(module_output_name)
    graph_kind = _classify_graph(
        modules,
        model_type=model_type,
        module_output_name=module_output_name,
    )
    routes: tuple[SentenceTransformerRouteSpec, ...] = ()
    route_mappings: tuple[SentenceTransformerRouteMapping, ...] = ()
    default_route: str | None = None
    allow_empty_key: bool | None = None
    if graph_kind is SentenceTransformerGraphKind.ROUTED_ENCODER:
        routes, route_mappings, default_route, allow_empty_key = _router_spec(
            root,
            first,
        )
    inputs = _input_specs(
        transformer_config,
        legacy_clip=graph_kind is SentenceTransformerGraphKind.LEGACY_CLIP,
    )
    if graph_kind is SentenceTransformerGraphKind.ROUTED_ENCODER:
        inputs = ()
    return SentenceTransformerGraphSpec(
        kind=graph_kind,
        model_type=model_type,
        transformer_task=(
            None
            if transformer_config.get("transformer_task") is None
            else str(transformer_config["transformer_task"])
        ),
        module_output_name=module_output_name,
        modules=modules,
        inputs=inputs,
        routes=routes,
        route_mappings=route_mappings,
        default_route=default_route,
        allow_empty_key=allow_empty_key,
    )


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
    graph = load_sentence_transformer_graph(resolved.path)
    if graph.kind is not SentenceTransformerGraphKind.DENSE_EMBEDDING:
        raise ValueError(
            f"Sentence Transformers graph is {graph.kind.value!r}; "
            "the native dense loader requires 'dense_embedding'"
        )
    modules = graph.modules
    transformer = modules[0]
    if transformer.role is not SentenceTransformerModuleRole.TRANSFORMER:
        raise ValueError(
            f"Sentence Transformers input module {transformer.type!r} is "
            "catalogued but not a registered native backbone"
        )
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
        if module.role is SentenceTransformerModuleRole.POOLING:
            directory = _module_directory(resolved.path, module.path)
            if pooling is not None or postprocessors:
                raise ValueError("Pooling must occur exactly once before dense modules")
            pooling = _load_pooling(directory, input_dimension=input_dimension)
            current_dimension = pooling.output_dimension
        elif module.role is SentenceTransformerModuleRole.DENSE:
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
        elif module.role is SentenceTransformerModuleRole.NORMALIZE:
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
            modalities=graph.modalities,
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
    sequence_length_buckets: Sequence[int] | None = None,
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
    native_processor = make_text_processor(
        tokenizer=processor,
        batch_builder=type(loaded.encoder.backbone).make_batch,
        max_sequence_length=loaded.max_sequence_length,
        sequence_length_buckets=sequence_length_buckets,
        prompts=loaded.prompts,
        default_prompt_name=loaded.default_prompt_name,
        include_prompt=loaded.encoder.pooling.include_prompt,
        pooling_batch_builder=SentenceBatch,
    )
    return TextEmbeddingModel(
        model=loaded.encoder,
        processor=native_processor,
        similarity_function=loaded.similarity_function,
    )


__all__ = [
    "LoadedSentenceTransformer",
    "SENTENCE_TRANSFORMERS_ORACLE_VERSION",
    "SentenceTransformerGraphKind",
    "SentenceTransformerGraphSpec",
    "SentenceTransformerInputSpec",
    "SentenceTransformerModuleSpec",
    "SentenceTransformerModuleRole",
    "SentenceTransformerRouteMapping",
    "SentenceTransformerRouteSpec",
    "SimilarityFunction",
    "load_sentence_transformer",
    "load_sentence_transformer_artifact",
    "load_sentence_transformer_graph",
    "load_sentence_transformer_modules",
]
