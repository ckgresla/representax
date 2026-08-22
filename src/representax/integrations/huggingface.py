"""Small, Torch-free Hugging Face checkpoint boundary.

Representax keeps checkpoint transport generic and lets each native model own
its configuration and parameter mapping.  This follows the useful separation
found in Levanter and MaxText without coupling the training runtime to either
project or to PyTorch.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import jax
import jax.numpy as jnp

ModelT = TypeVar("ModelT")


@dataclass(frozen=True, slots=True)
class ResolvedHuggingFaceCheckpoint:
    """A local checkpoint directory with its immutable external identity."""

    path: Path
    model_id: str
    revision: str


@runtime_checkable
class HuggingFaceCheckpointAdapter(Protocol[ModelT]):
    """Bidirectional mapping owned by one native model family."""

    def load(
        self,
        checkpoint: str | Path,
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
    ) -> ModelT: ...

    def state_dict(self, model: ModelT) -> Mapping[str, jax.Array]: ...


def resolve_hf_checkpoint(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    token: bool | str | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> ResolvedHuggingFaceCheckpoint:
    """Resolve a local path or Hub model to one immutable local snapshot."""

    candidate = Path(model_name_or_path).expanduser()
    if candidate.is_dir():
        return ResolvedHuggingFaceCheckpoint(
            path=candidate.resolve(),
            model_id=candidate.name,
            revision=revision or "local",
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "Hugging Face Hub loading requires `pip install representax[hf]`"
        ) from error

    model_id = str(model_name_or_path)
    path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=None if cache_directory is None else str(cache_directory),
            local_files_only=local_files_only,
            token=token,
            allow_patterns=None if allow_patterns is None else list(allow_patterns),
        )
    ).resolve()
    resolved_revision = path.name
    if not (
        len(resolved_revision) == 40
        and all(character in "0123456789abcdef" for character in resolved_revision)
    ):
        resolved_revision = revision or "resolved"
    return ResolvedHuggingFaceCheckpoint(
        path=path,
        model_id=model_id,
        revision=resolved_revision,
    )


def load_hf_config(checkpoint: str | Path) -> dict[str, Any]:
    """Load a local Hugging Face ``config.json`` without Transformers."""

    path = Path(checkpoint) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face config not found: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("Hugging Face config must contain a JSON object")
    return value


def _safetensor_shards(
    checkpoint: Path,
    requested: frozenset[str],
) -> dict[Path, frozenset[str]]:
    index_path = checkpoint / "model.safetensors.index.json"
    single_path = checkpoint / "model.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid safetensor index: {index_path}")
        missing = requested.difference(weight_map)
        if missing:
            rendered = ", ".join(sorted(missing))
            raise KeyError(f"checkpoint index is missing tensors: {rendered}")
        grouped: dict[Path, set[str]] = {}
        for name in requested:
            shard = checkpoint / str(weight_map[name])
            grouped.setdefault(shard, set()).add(name)
        return {path: frozenset(names) for path, names in grouped.items()}
    if single_path.is_file():
        return {single_path: requested}
    raise FileNotFoundError(
        f"no model.safetensors or model.safetensors.index.json in {checkpoint}"
    )


def load_safetensor_subset(
    checkpoint: str | Path,
    names: set[str] | frozenset[str],
    *,
    dtype: jnp.dtype = jnp.float32,
    device: jax.Device | None = None,
) -> dict[str, jax.Array]:
    """Read only selected tensors from a local single- or multi-shard model.

    ``safetensors`` remains an optional dependency. Checkpoints are materialized
    on CPU by default so parsing, dtype conversion, and model-specific layout
    changes never consume accelerator memory. Runtime placement is an explicit
    later boundary. ``device`` exists for controlled low-level use only.
    """

    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - broken installation
        raise ImportError(
            "safetensors is required for checkpoint loading; reinstall representax"
        ) from error

    requested = frozenset(names)
    if not requested:
        return {}
    checkpoint_path = Path(checkpoint)
    result: dict[str, jax.Array] = {}
    target = jax.devices("cpu")[0] if device is None else device
    for shard, shard_names in _safetensor_shards(checkpoint_path, requested).items():
        if not shard.is_file():
            raise FileNotFoundError(f"safetensor shard not found: {shard}")
        with safe_open(shard, framework="np") as handle:
            available = frozenset(handle.keys())
            missing = shard_names.difference(available)
            if missing:
                rendered = ", ".join(sorted(missing))
                raise KeyError(f"{shard.name} is missing tensors: {rendered}")
            with jax.default_device(target):
                for name in shard_names:
                    result[name] = jnp.asarray(handle.get_tensor(name), dtype=dtype)
    return result
