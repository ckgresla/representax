"""Atomic, inference-ready native and Hugging Face model publication."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np

from representax.config import JobConfig

EXPORT_SCHEMA = "representax-inference-bundle-v1"
COMPLETE_MARKER = "REPRESENTAX_COMPLETE"


@dataclass(frozen=True)
class InferenceBundle:
    """A complete, reloadable model publication."""

    path: Path
    native_path: Path
    huggingface_path: Path | None
    iteration: int


def _write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _fingerprint(document: Any) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _resolve_huggingface_source(source: str) -> Path:
    local = Path(source).expanduser()
    if local.exists():
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "remote Hugging Face export sources require representax[hf]"
        ) from error
    return Path(snapshot_download(source)).resolve()


def _state_dict(adapter: Any, model: eqx.Module) -> dict[str, Any]:
    state = adapter.state_dict(model)
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint adapter state_dict() must return a non-empty dict")
    return state


def _export_huggingface(
    model: eqx.Module,
    job: JobConfig,
    target: Path,
) -> None:
    config = job.export.huggingface
    if config is None:
        raise AssertionError("Hugging Face export is not configured")
    from representax.train.job import build_component

    adapter = build_component(config.adapter)
    from representax.models import merge_quantized_lora

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        model = jax.tree.map(
            lambda value: jax.device_put(value, cpu) if eqx.is_array(value) else value,
            model,
        )
        model = merge_quantized_lora(model)
        jax.block_until_ready(model)
        source = _resolve_huggingface_source(config.source_checkpoint)
        shutil.copytree(source, target)
        for pattern in (
            "*.safetensors",
            "*.safetensors.index.json",
            "pytorch_model*.bin",
        ):
            for stale in target.glob(pattern):
                stale.unlink()
        save = getattr(adapter, "save", None)
        if callable(save):
            save(model, target)
        else:
            from safetensors.numpy import save_file

            save_file(
                {
                    name: np.asarray(jax.device_get(value))
                    for name, value in _state_dict(adapter, model).items()
                },
                target / "model.safetensors",
            )
        if config.verify_reload:
            load = getattr(adapter, "load", None)
            if not callable(load):
                raise TypeError("verified Hugging Face export requires adapter.load()")
            parameters = inspect.signature(load).parameters
            load_options = {
                name: jax.numpy.float32
                for name in ("parameter_dtype", "compute_dtype")
                if name in parameters
            }
            restored = load(target, **load_options)
            expected = _state_dict(adapter, model)
            actual = _state_dict(adapter, restored)
            if set(actual) != set(expected):
                raise ValueError("Hugging Face export changed checkpoint tensor names")
            for name in sorted(expected):
                if not np.array_equal(
                    np.asarray(jax.device_get(actual[name])),
                    np.asarray(jax.device_get(expected[name])),
                ):
                    raise ValueError(
                        f"Hugging Face export failed exact reload verification: {name}"
                    )


def export_inference_bundle(
    model: eqx.Module,
    job: JobConfig,
    directory: str | Path,
    *,
    iteration: int | None,
) -> InferenceBundle:
    """Atomically publish a native model and optional verified HF checkpoint."""

    if iteration is None or iteration < 0:
        raise ValueError("export iteration must be non-negative")
    target = Path(directory).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"inference bundle already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent)
    ).resolve()
    try:
        native = temporary / "native"
        native.mkdir()
        config_path = native / "job.json"
        model_path = native / "model.eqx"
        config_path.write_text(job.model_dump_json(indent=2) + "\n")
        eqx.tree_serialise_leaves(model_path, model)
        huggingface_config = job.export.huggingface
        huggingface = None
        huggingface_manifest = None
        if huggingface_config is not None:
            huggingface = temporary / "huggingface"
            _export_huggingface(model, job, huggingface)
            huggingface_manifest = {
                "directory": "huggingface",
                "verified": huggingface_config.verify_reload,
            }
        manifest = {
            "schema_version": EXPORT_SCHEMA,
            "iteration": iteration,
            "selection": job.export.selection,
            "native": {
                "config": "native/job.json",
                "model": "native/model.eqx",
                "model_sha256": _sha256(model_path),
            },
            "huggingface": huggingface_manifest,
        }
        manifest["fingerprint"] = _fingerprint(manifest)
        _write_json(temporary / "manifest.json", manifest)
        (temporary / COMPLETE_MARKER).write_text(manifest["fingerprint"] + "\n")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return InferenceBundle(
        path=target,
        native_path=target / "native",
        huggingface_path=(
            None if job.export.huggingface is None else target / "huggingface"
        ),
        iteration=iteration,
    )


def load_inference_bundle(directory: str | Path) -> tuple[eqx.Module, JobConfig]:
    """Reconstruct and exactly deserialize a native inference model."""

    bundle = Path(directory).expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    complete_path = bundle / COMPLETE_MARKER
    if not manifest_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(f"inference bundle is incomplete: {bundle}")
    manifest = json.loads(manifest_path.read_text())
    fingerprint = manifest.pop("fingerprint", None)
    if (
        manifest.get("schema_version") != EXPORT_SCHEMA
        or fingerprint != _fingerprint(manifest)
        or complete_path.read_text().strip() != fingerprint
    ):
        raise ValueError(f"inference bundle manifest is invalid: {bundle}")
    native = bundle / "native"
    model_path = native / "model.eqx"
    if _sha256(model_path) != manifest["native"]["model_sha256"]:
        raise ValueError(f"native model digest differs: {model_path}")
    job = JobConfig.model_validate_json((native / "job.json").read_text())
    from representax.precision import prepare_master_model, resolve_precision_policy
    from representax.train.job import load_model, prepare_model

    template, _ = load_model(
        job.model,
        key=jax.random.fold_in(jax.random.key(job.training.seed), 0),
        activation_rematerialization=job.training.activation_rematerialization,
    )
    template, trainable_filter = prepare_model(
        template,
        adapter=job.training.adapter,
        key=jax.random.fold_in(jax.random.key(job.training.seed), 1),
    )
    template = prepare_master_model(
        template,
        resolve_precision_policy(job.training.precision),
        trainable_filter=trainable_filter,
    )
    return eqx.tree_deserialise_leaves(model_path, template), job


__all__ = [
    "COMPLETE_MARKER",
    "EXPORT_SCHEMA",
    "InferenceBundle",
    "export_inference_bundle",
    "load_inference_bundle",
]
