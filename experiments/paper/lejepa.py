"""Canonical LeJEPA ViT-B/16 ImageNet-1K readiness lifecycle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OFFICIAL_REFERENCE = Path("/home/ckg/representax-references/lejepa")
OFFICIAL_COMMIT = "c293d291ca87cd4fddee9d3fffe4e914c7272052"
STABLE_REFERENCE = Path("/home/ckg/representax-references/stable-pretraining")
STABLE_COMMIT = "9aa93f8b6153eebb73f57d4853ccf8a13d848310"
DEFAULT_IMAGENET_ROOT = Path("/raid/datasets/imagenet")
IMAGENET_VALIDATION_PROVENANCE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/imagenet-validation-provenance-20260831/result.json"
)
PHYSICAL_GPUS = (4, 5)
CAPACITY_PHYSICAL_GPUS = (0, 1)
IMAGENET_TRAIN_IMAGES = 1_281_167
CLASS_COUNT = 10
PROBE_TRAIN_PER_CLASS = 3
PROBE_VALIDATION_PER_CLASS = 2
PROBE_TEST_PER_CLASS = 2
CANARY_ARCHITECTURE = {
    "image_size": 224,
    "local_image_size": 98,
    "patch_size": 16,
    "hidden_size": 384,
    "depth": 12,
    "heads": 6,
    "mlp_ratio": 4.0,
    "projection_dimension": 512,
}
PAPER_VIT_LARGE_ARCHITECTURE = {
    "image_size": 224,
    "local_image_size": 98,
    "patch_size": 16,
    "hidden_size": 1024,
    "depth": 24,
    "heads": 16,
    "mlp_ratio": 4.0,
    "projection_dimension": 512,
}


@dataclass(frozen=True)
class LeJEPAReferenceContract:
    official_commit: str
    stable_pretraining_commit: str
    backbone: str = "timm/vit_base_patch16_224"
    image_size: int = 224
    local_image_size: int = 98
    patch_size: int = 16
    hidden_size: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    global_views: int = 2
    local_views: int = 6
    projector: str = "training-only MLP"
    evaluation_representation: str = (
        "concatenated CLS from final two blocks followed by LayerNorm"
    )
    regularization_weight: float = 0.02
    sigreg_knots: int = 17
    sigreg_slices: int = 1024
    sigreg_max_frequency: float = 3.0


class ImageNetAccessError(RuntimeError):
    """The requested local ImageNet-1K source is absent or incomplete."""


def _git_head(path: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def reference_contract() -> LeJEPAReferenceContract:
    official = _git_head(OFFICIAL_REFERENCE)
    stable = _git_head(STABLE_REFERENCE)
    if official != OFFICIAL_COMMIT:
        raise ValueError(f"expected LeJEPA commit {OFFICIAL_COMMIT}, found {official}")
    if stable != STABLE_COMMIT:
        raise ValueError(
            f"expected stable-pretraining commit {STABLE_COMMIT}, found {stable}"
        )
    return LeJEPAReferenceContract(official, stable)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def _validation_status(root: Path, provenance_path: Path) -> dict[str, Any]:
    validation = root / "val"
    validation_classes = (
        tuple(path for path in validation.iterdir() if path.is_dir())
        if validation.is_dir()
        else ()
    )
    status: dict[str, Any] = {
        "validation_directory_observed": validation.is_dir(),
        "validation_class_directories_observed": len(validation_classes),
        "accepted_for_quality": False,
    }
    if not provenance_path.is_file():
        status["blocker"] = "official validation provenance evidence is missing"
        return status
    evidence = json.loads(provenance_path.read_text(encoding="utf-8"))
    local = evidence.get("local_dataset", {})
    evidence_root = Path(str(local.get("validation_root", ""))).resolve()
    accepted = (
        evidence.get("schema_version")
        == "representax-imagenet-validation-provenance-v1"
        and evidence.get("status") == "pass"
        and evidence.get("official_quality_eval_admissible") is True
        and evidence_root == validation.resolve()
        and local.get("image_count") == 50_000
        and local.get("class_count") == 1_000
        and local.get("exact_filename_set") is True
        and local.get("exact_parent_synset_assignments") is True
        and local.get("local_files_byte_identical_to_archive") is True
    )
    status.update(
        {
            "accepted_for_quality": accepted,
            "provenance": str(provenance_path.resolve()),
            "provenance_sha256": _sha256(provenance_path),
        }
    )
    if not accepted:
        status["blocker"] = "validation provenance does not match this ImageNet tree"
    return status


def _imagenet_layout(
    root: Path,
    *,
    validation_provenance: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    train = root / "train"
    class_index = root / "imagenet_class_index.json"
    missing = tuple(str(path) for path in (train, class_index) if not path.exists())
    if missing:
        raise ImageNetAccessError(
            "local ImageNet-1K access is incomplete; missing " + ", ".join(missing)
        )
    train_classes = tuple(path for path in train.iterdir() if path.is_dir())
    if len(train_classes) != 1000:
        raise ImageNetAccessError(
            "local ImageNet-1K extraction must expose exactly 1000 train classes; "
            f"found train={len(train_classes)}"
        )
    validation_status = _validation_status(root, validation_provenance)
    return train, class_index, validation_status


def prepare_imagenet(
    output: Path,
    *,
    imagenet_root: Path = DEFAULT_IMAGENET_ROOT,
    validation_provenance: Path = IMAGENET_VALIDATION_PROVENANCE,
    class_count: int = CLASS_COUNT,
    seed: int = 17,
) -> dict[str, Any]:
    """Materialize bounded real ImageNet-1K manifests without copying images."""

    if class_count < 2:
        raise ValueError("LeJEPA readiness requires at least two ImageNet classes")
    contract = reference_contract()
    train_root, class_index_path, validation_status = _imagenet_layout(
        imagenet_root,
        validation_provenance=validation_provenance,
    )
    classes = json.loads(class_index_path.read_text(encoding="utf-8"))
    selected = tuple(classes[str(index)] for index in range(class_count))
    training_rows = []
    evaluation_rows = []
    source_files = []
    for label, (synset, name) in enumerate(selected):
        train_files = sorted((train_root / synset).glob("*.JPEG"))
        needed_train = (
            PROBE_TRAIN_PER_CLASS + PROBE_VALIDATION_PER_CLASS + PROBE_TEST_PER_CLASS
        )
        if len(train_files) < needed_train:
            raise ImageNetAccessError(
                f"ImageNet train class {synset} lacks the bounded readiness rows "
                "required"
            )
        for index, path in enumerate(train_files[:needed_train]):
            source_files.append(path)
            training_rows.append(
                {
                    "image": str(path.resolve()),
                    "label": label,
                    "synset": synset,
                    "class_name": name,
                    "view_seed": seed + label * needed_train + index,
                }
            )
            if index < PROBE_TRAIN_PER_CLASS:
                split = 0
            elif index < PROBE_TRAIN_PER_CLASS + PROBE_VALIDATION_PER_CLASS:
                split = 1
            else:
                split = 2
            evaluation_rows.append(
                {
                    "image": str(path.resolve()),
                    "label": label,
                    "synset": synset,
                    "split": split,
                    "split_scope": "readiness-only disjoint ImageNet train subset",
                }
            )
    output.mkdir(parents=True, exist_ok=False)
    train_path = output / "train.jsonl"
    evaluation_path = output / "evaluation.jsonl"
    _write_jsonl(train_path, training_rows)
    _write_jsonl(evaluation_path, evaluation_rows)
    split_counts = {
        str(split): sum(row["split"] == split for row in evaluation_rows)
        for split in range(3)
    }
    manifest = {
        "schema_version": "representax-lejepa-imagenet1k-inputs-v1",
        "status": "accepted",
        "acceptance_scope": (
            "readiness-only deterministic disjoint subsets of real ImageNet-1K train"
        ),
        "reference_contract": asdict(contract),
        "imagenet": {
            "root": str(imagenet_root.resolve()),
            "source": "local extracted ILSVRC2012 train classes",
            "access": "available",
            "classes_in_source": 1000,
            "bounded_classes": class_count,
            "class_index": str(class_index_path.resolve()),
            "class_index_sha256": _sha256(class_index_path),
            "selected_classes": [
                {"label": index, "synset": synset, "name": name}
                for index, (synset, name) in enumerate(selected)
            ],
            "official_validation": validation_status,
            "quality_claim_allowed": False,
        },
        "rows": {
            "training": len(training_rows),
            "evaluation": len(evaluation_rows),
            "evaluation_splits": split_counts,
            "unique_source_images": len(set(source_files)),
        },
        "files": {
            "train.jsonl": _sha256(train_path),
            "evaluation.jsonl": _sha256(evaluation_path),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def load_lejepa_model(
    *,
    config: Mapping[str, Any] | None = None,
    initialization_path: str | None = None,
    parameter_dtype: str = "float32",
    rematerialization: str = "full",
    initialization_device: str = "default",
    key: Any,
) -> tuple[Any, None]:
    import jax
    import jax.numpy as jnp

    from representax.models.lejepa import LeJEPAModel, LeJEPAViTConfig

    architecture = LeJEPAViTConfig.model_validate(config or {})
    if initialization_device not in {"cpu", "default"}:
        raise ValueError("initialization_device must be 'cpu' or 'default'")
    dtype = jnp.dtype(parameter_dtype)
    cpu_device = jax.devices("cpu")[0] if initialization_device == "cpu" else None
    with jax.default_device(cpu_device):
        initialization_key = (
            jax.device_put(key, cpu_device) if cpu_device is not None else key
        )
        if initialization_path is None:
            model = LeJEPAModel.init(
                architecture,
                key=initialization_key,
                parameter_dtype=dtype,
                rematerialization=rematerialization,
                revision=f"lejepa-{OFFICIAL_COMMIT[:12]}-random-init",
            )
        else:
            import torch

            payload = torch.load(
                initialization_path,
                map_location="cpu",
                weights_only=True,
            )
            model = LeJEPAModel.from_timm_state_dict(
                architecture,
                payload["model"],
                parameter_dtype=dtype,
                rematerialization=rematerialization,
                revision=f"lejepa-shared-timm-{STABLE_COMMIT[:12]}",
            )
    return model, None


def build_job(
    data_directory: Path,
    *,
    steps: int = 1,
    seed: int = 17,
    architecture: Mapping[str, Any] | None = None,
    initialization_path: Path | None = None,
    distributed: bool = True,
) -> Any:
    """Build the canonical one-update LeJEPA job and frozen evaluator."""

    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        EvaluationConfig,
        ExportConfig,
        FSDPConfig,
        JEPARepresentationEvaluatorConfig,
        JobConfig,
        LoggingConfig,
        MeshConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.jepa import JEPAConfig, LeJEPAConfig

    if steps != 1:
        raise ValueError("the readiness lifecycle is intentionally exactly one update")
    model_config = dict(architecture or {})
    initialization = initialization_path or data_directory / "shared-initialization.pt"
    image_size = int(model_config.get("image_size", 224))
    local_image_size = int(model_config.get("local_image_size", 98))
    device_count = 2 if distributed else 1
    training_arguments: dict[str, Any] = {}
    if distributed:
        training_arguments = {
            "mesh": MeshConfig(axis_shapes=(2,), axis_names=("model",)),
            "sharding": FSDPConfig(
                data_axis=None,
                parameter_axis="model",
                minimum_parameter_elements=2**18,
            ),
        }
    return JobConfig(
        name="paper-lejepa-vit-b16-imagenet1k-readiness",
        model=ModelConfig(
            target="experiments.paper.lejepa:load_lejepa_model",
            parameters={
                "config": model_config,
                "initialization_path": str(initialization.resolve()),
                "parameter_dtype": "float32",
                "rematerialization": "full",
            },
        ),
        task=JEPAConfig(),
        loss=LeJEPAConfig(
            regularization_weight=0.02,
            global_views=2,
            knots=17,
            slices=1024,
            max_frequency=3.0,
        ),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={
                    "learning_rate": 5e-4,
                    "b1": 0.9,
                    "b2": 0.999,
                    "weight_decay": 0.05,
                },
            )
        ),
        data=DataConfig(
            distribution=mix(
                source(str(data_directory / "train.jsonl"), map=identity),
                shuffle=False,
            ),
            collate=ComponentConfig(
                target="representax.models.lejepa:LeJEPATrainCollator",
                parameters={
                    "image_size": image_size,
                    "local_image_size": local_image_size,
                    "seed": seed,
                },
            ),
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        training=TrainingConfig(
            global_batch_size=device_count,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(micro_batch_size=device_count),
            precision=PrecisionConfig.bfloat16_mixed(),
            activation_rematerialization="full",
            donate_buffers=True,
            **training_arguments,
        ),
        checkpointing=CheckpointConfig(
            every=1,
            keep=1,
            save_final=True,
            asynchronous=False,
        ),
        evaluation=EvaluationConfig(
            data=DataConfig(
                distribution=mix(
                    source(
                        str(data_directory / "evaluation.jsonl"),
                        map=identity,
                    ),
                    shuffle=False,
                ),
                collate=ComponentConfig(
                    target="representax.models.lejepa:LeJEPAEvaluationCollator",
                    parameters={"image_size": image_size},
                ),
                num_threads=0,
                prefetch_buffer_size=0,
            ),
            batch_size=10 if architecture is None else 2,
            evaluators=(
                JEPARepresentationEvaluatorConfig(
                    inverse_regularization=(0.01, 0.1, 1.0, 10.0),
                    max_iterations=1000,
                    neighbors=20,
                    seed=seed,
                ),
            ),
            on_start=False,
            on_end=False,
            every_steps=None,
            primary_metric=("valid/jepa_representation/linear_probe_accuracy"),
            primary_metric_mode="max",
            save_best=False,
        ),
        logging=LoggingConfig(console_every=1, timing=True, accelerator=True),
        export=ExportConfig(enabled=True, selection="final"),
    )


def build_capacity_job(
    data_directory: Path,
    *,
    global_batch_size: int,
    steps: int = 3,
    seed: int = 7,
    architecture: Mapping[str, Any] | None = None,
) -> Any:
    """Build the promoted ViT-L training-only capacity measurement."""

    from representax.config import (
        BatchConfig,
        ComponentConfig,
        DataConfig,
        ExportConfig,
        FSDPConfig,
        JobConfig,
        LoggingConfig,
        MeshConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.jepa import JEPAConfig, LeJEPAConfig

    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if steps < 3:
        raise ValueError("capacity measurement requires at least three updates")
    model_config = dict(architecture or PAPER_VIT_LARGE_ARCHITECTURE)
    return JobConfig(
        name="paper-lejepa-vit-l16-imagenet1k-capacity",
        model=ModelConfig(
            target="experiments.paper.lejepa:load_lejepa_model",
            parameters={
                "config": model_config,
                "initialization_path": None,
                "parameter_dtype": "float32",
                "rematerialization": "full",
                "initialization_device": "cpu",
            },
        ),
        task=JEPAConfig(),
        loss=LeJEPAConfig(
            regularization_weight=0.02,
            global_views=2,
            knots=17,
            slices=1024,
            max_frequency=3.0,
        ),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={
                    "learning_rate": 5e-4,
                    "b1": 0.9,
                    "b2": 0.999,
                    "weight_decay": 0.05,
                },
            )
        ),
        data=DataConfig(
            distribution=mix(
                source(str(data_directory / "train.jsonl"), map=identity),
                shuffle=False,
            ),
            collate=ComponentConfig(
                target="representax.models.lejepa:LeJEPATrainCollator",
                parameters={
                    "image_size": int(model_config.get("image_size", 224)),
                    "local_image_size": int(
                        model_config.get("local_image_size", 98)
                    ),
                    "seed": seed,
                },
            ),
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        training=TrainingConfig(
            global_batch_size=global_batch_size,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(micro_batch_size=global_batch_size),
            precision=PrecisionConfig.bfloat16_mixed(),
            activation_rematerialization="full",
            donate_buffers=True,
            mesh=MeshConfig(axis_shapes=(2,), axis_names=("model",)),
            sharding=FSDPConfig(
                data_axis=None,
                parameter_axis="model",
                minimum_parameter_elements=2**18,
            ),
        ),
        checkpointing=None,
        logging=LoggingConfig(console_every=1, timing=True, accelerator=True),
        evaluation=None,
        export=ExportConfig(enabled=False),
    )


def _visible_physical_gpus() -> tuple[int, ...]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain physical GPU indices"
        ) from error


def _parameter_count(model: Any) -> int:
    import equinox as eqx
    import jax

    return sum(
        int(value.size) for value in jax.tree.leaves(model) if eqx.is_array(value)
    )


def _array_tree_digest(tree: Any) -> str:
    import equinox as eqx
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for value in jax.tree.leaves(tree):
        if not eqx.is_array(value):
            continue
        array = np.asarray(jax.device_get(value))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _metric_rows(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _device_memory() -> tuple[dict[str, Any], ...]:
    import jax

    return tuple(
        {
            "logical_id": device.id,
            "device_kind": device.device_kind,
            **{
                name: (device.memory_stats() or {}).get(name)
                for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
            },
        }
        for device in jax.devices()
    )


def stable_reference_decision() -> dict[str, Any]:
    """Describe the experiment-owned correction to the pinned helper objective."""

    return {
        "status": "experiment-owned-runner-required",
        "architecture_match": True,
        "objective_match": True,
        "pinned_helper_used_as_result": False,
        "correction": (
            "the pinned stable_pretraining.methods.LeJEPA computes "
            "invariance + lambda * SIGReg, while the pinned paper formula used here "
            "is (1 - lambda) * invariance + lambda * SIGReg; its default projector "
            "is retained, but the experiment owns the corrected objective and the "
            "separate last-two-CLS evaluation path; the "
            "stable Imagenette recipe uses 96px local crops while the paper README "
            "specifies 98px and is authoritative here"
        ),
        "policy": (
            "run the pinned stable-pretraining MLP and timm backbone through the "
            "experiment-owned exact paper objective"
        ),
    }


def _torch_paper_loss(
    projections: Any,
    directions: Any,
    *,
    regularization_weight: float = 0.02,
    knots: int = 17,
    max_frequency: float = 3.0,
) -> tuple[Any, Any, Any]:
    """Exact torch oracle for the paper-weighted per-view LeJEPA objective."""

    import torch

    directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
    center = projections[:, :2].mean(dim=1, keepdim=True)
    invariance = (projections - center).square().mean()
    t = torch.linspace(
        0.0,
        max_frequency,
        knots,
        device=projections.device,
        dtype=torch.float32,
    )
    dt = max_frequency / (knots - 1)
    weights = torch.full_like(t, 2.0 * dt)
    weights[[0, -1]] = dt
    phi = torch.exp(-t.square() / 2.0)
    weights = weights * phi
    sliced = torch.einsum("bvd,ds->vbs", projections.float(), directions.float())
    values = sliced[..., None] * t
    error = (values.cos().mean(dim=1) - phi).square()
    error = error + values.sin().mean(dim=1).square()
    sigreg = ((error * weights).sum(dim=-1) * projections.shape[0]).mean()
    total = (
        1.0 - regularization_weight
    ) * invariance.float() + regularization_weight * sigreg
    return total, invariance, sigreg


def exact_objective_parity() -> dict[str, Any]:
    """Verify value and projection-gradient parity for the exact eight-view loss."""

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    from representax.tasks.jepa import invariance_loss, sigreg_loss

    projections = np.arange(3 * 8 * 5, dtype=np.float32).reshape(3, 8, 5) / 20.0
    directions = np.arange(5 * 1024, dtype=np.float32).reshape(5, 1024) / 1000.0 + 0.1
    valid = jnp.ones((3, 8), dtype=jnp.bool_)

    def native_objective(value: Any) -> Any:
        invariance = invariance_loss(value, valid, global_views=2)
        sigreg = sigreg_loss(
            value,
            valid,
            jnp.asarray(directions),
            knots=17,
            max_frequency=3.0,
        )
        return 0.98 * invariance + 0.02 * sigreg

    native_value, native_gradient = jax.value_and_grad(native_objective)(
        jnp.asarray(projections)
    )
    torch_projections = torch.tensor(projections, requires_grad=True)
    torch_value, _, _ = _torch_paper_loss(
        torch_projections,
        torch.tensor(directions),
    )
    torch_value.backward()
    reference_gradient = torch_projections.grad
    if reference_gradient is None:  # pragma: no cover - torch contract
        raise AssertionError("torch objective did not produce a projection gradient")
    native_gradient_array = np.asarray(native_gradient)
    reference_gradient_array = reference_gradient.detach().numpy()
    value_error = abs(float(native_value) - float(torch_value.detach()))
    gradient_error = float(
        np.max(np.abs(native_gradient_array - reference_gradient_array))
    )
    value_tolerance = 1e-5
    gradient_tolerance = 2e-5
    accepted = value_error <= value_tolerance and gradient_error <= gradient_tolerance
    result = {
        "schema_version": "representax-lejepa-objective-parity-v1",
        "status": "accepted" if accepted else "rejected",
        "contract": {
            "batch": 3,
            "views": 8,
            "global_views": 2,
            "local_views": 6,
            "projection_dimension": 5,
            "slices": 1024,
            "knots": 17,
            "regularization_weight": 0.02,
            "formula": "(1-lambda)*invariance + lambda*per-view-SIGReg",
        },
        "combined_value": {
            "representax": float(native_value),
            "torch": float(torch_value.detach()),
            "absolute_error": value_error,
            "tolerance": value_tolerance,
        },
        "projection_gradient": {
            "shape": list(native_gradient_array.shape),
            "max_absolute_error": gradient_error,
            "tolerance": gradient_tolerance,
        },
    }
    if not accepted:
        raise RuntimeError("exact LeJEPA objective value/gradient parity failed")
    return result


def _build_timm_reference_model(*, device: Any) -> Any:
    """Build the matched timm ViT-S and pinned stable-pretraining projector."""

    import importlib.util

    import timm
    import torch
    import torch.nn as nn

    mlp_path = STABLE_REFERENCE / "stable_pretraining" / "backbone" / "mlp.py"
    spec = importlib.util.spec_from_file_location("_lejepa_stable_mlp", mlp_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned stable-pretraining MLP: {mlp_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MLP = module.MLP

    class ReferenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(
                "vit_small_patch16_224",
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                drop_path_rate=0.1,
            )
            # Paper locals are 98px for patch-16 ViTs. Preserve Conv2d's
            # unpadded floor semantics while allowing timm's dynamic positions.
            self.backbone.patch_embed.img_size = None
            self.projector = nn.Sequential(
                nn.Linear(384, 512, bias=True),
                MLP(
                    in_channels=512,
                    hidden_channels=[2048, 2048, 512],
                    norm_layer="batch_norm",
                    activation_layer=nn.ReLU,
                    inplace=True,
                    dropout=0.0,
                ),
            )
            self.evaluation_norm = nn.LayerNorm(768, eps=1e-6)

        def project(self, pixels: Any) -> Any:
            batch = pixels.shape[0]
            global_pixels = pixels[:, :2].transpose(0, 1).flatten(0, 1)
            local_pixels = pixels[:, 2:, :, :98, :98].transpose(0, 1).flatten(0, 1)
            features = torch.cat(
                (self.backbone(global_pixels), self.backbone(local_pixels)),
                dim=0,
            )
            return self.projector(features).reshape(8, batch, 512).transpose(0, 1)

        def encode(self, pixels: Any) -> Any:
            intermediates = self.backbone.forward_intermediates(
                pixels,
                indices=2,
                return_prefix_tokens=True,
                norm=False,
                output_fmt="NLC",
                intermediates_only=True,
            )
            cls = [value[1][:, 0] for value in intermediates]
            return self.evaluation_norm(torch.cat(cls, dim=-1))

    model = ReferenceModel().to(device)
    backbone = model.backbone
    if (
        backbone.embed_dim != 384
        or len(backbone.blocks) != 12
        or tuple(backbone.patch_embed.patch_size) != (16, 16)
        or backbone.blocks[0].attn.num_heads != 6
    ):
        raise RuntimeError("timm did not construct the exact ViT-S/16 canary profile")
    return model


def _torch_tree_digest(tree: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous()
            digest.update(str(array.dtype).encode())
            digest.update(str(tuple(array.shape)).encode())
            digest.update(
                array.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            )
        elif isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: repr(item)):
                digest.update(repr(key).encode())
                update(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                update(item)
        else:
            digest.update(repr(value).encode())

    update(tree)
    return "sha256:" + digest.hexdigest()


def _reference_optimizer(model: Any) -> Any:
    import torch

    return torch.optim.AdamW(
        (*model.backbone.parameters(), *model.projector.parameters()),
        lr=5e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    )


def materialize_shared_initialization(
    output: Path, *, seed: int = 17
) -> dict[str, Any]:
    """Materialize one timm initialization and the native training sketch."""

    import jax
    import numpy as np
    import timm
    import torch

    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"shared LeJEPA initialization already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = _build_timm_reference_model(device=torch.device("cpu"))
    step_key = jax.random.fold_in(jax.random.key(seed), 0)
    _, loss_key = jax.random.split(step_key)
    directions = np.array(
        jax.random.normal(loss_key, (512, 1024), dtype=jax.numpy.float32),
        copy=True,
    )
    payload = {
        "schema_version": "representax-lejepa-shared-initialization-v1",
        "model": model.state_dict(),
        "directions": torch.from_numpy(directions),
    }
    torch.save(payload, output)
    parameter_state = dict(model.named_parameters())
    manifest = {
        "schema_version": "representax-lejepa-shared-initialization-v1",
        "status": "accepted",
        "seed": seed,
        "architecture": "timm/vit_small_patch16_224",
        "architecture_parameters": CANARY_ARCHITECTURE,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "state_fingerprint": _torch_tree_digest(payload["model"]),
        "parameter_fingerprint": _torch_tree_digest(parameter_state),
        "directions_fingerprint": _torch_tree_digest(payload["directions"]),
        "directions_shape": list(directions.shape),
        "file": str(output.resolve()),
        "file_sha256": _sha256(output),
        "timm_version": timm.__version__,
        "official_commit": OFFICIAL_COMMIT,
        "stable_pretraining_commit": STABLE_COMMIT,
        "projector": "stable-pretraining 384->512->2048->2048->512",
        "evaluation_norm": "LayerNorm(768, eps=1e-6)",
        "local_patch_semantics": "unpadded Conv2d floor grid (98px -> 6x6)",
    }
    _write_json(output.with_suffix(".json"), manifest)
    return manifest


def _load_shared_initialization(
    initialization_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    manifest = json.loads(
        initialization_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    if _sha256(initialization_path) != manifest["file_sha256"]:
        raise RuntimeError("shared LeJEPA initialization file hash changed")
    payload = torch.load(initialization_path, map_location="cpu", weights_only=True)
    if _torch_tree_digest(payload["model"]) != manifest["state_fingerprint"]:
        raise RuntimeError("shared LeJEPA initialization state fingerprint changed")
    if _torch_tree_digest(payload["directions"]) != manifest["directions_fingerprint"]:
        raise RuntimeError("shared LeJEPA SIGReg directions changed")
    return payload, manifest


def _real_multicrop_batch(
    data_directory: Path,
    *,
    count: int = 2,
) -> tuple[Any, Any, tuple[str, ...]]:
    import numpy as np
    from PIL import Image

    from representax.models.lejepa.processing import canonical_multicrop_views

    rows = _metric_rows(data_directory / "train.jsonl")[:count]
    pixels = []
    sizes = []
    paths = []
    for row in rows:
        path = Path(str(row["image"]))
        with Image.open(path) as source:
            views, crop_sizes = canonical_multicrop_views(
                source.convert("RGB"),
                image_size=224,
                local_image_size=98,
                seed=17 + int(row["view_seed"]),
            )
        pixels.append(views)
        sizes.append(crop_sizes)
        paths.append(str(path.resolve()))
    return np.stack(pixels), np.stack(sizes), tuple(paths)


def _native_pre_update_objective(
    model: Any,
    pixels: Any,
    sizes: Any,
    directions: Any,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from representax.models.lejepa import LeJEPAMulticropImages
    from representax.tasks.jepa import invariance_loss, sigreg_loss

    batch = pixels.shape[0]
    flattened = LeJEPAMulticropImages(
        pixel_values=jnp.asarray(pixels).reshape((-1, *pixels.shape[2:])),
        crop_sizes=jnp.asarray(sizes).reshape(-1),
    )
    projections = model.project(flattened, key=None).reshape(batch, 8, 512)
    valid = jnp.ones((batch, 8), dtype=jnp.bool_)
    invariance = invariance_loss(projections, valid, global_views=2)
    sigreg = sigreg_loss(
        projections.astype(jnp.float32),
        valid,
        jnp.asarray(directions),
        knots=17,
        max_frequency=3.0,
    )
    loss = 0.98 * invariance + 0.02 * sigreg
    loss, invariance, sigreg = jax.device_get((loss, invariance, sigreg))
    return {
        "loss": float(loss),
        "invariance": float(invariance),
        "sigreg": float(sigreg),
        "deterministic_drop_path_disabled": True,
        "slices": 1024,
    }


def run_timm_reference_canary(
    *,
    data_directory: Path,
    initialization_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Run one matched timm update and the canonical frozen evaluator."""

    import numpy as np
    import torch
    from PIL import Image

    from representax.evaluation import (
        knn_accuracy,
        linear_probe_metrics,
        representation_geometry_metrics,
    )
    from representax.models.lejepa.processing import evaluation_image

    if _visible_physical_gpus() != (5,):
        raise RuntimeError(
            "timm reference canary requires CUDA_VISIBLE_DEVICES=5; "
            f"found {_visible_physical_gpus()}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("timm reference canary requires one visible CUDA GPU")
    contract = reference_contract()
    input_manifest = json.loads(
        (data_directory / "manifest.json").read_text(encoding="utf-8")
    )
    expected_scope = (
        "readiness-only deterministic disjoint subsets of real ImageNet-1K train"
    )
    if input_manifest.get("acceptance_scope") != expected_scope:
        raise ValueError("timm reference requires the bounded real ImageNet manifest")
    evaluation_rows = _metric_rows(data_directory / "evaluation.jsonl")
    shared_payload, shared_manifest = _load_shared_initialization(initialization_path)
    device = torch.device("cuda:0")
    model = _build_timm_reference_model(device=device)
    model.load_state_dict(shared_payload["model"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != shared_manifest["parameter_count"]:
        raise RuntimeError(
            "timm worker changed the shared architecture parameter count"
        )
    initial_parameter_digest = _torch_tree_digest(model.state_dict())
    if initial_parameter_digest != shared_manifest["state_fingerprint"]:
        raise RuntimeError("timm worker did not load the shared initial state exactly")
    view_values, view_sizes, source_images = _real_multicrop_batch(data_directory)
    expected_sizes = (224, 224, 98, 98, 98, 98, 98, 98)
    if any(tuple(value) != expected_sizes for value in view_sizes):
        raise RuntimeError("reference multicrop sizes differ from the paper")
    pixels = torch.from_numpy(view_values).to(device)
    directions = shared_payload["directions"].to(device)
    model.backbone.eval()
    model.projector.train()
    with torch.no_grad():
        pre_update_projections = model.project(pixels)
        pre_update_loss, pre_update_invariance, pre_update_sigreg = _torch_paper_loss(
            pre_update_projections,
            directions,
        )
    pre_update = {
        "loss": float(pre_update_loss.cpu()),
        "invariance": float(pre_update_invariance.cpu()),
        "sigreg": float(pre_update_sigreg.cpu()),
        "deterministic_drop_path_disabled": True,
        "slices": 1024,
    }
    model.load_state_dict(shared_payload["model"], strict=True)
    optimizer = _reference_optimizer(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        projections = model.project(pixels)
        loss, invariance, sigreg = _torch_paper_loss(projections, directions)
    loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in (
                *model.backbone.parameters(),
                *model.projector.parameters(),
            )
            if parameter.grad is not None
        )
    )
    optimizer.step()
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - started
    if not all(
        math.isfinite(float(value.detach().cpu()))
        for value in (loss, invariance, sigreg, gradient_norm)
    ):
        raise RuntimeError("timm reference update produced non-finite values")
    updated_model_digest = _torch_tree_digest(model.state_dict())
    updated_optimizer_digest = _torch_tree_digest(optimizer.state_dict())

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "updates": 1,
        },
        checkpoint,
    )
    del model, optimizer, projections, pixels
    torch.cuda.empty_cache()
    restored = _build_timm_reference_model(device=device)
    restored_optimizer = _reference_optimizer(restored)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    restored.load_state_dict(payload["model"], strict=True)
    restored_optimizer.load_state_dict(payload["optimizer"])
    if payload["updates"] != 1:
        raise RuntimeError("timm checkpoint did not restore update one")
    if _torch_tree_digest(restored.state_dict()) != updated_model_digest:
        raise RuntimeError("timm checkpoint changed model state during reload")
    if _torch_tree_digest(restored_optimizer.state_dict()) != updated_optimizer_digest:
        raise RuntimeError("timm checkpoint changed optimizer state during reload")

    restored.eval()
    embeddings = []
    labels = []
    splits = []
    batch_size = 10
    with torch.inference_mode():
        for start in range(0, len(evaluation_rows), batch_size):
            rows = evaluation_rows[start : start + batch_size]
            images = []
            for row in rows:
                with Image.open(Path(str(row["image"]))) as source:
                    images.append(evaluation_image(source.convert("RGB")))
                labels.append(int(row["label"]))
                splits.append(int(row["split"]))
            tensor = torch.from_numpy(np.stack(images)).to(device)
            embeddings.append(restored.encode(tensor).float().cpu().numpy())
    values = np.concatenate(embeddings)
    label_values = np.asarray(labels, dtype=np.int64)
    split_values = np.asarray(splits, dtype=np.int64)
    probe = linear_probe_metrics(
        values,
        label_values,
        split_values,
        inverse_regularization=(0.01, 0.1, 1.0, 10.0),
        max_iterations=1000,
        seed=17,
    )
    metrics = {
        "linear_probe_accuracy": probe["accuracy"],
        "linear_probe_f1_macro": probe["f1_macro"],
        "linear_probe_validation_accuracy": probe["validation_accuracy"],
        "linear_probe_selected_inverse_regularization": probe[
            "selected_inverse_regularization"
        ],
        "knn_accuracy": knn_accuracy(
            values,
            label_values,
            split_values,
            neighbors=20,
        ),
        **representation_geometry_metrics(values),
    }
    result = {
        "schema_version": "representax-lejepa-timm-reference-canary-v1",
        "status": "accepted",
        "scope": (
            "experiment-owned exact-paper one-update reference on disjoint ImageNet "
            "train subsets; not official ImageNet quality"
        ),
        "reference_contract": asdict(contract),
        "implementation": {
            "backbone": "timm vit_small_patch16_224",
            "projector": ("pinned stable-pretraining MLP 384->512->2048->2048->512"),
            "objective": "(1-lambda)*invariance + lambda*per-view-SIGReg",
            "local_crop_pixels": 98,
            "local_patch_semantics": "unpadded Conv2d floor grid (98px -> 6x6)",
            "pinned_helper_used_as_result": False,
            "parameter_count": parameter_count,
            "initial_parameter_digest": initial_parameter_digest,
            "initial_parameters_shared_with_representax": True,
            "shared_initialization": shared_manifest,
            "quality_trajectory_parity_claimed": False,
        },
        "training": {
            "updates": 1,
            "real_imagenet_examples": len(view_values),
            "real_imagenet_source_images": list(source_images),
            "views_per_example": 8,
            "global_views": 2,
            "local_views": 6,
            "loss": float(loss.detach().cpu()),
            "invariance": float(invariance.detach().cpu()),
            "sigreg": float(sigreg.detach().cpu()),
            "gradient_global_norm": float(gradient_norm.detach().cpu()),
            "update_seconds": update_seconds,
            "pre_update_objective": pre_update,
        },
        "checkpoint_reload": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "updates": 1,
            "model_state_exact": True,
            "optimizer_state_exact": True,
            "resumable_next_update": 2,
        },
        "evaluation": {
            "examples": len(values),
            "representation_dimension": values.shape[1],
            "metrics": metrics,
        },
        "devices": {
            "physical_gpu_indices": [5],
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        },
        "data": input_manifest,
    }
    _write_json(output / "result.json", result)
    return result


def run_canary(
    *,
    data_directory: Path,
    initialization_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Run one update, restore its checkpoint, then run one frozen evaluation."""

    import equinox as eqx
    import jax
    import numpy as np

    from representax import load_inference_bundle
    from representax.train import (
        CheckpointManager,
        run_job,
        scientific_fingerprint,
    )
    from representax.train.job import build_job_runtime

    if _visible_physical_gpus() != PHYSICAL_GPUS:
        raise RuntimeError(
            f"LeJEPA canary requires physical GPUs {PHYSICAL_GPUS}; "
            f"CUDA_VISIBLE_DEVICES exposes {_visible_physical_gpus()}"
        )
    if jax.default_backend() != "gpu" or len(jax.devices()) != 2:
        raise RuntimeError("LeJEPA canary requires exactly two visible JAX GPU devices")
    input_manifest = json.loads(
        (data_directory / "manifest.json").read_text(encoding="utf-8")
    )
    expected_scope = (
        "readiness-only deterministic disjoint subsets of real ImageNet-1K train"
    )
    if input_manifest.get("acceptance_scope") != expected_scope:
        raise ValueError(
            "LeJEPA canary requires the bounded real ImageNet input manifest"
        )
    shared_payload, shared_manifest = _load_shared_initialization(initialization_path)
    job = build_job(
        data_directory,
        architecture=CANARY_ARCHITECTURE,
        initialization_path=initialization_path,
    )
    initial_runtime = build_job_runtime(job, place_initial_state=False)
    initial_parameter_count = _parameter_count(initial_runtime.state.model)
    if initial_parameter_count != shared_manifest["parameter_count"]:
        raise RuntimeError(
            "Representax worker changed the shared architecture parameter count"
        )
    initial_parameter_digest = _array_tree_digest(initial_runtime.state.model)
    view_values, view_sizes, source_images = _real_multicrop_batch(data_directory)
    pre_update = _native_pre_update_objective(
        initial_runtime.state.model,
        view_values,
        view_sizes,
        shared_payload["directions"].numpy(),
    )
    del initial_runtime
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    run_directory = output / "run"
    started = time.perf_counter()
    completed = run_job(job, run_directory)
    jax.block_until_ready(completed.state)
    if completed.completed_iterations != 1 or int(completed.state.step) != 1:
        raise RuntimeError("LeJEPA lifecycle did not perform exactly one update")
    parameter_count_before_reload = _parameter_count(completed.state.model)
    updated_model_digest = _array_tree_digest(completed.state.model)
    updated_optimizer_digest = _array_tree_digest(completed.state.optimizer_state)
    inference_bundle = completed.inference_bundle
    if inference_bundle is None:
        raise RuntimeError("LeJEPA lifecycle did not publish a final inference bundle")
    inference_model, inference_job = load_inference_bundle(inference_bundle)
    if scientific_fingerprint(inference_job) != scientific_fingerprint(job):
        raise RuntimeError("reloaded LeJEPA inference job changed scientifically")
    parameter_count_after_export_reload = _parameter_count(inference_model)
    export_model_digest = _array_tree_digest(inference_model)
    if export_model_digest != updated_model_digest:
        raise RuntimeError("native inference export changed the trained model state")
    del completed
    gc.collect()

    runtime = build_job_runtime(job, place_initial_state=False)
    data_fingerprint = runtime.batches.data_fingerprint
    manager = CheckpointManager(
        run_directory,
        scientific_fingerprint=scientific_fingerprint(job),
        data_fingerprint=data_fingerprint,
        keep=1,
        asynchronous=False,
    )
    try:
        restored = manager.restore_training_state(runtime.state)
    finally:
        manager.close()
    if restored.iteration != 1 or int(restored.state.step) != 1:
        raise RuntimeError("LeJEPA checkpoint did not restore update one")
    restored_state = runtime.place_state(restored.state)
    parameter_count_after_reload = _parameter_count(restored_state.model)
    restored_model_digest = _array_tree_digest(restored_state.model)
    restored_optimizer_digest = _array_tree_digest(restored_state.optimizer_state)
    if (
        initial_parameter_count != parameter_count_before_reload
        or parameter_count_after_reload != parameter_count_before_reload
        or parameter_count_after_export_reload != parameter_count_before_reload
    ):
        raise RuntimeError("LeJEPA parameter count changed across checkpoint reload")
    if restored_model_digest != updated_model_digest:
        raise RuntimeError("LeJEPA checkpoint changed model state during reload")
    if restored_optimizer_digest != updated_optimizer_digest:
        raise RuntimeError("LeJEPA checkpoint changed optimizer state during reload")
    if runtime.evaluation_batches is None or len(runtime.evaluation_runners) != 1:
        raise RuntimeError("LeJEPA canonical evaluator was not constructed")
    checkpoint_evaluation = runtime.evaluation_runners[0].run(
        restored_state.model,
        runtime.evaluation_batches(),
        iteration=1,
        key=jax.random.key(job.training.seed + 1),
        place_batch=runtime.place_batch,
    )
    export_state = eqx.tree_at(
        lambda state: state.model, runtime.state, inference_model
    )
    exported_model = runtime.place_state(export_state).model
    export_evaluation = runtime.evaluation_runners[0].run(
        exported_model,
        runtime.evaluation_batches(),
        iteration=1,
        key=jax.random.key(job.training.seed + 1),
        place_batch=runtime.place_batch,
    )
    checkpoint_metrics = dict(checkpoint_evaluation.metrics)
    metrics = dict(export_evaluation.metrics)
    if checkpoint_metrics.keys() != metrics.keys() or any(
        not np.array_equal(
            np.asarray(checkpoint_metrics[name]),
            np.asarray(metrics[name]),
            equal_nan=True,
        )
        for name in metrics
    ):
        raise RuntimeError(
            "reloaded inference artifact did not reproduce checkpoint "
            "evaluation metrics"
        )
    required = {
        "valid/jepa_representation/linear_probe_accuracy",
        "valid/jepa_representation/knn_accuracy",
        "valid/jepa_representation/effective_rank",
        "valid/jepa_representation/feature_std_mean",
        "valid/jepa_representation/feature_std_min",
        "valid/jepa_representation/condition_number",
    }
    if not required <= metrics.keys():
        raise RuntimeError("LeJEPA evaluation omitted canonical representation metrics")
    finite_required = required - {"valid/jepa_representation/condition_number"}
    if not all(math.isfinite(metrics[name]) for name in finite_required):
        raise RuntimeError("LeJEPA evaluation produced non-finite required metrics")

    rows = _metric_rows(run_directory / "metrics.jsonl")
    updates = tuple(row for row in rows if row.get("event") == "training_step")
    if len(updates) != 1:
        raise RuntimeError("LeJEPA metric stream does not contain exactly one update")
    update = updates[0]["metrics"]
    loss = float(update["train/loss"])
    update_norm = float(update["train/update_global_norm"])
    if not math.isfinite(loss) or not math.isfinite(update_norm) or update_norm <= 0:
        raise RuntimeError("LeJEPA optimizer update was not finite and non-zero")
    resume_iterator = iter(runtime.batches)
    set_state = getattr(resume_iterator, "set_state", None)
    if not callable(set_state):
        raise RuntimeError("LeJEPA training iterator cannot restore its data cursor")
    set_state(restored.data_state)
    try:
        resume_batch = runtime.place_batch(next(resume_iterator))
        resume_update = runtime.step(
            restored_state,
            resume_batch,
            jax.random.fold_in(restored.rng, restored.iteration),
        )
        jax.block_until_ready(resume_update)
    finally:
        close = getattr(resume_iterator, "close", None)
        if callable(close):
            close()
    resume_loss = float(resume_update.metrics.loss)
    resume_update_norm = float(resume_update.metrics.update_global_norm)
    if (
        int(resume_update.state.step) != 2
        or not bool(resume_update.metrics.numeric_finite)
        or bool(resume_update.metrics.skipped_update)
        or not math.isfinite(resume_loss)
        or not math.isfinite(resume_update_norm)
        or resume_update_norm <= 0
    ):
        raise RuntimeError(
            "LeJEPA checkpoint resume update was not finite and non-zero"
        )
    checkpoint = restored.record
    result = {
        "schema_version": "representax-lejepa-imagenet1k-canary-v1",
        "status": "accepted",
        "scope": (
            "one-update one-evaluation lifecycle readiness on disjoint ImageNet train "
            "subsets; not model quality or official ImageNet validation"
        ),
        "reference_contract": asdict(reference_contract()),
        "data": input_manifest,
        "model": {
            "canary_profile": "timm/vit_small_patch16_224",
            "canary_architecture": CANARY_ARCHITECTURE,
            "serious_supported_profile": {
                "name": "paper ViT-L/16 304M-class backbone",
                "architecture": PAPER_VIT_LARGE_ARCHITECTURE,
            },
            "parameter_count": initial_parameter_count,
            "initial_parameter_digest": initial_parameter_digest,
            "initial_parameters_shared_with_torch": True,
            "shared_initialization": shared_manifest,
            "quality_trajectory_parity_claimed": False,
            "parameter_count_before_reload": parameter_count_before_reload,
            "parameter_count_after_reload": parameter_count_after_reload,
            "parameter_count_after_export_reload": (
                parameter_count_after_export_reload
            ),
            "training_output": "canonical projection MLP, 512 dimensions",
            "evaluation_output": (
                "frozen normalized concatenation of final two backbone CLS states, "
                "768 dimensions"
            ),
        },
        "devices": {
            "physical_gpu_indices": list(PHYSICAL_GPUS),
            "jax": list(_device_memory()),
        },
        "training": {
            "updates": 1,
            "real_imagenet_examples": job.training.global_batch_size,
            "real_imagenet_source_images": list(source_images),
            "views_per_example": 8,
            "global_views": 2,
            "local_views": 6,
            "objective": "(1-lambda)*invariance + lambda*per-view-SIGReg",
            "loss": loss,
            "invariance": float(update["train/invariance"]),
            "sigreg": float(update["train/sigreg"]),
            "update_global_norm": update_norm,
            "numeric_finite": bool(update["train/numeric_finite"]),
            "skipped_update": bool(update["train/skipped_update"]),
            "compile_and_first_update_seconds": float(
                update["perf/compilation_and_first_step_seconds"]
            ),
            "pre_update_objective": pre_update,
        },
        "checkpoint_reload": {
            "iteration": checkpoint.iteration,
            "path": str(checkpoint.path),
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "scientific_fingerprint": checkpoint.scientific_fingerprint,
            "data_fingerprint": checkpoint.data_fingerprint,
            "model_state_exact": True,
            "optimizer_state_exact": True,
            "resumable_next_update": 2,
            "resume_probe": {
                "executed": True,
                "from_iteration": 1,
                "optimizer_step_after": 2,
                "data_cursor_restored": True,
                "rng_restored": True,
                "loss": resume_loss,
                "update_global_norm": resume_update_norm,
                "numeric_finite": True,
                "skipped_update": False,
                "persisted": False,
            },
        },
        "evaluation": {
            "source": "reloaded native inference bundle",
            "invocation": "manual because evaluation.on_start/on_end are false",
            "configured_on_start": False,
            "configured_on_end": False,
            "batches": export_evaluation.batches,
            "examples": export_evaluation.examples,
            "duration_seconds": export_evaluation.duration_seconds,
            "compilation_seconds": export_evaluation.compilation_seconds,
            "metrics": metrics,
            "checkpoint_metrics": checkpoint_metrics,
            "metrics_reproduced_exactly": True,
        },
        "native_export_reload": {
            "path": str(inference_bundle),
            "manifest": str(Path(inference_bundle) / "manifest.json"),
            "complete_marker": str(Path(inference_bundle) / "REPRESENTAX_COMPLETE"),
            "model_state_reproduced_exactly": True,
            "metrics_reproduced_exactly": True,
        },
        "stable_pretraining_reference": stable_reference_decision(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(
        output / "manual-evaluation.json",
        {
            "schema_version": "representax-lejepa-manual-evaluation-v1",
            "status": "accepted",
            "iteration": 1,
            "source": "reloaded native inference bundle",
            "checkpoint_metrics": checkpoint_metrics,
            "exported_artifact_metrics": metrics,
            "metrics_reproduced_exactly": True,
            "batches": export_evaluation.batches,
            "examples": export_evaluation.examples,
        },
    )
    _write_json(output / "result.json", result)
    return result


def _capacity_projection(
    *,
    global_batch_size: int,
    steady_step_seconds: float,
    epochs: int = 100,
    train_images: int = IMAGENET_TRAIN_IMAGES,
    devices: int = 2,
) -> dict[str, float | int | str]:
    updates_per_epoch = math.ceil(train_images / global_batch_size)
    updates = updates_per_epoch * epochs
    training_seconds = updates * steady_step_seconds
    return {
        "scope": (
            "steady-training-only; excludes compilation, evaluation and checkpoints"
        ),
        "train_images": train_images,
        "epochs": epochs,
        "global_batch_size": global_batch_size,
        "updates_per_epoch": updates_per_epoch,
        "updates": updates,
        "training_seconds": training_seconds,
        "wall_hours": training_seconds / 3600.0,
        "logical_gpu_hours": training_seconds * devices / 3600.0,
    }


def run_capacity_canary(
    *,
    data_directory: Path,
    output: Path,
    global_batch_size: int,
    steps: int = 3,
) -> dict[str, Any]:
    """Measure the promoted ViT-L profile without lifecycle overhead."""

    import jax
    import numpy as np

    from representax.train import run_job

    if _visible_physical_gpus() != CAPACITY_PHYSICAL_GPUS:
        raise RuntimeError(
            f"LeJEPA capacity canary requires physical GPUs {CAPACITY_PHYSICAL_GPUS}; "
            f"CUDA_VISIBLE_DEVICES exposes {_visible_physical_gpus()}"
        )
    if jax.default_backend() != "gpu" or len(jax.devices()) != 2:
        raise RuntimeError(
            "LeJEPA capacity canary requires exactly two visible JAX GPU devices"
        )
    manifest_path = data_directory / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if input_manifest.get("status") != "accepted":
        raise ValueError("LeJEPA capacity canary requires accepted real-image inputs")

    job = build_capacity_job(
        data_directory,
        global_batch_size=global_batch_size,
        steps=steps,
    )
    started = time.perf_counter()
    completed = run_job(job, output)
    jax.block_until_ready(completed.state)
    elapsed_seconds = time.perf_counter() - started
    if completed.completed_iterations != steps or int(completed.state.step) != steps:
        raise RuntimeError("LeJEPA capacity canary did not complete every update")

    rows = _metric_rows(output / "metrics.jsonl")
    updates = tuple(row for row in rows if row.get("event") == "training_step")
    if len(updates) != steps:
        raise RuntimeError("LeJEPA capacity metric stream omitted training updates")
    update_metrics = tuple(row["metrics"] for row in updates)
    if any(
        not bool(metrics["train/numeric_finite"])
        or bool(metrics["train/skipped_update"])
        or not math.isfinite(float(metrics["train/loss"]))
        or float(metrics["train/update_global_norm"]) <= 0
        for metrics in update_metrics
    ):
        raise RuntimeError("LeJEPA capacity canary produced an invalid update")
    steady_step_seconds = tuple(
        float(metrics["perf/step_seconds"]) for metrics in update_metrics[1:]
    )
    steady_examples_per_second = tuple(
        float(metrics["perf/examples_per_second"]) for metrics in update_metrics[1:]
    )
    devices = _device_memory()
    result = {
        "schema_version": "representax-lejepa-vit-l16-capacity-v1",
        "status": "accepted",
        "scope": "training-only ViT-L/16 capacity and steady-throughput canary",
        "model": {
            "name": "LeJEPA ViT-L/16",
            "architecture": PAPER_VIT_LARGE_ARCHITECTURE,
            "parameter_count": _parameter_count(completed.state.model),
            "initialization": "deterministic random seed 7",
            "parameter_dtype": "float32",
            "compute_dtype": "bfloat16",
            "activation_rematerialization": "full",
        },
        "objective": {
            "views": {"global": 2, "local": 6},
            "global_resolution": 224,
            "local_resolution": 98,
            "sigreg_slices": 1024,
            "regularization_weight": 0.02,
        },
        "execution": {
            "physical_gpu_indices": list(CAPACITY_PHYSICAL_GPUS),
            "topology": "two-device FSDP over model axis",
            "global_batch_size": global_batch_size,
            "updates": steps,
            "compile_and_first_update_seconds": float(
                update_metrics[0]["perf/compilation_and_first_step_seconds"]
            ),
            "steady_step_seconds": list(steady_step_seconds),
            "median_steady_step_seconds": float(np.median(steady_step_seconds)),
            "steady_examples_per_second": list(steady_examples_per_second),
            "median_steady_examples_per_second": float(
                np.median(steady_examples_per_second)
            ),
            "losses": [float(metrics["train/loss"]) for metrics in update_metrics],
            "update_global_norms": [
                float(metrics["train/update_global_norm"])
                for metrics in update_metrics
            ],
            "elapsed_seconds": elapsed_seconds,
            "device_memory": list(devices),
        },
        "hundred_epoch_projection": _capacity_projection(
            global_batch_size=global_batch_size,
            steady_step_seconds=float(np.median(steady_step_seconds)),
        ),
        "excluded_overhead": ["evaluation", "checkpointing", "export"],
        "data_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
            "acceptance_scope": input_manifest.get("acceptance_scope"),
        },
    }
    _write_json(output / "capacity-result.json", result)
    return result


def aggregate_pair_results(output: Path) -> dict[str, Any]:
    """Apply every paired-framework acceptance gate to completed worker artifacts."""

    representax_result = json.loads(
        (output / "representax" / "result.json").read_text()
    )
    reference_result = json.loads(
        (output / "timm-reference" / "result.json").read_text()
    )
    objective_parity = json.loads(
        (output / "objective-parity" / "result.json").read_text()
    )
    parameter_counts_equal = (
        representax_result["model"]["parameter_count"]
        == reference_result["implementation"]["parameter_count"]
    )
    shared_state_fingerprint = representax_result["model"]["shared_initialization"][
        "state_fingerprint"
    ]
    shared_fingerprint_equal = (
        shared_state_fingerprint
        == reference_result["implementation"]["shared_initialization"][
            "state_fingerprint"
        ]
    )
    representax_pre_update = representax_result["training"]["pre_update_objective"]
    reference_pre_update = reference_result["training"]["pre_update_objective"]
    pre_update_tolerance = {"relative": 1e-4, "absolute": 1e-5}
    pre_update_agreement = {
        name: math.isclose(
            representax_pre_update[name],
            reference_pre_update[name],
            rel_tol=pre_update_tolerance["relative"],
            abs_tol=pre_update_tolerance["absolute"],
        )
        for name in ("loss", "invariance", "sigreg")
    }
    pre_update_agrees = all(pre_update_agreement.values())
    accepted = (
        representax_result["status"] == "accepted"
        and reference_result["status"] == "accepted"
        and objective_parity["status"] == "accepted"
        and parameter_counts_equal
        and shared_fingerprint_equal
        and pre_update_agrees
    )
    result = {
        "schema_version": "representax-lejepa-framework-pair-v1",
        "status": "accepted" if accepted else "rejected",
        "scope": "paired readiness canaries; not official ImageNet quality",
        "cross_framework_comparison": {
            "exact_architecture_parameter_count": parameter_counts_equal,
            "initial_parameters_identical": shared_fingerprint_equal,
            "initial_state_fingerprint": shared_state_fingerprint,
            "initialization_note": (
                "both workers load the same materialized timm backbone, projector, "
                "and evaluation LayerNorm arrays"
            ),
            "pre_update_objective": {
                "representax": representax_pre_update,
                "torch": reference_pre_update,
                "component_agreement": pre_update_agreement,
                "tolerance": pre_update_tolerance,
            },
            "quality_trajectory_parity_claimed": False,
            "quality_result_claimed": False,
        },
        "objective_parity": objective_parity,
        "representax": representax_result,
        "timm_reference": reference_result,
    }
    _write_json(output / "result.json", result)
    if not accepted:
        raise RuntimeError("paired LeJEPA acceptance gates did not all pass")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--imagenet-root", type=Path, default=DEFAULT_IMAGENET_ROOT)
    run = subparsers.add_parser("run")
    run.add_argument("--data-directory", type=Path, required=True)
    run.add_argument("--initialization", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--gpus", default="4,5")
    reference = subparsers.add_parser("reference")
    reference.add_argument("--data-directory", type=Path, required=True)
    reference.add_argument("--initialization", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--gpu", default="5")
    parity = subparsers.add_parser("parity")
    parity.add_argument("--output", type=Path, required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output", type=Path, required=True)
    all_command = subparsers.add_parser("all")
    all_command.add_argument("--output", type=Path, required=True)
    all_command.add_argument(
        "--imagenet-root", type=Path, default=DEFAULT_IMAGENET_ROOT
    )
    all_command.add_argument("--gpus", default="4,5")
    capacity = subparsers.add_parser("capacity")
    capacity.add_argument("--data-directory", type=Path, required=True)
    capacity.add_argument("--output", type=Path, required=True)
    capacity.add_argument("--gpus", default="0,1")
    capacity.add_argument("--global-batch-size", type=int, required=True)
    capacity.add_argument("--steps", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        result = prepare_imagenet(
            arguments.output,
            imagenet_root=arguments.imagenet_root,
        )
    elif arguments.command == "run":
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpus
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        result = run_canary(
            data_directory=arguments.data_directory,
            initialization_path=arguments.initialization,
            output=arguments.output,
        )
    elif arguments.command == "reference":
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
        result = run_timm_reference_canary(
            data_directory=arguments.data_directory,
            initialization_path=arguments.initialization,
            output=arguments.output,
        )
    elif arguments.command == "capacity":
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpus
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
        result = run_capacity_canary(
            data_directory=arguments.data_directory,
            output=arguments.output,
            global_batch_size=arguments.global_batch_size,
            steps=arguments.steps,
        )
    elif arguments.command == "parity":
        arguments.output.mkdir(parents=True, exist_ok=False)
        result = exact_objective_parity()
        _write_json(arguments.output / "result.json", result)
    elif arguments.command == "initialize":
        result = materialize_shared_initialization(arguments.output)
    elif arguments.command == "aggregate":
        result = aggregate_pair_results(arguments.output)
    else:
        data_directory = arguments.output / "data"
        prepare_imagenet(data_directory, imagenet_root=arguments.imagenet_root)
        script = Path(__file__).resolve()
        environment = dict(os.environ)
        environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        parity_environment = dict(environment)
        parity_environment["CUDA_VISIBLE_DEVICES"] = ""
        parity_environment["JAX_PLATFORMS"] = "cpu"
        subprocess.run(
            (
                sys.executable,
                str(script),
                "parity",
                "--output",
                str(arguments.output / "objective-parity"),
            ),
            check=True,
            env=parity_environment,
        )
        initialization_path = arguments.output / "shared-initialization.pt"
        subprocess.run(
            (
                sys.executable,
                str(script),
                "initialize",
                "--output",
                str(initialization_path),
            ),
            check=True,
            env=parity_environment,
        )
        subprocess.run(
            (
                sys.executable,
                str(script),
                "run",
                "--data-directory",
                str(data_directory),
                "--initialization",
                str(initialization_path),
                "--output",
                str(arguments.output / "representax"),
                "--gpus",
                arguments.gpus,
            ),
            check=True,
            env=environment,
        )
        subprocess.run(
            (
                sys.executable,
                str(script),
                "reference",
                "--data-directory",
                str(data_directory),
                "--initialization",
                str(initialization_path),
                "--output",
                str(arguments.output / "timm-reference"),
                "--gpu",
                "5",
            ),
            check=True,
            env=environment,
        )
        result = aggregate_pair_results(arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANARY_ARCHITECTURE",
    "ImageNetAccessError",
    "LeJEPAReferenceContract",
    "PAPER_VIT_LARGE_ARCHITECTURE",
    "build_job",
    "build_capacity_job",
    "load_lejepa_model",
    "prepare_imagenet",
    "reference_contract",
    "run_canary",
    "run_capacity_canary",
    "run_timm_reference_canary",
    "stable_reference_decision",
]
