"""Matched V-JEPA 2.1 ViT-B/16 paper-readiness preflight."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
PANEL_MANIFEST = ROOT / "benchmarks/configs/paper-multimodal-jepa-v1.json"
FRAMEWORKS = ("representax", "facebookresearch-vjepa2")
PREFLIGHT_BATCH_SIZE = 1
PREFLIGHT_STEPS = 4
PREFLIGHT_TRAIN_VIDEOS = 4
PREFLIGHT_EVALUATION_VIDEOS = 4

MASK_PATTERNS = (
    {
        "spatial_scale": (0.15, 0.15),
        "temporal_scale": (1.0, 1.0),
        "aspect_ratio": (0.75, 1.5),
        "num_blocks": 8,
    },
    {
        "spatial_scale": (0.7, 0.7),
        "temporal_scale": (1.0, 1.0),
        "aspect_ratio": (0.75, 1.5),
        "num_blocks": 2,
    },
)


@dataclass(frozen=True, slots=True)
class FrozenContract:
    reference_commit: str
    reference_config: str
    dataset: Mapping[str, Any]
    global_batch_size: int
    video_frames: int
    image_resolution: int


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract() -> FrozenContract:
    """Resolve and validate the V-JEPA row frozen across paper manifests."""

    campaign = _document(CAMPAIGN_MANIFEST)
    panel = _document(PANEL_MANIFEST)
    campaign_row = next(
        row
        for row in campaign["workloads"]
        if row["name"] == "vjepa2-1-video-representation"
    )
    panel_row = next(
        row
        for row in panel["workloads"]
        if row["name"] == "vjepa2-1-video-representation"
    )
    if campaign_row["frameworks"] != list(FRAMEWORKS):
        raise ValueError("the frozen V-JEPA frameworks changed")
    if panel_row["reference"] != "vjepa2":
        raise ValueError("the frozen V-JEPA reference changed")
    model = panel["models"][panel_row["model"]]
    reference = panel["references"][panel_row["reference"]]
    if model["revision"] != reference:
        raise ValueError("the V-JEPA model and reference revisions diverged")
    return FrozenContract(
        reference_commit=reference,
        reference_config=model["config"],
        dataset=panel["datasets"][panel_row["train"]],
        global_batch_size=int(campaign_row["global_batch"]),
        video_frames=int(campaign_row["video_frames"]),
        image_resolution=int(campaign_row["image_resolution"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _decode_video(payload: bytes, scratch: Path) -> np.ndarray:
    """Decode one source video to RGB frames without adding a Python codec stack."""

    scratch.write_bytes(payload)
    probe = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(scratch),
        ),
        check=True,
        capture_output=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    decoded = subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(scratch),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ),
        check=True,
        capture_output=True,
    ).stdout
    frame_elements = height * width * 3
    if not decoded or len(decoded) % frame_elements:
        raise ValueError(f"ffmpeg produced an invalid RGB stream for {scratch.name}")
    return np.frombuffer(decoded, dtype=np.uint8).reshape(-1, height, width, 3)


def _sample_frames(frames: np.ndarray, count: int) -> np.ndarray:
    if not len(frames):
        raise ValueError("decoded video contains no frames")
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(np.int64)
    return np.ascontiguousarray(frames[indices])


def _process_frames(frames: np.ndarray, *, training: bool, seed: int) -> np.ndarray:
    from representax.models.vjepa2_1.processing import _random_resized_crop, _resize

    contract = frozen_contract()
    sampled = _sample_frames(frames, contract.video_frames)
    if training:
        sampled = _random_resized_crop(
            sampled,
            size=contract.image_resolution,
            scale=(0.3, 1.0),
            ratio=(0.75, 1.35),
            rng=np.random.default_rng(seed),
        )
    else:
        sampled = _resize(
            sampled,
            contract.image_resolution,
            contract.image_resolution,
        )
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (sampled.astype(np.float32) / 255.0 - mean) / std
    return np.ascontiguousarray(np.moveaxis(normalized, -1, 0), dtype=np.float32)


def _materialize_split(
    directory: Path,
    *,
    split: str,
    count: int,
    training: bool,
    seed: int,
    skip: int = 0,
) -> tuple[dict[str, Any], ...]:
    import datasets

    contract = frozen_contract()
    source: Any = datasets.load_dataset(
        contract.dataset["repo_id"],
        revision=contract.dataset["revision"],
        split=split,
        streaming=True,
    ).cast_column("video", datasets.Video(decode=False))
    rows = []
    for source_index, row in enumerate(source):
        if source_index < skip:
            continue
        if len(rows) >= count:
            break
        video = row["video"]
        payload = video.get("bytes")
        if payload is None:
            payload = Path(video["path"]).read_bytes()
        stem = f"{len(rows):04d}"
        raw = directory / "raw" / split / f"{stem}.webm"
        raw.parent.mkdir(parents=True, exist_ok=True)
        try:
            frames = _decode_video(payload, raw)
        except (ValueError, subprocess.CalledProcessError):
            continue
        tensor = directory / "tensors" / split / f"{stem}.npy"
        tensor.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            tensor,
            _process_frames(frames, training=training, seed=seed + source_index),
            allow_pickle=False,
        )
        rows.append(
            {
                "source_index": source_index,
                "source_path": str(video.get("path", "")),
                "tensor": str(tensor.relative_to(directory)),
                "tensor_sha256": _sha256(tensor),
            }
        )
    if len(rows) != count:
        raise ValueError(
            f"SSV2 {split} produced {len(rows)} usable videos; expected {count}"
        )
    return tuple(rows)


def _official_modules(reference: Path) -> tuple[Any, Any]:
    reference_text = str(reference.resolve())
    if reference_text not in sys.path:
        sys.path.insert(0, reference_text)
    encoder_module = import_module("app.vjepa_2_1.models.vision_transformer")
    predictor_module = import_module("app.vjepa_2_1.models.predictor")
    return encoder_module.VisionTransformer, predictor_module.VisionTransformerPredictor


def _official_models(
    reference: Path,
    *,
    device: Any,
    activation_checkpointing: bool,
) -> tuple[Any, Any, Any]:
    import torch

    encoder_type, predictor_type = _official_modules(reference)

    def norm(size: int) -> Any:
        return torch.nn.LayerNorm(size, eps=1e-6)

    encoder = encoder_type(
        img_size=(256, 256),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=norm,
        use_rope=True,
        interpolate_rope=True,
        img_temporal_dim_size=1,
        modality_embedding=True,
        n_output_distillation=4,
        use_sdpa=True,
        use_activation_checkpointing=activation_checkpointing,
    )
    predictor = predictor_type(
        img_size=(256, 256),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=norm,
        use_rope=True,
        interpolate_rope=True,
        img_temporal_dim_size=1,
        modality_embedding=True,
        n_output_distillation=4,
        use_sdpa=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        return_all_tokens=True,
        use_activation_checkpointing=activation_checkpointing,
    )
    target = copy.deepcopy(encoder)
    return encoder.to(device), predictor.to(device), target.to(device)


def _write_initial_checkpoint(reference: Path, output: Path, seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    encoder, predictor, target = _official_models(
        reference,
        device=torch.device("cpu"),
        activation_checkpointing=False,
    )
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target.state_dict(),
        },
        output,
    )


def prepare_data(
    output: Path,
    *,
    reference: Path,
    train_videos: int = PREFLIGHT_TRAIN_VIDEOS,
    evaluation_videos: int = PREFLIGHT_EVALUATION_VIDEOS,
    seed: int = 7,
) -> dict[str, Any]:
    """Materialize a bounded, deterministic SSV2 tensor and initialization set."""

    if min(train_videos, evaluation_videos) <= 1:
        raise ValueError("V-JEPA preflight needs at least two videos per split")
    contract = frozen_contract()
    actual_commit = subprocess.run(
        ("git", "-C", str(reference), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != contract.reference_commit:
        raise ValueError(
            f"expected Meta commit {contract.reference_commit}, found {actual_commit}"
        )
    output.mkdir(parents=True, exist_ok=False)
    import datasets

    available_splits = datasets.get_dataset_split_names(
        contract.dataset["repo_id"],
        revision=contract.dataset["revision"],
    )
    requested_evaluation_split = str(contract.dataset["evaluate_split"])
    if requested_evaluation_split in available_splits:
        evaluation_split = requested_evaluation_split
        evaluation_skip = 0
        split_deviation = None
    else:
        evaluation_split = str(contract.dataset["train_split"])
        evaluation_skip = train_videos
        split_deviation = {
            "requested": requested_evaluation_split,
            "actual": evaluation_split,
            "reason": "pinned Hugging Face revision exposes only the train split",
            "policy": "disjoint deterministic SSV2 holdout for readiness only",
        }
    train = _materialize_split(
        output,
        split=str(contract.dataset["train_split"]),
        count=train_videos,
        training=True,
        seed=seed,
    )
    evaluation = _materialize_split(
        output,
        split=evaluation_split,
        count=evaluation_videos,
        training=False,
        seed=seed + 10_000,
        skip=evaluation_skip,
    )
    _write_jsonl(output / "train.jsonl", train)
    _write_jsonl(output / "evaluation.jsonl", evaluation)

    from representax.models.vjepa2_1 import VJEPAMaskConfig, sample_vjepa_masks

    masks = sample_vjepa_masks(
        tuple(VJEPAMaskConfig.model_validate(pattern) for pattern in MASK_PATTERNS),
        batch_size=PREFLIGHT_BATCH_SIZE,
        grid=(8, 16, 16),
        seed=seed,
    )
    np.savez_compressed(
        output / "masks.npz",
        context_ids=masks[0],
        target_ids=masks[1],
        context_valid=masks[2],
        target_valid=masks[3],
    )
    checkpoint = output / "official-initialization.pth.tar"
    _write_initial_checkpoint(reference, checkpoint, seed)
    manifest = {
        "schema_version": "representax-vjepa-preflight-data-v1",
        "contract": asdict(contract),
        "reference_checkout": str(reference.resolve()),
        "seed": seed,
        "training_videos": train_videos,
        "evaluation_videos": evaluation_videos,
        "available_dataset_splits": available_splits,
        "evaluation_split_deviation": split_deviation,
        "mask_patterns": MASK_PATTERNS,
        "files": {
            "train.jsonl": _sha256(output / "train.jsonl"),
            "evaluation.jsonl": _sha256(output / "evaluation.jsonl"),
            "masks.npz": _sha256(output / "masks.npz"),
            "official-initialization.pth.tar": _sha256(checkpoint),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


class VJEPAPreflightCollator:
    """Load the preprocessed SSV2 clips and shared official mask plan."""

    def __init__(self, *, root_directory: str | Path) -> None:
        self.root_directory = Path(root_directory).resolve()

    def data_contract(self) -> Mapping[str, Any]:
        manifest = _document(self.root_directory / "manifest.json")
        return {
            "schema_version": "representax-vjepa-preflight-collator-v1",
            "manifest": manifest["files"],
            "root_directory": str(self.root_directory),
        }

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        import jax.numpy as jnp

        from representax.tasks.jepa import VJEPA2_1Batch

        pixels = np.stack(
            tuple(
                np.load(self.root_directory / str(row["tensor"]), allow_pickle=False)
                for row in rows
            )
        )
        with np.load(self.root_directory / "masks.npz") as masks:
            repeat = len(rows)
            values = {
                name: np.repeat(masks[name], repeat, axis=0)
                for name in (
                    "context_ids",
                    "target_ids",
                    "context_valid",
                    "target_valid",
                )
            }
        return VJEPA2_1Batch(
            pixels=jnp.asarray(pixels),
            context_ids=jnp.asarray(values["context_ids"]),
            target_ids=jnp.asarray(values["target_ids"]),
            context_valid=jnp.asarray(values["context_valid"]),
            target_valid=jnp.asarray(values["target_valid"]),
        )


def _representax_job(
    *,
    data_directory: Path,
    steps: int,
    seed: int,
) -> Any:
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even integer of at least four")
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        ExportConfig,
        JobConfig,
        LoggingConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.jepa import VJEPA2_1DenseConfig, VJEPA2_1TaskConfig

    data = DataConfig(
        distribution=mix(
            source(str(data_directory / "train.jsonl"), map=identity),
            shuffle=False,
        ),
        collate=ComponentConfig(
            target="experiments.paper.vjepa:VJEPAPreflightCollator",
            parameters={"root_directory": str(data_directory)},
        ),
        drop_remainder=True,
        num_threads=0,
        prefetch_buffer_size=0,
    )
    return JobConfig(
        name="paper-preflight-vjepa2-1-video-representation",
        model=ModelConfig(
            target="representax.models.vjepa2_1:load_vjepa2_1",
            parameters={
                "config": {
                    "image_size": 256,
                    "patch_size": 16,
                    "video_frames": 16,
                    "tubelet_size": 2,
                    "hidden_size": 768,
                    "depth": 12,
                    "heads": 12,
                    "predictor_hidden_size": 384,
                    "predictor_depth": 12,
                    "predictor_heads": 12,
                    "supervision_layers": [2, 5, 8, 11],
                },
                "modality": "video",
                "checkpoint": str(data_directory / "official-initialization.pth.tar"),
                "training": False,
                "dtype": "float32",
                "implementation": "xla",
                "rematerialization": "full",
            },
        ),
        task=VJEPA2_1TaskConfig(),
        loss=VJEPA2_1DenseConfig(
            context_weight=0.5,
            ema_start=0.99925,
            ema_end=0.99925,
            ema_steps=steps,
        ),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1e-8,
                    "weight_decay": 0.04,
                },
            ),
            schedule=ComponentConfig(
                target="optax.linear_schedule",
                parameters={
                    "init_value": 1e-4,
                    "end_value": 6e-4,
                    "transition_steps": 12_000,
                },
            ),
            max_gradient_norm=None,
        ),
        data=data,
        training=TrainingConfig(
            global_batch_size=PREFLIGHT_BATCH_SIZE,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(micro_batch_size=PREFLIGHT_BATCH_SIZE),
            activation_rematerialization="full",
            donate_buffers=True,
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=CheckpointConfig(
            every=steps // 2,
            keep=2,
            save_final=True,
            asynchronous=True,
        ),
        logging=LoggingConfig(console_every=1, timing=True, accelerator=True),
        export=ExportConfig(),
    )


def _geometry(embeddings: np.ndarray) -> dict[str, float]:
    from representax.evaluation.jepa_representation import (
        representation_geometry_metrics,
    )

    return representation_geometry_metrics(np.asarray(embeddings, dtype=np.float32))


def _native_embeddings(model: Any, data_directory: Path) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    from representax.core import Route

    records = _read_jsonl(data_directory / "evaluation.jsonl")
    values = []
    for row in records:
        pixels = np.load(data_directory / str(row["tensor"]), allow_pickle=False)
        values.append(
            np.asarray(model.encode(jnp.asarray(pixels[None]), route=Route.GENERIC))
        )
    jax.block_until_ready(values)
    return np.concatenate(values)


def _steady_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    durations = []
    compilations = []
    for row in rows:
        if row.get("event") != "training_step":
            continue
        metrics = row["metrics"]
        compilation = metrics.get("perf/compilation_and_first_step_seconds")
        if compilation is not None:
            compilations.append(float(compilation))
            continue
        duration = metrics.get("perf/step_seconds")
        if duration is not None and float(duration) > 0:
            durations.append(float(duration))
    report: dict[str, Any] = {
        "compilation_seconds": sum(compilations),
        "compilation_events": len(compilations),
        "measured_steps": len(durations),
    }
    if durations:
        report.update(
            {
                "median_step_seconds": statistics.median(durations),
                "examples_per_second": len(durations) / sum(durations),
            }
        )
    return report


def _representax_worker(
    *,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    import jax

    from representax import load_inference_bundle
    from representax.models.vjepa2_1 import VJEPA2_1Model
    from representax.train import run_job

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("V-JEPA preflight requires exactly one visible GPU")
    job = _representax_job(
        data_directory=data_directory,
        steps=steps,
        seed=seed,
    )
    from representax.models.vjepa2_1 import load_vjepa2_1

    initial_model, _ = load_vjepa2_1(
        key=jax.random.key(seed),
        **job.model.parameters,
    )
    initial_evaluation = _geometry(_native_embeddings(initial_model, data_directory))
    del initial_model
    gc.collect()
    jax.clear_caches()

    started = time.perf_counter()
    paused = run_job(job, run_directory, stop_after=steps // 2)
    if paused.completed_iterations != steps // 2:
        raise RuntimeError("Representax did not stop at the midpoint checkpoint")
    del paused
    gc.collect()
    jax.clear_caches()
    completed = run_job(job, run_directory, resume=True)
    jax.block_until_ready(completed.state)
    elapsed = time.perf_counter() - started
    if not completed.resumed or completed.completed_iterations != steps:
        raise RuntimeError("Representax did not resume to the final update")
    if completed.inference_bundle is None:
        raise RuntimeError("Representax did not export an inference bundle")
    if not isinstance(completed.state.model, VJEPA2_1Model):
        raise TypeError("Representax returned a different model family")
    expected = _native_embeddings(completed.state.model, data_directory)
    final_evaluation = _geometry(expected)
    reloaded, reloaded_job = load_inference_bundle(completed.inference_bundle)
    if reloaded_job != job or not isinstance(reloaded, VJEPA2_1Model):
        raise RuntimeError("native reload reconstructed a different model or job")
    actual = _native_embeddings(reloaded, data_directory)
    reload_difference = float(np.max(np.abs(expected - actual)))
    if reload_difference != 0.0:
        raise RuntimeError("native inference reload changed V-JEPA embeddings")

    rows = _read_jsonl(run_directory / "metrics.jsonl")
    updates = [row for row in rows if row.get("event") == "training_step"]
    if len(updates) != steps:
        raise RuntimeError("Representax evidence is missing training updates")
    losses = [float(row["metrics"]["train/loss"]) for row in updates]
    if not np.all(np.isfinite(losses)):
        raise RuntimeError("Representax produced a non-finite loss")
    return {
        "schema_version": "representax-vjepa-worker-v1",
        "framework": "representax",
        "steps": steps,
        "global_batch_size": PREFLIGHT_BATCH_SIZE,
        "elapsed_seconds": elapsed,
        "timing": _steady_state(rows),
        "losses": losses,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "resumed": completed.resumed,
        "checkpoint": str(run_directory / "checkpoints" / str(steps // 2)),
        "inference_bundle": str(completed.inference_bundle),
        "reload_maximum_absolute_difference": reload_difference,
        "device": jax.devices()[0].device_kind,
        "imagenet_probe": "gated-not-run-in-bounded-preflight",
    }


def _load_reference_state(
    reference: Path,
    checkpoint: Path,
    *,
    device: Any,
    activation_checkpointing: bool,
) -> tuple[Any, Any, Any]:
    import torch

    encoder, predictor, target = _official_models(
        reference,
        device=device,
        activation_checkpointing=activation_checkpointing,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    target.load_state_dict(state["target_encoder"])
    return encoder, predictor, target


def _reference_embeddings(model: Any, data_directory: Path, device: Any) -> np.ndarray:
    import torch

    model.eval()
    values = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in _read_jsonl(data_directory / "evaluation.jsonl"):
            pixels = torch.from_numpy(
                np.load(data_directory / str(row["tensor"]), allow_pickle=False)
            )[None].to(device)
            tokens = model(pixels, training=False)
            values.append(tokens.mean(dim=1).float().cpu().numpy())
    return np.concatenate(values)


def _reference_loss(
    encoder: Any,
    predictor: Any,
    target: Any,
    pixels: Any,
    masks: Mapping[str, np.ndarray],
) -> Any:
    import torch
    import torch.nn.functional as functional

    compute_mask_distance = import_module(
        "app.vjepa_2_1.models.utils.masks_dist"
    ).compute_mask_distance

    target_features = target(pixels, training=True)
    target_features = torch.cat(
        tuple(
            functional.layer_norm(
                target_features[..., start : start + 768],
                (768,),
                eps=1e-6,
            )
            for start in range(0, 3072, 768)
        ),
        dim=-1,
    )
    losses = []
    for mask_index in range(masks["context_ids"].shape[1]):
        context = torch.from_numpy(masks["context_ids"][:, mask_index]).to(
            pixels.device
        )
        target_ids = torch.from_numpy(masks["target_ids"][:, mask_index]).to(
            pixels.device
        )
        context_valid = torch.from_numpy(masks["context_valid"][:, mask_index]).to(
            pixels.device
        )
        target_valid = torch.from_numpy(masks["target_valid"][:, mask_index]).to(
            pixels.device
        )
        context = context[:, : int(context_valid.sum(dim=1).min())]
        target_ids = target_ids[:, : int(target_valid.sum(dim=1).min())]
        context_features = encoder(pixels, masks=context, training=True)
        predicted_target, predicted_context = predictor(
            context_features,
            context,
            target_ids,
            mod="video",
            mask_index=0,
        )
        batch_indices = torch.arange(len(pixels), device=pixels.device)[:, None]
        target_for_prediction = target_features[batch_indices, target_ids]
        target_for_context = target_features[batch_indices, context]
        distance = compute_mask_distance(
            [[target_ids]],
            [[context]],
            grid_size=16,
            offset_context_loss=False,
        )[0][0]
        prediction_loss = torch.mean(
            torch.abs(predicted_target - target_for_prediction)
        )
        context_loss = torch.mean(
            torch.abs(predicted_context - target_for_context)
            * (1.0 / distance.unsqueeze(-1))
        )
        losses.append(prediction_loss + 0.5 * context_loss)
    return torch.stack(losses).mean()


def _save_reference_checkpoint(
    path: Path,
    *,
    encoder: Any,
    predictor: Any,
    target: Any,
    optimizer: Any,
    step: int,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def _facebookresearch_worker(
    *,
    reference: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Meta V-JEPA preflight requires exactly one visible GPU")
    contract = frozen_contract()
    actual_commit = subprocess.run(
        ("git", "-C", str(reference), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != contract.reference_commit:
        raise RuntimeError("Meta reference checkout does not match the frozen commit")
    run_directory.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial_checkpoint = data_directory / "official-initialization.pth.tar"
    encoder, predictor, target = _load_reference_state(
        reference,
        initial_checkpoint,
        device=device,
        activation_checkpointing=True,
    )
    initial_evaluation = _geometry(
        _reference_embeddings(target, data_directory, device)
    )
    for parameter in target.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (*encoder.parameters(), *predictor.parameters()),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.04,
    )
    records = _read_jsonl(data_directory / "train.jsonl")
    with np.load(data_directory / "masks.npz") as loaded_masks:
        masks = {name: loaded_masks[name] for name in loaded_masks.files}
    losses = []
    durations = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def run_updates(start: int, stop: int) -> None:
        for iteration in range(start, stop):
            lr = 1e-4 + (6e-4 - 1e-4) * min(iteration, 12_000) / 12_000
            for group in optimizer.param_groups:
                group["lr"] = lr
            pixels = torch.from_numpy(
                np.load(
                    data_directory / str(records[iteration]["tensor"]),
                    allow_pickle=False,
                )
            )[None].to(device)
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = _reference_loss(
                    encoder,
                    predictor,
                    target,
                    pixels,
                    masks,
                )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for online, target_parameter in zip(
                    encoder.parameters(), target.parameters(), strict=True
                ):
                    target_parameter.mul_(0.99925).add_(online, alpha=0.00075)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - step_started)
            losses.append(float(loss.detach()))

    midpoint = steps // 2
    run_updates(0, midpoint)
    midpoint_path = run_directory / "checkpoints" / f"checkpoint-{midpoint}.pt"
    _save_reference_checkpoint(
        midpoint_path,
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        step=midpoint,
    )
    del encoder, predictor, target, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    encoder, predictor, target = _load_reference_state(
        reference,
        midpoint_path,
        device=device,
        activation_checkpointing=True,
    )
    for parameter in target.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (*encoder.parameters(), *predictor.parameters()),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.04,
    )
    checkpoint_state = torch.load(
        midpoint_path, map_location=device, weights_only=False
    )
    optimizer.load_state_dict(checkpoint_state["optimizer"])
    run_updates(midpoint, steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final_evaluation = _geometry(_reference_embeddings(target, data_directory, device))
    final_path = run_directory / "final-model.pt"
    _save_reference_checkpoint(
        final_path,
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        step=steps,
    )
    expected = _reference_embeddings(target, data_directory, device)
    _, _, reloaded_target = _load_reference_state(
        reference,
        final_path,
        device=device,
        activation_checkpointing=False,
    )
    actual = _reference_embeddings(reloaded_target, data_directory, device)
    reload_difference = float(np.max(np.abs(expected - actual)))
    if reload_difference != 0.0 or not np.all(np.isfinite(losses)):
        raise RuntimeError("Meta checkpoint reload or finite-loss acceptance failed")
    warm = durations[1:]
    return {
        "schema_version": "representax-vjepa-worker-v1",
        "framework": "facebookresearch-vjepa2",
        "reference_commit": actual_commit,
        "torch_version": torch.__version__,
        "steps": steps,
        "global_batch_size": PREFLIGHT_BATCH_SIZE,
        "elapsed_seconds": elapsed,
        "timing": {
            "compilation_seconds": 0.0,
            "compilation_events": 0,
            "measured_steps": len(warm),
            "median_step_seconds": statistics.median(warm),
            "examples_per_second": len(warm) / sum(warm),
        },
        "losses": losses,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "resumed": True,
        "checkpoint": str(midpoint_path),
        "inference_bundle": str(final_path),
        "reload_maximum_absolute_difference": reload_difference,
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(),
        "imagenet_probe": "gated-not-run-in-bounded-preflight",
    }


def _worker(arguments: argparse.Namespace) -> None:
    if arguments.framework == "representax":
        report = _representax_worker(
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            steps=arguments.steps,
            seed=arguments.seed,
        )
    else:
        report = _facebookresearch_worker(
            reference=arguments.reference,
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            steps=arguments.steps,
            seed=arguments.seed,
        )
    _write_json(arguments.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _pair(arguments: argparse.Namespace) -> None:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    commands = {}
    for framework in FRAMEWORKS:
        report = output / f"{framework}.json"
        log = output / f"{framework}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.paper.vjepa",
            "worker",
            "--framework",
            framework,
            "--reference",
            str(arguments.reference),
            "--data-directory",
            str(arguments.data_directory),
            "--run-directory",
            str(output / framework),
            "--report",
            str(report),
            "--steps",
            str(arguments.steps),
            "--seed",
            str(arguments.seed),
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(arguments.gpu),
            "PYTHONUNBUFFERED": "1",
        }
        if framework == "representax":
            environment.update(
                {
                    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                    "JAX_COMPILATION_CACHE_DIR": str(output / "jax-cache"),
                }
            )
        commands[framework] = command
        with log.open("x", encoding="utf-8") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        reports[framework] = _document(report)
    summary = {
        "schema_version": "representax-vjepa-preflight-v1",
        "contract": {
            **asdict(frozen_contract()),
            "steps": arguments.steps,
            "seed": arguments.seed,
            "gpu": arguments.gpu,
            "data_manifest": _document(arguments.data_directory / "manifest.json"),
        },
        "commands": commands,
        **reports,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(reports, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--reference", type=Path, required=True)
    prepare.add_argument("--train-videos", type=int, default=PREFLIGHT_TRAIN_VIDEOS)
    prepare.add_argument(
        "--evaluation-videos", type=int, default=PREFLIGHT_EVALUATION_VIDEOS
    )
    prepare.add_argument("--seed", type=int, default=7)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--framework", choices=FRAMEWORKS, required=True)
    worker.add_argument("--reference", type=Path, required=True)
    worker.add_argument("--data-directory", type=Path, required=True)
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=PREFLIGHT_STEPS)
    worker.add_argument("--seed", type=int, default=7)

    pair = subparsers.add_parser("pair")
    pair.add_argument("--reference", type=Path, required=True)
    pair.add_argument("--data-directory", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--steps", type=int, default=PREFLIGHT_STEPS)
    pair.add_argument("--seed", type=int, default=7)
    pair.add_argument("--gpu", type=int, default=4)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        manifest = prepare_data(
            arguments.output,
            reference=arguments.reference,
            train_videos=arguments.train_videos,
            evaluation_videos=arguments.evaluation_videos,
            seed=arguments.seed,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif arguments.command == "worker":
        _worker(arguments)
    else:
        _pair(arguments)


if __name__ == "__main__":
    main()


__all__ = [
    "FrozenContract",
    "VJEPAPreflightCollator",
    "frozen_contract",
    "prepare_data",
]
