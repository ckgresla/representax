from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
from experiments.preflights.vjepa import (
    FRAMEWORKS,
    PREFLIGHT_BATCH_SIZE,
    VJEPAPreflightCollator,
    _parser,
    _representax_job,
    frozen_contract,
)

from representax.tasks.jepa.vjepa2_1 import (
    dense_prediction_loss,
    mask_distance_weights,
)


def test_frozen_vjepa_contract() -> None:
    contract = frozen_contract()

    assert contract.reference_commit == "204698b45b3712590f06245fbfba32d3be539812"
    assert contract.reference_config == (
        "configs/train_2_1/vitb16/pretrain-256px-16f.yaml"
    )
    assert contract.dataset["source"] == "qualcomm-something-something-v2"
    assert contract.dataset["evaluate_split"] == "validation"
    assert contract.dataset["preflight_mirror"] == {
        "repo_id": "jxie/something_something_v2",
        "revision": "86851581ad645fbce86488d474604ea2021ab3cc",
        "available_split": "train",
    }
    assert contract.global_batch_size == 128
    assert contract.video_frames == 16
    assert contract.image_resolution == 256
    assert FRAMEWORKS == ("representax", "facebookresearch-vjepa2")


def test_vjepa_collator_loads_shared_pixels_and_masks(tmp_path) -> None:
    tensor = np.zeros((3, 16, 8, 8), dtype=np.float32)
    np.save(tmp_path / "clip.npy", tensor)
    shape = (1, 2, 3)
    np.savez_compressed(
        tmp_path / "masks.npz",
        context_ids=np.arange(6, dtype=np.int32).reshape(shape),
        target_ids=np.arange(6, dtype=np.int32).reshape(shape),
        context_valid=np.ones(shape, dtype=bool),
        target_valid=np.ones(shape, dtype=bool),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {"masks.npz": "sha256:test"}})
    )

    batch = VJEPAPreflightCollator(root_directory=tmp_path)(({"tensor": "clip.npy"},))

    assert batch.pixels.shape == (1, 3, 16, 8, 8)
    assert batch.context_ids.shape == shape
    assert batch.target_ids.shape == shape
    assert np.all(np.asarray(batch.context_valid))
    assert np.all(np.asarray(batch.target_valid))


def test_representax_job_uses_full_frozen_architecture_and_lifecycle(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(
        "\n".join(json.dumps({"tensor": "clip.npy"}) for _ in range(4)) + "\n"
    )
    (data / "official-initialization.pth.tar").write_bytes(b"checkpoint")

    job = _representax_job(data_directory=data, steps=4, seed=7)

    config = job.model.parameters["config"]
    assert config["image_size"] == 256
    assert config["video_frames"] == 16
    assert config["hidden_size"] == 768
    assert config["depth"] == 12
    assert config["predictor_depth"] == 12
    assert job.training.global_batch_size == PREFLIGHT_BATCH_SIZE
    assert job.model.parameters["dtype"] == "float32"
    assert job.training.precision.parameter_dtype == "float32"
    assert job.training.precision.compute_dtype == "bfloat16"
    assert job.checkpointing is not None and job.checkpointing.every == 2
    assert job.checkpointing.save_final
    assert job.export.enabled
    assert job.evaluation is None


def test_pair_command_defaults_to_assigned_gpu_four() -> None:
    arguments = _parser().parse_args(
        [
            "pair",
            "--reference",
            "/reference",
            "--data-directory",
            "/data",
            "--output",
            "/output",
        ]
    )

    assert arguments.gpu == 4


def test_padded_target_ids_do_not_change_vjepa_distance_weights() -> None:
    context = jnp.asarray([[[0]]], dtype=jnp.int32)
    target = jnp.asarray([[[1, 0]]], dtype=jnp.int32)
    target_valid = jnp.asarray([[[True, False]]])

    weights = mask_distance_weights(
        context,
        target,
        grid_height=2,
        grid_width=2,
        target_valid=target_valid,
    )

    np.testing.assert_allclose(weights, [[[1.0]]])


def test_vjepa_loss_averages_mask_families_instead_of_pooling_tokens() -> None:
    prediction = jnp.asarray([[[[1.0], [100.0], [100.0]], [[3.0], [3.0], [3.0]]]])
    target = jnp.zeros_like(prediction)
    valid = jnp.asarray([[[True, False, False], [True, True, True]]])

    loss = dense_prediction_loss(prediction, target, valid)

    np.testing.assert_allclose(loss, 2.0)
