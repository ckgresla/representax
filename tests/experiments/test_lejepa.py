"""Canonical LeJEPA paper runner contracts."""

import json

import jax.numpy as jnp
import numpy as np
import torch
from experiments.paper.lejepa import (
    CANARY_ARCHITECTURE,
    OFFICIAL_COMMIT,
    PAPER_VIT_LARGE_ARCHITECTURE,
    STABLE_COMMIT,
    _capacity_projection,
    _parser,
    _torch_paper_loss,
    build_capacity_job,
    build_job,
    exact_objective_parity,
    prepare_imagenet,
    reference_contract,
    stable_reference_decision,
)
from PIL import Image

from representax.tasks.jepa import sigreg_loss


def _fake_imagenet(root, *, classes: int = 2) -> None:
    train = root / "train"
    train.mkdir(parents=True)
    index = {}
    for label in range(1000):
        synset = f"n{label:08d}"
        index[str(label)] = [synset, f"class-{label}"]
        directory = train / synset
        directory.mkdir()
        if label < classes:
            for image_index in range(7):
                Image.new(
                    "RGB",
                    (32 + image_index, 40),
                    (label * 20, image_index * 10, 100),
                ).save(directory / f"{synset}_{image_index}.JPEG")
    (root / "imagenet_class_index.json").write_text(json.dumps(index))


def test_reference_contract_pins_both_sources_and_paper_crop_choice() -> None:
    contract = reference_contract()

    assert contract.official_commit == OFFICIAL_COMMIT
    assert contract.stable_pretraining_commit == STABLE_COMMIT
    assert contract.global_views == 2
    assert contract.local_views == 6
    assert contract.local_image_size == 98
    assert contract.sigreg_slices == 1024


def test_prepare_labels_train_only_subsets_as_readiness_not_quality(tmp_path) -> None:
    source = tmp_path / "imagenet"
    _fake_imagenet(source)

    manifest = prepare_imagenet(
        tmp_path / "prepared",
        imagenet_root=source,
        class_count=2,
    )

    assert manifest["rows"]["training"] == 14
    assert manifest["rows"]["evaluation_splits"] == {"0": 6, "1": 4, "2": 4}
    assert "readiness-only" in manifest["acceptance_scope"]
    assert not manifest["imagenet"]["quality_claim_allowed"]
    official = manifest["imagenet"]["official_validation"]
    assert not official["accepted_for_quality"]
    assert "provenance" in official["blocker"]


def test_prepare_accepts_matching_official_validation_provenance(tmp_path) -> None:
    source = tmp_path / "imagenet"
    _fake_imagenet(source)
    validation = source / "val"
    validation.mkdir()
    provenance = tmp_path / "validation-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "representax-imagenet-validation-provenance-v1",
                "status": "pass",
                "official_quality_eval_admissible": True,
                "local_dataset": {
                    "validation_root": str(validation.resolve()),
                    "image_count": 50_000,
                    "class_count": 1_000,
                    "exact_filename_set": True,
                    "exact_parent_synset_assignments": True,
                    "local_files_byte_identical_to_archive": True,
                },
            }
        )
    )

    manifest = prepare_imagenet(
        tmp_path / "prepared",
        imagenet_root=source,
        validation_provenance=provenance,
        class_count=2,
    )

    official = manifest["imagenet"]["official_validation"]
    assert official["accepted_for_quality"] is True
    assert official["provenance_sha256"].startswith("sha256:")


def test_job_uses_canonical_views_evaluator_export_and_one_update(tmp_path) -> None:
    architecture = {
        "image_size": 16,
        "local_image_size": 8,
        "patch_size": 4,
        "hidden_size": 12,
        "depth": 2,
        "heads": 3,
        "projector_bottleneck": 8,
        "projector_hidden_size": 24,
        "projection_dimension": 7,
        "drop_path_rate": 0.0,
    }

    job = build_job(tmp_path, architecture=architecture, distributed=False)

    assert job.training.max_steps == 1
    assert job.loss.global_views == 2
    assert job.loss.slices == 1024
    assert job.data.collate.parameters["image_size"] == 16
    assert job.data.collate.parameters["local_image_size"] == 8
    assert job.evaluation is not None
    assert job.evaluation.evaluators[0].neighbors == 20
    assert job.export.enabled
    assert job.export.selection == "final"
    assert job.checkpointing is not None and job.checkpointing.every == 1


def test_capacity_job_is_training_only_vit_large_fsdp(tmp_path) -> None:
    job = build_capacity_job(tmp_path, global_batch_size=2)

    assert job.model.parameters["config"] == PAPER_VIT_LARGE_ARCHITECTURE
    assert job.model.parameters["initialization_path"] is None
    assert job.model.parameters["initialization_device"] == "cpu"
    assert job.training.global_batch_size == 2
    assert job.training.max_steps == 3
    assert job.training.mesh.axis_names == ("model",)
    assert job.training.sharding.parameter_axis == "model"
    assert job.checkpointing is None
    assert job.evaluation is None
    assert not job.export.enabled


def test_capacity_projection_is_labeled_and_uses_complete_batches() -> None:
    projection = _capacity_projection(
        global_batch_size=4,
        steady_step_seconds=2.0,
        epochs=3,
        train_images=10,
        devices=2,
    )

    assert projection["updates_per_epoch"] == 3
    assert projection["updates"] == 9
    assert projection["training_seconds"] == 18.0
    assert projection["logical_gpu_hours"] == 0.01
    assert projection["scope"].startswith("steady-training-only")


def test_torch_reference_objective_matches_native_formula_and_gradient() -> None:
    projections = np.arange(3 * 8 * 5, dtype=np.float32).reshape(3, 8, 5) / 20
    directions = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) / 10 + 0.1
    native_invariance = jnp.mean(
        jnp.square(
            jnp.asarray(projections)
            - jnp.mean(jnp.asarray(projections)[:, :2], axis=1, keepdims=True)
        )
    )
    native_sigreg = sigreg_loss(
        jnp.asarray(projections),
        jnp.ones((3, 8), dtype=jnp.bool_),
        jnp.asarray(directions),
        knots=17,
        max_frequency=3.0,
    )
    expected = 0.98 * native_invariance + 0.02 * native_sigreg

    torch_projections = torch.tensor(projections, requires_grad=True)
    actual, invariance, sigreg = _torch_paper_loss(
        torch_projections,
        torch.tensor(directions),
    )
    actual.backward()

    def native_objective(value):
        native_invariance = jnp.mean(
            jnp.square(value - jnp.mean(value[:, :2], axis=1, keepdims=True))
        )
        native_sigreg = sigreg_loss(
            value,
            jnp.ones((3, 8), dtype=jnp.bool_),
            jnp.asarray(directions),
            knots=17,
            max_frequency=3.0,
        )
        return 0.98 * native_invariance + 0.02 * native_sigreg

    import jax

    native_gradient = jax.grad(native_objective)(jnp.asarray(projections))

    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(
        invariance.detach().numpy(), native_invariance, rtol=1e-6
    )
    np.testing.assert_allclose(sigreg.detach().numpy(), native_sigreg, rtol=1e-5)
    np.testing.assert_allclose(
        torch_projections.grad.detach().numpy(),
        native_gradient,
        rtol=2e-5,
        atol=2e-5,
    )
    assert exact_objective_parity()["status"] == "accepted"


def test_profiles_and_reference_worker_policy_are_explicit() -> None:
    assert CANARY_ARCHITECTURE["hidden_size"] == 384
    assert CANARY_ARCHITECTURE["projection_dimension"] == 512
    assert PAPER_VIT_LARGE_ARCHITECTURE["hidden_size"] == 1024
    assert PAPER_VIT_LARGE_ARCHITECTURE["depth"] == 24
    assert PAPER_VIT_LARGE_ARCHITECTURE["projection_dimension"] == 512
    decision = stable_reference_decision()
    assert decision["objective_match"]
    assert not decision["pinned_helper_used_as_result"]


def test_cli_assigns_framework_workers_without_architecture_substitution() -> None:
    representax = _parser().parse_args(
        [
            "run",
            "--data-directory",
            "/data",
            "--initialization",
            "/initialization.pt",
            "--output",
            "/output",
        ]
    )
    reference = _parser().parse_args(
        [
            "reference",
            "--data-directory",
            "/data",
            "--initialization",
            "/initialization.pt",
            "--output",
            "/output",
        ]
    )
    parity = _parser().parse_args(["parity", "--output", "/output"])
    initialize = _parser().parse_args(["initialize", "--output", "/state.pt"])
    aggregate = _parser().parse_args(["aggregate", "--output", "/output"])
    capacity = _parser().parse_args(
        [
            "capacity",
            "--data-directory",
            "/data",
            "--output",
            "/output",
            "--global-batch-size",
            "2",
        ]
    )

    assert representax.gpus == "4,5"
    assert reference.gpu == "5"
    assert parity.command == "parity"
    assert initialize.command == "initialize"
    assert aggregate.command == "aggregate"
    assert capacity.gpus == "0,1"
    assert capacity.global_batch_size == 2
    assert capacity.steps == 3
