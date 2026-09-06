"""Contracts for paper experiment 10."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _module():
    path = (
        Path(__file__).parents[2]
        / "experiments/10-cross-accelerator-framework-comparison/run.py"
    )
    spec = importlib.util.spec_from_file_location("cross_accelerator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dense_contract_freezes_scientific_work_across_variants() -> None:
    module = _module()
    variants = (
        "representax-local",
        "representax-global",
        "sentence-transformers-local",
    )
    contracts = [
        module._contract(
            seed=42,
            variant=variant,
            steps=20,
            platform="tpu",
        )
        for variant in variants
    ]

    training = [contract["training"] for contract in contracts]
    scopes = [value.pop("negative_scope") for value in training]
    assert training[0] == training[1] == training[2]
    assert scopes == ["local", "global", "local"]
    assert training[0]["global_batch_size"] == 2048
    assert training[0]["query_length"] == 32
    assert training[0]["document_length"] == 256
    assert training[0]["compute_dtype"] == "bfloat16"


def test_cli_requires_explicit_representax_negative_scope(tmp_path: Path) -> None:
    module = _module()
    parser = module._parser()

    arguments = parser.parse_args(
        (
            "representax",
            "--checkpoint",
            str(tmp_path / "model"),
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "output"),
            "--seed",
            "7",
            "--platform",
            "tpu",
            "--negative-scope",
            "local",
        )
    )

    assert arguments.negative_scope == "local"
    assert arguments.steps == 20


def test_gpu_reference_control_records_inductor_without_changing_science() -> None:
    module = _module()
    eager = module._contract(
        seed=7,
        variant="sentence-transformers-local",
        steps=20,
        platform="gpu",
    )
    inductor = module._contract(
        seed=7,
        variant="sentence-transformers-local",
        steps=20,
        platform="gpu",
        torch_compile=True,
    )

    assert eager["training"] == inductor["training"]
    assert eager["data"] == inductor["data"]
    assert not eager["execution"]["torch_compile"]
    assert inductor["execution"]["torch_compile"]


def test_tpu_reference_uses_the_supported_uncached_loss() -> None:
    module = _module()
    gpu = module._contract(
        seed=7,
        variant="sentence-transformers-local",
        steps=20,
        platform="gpu",
    )
    tpu = module._contract(
        seed=7,
        variant="sentence-transformers-local",
        steps=20,
        platform="tpu",
    )

    assert gpu["training"] == tpu["training"]
    assert gpu["execution"]["sentence_transformers_loss"] == (
        "cached_multiple_negatives_ranking"
    )
    assert tpu["execution"]["sentence_transformers_loss"] == (
        "multiple_negatives_ranking"
    )
    assert gpu["execution"]["sentence_transformers_grad_cache_chunk"] == 128
    assert tpu["execution"]["sentence_transformers_grad_cache_chunk"] is None


def test_campaign_exposes_every_frozen_recipe() -> None:
    module = _module()

    assert len(module.RECIPES) == 13
    assert set(module.REFERENCE_FRAMEWORKS) == set(module.RECIPES)
    assert module.REFERENCE_FRAMEWORKS["late-interaction"] == "pylate"
    assert module.REFERENCE_FRAMEWORKS["outcome-reward"] == "trl"
    assert module.REFERENCE_FRAMEWORKS["v-jepa"] == "facebookresearch-vjepa2"
    assert set(module.RECIPE_ASSETS) == set(module.RECIPES)
    assert module.RECIPE_ASSETS["v-jepa"] == (
        None,
        "vjepa-data",
        "vjepa2-reference",
    )


def test_environment_state_records_reproducible_runtime(monkeypatch) -> None:
    module = _module()
    monkeypatch.setenv("PJRT_DEVICE", "TPU")

    state = module._environment_state()

    assert state["python"]
    assert state["executable"]
    assert state["packages"]
    assert state["accelerator_environment"]["PJRT_DEVICE"] == "TPU"


def test_environment_state_can_interrogate_worker_interpreter() -> None:
    module = _module()

    state = module._environment_state(Path(module.sys.executable))

    assert Path(state["executable"]).resolve() == Path(module.sys.executable).resolve()


def test_suite_selects_frozen_assets_and_reference_environments(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    calls = []
    monkeypatch.setattr(module, "_run_recipe", calls.append)
    arguments = module._parser().parse_args(
        (
            "suite",
            "--framework",
            "reference",
            "--asset-root",
            str(tmp_path / "assets"),
            "--output",
            str(tmp_path / "results"),
            "--platform",
            "tpu",
            "--recipe",
            "dense-retrieval",
            "--recipe",
            "late-interaction",
            "--recipe",
            "v-jepa",
        )
    )

    module._run_suite(arguments)

    assert [call.recipe for call in calls] == [
        "dense-retrieval",
        "late-interaction",
        "v-jepa",
    ]
    assert calls[0].checkpoint == tmp_path / "assets/all-mpnet-base-v2"
    assert calls[0].data == tmp_path / "assets/dense-msmarco-unique-v1"
    assert ".venv-torch-xla-late" in str(calls[1].worker_python)
    assert calls[2].checkpoint is None
    assert calls[2].reference == tmp_path / "assets/vjepa2-reference"


def test_recipe_command_preserves_frozen_tpu_shape(tmp_path: Path) -> None:
    module = _module()
    parser = module._parser()
    arguments = parser.parse_args(
        (
            "recipe",
            "--recipe",
            "v-jepa",
            "--framework",
            "reference",
            "--data",
            str(tmp_path / "data"),
            "--reference",
            str(tmp_path / "vjepa2"),
            "--output",
            str(tmp_path / "output"),
            "--seed",
            "7",
            "--platform",
            "tpu",
        )
    )

    command = module._recipe_command(arguments)

    assert "facebookresearch-vjepa2" in command
    assert command[command.index("--batch-size") + 1] == "128"
    assert command[command.index("--platform") + 1] == "tpu"
    assert "--gpu" not in command


def test_multimodal_recipe_matches_device_local_tpu_negatives(tmp_path: Path) -> None:
    module = _module()
    parser = module._parser()
    arguments = parser.parse_args(
        (
            "recipe",
            "--recipe",
            "image-text",
            "--framework",
            "representax",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "output"),
            "--seed",
            "7",
            "--platform",
            "tpu",
        )
    )

    command = module._recipe_command(arguments)

    assert command[command.index("--negative-scope") + 1] == "local"


def test_audio_tpu_recipe_uses_the_matched_feasible_batch(tmp_path: Path) -> None:
    module = _module()
    arguments = module._parser().parse_args(
        (
            "recipe",
            "--recipe",
            "audio-text",
            "--framework",
            "representax",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "output"),
            "--seed",
            "7",
            "--platform",
            "tpu",
        )
    )

    command = module._recipe_command(arguments)

    assert command[command.index("--batch-size") + 1] == "64"
    assert command[command.index("--sharding") + 1] == "ddp"
    assert command[command.index("--negative-scope") + 1] == "local"
    assert "--continuous" in command


def test_audio_gpu_recipe_keeps_the_frozen_batch(tmp_path: Path) -> None:
    module = _module()
    arguments = module._parser().parse_args(
        (
            "recipe",
            "--recipe",
            "audio-text",
            "--framework",
            "reference",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "output"),
            "--seed",
            "7",
            "--platform",
            "gpu",
        )
    )

    command = module._recipe_command(arguments)

    assert command[command.index("--batch-size") + 1] == "256"


def test_video_recipe_keeps_frozen_batch_and_negative_scope(tmp_path: Path) -> None:
    module = _module()
    arguments = module._parser().parse_args(
        (
            "recipe",
            "--recipe",
            "video-text",
            "--framework",
            "representax",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "output"),
            "--seed",
            "7",
            "--platform",
            "tpu",
        )
    )

    command = module._recipe_command(arguments)

    assert command[command.index("--batch-size") + 1] == "128"
    assert command[command.index("--negative-scope") + 1] == "local"


def test_reference_helpers_are_self_contained(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "pairs.parquet"
    pq.write_table(
        pa.table({"query": ["q"], "positive": ["p"], "ignored": [1]}),
        path,
    )

    assert module._training_rows(path) == [{"query": "q", "positive": "p"}]
    assert module._FixedPairCollator(object()).valid_label_columns == []


def test_checkpoint_files_are_content_addressed(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weight = checkpoint / "weight.bin"
    weight.write_bytes(b"weights")
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model": {"id": module.MODEL_ID, "revision": module.MODEL_REVISION},
                "files": {
                    "weight.bin": {
                        "bytes": weight.stat().st_size,
                        "sha256": module._sha256(weight),
                    }
                },
            }
        )
    )
    monkeypatch.setattr(module, "MODEL_MANIFEST", manifest)

    module._verify_checkpoint(checkpoint)
    weight.write_bytes(b"changed")

    try:
        module._verify_checkpoint(checkpoint)
    except ValueError as error:
        assert "hash changed" in str(error)
    else:
        raise AssertionError("changed checkpoint was accepted")


def test_tracked_data_manifest_matches_the_frozen_contract() -> None:
    module = _module()
    path = (
        Path(__file__).parents[2]
        / "experiments/10-cross-accelerator-framework-comparison/data-manifest.json"
    )
    manifest = json.loads(path.read_text())

    assert manifest["dataset"] == {
        "id": module.DATASET_ID,
        "revision": module.DATASET_REVISION,
    }
    assert manifest["model_tokenizer"] == {
        "id": module.MODEL_ID,
        "revision": module.MODEL_REVISION,
    }
    assert manifest["selection"] == {
        "document_length_at_most": module.DOCUMENT_LENGTH,
        "order": "first eligible source occurrence",
        "query_length_at_most": module.QUERY_LENGTH,
        "rows": module.DATA_POOL_SIZE,
        "unique_positive": True,
        "unique_query": True,
    }
    assert set(manifest["seed_files"]) == {str(seed) for seed in module.SEEDS}
