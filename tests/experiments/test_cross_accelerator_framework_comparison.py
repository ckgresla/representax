"""Contracts for paper experiment 10."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
