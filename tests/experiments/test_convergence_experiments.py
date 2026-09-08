"""Frozen contracts for paper convergence experiments 11 through 13."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _experiment(number: int, name: str) -> ModuleType:
    path = ROOT / "experiments" / f"{number:02d}-{name}" / "run.py"
    spec = importlib.util.spec_from_file_location(f"paper_experiment_{number}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_dense_retrieval_contract_and_command_are_frozen() -> None:
    experiment = _experiment(11, "dense-retrieval-convergence")
    contract = experiment.contract()
    command = experiment.worker_command(42, 1, steps=3_817)

    assert contract["model"] == {
        "id": "jhu-clsp/ettin-encoder-150m",
        "revision": "45d08642849e5c5701b162671ac811b7654bfd9f",
    }
    assert contract["seeds"] == [7, 42, 773]
    assert contract["global_batch_size"] == 128
    assert contract["sequence_length_buckets"] == [16, 96, 128]
    assert contract["grad_cache"] == {
        "implementation": "custom_vjp",
        "query_micro_batch_size": 128,
        "document_micro_batch_size": 64,
        "loss_row_chunk_size": 64,
    }
    assert contract["evaluation"]["batch_size"] == 4_096
    assert _argument(command, "--steps") == "3817"
    assert _argument(command, "--training-parquet").endswith("seed-42.parquet")
    assert _argument(command, "--checkpoint-every") == "1908"
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--sequence-length-bucket"
    ] == ["16", "96", "128"]


def test_dense_transfer_commands_are_explicit() -> None:
    experiment = _experiment(11, "dense-retrieval-convergence")
    parser = experiment._parser()

    assert parser.parse_args(["prepare-transfer"]).command == "prepare-transfer"
    assert parser.parse_args(["evaluate-seed", "--seed", "42"]).seed == 42
    assert parser.parse_args(["evaluate-all", "--gpus", "0", "1", "2"]).gpus == [
        0,
        1,
        2,
    ]
    assert str(ROOT) in experiment._environment(0)["PYTHONPATH"].split(
        experiment.os.pathsep
    )


def test_late_interaction_contract_and_command_are_frozen() -> None:
    experiment = _experiment(12, "late-interaction-convergence")
    contract = experiment.contract()
    command = experiment.worker_command(773, 2)

    assert contract["model"] == {
        "id": "lightonai/GTE-ModernColBERT-v1",
        "revision": "cbbe53366e564450558f5e639dd499171f127538",
    }
    assert contract["optimizer_steps"] == 1_000
    assert contract["global_batch_size"] == 512
    assert contract["query_sequence_length_buckets"] == [16, 32]
    assert contract["document_sequence_length_buckets"] == [32, 64, 128, 256]
    assert contract["training_presentations"] == 512_000
    assert _argument(command, "--steps") == "1000"
    assert _argument(command, "--warmup-steps") == "60"
    assert _argument(command, "--negative-scope") == "global"


def test_image_text_contract_and_command_are_frozen() -> None:
    experiment = _experiment(13, "image-text-convergence")
    contract = experiment.contract()
    command = experiment.worker_command(7, 0)

    assert contract["model"] == {
        "id": "sentence-transformers/clip-ViT-B-32",
        "revision": "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f",
    }
    assert contract["training_data"]["images"] == 117_760
    assert contract["training_data"]["distinct_captions_per_image"] == 5
    assert contract["optimizer_steps"] == 1_150
    assert contract["global_batch_size"] == 512
    assert contract["loss"]["symmetric"] is True
    assert contract["evaluation"]["directions"] == [
        "text-to-image",
        "image-to-text",
    ]
    assert _argument(command, "--steps") == "1150"
    assert _argument(command, "--warmup-steps") == "69"
    assert _argument(command, "--training-file") == "train-seed-7.jsonl"
    assert "--symmetric" in command


def test_image_text_order_distributes_repeated_captions_across_batches() -> None:
    experiment = _experiment(13, "image-text-convergence")
    previous_batch_size = experiment.GLOBAL_BATCH_SIZE
    experiment.GLOBAL_BATCH_SIZE = 3
    rows = [
        {"image_id": index, "caption": caption}
        for index, caption in enumerate(("same", "same", "a", "b", "c", "d"))
    ]
    try:
        ordered = experiment._batch_unique_caption_order(rows, batch_size=3, seed=7)
    finally:
        experiment.GLOBAL_BATCH_SIZE = previous_batch_size

    batches = (ordered[:3], ordered[3:])
    assert all(len({row["image_id"] for row in batch}) == 3 for batch in batches)
    assert all(len({row["caption"] for row in batch}) == 3 for batch in batches)


def test_image_text_bidirectional_evaluation_commands_are_explicit() -> None:
    experiment = _experiment(13, "image-text-convergence")
    parser = experiment._parser()

    assert parser.parse_args(["prepare-evaluation"]).command == "prepare-evaluation"
    assert parser.parse_args(["evaluate-seed", "--seed", "42", "--gpu", "1"]).seed == 42
    assert parser.parse_args(
        ["evaluate-all", "--gpus", "0", "1", "2"]
    ).gpus == [0, 1, 2]


@pytest.mark.parametrize(
    ("number", "name"),
    (
        (11, "dense-retrieval-convergence"),
        (12, "late-interaction-convergence"),
        (13, "image-text-convergence"),
    ),
)
def test_all_commands_require_three_distinct_gpus(number: int, name: str) -> None:
    parser = _experiment(number, name)._parser()

    assert parser.parse_args(["all", "--gpus", "0", "1", "2"]).gpus == [0, 1, 2]
    with pytest.raises(SystemExit):
        parser.parse_args(["all", "--gpus", "0", "1"])
