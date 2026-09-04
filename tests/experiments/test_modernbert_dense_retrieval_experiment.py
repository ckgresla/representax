from __future__ import annotations

import importlib.util
from pathlib import Path


def _experiment():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "09-modernbert-dense-retrieval"
        / "run.py"
    )
    spec = importlib.util.spec_from_file_location("modernbert_dense_retrieval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trajectory_commands_share_the_scientific_recipe(tmp_path: Path) -> None:
    experiment = _experiment()
    commands = {
        name: experiment._trajectory_command(name, tmp_path / name)
        for name in ("custom-vjp", "rematerialized", "st-eager", "st-inductor")
    }

    for command in commands.values():
        assert command[command.index("--batch-size") + 1] == "128"
        assert command[command.index("--steps") + 1] == "30"
        assert command[command.index("--maximum-length") + 1] == "128"
        assert command[command.index("--seed") + 1] == "17"
        assert "--mixed-precision" in command

    assert "custom_vjp" in commands["custom-vjp"]
    assert "rematerialized" in commands["rematerialized"]
    assert "--sentence-transformers-torch-compile" not in commands["st-eager"]
    assert "--sentence-transformers-torch-compile" in commands["st-inductor"]


def test_padding_ablation_commands_change_only_sequence_shapes(tmp_path: Path) -> None:
    experiment = _experiment()
    fixed = experiment._trajectory_command(
        "st-eager",
        tmp_path / "st-eager-fixed",
        fixed_lengths=(16, 128),
    )
    fine = experiment._trajectory_command(
        "custom-vjp",
        tmp_path / "rx-fine-buckets",
        sequence_length_buckets=(16, 64, 80, 96, 112, 128),
    )

    assert fixed[fixed.index("--sentence-transformers-query-length") + 1] == "16"
    assert fixed[fixed.index("--sentence-transformers-document-length") + 1] == "128"
    assert [
        fine[index + 1]
        for index, value in enumerate(fine)
        if value == "--sequence-length-bucket"
    ] == ["16", "64", "80", "96", "112", "128"]
