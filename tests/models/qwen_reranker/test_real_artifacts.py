"""Native and upstream reload acceptance for real Qwen reranker artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.models.qwen_reranker import (
    QwenRerankerCheckpointAdapter,
    load_qwen_reranker,
)
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
)

pytestmark = pytest.mark.performance

CASES = (
    ("qwen3", "REPRESENTAX_QWEN3_RERANKER_CHECKPOINT"),
    ("contextual", "REPRESENTAX_CONTEXTUAL_RERANKER_CHECKPOINT"),
    ("mixedbread", "REPRESENTAX_MXBAI_RERANKER_CHECKPOINT"),
)


@pytest.mark.parametrize(("name", "environment"), CASES)
def test_export_reloads_natively_and_in_sentence_transformers(
    name: str,
    environment: str,
    tmp_path: Path,
) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Qwen reranker export acceptance requires a GPU")
    checkpoint_value = os.environ.get(environment)
    oracle_directory = os.environ.get("REPRESENTAX_QWEN_RERANKER_ORACLES")
    if checkpoint_value is None or oracle_directory is None:
        pytest.skip(f"set {environment} and REPRESENTAX_QWEN_RERANKER_ORACLES")
    checkpoint = Path(checkpoint_value)
    oracle = np.load(Path(oracle_directory) / f"{name}.npz")
    length = int(oracle["input_ids"].shape[1])
    model, _ = load_qwen_reranker(
        checkpoint,
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(length,),
        rematerialization="none",
    )
    adapter = QwenRerankerCheckpointAdapter(rematerialization="none")
    export = adapter.save(model, tmp_path / name, source_checkpoint=checkpoint)
    native_reload = adapter.load(
        export,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
    )
    reloaded_state = adapter.state_dict(native_reload)
    for tensor_name, expected in adapter.state_dict(model).items():
        np.testing.assert_array_equal(reloaded_state[tensor_name], expected)

    output = tmp_path / f"{name}.npz"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.models.qwen_reranker.sentence_transformers_reload",
            "--checkpoint",
            str(export),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    reloaded = np.load(output)
    np.testing.assert_array_equal(reloaded["input_ids"], oracle["input_ids"])
    np.testing.assert_array_equal(reloaded["attention_mask"], oracle["attention_mask"])
    tolerance = NumericalTolerance(absolute=0.02, relative=0.02, cosine=0.99999)
    assert_numerically_equivalent(reloaded["logits"], oracle["logits"], tolerance)
    assert_numerically_equivalent(reloaded["scores"], oracle["scores"], tolerance)
