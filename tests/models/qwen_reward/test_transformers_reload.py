"""Hugging Face reload acceptance for native Qwen3 reward export."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import score_logits
from representax.models.qwen_reward import QwenRewardCheckpointAdapter
from tests.models.qwen_reward.test_model import _batch, _source_checkpoint

pytestmark = pytest.mark.parity


def _upstream_python() -> str:
    executable = os.environ.get("REPRESENTAX_QWEN_REWARD_TRANSFORMERS_PYTHON")
    if executable is None:
        pytest.skip("set REPRESENTAX_QWEN_REWARD_TRANSFORMERS_PYTHON for parity")
    return executable


def test_native_reward_export_reloads_in_transformers(tmp_path: Path) -> None:
    source = _source_checkpoint(tmp_path / "source")
    adapter = QwenRewardCheckpointAdapter()
    model = adapter.load(
        source,
        head_key=jax.random.key(5),
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    export = adapter.save(model, tmp_path / "export", source_checkpoint=source)
    inputs = tmp_path / "inputs.npz"
    np.savez(
        inputs,
        input_ids=np.asarray(_batch().input_ids),
        attention_mask=np.asarray(_batch().attention_mask),
    )
    output = tmp_path / "transformers.npz"
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.qwen_reward.transformers_reload",
            "--checkpoint",
            str(export),
            "--inputs",
            str(inputs),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    reference = np.asarray(score_logits(model, _batch()))[:, None]
    with np.load(output) as result:
        np.testing.assert_allclose(result["logits"], reference, rtol=1e-5, atol=1e-5)
