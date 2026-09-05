"""Tests for Hugging Face checkpoint utilities."""

from __future__ import annotations

import jax
import numpy as np
from safetensors.numpy import save_file

from representax.integrations.huggingface import load_safetensor_subset


def test_safetensors_default_to_a_process_local_cpu(monkeypatch, tmp_path) -> None:
    save_file(
        {"weight": np.arange(4, dtype=np.float32)},
        tmp_path / "model.safetensors",
    )
    local_devices = jax.local_devices
    calls = []

    def record_local_devices(*, backend):
        calls.append(backend)
        return local_devices(backend=backend)

    monkeypatch.setattr(jax, "local_devices", record_local_devices)

    result = load_safetensor_subset(tmp_path, {"weight"})

    assert calls == ["cpu"]
    np.testing.assert_array_equal(result["weight"], np.arange(4, dtype=np.float32))
