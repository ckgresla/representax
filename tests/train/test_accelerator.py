from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from representax.train.accelerator import AcceleratorMonitor


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0

    def __init__(self) -> None:
        self.initialized = False
        self.shutdown = False

    def nvmlInit(self) -> None:
        self.initialized = True

    def nvmlShutdown(self) -> None:
        self.shutdown = True

    def nvmlDeviceGetCount(self) -> int:
        return 2

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetHandleByUUID(self, identifier: str) -> str:
        return identifier

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=80 + int(handle))

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(used=100 + int(handle), total=1000)

    def nvmlDeviceGetPowerUsage(self, handle) -> int:
        return 250_000 + int(handle)

    def nvmlDeviceGetTemperature(self, handle, sensor: int) -> int:
        assert sensor == self.NVML_TEMPERATURE_GPU
        return 60 + int(handle)


def test_accelerator_monitor_publishes_visible_devices(monkeypatch) -> None:
    nvml = FakeNvml()
    monkeypatch.setattr(
        "representax.train.accelerator.jax.default_backend",
        lambda: "gpu",
    )
    monkeypatch.setattr(
        "representax.train.accelerator._load_nvml",
        lambda: nvml,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    published = []
    received = threading.Event()

    def publish(values):
        published.append(values)
        received.set()

    monitor = AcceleratorMonitor(publish, interval_seconds=10.0)
    monitor.start()
    assert received.wait(timeout=2)
    monitor.close()

    assert nvml.initialized is True
    assert nvml.shutdown is True
    assert published[0] == {
        "accelerator/0/utilization_percent": 81.0,
        "accelerator/0/memory_used_bytes": 101,
        "accelerator/0/memory_total_bytes": 1000,
        "accelerator/0/power_watts": 250.001,
        "accelerator/0/temperature_celsius": 61.0,
    }


class FakeTpuMetric:
    def __init__(self, values) -> None:
        self._values = values

    def data(self):
        return self._values


class FakeTpuMonitoring:
    values = {
        "duty_cycle_pct": ["74.5", "80.0"],
        "tensorcore_util": ["62.0", "65.5"],
        "hbm_capacity_usage": ["1000", "2000"],
        "hbm_capacity_total": ["16000", "16000"],
    }

    def list_supported_metrics(self):
        return tuple(self.values)

    def get_metric(self, name):
        return FakeTpuMetric(self.values[name])


def test_accelerator_monitor_publishes_tpu_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        "representax.train.accelerator.jax.default_backend",
        lambda: "tpu",
    )
    monkeypatch.setattr(
        "representax.train.accelerator._load_tpu_monitoring",
        FakeTpuMonitoring,
    )
    published = []
    received = threading.Event()

    def publish(values):
        published.append(values)
        received.set()

    monitor = AcceleratorMonitor(publish, interval_seconds=10.0)
    monitor.start()
    assert received.wait(timeout=2)
    monitor.close()

    assert published[0] == {
        "accelerator/0/utilization_percent": 74.5,
        "accelerator/0/tensorcore_utilization_percent": 62.0,
        "accelerator/0/memory_used_bytes": 1000,
        "accelerator/0/memory_total_bytes": 16000,
        "accelerator/1/utilization_percent": 80.0,
        "accelerator/1/tensorcore_utilization_percent": 65.5,
        "accelerator/1/memory_used_bytes": 2000,
        "accelerator/1/memory_total_bytes": 16000,
    }


def test_accelerator_monitor_rejects_unsupported_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "representax.train.accelerator.jax.default_backend",
        lambda: "cpu",
    )

    with pytest.raises(RuntimeError, match="Google TPU"):
        AcceleratorMonitor(lambda _values: None)
