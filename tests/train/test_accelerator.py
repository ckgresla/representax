from __future__ import annotations

import threading
from types import SimpleNamespace

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
