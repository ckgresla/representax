"""Optional background accelerator telemetry for the canonical run log."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from typing import Any


def _load_nvml() -> Any:
    try:
        import pynvml
    except ImportError as error:
        raise RuntimeError(
            "logging.accelerator requires the 'performance' dependency group"
        ) from error
    return pynvml


def _visible_device_handles(pynvml: Any) -> tuple[Any, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return tuple(
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        )
    identifiers = tuple(item.strip() for item in visible.split(",") if item.strip())
    handles = []
    for identifier in identifiers:
        if identifier.isdecimal():
            handles.append(pynvml.nvmlDeviceGetHandleByIndex(int(identifier)))
        else:
            handles.append(pynvml.nvmlDeviceGetHandleByUUID(identifier))
    return tuple(handles)


class AcceleratorMonitor:
    """Sample visible NVIDIA devices without blocking the training thread."""

    def __init__(
        self,
        publish: Callable[[Mapping[str, float | int]], None],
        *,
        interval_seconds: float = 1.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("accelerator sampling interval must be positive")
        self._publish = publish
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._pynvml = _load_nvml()
        try:
            self._pynvml.nvmlInit()
            self._handles = _visible_device_handles(self._pynvml)
            if not self._handles:
                raise RuntimeError(
                    "logging.accelerator found no visible NVIDIA devices"
                )
        except BaseException:
            self._pynvml.nvmlShutdown()
            raise
        self._thread = threading.Thread(
            target=self._run,
            name="representax-accelerator-monitor",
            daemon=False,
        )

    def _sample(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {}
        for index, handle in enumerate(self._handles):
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            prefix = f"accelerator/{index}"
            values[f"{prefix}/utilization_percent"] = float(utilization.gpu)
            values[f"{prefix}/memory_used_bytes"] = int(memory.used)
            values[f"{prefix}/memory_total_bytes"] = int(memory.total)
            values[f"{prefix}/power_watts"] = (
                float(self._pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            )
            values[f"{prefix}/temperature_celsius"] = float(
                self._pynvml.nvmlDeviceGetTemperature(
                    handle,
                    self._pynvml.NVML_TEMPERATURE_GPU,
                )
            )
        return values

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._publish(self._sample())
                self._stop.wait(self._interval_seconds)
        except BaseException as error:
            self._error = error

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        try:
            self._pynvml.nvmlShutdown()
        finally:
            if self._error is not None:
                raise RuntimeError("accelerator telemetry failed") from self._error


__all__ = ["AcceleratorMonitor"]
