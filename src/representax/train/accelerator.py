"""Optional background accelerator telemetry for the canonical run log."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, Protocol

import jax


def _load_nvml() -> Any:
    try:
        import pynvml
    except ImportError as error:
        raise RuntimeError(
            "logging.accelerator requires the 'performance' dependency group"
        ) from error
    return pynvml


def _load_tpu_monitoring() -> Any:
    try:
        return import_module("libtpu.sdk").tpumonitoring
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "logging.accelerator on TPU requires the 'tpu' dependency group"
        ) from error


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


class _Sampler(Protocol):
    def sample(self) -> Mapping[str, float | int]: ...

    def close(self) -> None: ...


class _NvidiaSampler:
    def __init__(self) -> None:
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

    def sample(self) -> Mapping[str, float | int]:
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

    def close(self) -> None:
        self._pynvml.nvmlShutdown()


_TPU_METRICS = (
    "duty_cycle_pct",
    "tensorcore_util",
    "hbm_capacity_usage",
    "hbm_capacity_total",
)


def _tpu_metric_values(monitoring: Any, name: str) -> tuple[float, ...]:
    metric = monitoring.get_metric(name)
    data = metric.data()
    if not isinstance(data, (list, tuple)) or not data:
        raise RuntimeError(f"TPU metric {name!r} returned no per-device values")
    try:
        return tuple(float(value) for value in data)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"TPU metric {name!r} returned invalid values") from error


class _TpuSampler:
    def __init__(self) -> None:
        self._monitoring = _load_tpu_monitoring()
        supported = set(self._monitoring.list_supported_metrics())
        missing = set(_TPU_METRICS) - supported
        if missing:
            raise RuntimeError(
                "logging.accelerator requires unavailable TPU metrics: "
                f"{sorted(missing)}"
            )

    def sample(self) -> Mapping[str, float | int]:
        samples = {
            name: _tpu_metric_values(self._monitoring, name) for name in _TPU_METRICS
        }
        counts = {len(values) for values in samples.values()}
        if len(counts) != 1:
            raise RuntimeError("TPU accelerator metrics disagree on device count")
        values: dict[str, float | int] = {}
        for index in range(counts.pop()):
            prefix = f"accelerator/{index}"
            values[f"{prefix}/utilization_percent"] = samples["duty_cycle_pct"][index]
            values[f"{prefix}/tensorcore_utilization_percent"] = samples[
                "tensorcore_util"
            ][index]
            values[f"{prefix}/memory_used_bytes"] = int(
                samples["hbm_capacity_usage"][index]
            )
            values[f"{prefix}/memory_total_bytes"] = int(
                samples["hbm_capacity_total"][index]
            )
        return values

    def close(self) -> None:
        pass


def _sampler() -> _Sampler:
    platform = jax.default_backend()
    if platform == "gpu":
        return _NvidiaSampler()
    if platform == "tpu":
        return _TpuSampler()
    raise RuntimeError(
        "logging.accelerator supports NVIDIA GPU and Google TPU backends; "
        f"found {platform!r}"
    )


class AcceleratorMonitor:
    """Sample the active accelerator without blocking the training thread."""

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
        self._sampler = _sampler()
        self._thread = threading.Thread(
            target=self._run,
            name="representax-accelerator-monitor",
            daemon=False,
        )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._publish(self._sampler.sample())
                self._stop.wait(self._interval_seconds)
        except BaseException as error:
            self._error = error

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        try:
            self._sampler.close()
        finally:
            if self._error is not None:
                raise RuntimeError("accelerator telemetry failed") from self._error


__all__ = ["AcceleratorMonitor"]
