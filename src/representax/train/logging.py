"""Structured local records for one training run."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TextIO

import jax


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _json_value(value: Any) -> Any:
    value = jax.device_get(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        converted = value.tolist()
        return converted.item() if hasattr(converted, "item") else converted
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json_line(stream: TextIO, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(_json_value(row), sort_keys=True) + "\n")
    stream.flush()


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(document), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class TrainingStepRecord:
    """Host-visible facts from one attempted optimizer update."""

    iteration: int
    optimizer_step: int
    loss: float
    task: Mapping[str, Any]
    gradient_global_norm: float
    clipped_gradient_global_norm: float
    update_global_norm: float
    numeric_finite: bool
    skipped_update: bool
    data_wait_seconds: float
    placement_seconds: float
    compiled_step_seconds: float | None
    compilation_and_first_step_seconds: float | None
    end_to_end_seconds: float
    examples_per_second: float | None
    end_to_end_examples_per_second: float

    def as_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


class EventSink(Protocol):
    """Optional mirror for structured rows, such as W&B or TensorBoard."""

    def write(self, row: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class RunLogger:
    """Durable JSONL source of truth with optional external mirrors."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        manifest: Mapping[str, Any],
        sinks: Sequence[EventSink] = (),
    ) -> None:
        self.run_directory = Path(run_directory).expanduser().resolve()
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self._events = (self.run_directory / "events.jsonl").open(
            "x", encoding="utf-8", buffering=1
        )
        self._metrics = (self.run_directory / "metrics.jsonl").open(
            "x", encoding="utf-8", buffering=1
        )
        self._sinks = tuple(sinks)
        self._sequence = 0
        self._closed = False
        self._manifest = {
            **_json_value(manifest),
            "schema_version": "representax-run-v1",
            "kind": "train",
            "status": "running",
            "started_at": _timestamp(),
        }
        _atomic_write_json(self.run_directory / "run.json", self._manifest)

    def _publish(self, row: Mapping[str, Any], *, metric: bool = False) -> None:
        _write_json_line(self._events, row)
        if metric:
            _write_json_line(self._metrics, row)
        for sink in self._sinks:
            sink.write(row)

    def event(self, event: str, **fields: Any) -> None:
        row = {
            **fields,
            "schema_version": "representax-event-v1",
            "timestamp": _timestamp(),
            "sequence": self._sequence,
            "category": "event",
            "event": event,
        }
        self._sequence += 1
        self._publish(row)

    def metrics(self, record: TrainingStepRecord) -> None:
        row = {
            "schema_version": "representax-metrics-v1",
            "timestamp": _timestamp(),
            "sequence": self._sequence,
            "category": "metric",
            "event": "training_step",
            **record.as_dict(),
        }
        self._sequence += 1
        self._publish(row, metric=True)

    def console_metrics(self, record: TrainingStepRecord) -> None:
        throughput = (
            "compiling"
            if record.examples_per_second is None
            else f"{record.examples_per_second:.1f} examples/s"
        )
        print(
            f"train iteration={record.iteration} step={record.optimizer_step} "
            f"loss={record.loss:.6g} {throughput}",
            flush=True,
        )

    def finish(self, status: str, **fields: Any) -> None:
        self._manifest = {
            **self._manifest,
            **_json_value(fields),
            "status": status,
            "finished_at": _timestamp(),
        }
        _atomic_write_json(self.run_directory / "run.json", self._manifest)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._events.close()
            self._metrics.close()
        finally:
            for sink in self._sinks:
                sink.close()


__all__ = ["EventSink", "RunLogger", "TrainingStepRecord"]
