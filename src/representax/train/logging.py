"""Structured local records for one training run."""

from __future__ import annotations

import json
import os
import threading
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
        resume_cursor: Mapping[str, int] | None = None,
    ) -> None:
        self.run_directory = Path(run_directory).expanduser().resolve()
        self._events_path = self.run_directory / "events.jsonl"
        self._metrics_path = self.run_directory / "metrics.jsonl"
        self._sinks = tuple(sinks)
        self._lock = threading.RLock()
        self._closed = False
        if resume_cursor is None:
            self.run_directory.mkdir(parents=True, exist_ok=False)
            self._events = self._events_path.open("x", encoding="utf-8", buffering=1)
            self._metrics = self._metrics_path.open("x", encoding="utf-8", buffering=1)
            self._sequence = 0
            self._manifest = {
                **_json_value(manifest),
                "schema_version": "representax-run-v1",
                "kind": "train",
                "status": "running",
                "started_at": _timestamp(),
            }
        else:
            self._manifest = self._resume(manifest, resume_cursor)
        _atomic_write_json(self.run_directory / "run.json", self._manifest)

    def _resume(
        self,
        manifest: Mapping[str, Any],
        cursor: Mapping[str, int],
    ) -> dict[str, Any]:
        run_path = self.run_directory / "run.json"
        if not run_path.is_file():
            raise FileNotFoundError(f"run manifest is missing: {run_path}")
        previous = json.loads(run_path.read_text())
        if (
            previous.get("schema_version") != "representax-run-v1"
            or previous.get("kind") != "train"
        ):
            raise ValueError(f"run manifest is incompatible: {run_path}")
        immutable = ("task", "global_batch_size", "max_steps", "seed")
        for name in immutable:
            if previous.get(name) != manifest.get(name):
                raise ValueError(f"resumed run {name} differs")
        required = {"events_bytes", "metrics_bytes", "sequence"}
        if set(cursor) != required:
            raise ValueError(
                "logging cursor keys differ; "
                f"expected={sorted(required)}, actual={sorted(cursor)}"
            )
        positions = {name: int(value) for name, value in cursor.items()}
        if any(value < 0 for value in positions.values()):
            raise ValueError("logging cursor values must be non-negative")
        self._truncate(self._events_path, positions["events_bytes"])
        self._truncate(self._metrics_path, positions["metrics_bytes"])
        self._events = self._events_path.open("a", encoding="utf-8", buffering=1)
        self._metrics = self._metrics_path.open("a", encoding="utf-8", buffering=1)
        self._sequence = positions["sequence"]
        terminal_fields = {
            "completed_iterations",
            "error",
            "error_type",
            "finished_at",
            "optimizer_steps",
        }
        active = {
            key: value for key, value in previous.items() if key not in terminal_fields
        }
        return {
            **active,
            "checkpoint": _json_value(manifest.get("checkpoint")),
            "status": "running",
            "resumed_at": _timestamp(),
            "resume_count": int(previous.get("resume_count", 0)) + 1,
        }

    @staticmethod
    def _truncate(path: Path, position: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"append-only run log is missing: {path}")
        size = path.stat().st_size
        if size < position:
            raise ValueError(
                f"run log {path} is shorter than its checkpoint cursor: "
                f"{size} < {position}"
            )
        with path.open("r+b") as stream:
            stream.truncate(position)

    def _publish(self, row: Mapping[str, Any], *, metric: bool = False) -> None:
        _write_json_line(self._events, row)
        if metric:
            _write_json_line(self._metrics, row)
        for sink in self._sinks:
            sink.write(row)

    def event(self, event: str, **fields: Any) -> None:
        with self._lock:
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
        with self._lock:
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

    def cursor(self) -> dict[str, int]:
        """Return byte positions that define the durable append-only boundary."""

        with self._lock:
            self._events.flush()
            self._metrics.flush()
            return {
                "events_bytes": self._events_path.stat().st_size,
                "metrics_bytes": self._metrics_path.stat().st_size,
                "sequence": self._sequence,
            }

    def finish(self, status: str, **fields: Any) -> None:
        with self._lock:
            self._manifest = {
                **self._manifest,
                **_json_value(fields),
                "status": status,
                "finished_at": _timestamp(),
            }
            _atomic_write_json(self.run_directory / "run.json", self._manifest)

    def close(self) -> None:
        with self._lock:
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
