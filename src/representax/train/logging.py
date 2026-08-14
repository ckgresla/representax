"""Ordered asynchronous reporting for one training run."""

from __future__ import annotations

import json
import os
import queue
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TextIO

import jax


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _host_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _host_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_host_json_value(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        converted = value.tolist()
        return converted.item() if hasattr(converted, "item") else converted
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_value(value: Any) -> Any:
    """Request one PyTree-wide device transfer, then normalize it for JSON."""

    return _host_json_value(jax.device_get(value))


def _write_json_line(stream: TextIO, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class MetricRecord:
    """One reporter-ready metric mapping, possibly still backed by device arrays."""

    iteration: int
    values: Mapping[str, Any]
    event: str = "training_step"

    def __post_init__(self) -> None:
        if self.iteration <= 0:
            raise ValueError("metric iteration must be positive")
        if not self.event:
            raise ValueError("metric event must be non-empty")
        if not self.values:
            raise ValueError("metric values must be non-empty")
        required = {"train/loss", "train/skipped_update"}
        missing = required - set(self.values)
        if missing:
            raise ValueError(f"training metric values are missing {sorted(missing)}")
        invalid = [
            name for name in self.values if not isinstance(name, str) or "/" not in name
        ]
        if invalid:
            raise ValueError(
                "metric names must be namespaced, for example train/loss: "
                f"{sorted(map(repr, invalid))}"
            )


class Reporter(Protocol):
    """Host-row consumer such as W&B, TensorBoard, or a test recorder."""

    def write(self, row: Mapping[str, Any]) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@dataclass
class _EventItem:
    event: str
    fields: Mapping[str, Any]


@dataclass
class _MetricItem:
    record: MetricRecord
    console: bool


@dataclass
class _BarrierItem:
    done: threading.Event
    cursor: dict[str, int] | None = None


@dataclass
class _FinishItem:
    status: str
    fields: Mapping[str, Any]
    done: threading.Event


@dataclass
class _CloseItem:
    done: threading.Event


_WorkItem = _EventItem | _MetricItem | _BarrierItem | _FinishItem | _CloseItem


class RunLogger:
    """Durable JSONL source of truth with bounded asynchronous reporters."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        manifest: Mapping[str, Any],
        reporters: Sequence[Reporter] = (),
        resume_cursor: Mapping[str, int] | None = None,
        queue_size: int = 16,
        initial_optimizer_step: int = 0,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("reporter queue_size must be positive")
        self.run_directory = Path(run_directory).expanduser().resolve()
        self._events_path = self.run_directory / "events.jsonl"
        self._metrics_path = self.run_directory / "metrics.jsonl"
        self._reporters = tuple(reporters)
        self._queue: queue.Queue[_WorkItem] = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._closed = False
        if initial_optimizer_step < 0:
            raise ValueError("initial_optimizer_step must be non-negative")
        self._optimizer_step = initial_optimizer_step
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
        self._worker = threading.Thread(
            target=self._run,
            name="representax-reporter",
            daemon=False,
        )
        self._worker.start()

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
        immutable = (
            "scientific",
            "scientific_fingerprint",
            "data_contract",
            "data_fingerprint",
        )
        normalized = _json_value(manifest)
        for name in immutable:
            if previous.get(name) != normalized.get(name):
                raise ValueError(f"resumed run {name} differs")
        required = {
            "events_bytes",
            "metrics_bytes",
            "optimizer_step",
            "sequence",
        }
        if set(cursor) != required:
            raise ValueError(
                "logging cursor keys differ; "
                f"expected={sorted(required)}, actual={sorted(cursor)}"
            )
        positions = {name: int(value) for name, value in cursor.items()}
        if any(value < 0 for value in positions.values()):
            raise ValueError("logging cursor values must be non-negative")
        if positions["optimizer_step"] != self._optimizer_step:
            raise ValueError(
                "logging cursor optimizer_step differs from restored training state"
            )
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
            "execution": normalized.get("execution"),
            "scientific": normalized.get("scientific"),
            "runtime": normalized.get("runtime"),
            "checkpoint": normalized.get("checkpoint"),
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
        for reporter in self._reporters:
            reporter.write(row)

    def _publish_event(self, event: str, fields: Mapping[str, Any]) -> None:
        row = {
            **_json_value(fields),
            "schema_version": "representax-event-v1",
            "timestamp": _timestamp(),
            "sequence": self._sequence,
            "category": "event",
            "event": event,
        }
        self._sequence += 1
        self._publish(row)

    def _publish_metric(self, item: _MetricItem) -> None:
        record = item.record
        values = _json_value(record.values)
        skipped_update = bool(values["train/skipped_update"])
        if not skipped_update:
            self._optimizer_step += 1
        optimizer_step = self._optimizer_step
        row = {
            "schema_version": "representax-metrics-v1",
            "timestamp": _timestamp(),
            "sequence": self._sequence,
            "category": "metric",
            "event": record.event,
            "iteration": record.iteration,
            "optimizer_step": optimizer_step,
            "metrics": values,
        }
        self._sequence += 1
        self._publish(row, metric=True)
        if values.get("train/skipped_update"):
            self._publish_event(
                "nonfinite_update_skipped",
                {"iteration": record.iteration, "optimizer_step": optimizer_step},
            )
        if item.console:
            throughput = values.get("perf/examples_per_second")
            suffix = "" if throughput is None else f" {throughput:.1f} examples/s"
            print(
                f"train iteration={record.iteration} step={optimizer_step} "
                f"loss={values['train/loss']:.6g}{suffix}",
                flush=True,
            )

    def _flush_outputs(self) -> None:
        self._events.flush()
        self._metrics.flush()
        os.fsync(self._events.fileno())
        os.fsync(self._metrics.fileno())
        for reporter in self._reporters:
            reporter.flush()

    def _cursor(self) -> dict[str, int]:
        self._flush_outputs()
        return {
            "events_bytes": self._events_path.stat().st_size,
            "metrics_bytes": self._metrics_path.stat().st_size,
            "optimizer_step": self._optimizer_step,
            "sequence": self._sequence,
        }

    def _run(self) -> None:
        close_item: _CloseItem | None = None
        while close_item is None:
            item = self._queue.get()
            try:
                if isinstance(item, _CloseItem):
                    close_item = item
                elif self._error is not None:
                    pass
                elif isinstance(item, _EventItem):
                    self._publish_event(item.event, item.fields)
                elif isinstance(item, _MetricItem):
                    self._publish_metric(item)
                elif isinstance(item, _BarrierItem):
                    item.cursor = self._cursor()
                elif isinstance(item, _FinishItem):
                    self._flush_outputs()
                    self._manifest = {
                        **self._manifest,
                        **_json_value(item.fields),
                        "status": item.status,
                        "finished_at": _timestamp(),
                    }
                    _atomic_write_json(
                        self.run_directory / "run.json",
                        self._manifest,
                    )
            except BaseException as error:
                if self._error is None:
                    self._error = error
            finally:
                if isinstance(item, (_BarrierItem, _FinishItem, _CloseItem)):
                    item.done.set()
                self._queue.task_done()
        try:
            self._flush_outputs()
            self._events.close()
            self._metrics.close()
            for reporter in self._reporters:
                reporter.close()
        except BaseException as error:
            if self._error is None:
                self._error = error
        finally:
            close_item.done.set()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous reporter failed") from self._error

    def _submit(self, item: _WorkItem) -> None:
        self._raise_if_failed()
        while True:
            try:
                self._queue.put(item, timeout=0.1)
                break
            except queue.Full:
                self._raise_if_failed()
        self._raise_if_failed()

    def event(self, event: str, **fields: Any) -> None:
        if not event:
            raise ValueError("event name must be non-empty")
        self._submit(_EventItem(event=event, fields=fields))

    def metrics(self, record: MetricRecord, *, console: bool = False) -> None:
        self._submit(_MetricItem(record=record, console=console))

    def cursor(self) -> dict[str, int]:
        """Drain through this boundary and return its durable log positions."""

        item = _BarrierItem(done=threading.Event())
        self._submit(item)
        item.done.wait()
        self._raise_if_failed()
        assert item.cursor is not None
        return item.cursor

    def finish(self, status: str, **fields: Any) -> None:
        item = _FinishItem(status=status, fields=fields, done=threading.Event())
        self._submit(item)
        item.done.wait()
        self._raise_if_failed()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        item = _CloseItem(done=threading.Event())
        while True:
            try:
                self._queue.put(item, timeout=0.1)
                break
            except queue.Full:
                if not self._worker.is_alive():
                    break
        item.done.wait(timeout=30)
        self._worker.join(timeout=30)
        if self._worker.is_alive():
            raise RuntimeError("asynchronous reporter did not close")
        self._raise_if_failed()


__all__ = ["MetricRecord", "Reporter", "RunLogger"]
