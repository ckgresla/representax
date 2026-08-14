"""One-in-flight asynchronous checkpoints through Orbax's V1 training API."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import jax
import numpy as np
from orbax.checkpoint import v1 as ocp

from representax.planning import ScientificSpec

from .state import TrainState

PyTree = Any
EventCallback = Callable[..., None]

CHECKPOINT_SCHEMA = "representax-orbax-checkpoint-v1"
LATEST_SCHEMA = "representax-checkpoint-latest-v1"
PROGRESS_SCHEMA = "representax-training-progress-v1"
ORBAX_MARKER = "_CHECKPOINT_METADATA"
COMPLETE_MARKER = "REPRESENTAX_COMPLETE"
CHECKPOINT_MANIFEST = "checkpoint.json"
LATEST_POINTER = "latest"
REQUIRED_CHECKPOINTABLES = frozenset(
    {"model", "optimizer", "optimizer_step", "rng", "progress"}
)


class AsyncResponse(Protocol):
    def result(self) -> bool: ...


class OrbaxCheckpointer(Protocol):
    def save_checkpointables_async(
        self,
        step: int,
        checkpointables: dict[str, Any],
        *,
        force: bool = False,
        overwrite: bool = False,
        metrics: Any | None = None,
        custom_metadata: Any | None = None,
    ) -> AsyncResponse | None: ...

    def load_checkpointables(
        self,
        step: int | None = None,
        abstract_checkpointables: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class CheckpointError(RuntimeError):
    """Base class for checkpoint publication and restoration errors."""


class IncompleteCheckpointError(CheckpointError):
    """Raised when a path is not a fully published Representax checkpoint."""


class CheckpointWriteError(CheckpointError):
    """Raised on the training thread when an asynchronous save failed."""


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint cadence and bounded retention."""

    every: int
    keep: int = 3
    additional_iterations: tuple[int, ...] = ()
    save_final: bool = True
    asynchronous: bool = True

    def __post_init__(self) -> None:
        if self.every <= 0:
            raise ValueError("checkpoint every must be positive")
        if self.keep <= 0:
            raise ValueError("checkpoint keep must be positive")
        if any(iteration <= 0 for iteration in self.additional_iterations):
            raise ValueError("additional checkpoint iterations must be positive")
        if len(set(self.additional_iterations)) != len(self.additional_iterations):
            raise ValueError("additional checkpoint iterations must be unique")

    def should_save(self, iteration: int, *, final: bool) -> bool:
        return (
            iteration % self.every == 0
            or iteration in self.additional_iterations
            or (final and self.save_final)
        )


@dataclass(frozen=True)
class CheckpointRecord:
    iteration: int
    path: Path
    checkpoint_fingerprint: str
    science_fingerprint: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class CheckpointTicket:
    """A scheduled save; durability is established by ``manager.wait()``."""

    iteration: int
    path: Path
    asynchronous: bool


@dataclass(frozen=True)
class RestoredTrainingState:
    """State required to continue at the exact next attempted iteration."""

    state: TrainState
    iteration: int
    rng: jax.Array
    data_state: Mapping[str, Any]
    logging_cursor: Mapping[str, int]
    record: CheckpointRecord


@dataclass
class _PendingSave:
    iteration: int
    path: Path
    response: AsyncResponse
    manifest: dict[str, Any]
    started_at: float
    thread: threading.Thread | None = None
    error: BaseException | None = None


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
    raise TypeError(f"checkpoint metadata is not JSON-compatible: {type(value)!r}")


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def science_fingerprint(science: ScientificSpec) -> str:
    """Fingerprint the scientific values that a resume may not change."""

    return _fingerprint(asdict(science))


def tree_structure_fingerprint(tree: PyTree) -> str:
    """Fingerprint paths, shapes, dtypes, and static PyTree structure."""

    leaves, structure = jax.tree.flatten(tree)
    return _fingerprint(
        {
            "structure": str(structure),
            "leaves": [
                {
                    "type": type(leaf).__qualname__,
                    "shape": tuple(getattr(leaf, "shape", ())),
                    "dtype": str(getattr(leaf, "dtype", "")),
                }
                for leaf in leaves
            ],
        }
    )


def training_checkpointables(
    *,
    state: TrainState,
    iteration: int,
    rng: jax.Array,
    data_state: Mapping[str, Any],
    logging_cursor: Mapping[str, int],
) -> dict[str, Any]:
    """Build the independently restorable training-state bundle."""

    if iteration < 0:
        raise ValueError("checkpoint iteration must be non-negative")
    cursor = {str(name): int(value) for name, value in logging_cursor.items()}
    if any(value < 0 for value in cursor.values()):
        raise ValueError("logging cursor values must be non-negative")
    return {
        "model": state.model,
        "optimizer": state.optimizer_state,
        "optimizer_step": state.step,
        "rng": rng,
        "progress": {
            "schema_version": PROGRESS_SCHEMA,
            "iteration": iteration,
            "optimizer_step": int(jax.device_get(state.step)),
            "data_state": _json_value(data_state),
            "logging_cursor": cursor,
        },
    }


def abstract_pytree(tree: PyTree) -> PyTree:
    """Replace arrays by explicit shape, dtype, and sharding restore specs."""

    def abstract_leaf(value: Any) -> Any:
        if isinstance(value, jax.ShapeDtypeStruct):
            return value
        if isinstance(value, jax.Array):
            return jax.ShapeDtypeStruct(
                value.shape,
                value.dtype,
                sharding=value.sharding,
            )
        if isinstance(value, np.ndarray):
            return jax.ShapeDtypeStruct(value.shape, value.dtype)
        if isinstance(value, np.generic):
            return jax.ShapeDtypeStruct((), value.dtype)
        return value

    return jax.tree.map(abstract_leaf, tree)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_value(document), indent=2, sort_keys=True) + "\n",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    return _fingerprint(
        {
            key: value
            for key, value in manifest.items()
            if key != "checkpoint_fingerprint"
        }
    )


def validate_complete_checkpoint(
    path: str | Path,
    *,
    expected_science_fingerprint: str | None = None,
) -> CheckpointRecord:
    """Accept only a complete Orbax checkpoint finalized by Representax."""

    directory = Path(path).expanduser().resolve()
    required = (
        directory / ORBAX_MARKER,
        directory / CHECKPOINT_MANIFEST,
        directory / COMPLETE_MARKER,
    )
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise IncompleteCheckpointError(
            f"incomplete checkpoint {directory}; missing {', '.join(missing)}"
        )
    manifest = json.loads((directory / CHECKPOINT_MANIFEST).read_text())
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise IncompleteCheckpointError(
            f"unsupported checkpoint manifest: {directory / CHECKPOINT_MANIFEST}"
        )
    fingerprint = manifest.get("checkpoint_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != _manifest_fingerprint(
        manifest
    ):
        raise IncompleteCheckpointError(
            f"checkpoint manifest fingerprint differs: {directory}"
        )
    if (directory / COMPLETE_MARKER).read_text().strip() != fingerprint:
        raise IncompleteCheckpointError(
            f"checkpoint completion marker differs: {directory}"
        )
    fingerprint_science = manifest.get("science_fingerprint")
    if not isinstance(fingerprint_science, str):
        raise IncompleteCheckpointError(
            f"checkpoint science fingerprint is missing: {directory}"
        )
    if (
        expected_science_fingerprint is not None
        and fingerprint_science != expected_science_fingerprint
    ):
        raise IncompleteCheckpointError(
            f"checkpoint scientific specification differs: {directory}"
        )
    iteration = manifest.get("iteration")
    if not isinstance(iteration, int) or iteration < 0:
        raise IncompleteCheckpointError(f"invalid checkpoint iteration: {directory}")
    return CheckpointRecord(
        iteration=iteration,
        path=directory,
        checkpoint_fingerprint=fingerprint,
        science_fingerprint=fingerprint_science,
        manifest=manifest,
    )


class CheckpointManager:
    """One-in-flight Orbax V1 sequence with explicit publication and failures.

    ``ocp.training.Checkpointer.save_checkpointables_async`` performs the
    donation-safe device-to-host snapshot before returning an ``AsyncResponse``.
    Representax then lets Orbax's storage work and its own durable publication
    markers finish in the background.
    """

    def __init__(
        self,
        run_directory: str | Path,
        *,
        science_fingerprint: str,
        keep: int = 3,
        asynchronous: bool = True,
        event: EventCallback | None = None,
        checkpointer: OrbaxCheckpointer | None = None,
        process_index: int | None = None,
    ) -> None:
        if keep <= 0:
            raise ValueError("checkpoint retention must be positive")
        if not science_fingerprint:
            raise ValueError("science_fingerprint must be non-empty")
        self.run_directory = Path(run_directory).expanduser().resolve()
        self.checkpoint_root = self.run_directory / "checkpoints"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.science_fingerprint = science_fingerprint
        self.keep = keep
        self.asynchronous = asynchronous
        self._event = event
        self.process_index = (
            jax.process_index() if process_index is None else process_index
        )
        self._checkpointer = checkpointer or ocp.training.Checkpointer(
            self.checkpoint_root,
            preservation_policy=ocp.training.preservation_policies.LatestN(keep),
            cleanup_tmp_directories=False,
        )
        self._pending: _PendingSave | None = None
        self._closed = False

    def set_event_callback(self, event: EventCallback | None) -> None:
        self._event = event

    def _emit(self, event: str, **fields: Any) -> None:
        if self.process_index == 0 and self._event is not None:
            self._event(event, **fields)

    def _iteration_path(self, iteration: int) -> Path:
        return (self.checkpoint_root / str(iteration)).resolve()

    def _checkpoint_manifest(
        self,
        *,
        iteration: int,
        checkpointables: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "iteration": iteration,
            "science_fingerprint": self.science_fingerprint,
            "checkpointables": sorted(checkpointables),
            "structure_fingerprints": {
                name: tree_structure_fingerprint(value)
                for name, value in sorted(checkpointables.items())
            },
        }

    def save(
        self,
        iteration: int,
        checkpointables: Mapping[str, Any],
        *,
        metrics: Mapping[str, Any] | None = None,
    ) -> CheckpointTicket:
        """Snapshot one state; disk publication may continue asynchronously."""

        if self._closed:
            raise RuntimeError("checkpoint manager is closed")
        if iteration < 0:
            raise ValueError("checkpoint iteration must be non-negative")
        missing = REQUIRED_CHECKPOINTABLES - set(checkpointables)
        unexpected = set(checkpointables) - REQUIRED_CHECKPOINTABLES
        if missing or unexpected:
            raise ValueError(
                "checkpointable names differ; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        progress = checkpointables["progress"]
        if not isinstance(progress, Mapping) or progress.get("iteration") != iteration:
            raise ValueError(
                "checkpoint progress.iteration must equal the save iteration"
            )
        self.wait(backpressure=True)
        manifest = self._checkpoint_manifest(
            iteration=iteration,
            checkpointables=checkpointables,
        )
        self._emit("checkpoint_snapshot_started", iteration=iteration)
        snapshot_started = time.perf_counter()
        try:
            response = self._checkpointer.save_checkpointables_async(
                iteration,
                dict(checkpointables),
                force=True,
                overwrite=False,
                metrics=None if metrics is None else dict(metrics),
                custom_metadata={
                    "schema_version": CHECKPOINT_SCHEMA,
                    "iteration": iteration,
                    "science_fingerprint": self.science_fingerprint,
                },
            )
        except BaseException as error:
            self._emit(
                "checkpoint_failed",
                iteration=iteration,
                phase="snapshot",
                duration_seconds=time.perf_counter() - snapshot_started,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        snapshot_seconds = time.perf_counter() - snapshot_started
        self._emit(
            "checkpoint_snapshot_finished",
            iteration=iteration,
            duration_seconds=snapshot_seconds,
        )
        if response is None:
            raise CheckpointWriteError(
                f"Orbax unexpectedly skipped forced checkpoint iteration {iteration}"
            )
        pending = _PendingSave(
            iteration=iteration,
            path=self._iteration_path(iteration),
            response=response,
            manifest=manifest,
            started_at=time.perf_counter(),
        )
        self._pending = pending
        self._emit(
            "checkpoint_queued",
            iteration=iteration,
            path=str(pending.path),
            snapshot_seconds=snapshot_seconds,
        )
        if self.asynchronous:
            pending.thread = threading.Thread(
                target=self._finalize,
                args=(pending,),
                name=f"representax-checkpoint-{iteration}",
                daemon=False,
            )
            pending.thread.start()
        else:
            self._finalize(pending)
            self.wait()
        return CheckpointTicket(
            iteration=iteration,
            path=pending.path,
            asynchronous=self.asynchronous,
        )

    def _finalize(self, pending: _PendingSave) -> None:
        try:
            if pending.response.result() is not True:
                raise CheckpointWriteError(
                    f"Orbax did not publish checkpoint iteration {pending.iteration}"
                )
            if self.process_index == 0:
                if not (pending.path / ORBAX_MARKER).is_file():
                    raise CheckpointWriteError(
                        f"Orbax completion metadata is missing: {pending.path}"
                    )
                pending.manifest["checkpoint_fingerprint"] = _manifest_fingerprint(
                    pending.manifest
                )
                _atomic_write_json(
                    pending.path / CHECKPOINT_MANIFEST,
                    pending.manifest,
                )
                _atomic_write_text(
                    pending.path / COMPLETE_MARKER,
                    pending.manifest["checkpoint_fingerprint"] + "\n",
                )
                _fsync_directory(pending.path)
                _atomic_write_json(
                    self.checkpoint_root / LATEST_POINTER,
                    {
                        "schema_version": LATEST_SCHEMA,
                        "iteration": pending.iteration,
                        "path": pending.path.name,
                        "checkpoint_fingerprint": pending.manifest[
                            "checkpoint_fingerprint"
                        ],
                    },
                )
                _fsync_directory(self.checkpoint_root)
                self._emit(
                    "checkpoint_saved",
                    iteration=pending.iteration,
                    path=str(pending.path),
                    duration_seconds=time.perf_counter() - pending.started_at,
                )
        except BaseException as error:
            pending.error = error
            self._emit(
                "checkpoint_failed",
                iteration=pending.iteration,
                phase="write",
                duration_seconds=time.perf_counter() - pending.started_at,
                error_type=type(error).__name__,
                error=str(error),
            )

    def wait(self, *, backpressure: bool = False) -> None:
        """Wait for the outstanding publication and surface its failure."""

        pending = self._pending
        if pending is None:
            return
        waiting = pending.thread is not None and pending.thread.is_alive()
        started = time.perf_counter()
        if backpressure and waiting:
            self._emit("checkpoint_backpressure_started", iteration=pending.iteration)
        if pending.thread is not None:
            pending.thread.join()
        if backpressure and waiting:
            self._emit(
                "checkpoint_backpressure_finished",
                iteration=pending.iteration,
                duration_seconds=time.perf_counter() - started,
            )
        self._pending = None
        if pending.error is not None:
            raise CheckpointWriteError(
                f"checkpoint iteration {pending.iteration} failed: {pending.error}"
            ) from pending.error

    def latest(self) -> CheckpointRecord:
        self.wait()
        pointer = self.checkpoint_root / LATEST_POINTER
        if not pointer.is_file():
            raise FileNotFoundError(f"no complete checkpoint pointer: {pointer}")
        document = json.loads(pointer.read_text())
        if document.get("schema_version") != LATEST_SCHEMA:
            raise IncompleteCheckpointError(f"invalid latest pointer: {pointer}")
        path_component = document.get("path")
        if (
            not isinstance(path_component, str)
            or Path(path_component).name != path_component
        ):
            raise IncompleteCheckpointError(f"invalid latest path: {pointer}")
        record = validate_complete_checkpoint(
            self.checkpoint_root / path_component,
            expected_science_fingerprint=self.science_fingerprint,
        )
        if record.iteration != document.get(
            "iteration"
        ) or record.checkpoint_fingerprint != document.get("checkpoint_fingerprint"):
            raise IncompleteCheckpointError(f"latest pointer differs: {pointer}")
        return record

    def record(self, iteration: int | None = None) -> CheckpointRecord:
        """Resolve and validate one complete checkpoint without restoring."""

        self.wait()
        if iteration is None:
            return self.latest()
        return validate_complete_checkpoint(
            self._iteration_path(iteration),
            expected_science_fingerprint=self.science_fingerprint,
        )

    def restore(
        self,
        checkpointables_like: Mapping[str, Any],
        *,
        iteration: int | None = None,
    ) -> tuple[dict[str, Any], CheckpointRecord]:
        """Restore training leaves into explicit shapes and shardings."""

        self.wait()
        record = self.record(iteration)
        requested = set(checkpointables_like)
        available = set(record.manifest.get("checkpointables", ()))
        if not requested or not requested <= available:
            raise ValueError(
                "restore keys must be a non-empty checkpoint subset; "
                f"requested={sorted(requested)}, available={sorted(available)}"
            )
        recorded_structures = record.manifest.get("structure_fingerprints")
        if not isinstance(recorded_structures, Mapping):
            raise IncompleteCheckpointError(
                f"checkpoint structure fingerprints are missing: {record.path}"
            )
        for name, value in checkpointables_like.items():
            if name == "progress":
                continue
            if recorded_structures.get(name) != tree_structure_fingerprint(value):
                raise IncompleteCheckpointError(
                    f"restore template structure differs for {name}: {record.path}"
                )
        started = time.perf_counter()
        restored = self._checkpointer.load_checkpointables(
            record.iteration,
            {
                name: None if name == "progress" else abstract_pytree(value)
                for name, value in checkpointables_like.items()
            },
        )
        progress = restored.get("progress")
        if progress is not None and (
            progress.get("schema_version") != PROGRESS_SCHEMA
            or progress.get("iteration") != record.iteration
        ):
            raise IncompleteCheckpointError(
                "restored progress differs from checkpoint iteration "
                f"{record.iteration}"
            )
        self._emit(
            "checkpoint_restored",
            iteration=record.iteration,
            path=str(record.path),
            checkpointables=sorted(restored),
            duration_seconds=time.perf_counter() - started,
        )
        return restored, record

    def restore_training_state(
        self,
        state_like: TrainState,
        *,
        iteration: int | None = None,
    ) -> RestoredTrainingState:
        """Restore the canonical bundle and reconstruct ``TrainState``."""

        like = training_checkpointables(
            state=state_like,
            iteration=0,
            rng=jax.random.key(0),
            data_state={},
            logging_cursor={"events_bytes": 0, "metrics_bytes": 0, "sequence": 0},
        )
        restored, record = self.restore(like, iteration=iteration)
        progress = restored["progress"]
        optimizer_step = int(jax.device_get(restored["optimizer_step"]))
        if optimizer_step != progress.get("optimizer_step"):
            raise IncompleteCheckpointError(
                "restored optimizer step differs from checkpoint progress"
            )
        if not isinstance(progress.get("data_state"), Mapping):
            raise IncompleteCheckpointError(
                "restored checkpoint has no Grain iterator state"
            )
        if not isinstance(progress.get("logging_cursor"), Mapping):
            raise IncompleteCheckpointError("restored checkpoint has no logging cursor")
        return RestoredTrainingState(
            state=TrainState(
                model=restored["model"],
                optimizer_state=restored["optimizer"],
                step=restored["optimizer_step"],
            ),
            iteration=int(progress["iteration"]),
            rng=restored["rng"],
            data_state=progress["data_state"],
            logging_cursor={
                str(key): int(value)
                for key, value in progress["logging_cursor"].items()
            },
            record=record,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.wait()
        finally:
            self._checkpointer.close()
            self._closed = True


__all__ = [
    "CHECKPOINT_MANIFEST",
    "COMPLETE_MARKER",
    "CheckpointConfig",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointRecord",
    "CheckpointTicket",
    "CheckpointWriteError",
    "IncompleteCheckpointError",
    "RestoredTrainingState",
    "science_fingerprint",
    "training_checkpointables",
    "validate_complete_checkpoint",
]
