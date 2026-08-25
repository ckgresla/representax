"""Optional W&B adapter for the asynchronous reporter protocol."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax

from representax.config import JobConfig, WandbConfig


def _runtime_metadata() -> dict[str, Any]:
    """Describe the visible JAX world as static run metadata."""

    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "global_device_count": jax.device_count(),
        "device_platforms": sorted({device.platform for device in devices}),
        "device_models": sorted(
            {str(getattr(device, "device_kind", "unknown")) for device in devices}
        ),
    }


class WandbReporter:
    """Forward canonical metric rows to one lazily initialized W&B run."""

    def __init__(
        self,
        config: WandbConfig,
        *,
        job: JobConfig,
        run_directory: str | Path,
        resume: bool = False,
        client: Any = None,
    ) -> None:
        self.config = config
        self.job = job
        self.run_directory = Path(run_directory).expanduser().resolve()
        self.resume = resume
        self._client = client
        self._run: Any = None
        self._status: str | None = None
        self._finished = False

    def _initialize(self) -> Any:
        if self._run is not None:
            return self._run
        if self._client is None:
            try:
                wandb = importlib.import_module("wandb")
            except ImportError as error:
                raise ImportError(
                    "W&B reporting requires `pip install representax[wandb]`"
                ) from error
            self._client = wandb
        run_id = self.config.run_id
        if run_id is None:
            run_id = hashlib.sha256(str(self.run_directory).encode()).hexdigest()[:16]
        job_config = self.job.model_dump(mode="json")
        self._run = self._client.init(
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            name=self.config.name or self.job.name,
            id=run_id,
            tags=self.config.tags,
            mode=self.config.mode,
            resume="must" if self.resume else "allow",
            dir=str(self.run_directory),
            config={**job_config, "runtime": _runtime_metadata()},
        )
        return self._run

    def write(self, row: Mapping[str, Any]) -> None:
        run = self._initialize()
        if row.get("category") != "metric":
            return
        values = dict(row["metrics"])
        if row.get("event") == "training_step":
            values["train/optimizer_step"] = int(row["optimizer_step"])
        run.log(values, step=int(row["iteration"]))

    def flush(self) -> None:
        """W&B's run.log is already durable through its own client process."""

    def finish(self, status: str, fields: Mapping[str, Any]) -> None:
        self._status = status
        if self._run is not None:
            summary = getattr(self._run, "summary", None)
            if summary is not None:
                summary.update(
                    {
                        "representax/status": status,
                        **{
                            f"representax/{name}": value
                            for name, value in fields.items()
                            if isinstance(value, (str, int, float, bool))
                        },
                    }
                )

    def close(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._run is not None:
            self._run.finish(exit_code=0 if self._status == "completed" else 1)


__all__ = ["WandbReporter"]
