"""W&B reporting through the bounded asynchronous logger."""

import json
from typing import Any, cast

from representax.config import JobConfig, LoggingConfig, WandbConfig
from representax.train import MetricRecord, RunLogger, WandbReporter


class FakeJob:
    name = "wandb-contract"

    def model_dump(self, *, mode):
        assert mode == "json"
        return {"name": self.name, "training": {"global_batch_size": 16}}


class FakeRun:
    def __init__(self):
        self.rows = []
        self.summary = {}
        self.exit_codes = []

    def log(self, values, *, step):
        self.rows.append((values, step))

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


class FakeWandb:
    def __init__(self):
        self.calls = []
        self.run = FakeRun()

    def init(self, **kwargs):
        self.calls.append(kwargs)
        return self.run


def test_wandb_config_is_optional_and_serializable() -> None:
    assert LoggingConfig().wandb is None
    assert LoggingConfig().timing is False
    assert LoggingConfig().accelerator is False
    config = LoggingConfig(
        wandb=WandbConfig(
            project="representax",
            entity="research",
            tags=("paper", "dense"),
            mode="offline",
        )
    )
    assert config.model_dump(mode="json")["wandb"]["project"] == "representax"


def test_wandb_receives_canonical_metrics_on_the_reporter_worker(tmp_path) -> None:
    client = FakeWandb()
    reporter = WandbReporter(
        WandbConfig(project="representax", mode="offline"),
        job=cast(JobConfig, FakeJob()),
        run_directory=tmp_path / "run",
        client=client,
    )
    logger = RunLogger(
        tmp_path / "run",
        manifest={},
        reporters=(reporter,),
        queue_size=2,
    )
    logger.event("training_started", iteration=0)
    logger.metrics(
        MetricRecord(
            iteration=1,
            values={
                "train/loss": 0.75,
                "train/skipped_update": False,
                "perf/examples_per_second": 128.0,
            },
        )
    )
    logger.metrics(
        MetricRecord(
            iteration=1,
            event="evaluation",
            values={"valid/retrieval/ndcg@10": 0.42},
        )
    )
    logger.finish("completed", completed_iterations=1)
    logger.close()

    metric_rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert metric_rows[0]["optimizer_step"] == 1
    assert "optimizer_step" not in metric_rows[1]

    assert len(client.calls) == 1
    assert client.calls[0]["project"] == "representax"
    assert client.calls[0]["name"] == "wandb-contract"
    assert client.calls[0]["resume"] == "allow"
    assert client.calls[0]["config"]["runtime"]["global_device_count"] >= 1
    assert client.calls[0]["config"]["runtime"]["process_count"] >= 1
    assert client.run.rows == [
        (
            {
                "train/loss": 0.75,
                "train/skipped_update": False,
                "perf/examples_per_second": 128.0,
                "train/optimizer_step": 1,
            },
            1,
        ),
        (
            {
                "valid/retrieval/ndcg@10": 0.42,
            },
            1,
        ),
    ]
    assert client.run.summary == {
        "representax/status": "completed",
        "representax/completed_iterations": 1,
    }
    assert client.run.exit_codes == [0]


def test_wandb_failure_status_uses_a_nonzero_exit_code(tmp_path) -> None:
    client = FakeWandb()
    reporter = WandbReporter(
        WandbConfig(project="representax", mode="disabled"),
        job=cast(JobConfig, FakeJob()),
        run_directory=tmp_path / "failed",
        client=client,
    )
    reporter.write(
        {
            "category": "event",
            "event": "training_started",
            "iteration": 0,
            "optimizer_step": 0,
        }
    )
    reporter.finish("failed", cast(dict[str, Any], {"error_type": "ValueError"}))
    reporter.close()
    assert len(client.calls) == 1
    assert client.run.rows == []
    assert client.run.exit_codes == [1]
