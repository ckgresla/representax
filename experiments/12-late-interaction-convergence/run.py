"""Train the three-seed GTE-ModernColBERT convergence result."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(
    os.environ.get(
        "REPRESENTAX_EXPERIMENT_PYTHON", ROOT / "experiments/.venv/bin/python"
    )
)
PAPER_ROOT = Path(os.environ.get("REPRESENTAX_PAPER_ROOT", "/raid/representax-paper"))
ASSET_ROOT = Path(
    os.environ.get("REPRESENTAX_PAPER_ASSETS", "/raid/representax-paper-assets")
)
OUTPUT = PAPER_ROOT / "12-late-interaction-convergence"
PREVIOUS_OUTPUT = OUTPUT
OUTPUT = OUTPUT / "hard-negatives"
DATA = ASSET_ROOT / "late-interaction-hard-negatives"
NANOBEIR = (
    Path("/raid/representax/data/dense-retrieval-msmarco-v1/artifacts")
    / "sentence-transformers/NanoBEIR-en"
)
CHECKPOINT = ASSET_ROOT / "late-checkpoint"

MODEL_ID = "lightonai/GTE-ModernColBERT-v1"
MODEL_REVISION = "cbbe53366e564450558f5e639dd499171f127538"
SEEDS = (7, 42, 773)
STEPS = 1_000
GLOBAL_BATCH_SIZE = 512
QUERY_BUCKETS = (16, 32)
DOCUMENT_BUCKETS = (32, 64, 128, 256)
GRAD_CACHE_MICRO_BATCH = 8
WARMUP_STEPS = round(STEPS * 0.06)
DATASET_ID = "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
DATASET_REVISION = "0d54352548089199bde15ad7e06efe895dc80b56"
EVALUATION_STEPS = (100, 250, 500, STEPS)
EVALUATION_SCORE_DTYPE = "float32"


class TripletCollator:
    """Add mined documents without treating copies of a positive as negatives."""

    def __init__(self, *, processor: Any) -> None:
        self.processor = processor

    def data_contract(self) -> dict[str, Any]:
        return {
            "processor": self.processor.data_contract(),
            "fields": ["query", "positive", "negative"],
        }

    def __call__(self, examples: Any) -> Any:
        import numpy as np

        from representax.core import Route
        from representax.tasks.retrieval import retrieval_batch

        queries = tuple(str(row["query"]) for row in examples)
        positives = tuple(str(row["positive"]) for row in examples)
        negatives = tuple(str(row["negative"]) for row in examples)
        if any(not value.strip() for value in (*queries, *positives, *negatives)):
            raise ValueError("triplet fields must be nonempty")
        if any(p == n for p, n in zip(positives, negatives, strict=True)):
            raise ValueError("a mined negative equals its positive")
        if len(set(queries)) != len(queries) or len(set(positives)) != len(positives):
            raise ValueError("a training batch contains duplicate queries or positives")
        documents = (*positives, *negatives)
        first_position = {}
        for index, text in enumerate(documents):
            first_position.setdefault(text, index)
        valid = np.array(
            [first_position[text] == index for index, text in enumerate(documents)]
        )
        return retrieval_batch(
            query=self.processor(queries, route=Route.QUERY),
            document=self.processor(documents, route=Route.DOCUMENT),
            positive_mask=np.array(
                [[positive == doc for doc in documents] for positive in positives]
            ),
            document_valid=valid,
        )


def job_config(seed: int) -> Any:
    """Own the executable configuration; the launch record is derived from it."""
    from experiments.preflights.late_interaction import _representax_job

    from representax.config import JobConfig

    values = _representax_job(
        checkpoint=CHECKPOINT,
        data_directory=DATA / f"seed-{seed}",
        steps=STEPS,
        seed=seed,
        warmup_steps=WARMUP_STEPS,
    ).model_dump(mode="json")
    values["name"] = "paper-12-late-interaction-hard-negatives"
    values["model"]["parameters"].update(
        revision=MODEL_REVISION,
        query_sequence_length_buckets=list(QUERY_BUCKETS),
        document_sequence_length_buckets=list(DOCUMENT_BUCKETS),
    )
    values["export"]["huggingface"]["adapter"]["parameters"].update(
        model_id=MODEL_ID, revision=MODEL_REVISION
    )
    values["data"]["collate"] = {
        "target": "experiments.12-late-interaction-convergence.run:TripletCollator",
    }
    values["training"]["global_batch_size"] = GLOBAL_BATCH_SIZE
    values["training"]["batch"]["micro_batch_size"] = GLOBAL_BATCH_SIZE
    values["training"]["grad_cache"]["micro_batch_size"] = GRAD_CACHE_MICRO_BATCH
    values["training"]["grad_cache"]["loss_row_chunk_size"] = GRAD_CACHE_MICRO_BATCH
    values["checkpointing"]["additional_iterations"] = [100, 250]
    values["checkpointing"]["keep"] = 4
    return JobConfig.model_validate(values)


def contract(job: Any = None) -> dict[str, Any]:
    job = job_config(SEEDS[0]) if job is None else job
    training = job.training
    parameters = job.model.parameters
    schedule = job.optimization.schedule.parameters
    return {
        "experiment": "12-late-interaction-convergence",
        "model": {"id": MODEL_ID, "revision": parameters["revision"]},
        "training_data": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "configuration": "triplet-hard",
            "mined_negatives_per_query": 1,
            "duplicate_queries": 0,
            "duplicate_positives": 0,
            "order": "independent deterministic permutation per seed",
        },
        "seeds": list(SEEDS),
        "optimizer_steps": training.max_steps,
        "global_batch_size": training.global_batch_size,
        "training_presentations": training.max_steps * training.global_batch_size,
        "query_sequence_length_buckets": parameters["query_sequence_length_buckets"],
        "document_sequence_length_buckets": parameters[
            "document_sequence_length_buckets"
        ],
        "grad_cache_micro_batch_size": training.grad_cache.micro_batch_size,
        "loss": {
            "name": "late-interaction-contrastive",
            "temperature": job.loss.temperature,
            "symmetric": job.loss.symmetric,
            "negative_scope": job.loss.negative_scope,
        },
        "optimization": {
            "optimizer": job.optimization.optimizer.target,
            "learning_rate": schedule["peak_value"],
            "weight_decay": job.optimization.optimizer.parameters["weight_decay"],
            "warmup_steps": schedule["warmup_steps"],
            "schedule": job.optimization.schedule.target,
            "gradient_clip_norm": job.optimization.max_gradient_norm,
            "precision": training.precision.model_dump(mode="json"),
        },
        "evaluation": {
            "dataset": "NanoMSMARCO",
            "steps": [0, *EVALUATION_STEPS],
            "score_dtype": EVALUATION_SCORE_DTYPE,
        },
        "checkpointing": job.checkpointing.model_dump(mode="json"),
        "export": job.export.model_dump(mode="json"),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = subprocess.run(
        ("git", "diff", "HEAD", "--binary"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "working_tree_clean": not bool(patch),
        "working_tree_patch_sha256": "sha256:" + hashlib.sha256(patch).hexdigest(),
    }


def prepare_data() -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as parquet
    from experiments.preflights import late_interaction as late
    from huggingface_hub import snapshot_download

    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for seed in SEEDS:
            path = DATA / f"seed-{seed}" / "train.jsonl"
            if late._sha256(path) != manifest["seed_files"][str(seed)]["sha256"]:
                raise ValueError(f"prepared data changed: {path}")
        return
    source = Path(
        snapshot_download(
            DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=["triplet-hard/*.parquet"],
        )
    )
    files = sorted((source / "triplet-hard").glob("*.parquet"))
    if len(files) != 20:
        raise ValueError("the pinned triplet-hard source must contain all 20 shards")
    rows = []
    queries, positives = set(), set()
    for path in files:
        for index, row in enumerate(late._parquet_rows(path)):
            query, positive, negative = (
                str(row[key]).strip() for key in ("query", "positive", "negative")
            )
            if not query or not positive or not negative or positive == negative:
                continue
            if query in queries or positive in positives:
                continue
            queries.add(query)
            positives.add(positive)
            rows.append(
                dict(
                    query=query,
                    positive=positive,
                    negative=negative,
                    source_file=path.name,
                    source_row=index,
                )
            )
    # Repeat complete batches only: no tail-to-head batch can duplicate a query.
    count = len(rows) - len(rows) % GLOBAL_BATCH_SIZE
    if count < GLOBAL_BATCH_SIZE:
        raise ValueError("filtered source did not yield one full batch")
    table = pa.Table.from_pylist(rows[:count])
    seed_files = {}
    for seed in SEEDS:
        destination = DATA / f"seed-{seed}"
        path = DATA / f"seed-{seed}.parquet"
        permutation = np.random.default_rng(seed).permutation(count)
        parquet.write_table(table.take(pa.array(permutation)), path, compression="zstd")
        manifest = late.prepare_data(
            destination,
            training_parquet=path,
            nanobeir_directory=NANOBEIR,
            training_rows=count,
            negative_field="negative",
        )
        seed_files[str(seed)] = manifest["training"]
    _write_json(
        manifest_path,
        {
            **contract()["training_data"],
            "usable_rows": count,
            "source_files": [
                {"file": str(path.relative_to(source)), "sha256": late._sha256(path)}
                for path in files
            ],
            "seed_files": seed_files,
        },
    )


def worker_command(seed: int) -> list[str]:
    return [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "worker",
        "--seed",
        str(seed),
    ]


def _environment(gpu: int) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))
        ),
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HF_HOME": os.environ.get("HF_HOME", "/raid/.cache/huggingface"),
        "JAX_COMPILATION_CACHE_DIR": str(OUTPUT / "jax-cache"),
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    }


def run_seed(seed: int, gpu: int) -> None:
    data = DATA / f"seed-{seed}" / "manifest.json"
    if not data.is_file():
        raise FileNotFoundError(f"prepared seed data is missing: {data}")
    run = OUTPUT / "runs" / f"seed-{seed}"
    if run.exists():
        raise FileExistsError(f"run already exists: {run}")
    run.mkdir(parents=True)
    command = worker_command(seed)
    print(shlex.join(command), flush=True)
    with (run / "worker.log").open("x", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def worker(seed: int) -> None:
    import jax
    from experiments.preflights import late_interaction as late

    from representax.train import run_job
    from representax.train.job import load_model

    job = job_config(seed)
    data = DATA / f"seed-{seed}"
    run = OUTPUT / "runs" / f"seed-{seed}"
    manifest = json.loads((DATA / "manifest.json").read_text())
    expected = manifest["seed_files"][str(seed)]["sha256"]
    if late._sha256(data / "train.jsonl") != expected:
        raise ValueError("training data no longer matches its prepared manifest")
    if (run / "launch.json").exists():
        raise FileExistsError(f"run already launched: {run}")
    run.mkdir(parents=True, exist_ok=True)
    patch = subprocess.run(
        ("git", "diff", "HEAD", "--binary"), cwd=ROOT, check=True, capture_output=True
    ).stdout
    (run / "working-tree.patch").write_bytes(patch)
    _write_json(
        run / "launch.json",
        {
            "job": job.model_dump(mode="json"),
            "contract": contract(job),
            "git": _git_state(),
            "command": worker_command(seed),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "data_manifest": manifest,
        },
    )
    initial, processor = load_model(
        job.model,
        key=jax.random.key(job.training.seed),
        activation_rematerialization=job.training.activation_rematerialization,
    )

    def evaluate(model: Any, step: int) -> dict[str, Any]:
        result = late._representax_evaluation(
            model,
            processor,
            data,
            index_directory=None,
            score_dtype=EVALUATION_SCORE_DTYPE,
            query_buckets=tuple(job.model.parameters["query_sequence_length_buckets"]),
            document_buckets=tuple(
                job.model.parameters["document_sequence_length_buckets"]
            ),
        )
        _write_json(run / f"evaluation-{step}.json", result)
        print(
            json.dumps({"iteration": step, "evaluation": result["metrics"]}), flush=True
        )
        return result

    initial_evaluation = evaluate(initial, 0)
    del initial
    gc.collect()
    milestones = EVALUATION_STEPS
    if (
        tuple(sorted(set(milestones))) != milestones
        or milestones[-1] != job.training.max_steps
    ):
        raise ValueError("evaluation milestones must end at the full training budget")
    for stage, step in enumerate(milestones):
        completed = run_job(job, run / "run", resume=stage > 0, stop_after=step)
        if completed.completed_iterations != step:
            raise RuntimeError("training did not reach the requested evaluation step")
        jax.block_until_ready(completed.state)
        final_evaluation = evaluate(completed.state.model, step)
        rows = late._metric_rows(run / "run" / "metrics.jsonl")
        training = [row for row in rows if row.get("event") == "training_step"]
        _write_json(
            run / "report.json",
            {
                "steps": completed.completed_iterations,
                "initial_evaluation": initial_evaluation,
                "final_evaluation": final_evaluation,
                "final_loss": float(training[-1]["metrics"]["train/loss"]),
                "steady_state": late.representax_steady_state(
                    training, job.training.global_batch_size
                ),
                "inference_bundle": str(completed.inference_bundle)
                if completed.inference_bundle
                else None,
            },
        )
        del completed
        gc.collect()


def evaluate_existing(seed: int) -> None:
    import jax
    from experiments.preflights import late_interaction as late

    from representax import load_inference_bundle
    from representax.train.job import load_model

    report = json.loads(
        (PREVIOUS_OUTPUT / "runs" / f"seed-{seed}" / "report.json").read_text()
    )
    model, job = load_inference_bundle(report["inference_bundle"])
    initial, processor = load_model(
        job.model,
        key=jax.random.key(seed),
        activation_rematerialization=job.training.activation_rematerialization,
    )
    data = ASSET_ROOT / "late-interaction-convergence" / f"seed-{seed}"
    for name, value in (("initial", initial), ("final", model)):
        result = late._representax_evaluation(
            value,
            processor,
            data,
            index_directory=None,
            score_dtype=EVALUATION_SCORE_DTYPE,
        )
        _write_json(
            OUTPUT / "previous-fp32-evaluation" / f"seed-{seed}-{name}.json", result
        )
        print(
            json.dumps({"seed": seed, "model": name, "metrics": result["metrics"]}),
            flush=True,
        )


def run_all(gpus: tuple[int, ...]) -> None:
    if len(gpus) != len(SEEDS) or len(set(gpus)) != len(gpus):
        raise ValueError("all requires three distinct GPU indices")
    processes = []
    for seed, gpu in zip(SEEDS, gpus, strict=True):
        command = [str(PYTHON), __file__, "run", "--seed", str(seed), "--gpu", str(gpu)]
        processes.append((seed, subprocess.Popen(command, cwd=ROOT)))
    failures = [(seed, process.wait()) for seed, process in processes]
    failed = [(seed, code) for seed, code in failures if code]
    if failed:
        raise RuntimeError(f"late-interaction convergence workers failed: {failed}")
    aggregate()


def aggregate() -> None:
    reports = {}
    for seed in SEEDS:
        path = OUTPUT / "runs" / f"seed-{seed}" / "report.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing seed report: {path}")
        reports[str(seed)] = json.loads(path.read_text())
    _write_json(OUTPUT / "summary.json", {"contract": contract(), "runs": reports})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("contract")
    commands.add_parser("prepare")
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluate = commands.add_parser("evaluate-existing")
    evaluate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run = commands.add_parser("run")
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run.add_argument("--gpu", type=int, required=True)
    all_runs = commands.add_parser("all")
    all_runs.add_argument("--gpus", type=int, nargs=3, required=True)
    commands.add_parser("aggregate")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "contract":
        print(json.dumps(contract(), indent=2, sort_keys=True))
    elif arguments.command == "prepare":
        prepare_data()
    elif arguments.command == "run":
        run_seed(arguments.seed, arguments.gpu)
    elif arguments.command == "worker":
        worker(arguments.seed)
    elif arguments.command == "evaluate-existing":
        evaluate_existing(arguments.seed)
    elif arguments.command == "all":
        run_all(tuple(arguments.gpus))
    else:
        aggregate()


if __name__ == "__main__":
    main()
