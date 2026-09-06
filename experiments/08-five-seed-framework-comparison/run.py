"""Run the bounded five-seed framework decision sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPOSITORY_ROOT / "experiments/.venv/bin/python"
PYLATE_PYTHON = REPOSITORY_ROOT / "experiments/.venv-late-interaction/bin/python"
OUTPUT_ROOT = Path(
    os.environ.get(
        "REPRESENTAX_DECISION_SWEEP_ROOT",
        "/raid/representax-paper/08-five-seed-framework-comparison",
    )
)
SEEDS = (7, 42, 773, 1234, 2026)
STEPS = 20
BUCKETS = (16, 32, 64, 128, 256)

MPNET = Path("/raid/representax/oracles/all-mpnet-base-v2")
DENSE_DATA = Path("/raid/representax/data/dense-retrieval-msmarco-v1")
PAIR_ROOT = Path(
    "/raid/representax-paper/02-semantic-similarity-pair-classification"
)
CROSS_CHECKPOINT = Path(
    "/raid/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/"
    "snapshots/233902d25c440f23af6f7d6e94d2946bac0bee0a"
)
CROSS_DATA = Path(
    "/raid/representax-paper/03-cross-encoder-reranking/"
    "screen-30-seed-7/data-v2"
)
LATE_CHECKPOINT = Path(
    "/raid/.cache/huggingface/hub/models--lightonai--GTE-ModernColBERT-v1/"
    "snapshots/cbbe53366e564450558f5e639dd499171f127538"
)
LATE_DATA = Path(
    "/raid/representax-paper/04-late-interaction/screen-30-seed-7/data"
)
QWEN = Path(
    "/raid/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
REWARD_ROOT = Path(
    "/raid/representax-paper/05-reward-modeling/screen-30-seed-7"
)
MULTIMODAL_ROOT = Path(
    "/raid/representax-paper/06-multimodal-retrieval/screen-30-seed-7"
)
CLIP = Path(
    "/raid/.cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/"
    "snapshots/327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"
)
OMNI = Path(
    "/raid/.cache/huggingface/hub/models--LCO-Embedding--LCO-Embedding-Omni-3B-2605/"
    "snapshots/5f6b5329da5141367da30e06a9826d1322d6c9b2"
)
VJEPA_REFERENCE = REPOSITORY_ROOT / "experiments/.references/vjepa2"
VJEPA_DATA = OUTPUT_ROOT / "vjepa-data"


@dataclass(frozen=True, slots=True)
class Recipe:
    name: str
    gpu_count: int
    loss_scale_comparable: bool = True
    serial: bool = False

    def command(self, seed: int, gpus: tuple[int, ...], output: Path) -> list[str]:
        python = str(PYTHON)
        if self.name == "dense-retrieval":
            command = [
                python,
                "-m",
                "benchmarks.dense_retrieval",
                "pair",
                "--model",
                "mpnet",
                "--checkpoint",
                str(MPNET),
                "--data-directory",
                str(DENSE_DATA),
                "--batch-size",
                "2048",
                "--steps",
                str(STEPS),
                "--maximum-length",
                "256",
                "--representax-cache-chunk-size",
                "32",
                "--sentence-transformers-cache-chunk-size",
                "128",
                "--data-threads",
                "4",
                "--prefetch-buffer-size",
                "8",
                "--sentence-transformers-data-threads",
                "0",
                "--sentence-transformers-torch-compile",
                "--mixed-precision",
                "--telemetry",
                "--seed",
                str(seed),
                "--result-directory",
                str(output),
                "--representax-gpu",
                str(gpus[0]),
                "--sentence-transformers-gpu",
                str(gpus[1]),
            ]
            for bucket in BUCKETS:
                command.extend(("--sequence-length-bucket", str(bucket)))
            return command

        if self.name.startswith(("semantic-similarity-", "pair-classification-")):
            workload, model = self.name.rsplit("-", 2)[0], self.name.rsplit("-", 2)[1]
            if self.name.startswith("semantic-similarity"):
                workload = "semantic-similarity"
                model = self.name.removeprefix("semantic-similarity-")
            else:
                workload = "pair-classification"
                model = self.name.removeprefix("pair-classification-")
            return [
                python,
                "-m",
                "experiments.preflights.semantic_pair",
                "pair",
                "--workload",
                workload,
                "--model",
                model,
                "--checkpoint",
                str(PAIR_ROOT / "checkpoints" / model),
                "--data-directory",
                str(PAIR_ROOT / "data"),
                "--output",
                str(output),
                "--steps",
                str(STEPS),
                "--seed",
                str(seed),
                "--representax-gpu",
                str(gpus[0]),
                "--reference-gpu",
                str(gpus[1]),
            ]

        module = {
            "cross-encoder": "cross_encoder",
            "late-interaction": "late_interaction",
            "outcome-reward": "outcome_reward",
            "process-reward": "process_reward",
            "image-text": "image_text",
            "audio-text": "audio_text",
            "video-text": "video_text",
            "v-jepa": "vjepa",
        }[self.name]
        command = [python, "-m", f"experiments.preflights.{module}", "pair"]
        checkpoint = {
            "cross-encoder": CROSS_CHECKPOINT,
            "late-interaction": LATE_CHECKPOINT,
            "outcome-reward": QWEN,
            "process-reward": QWEN,
            "image-text": CLIP,
            "audio-text": OMNI,
            "video-text": OMNI,
        }.get(self.name)
        data = {
            "cross-encoder": CROSS_DATA,
            "late-interaction": LATE_DATA,
            "outcome-reward": REWARD_ROOT / "outcome/data",
            "process-reward": REWARD_ROOT / "process/data",
            "image-text": MULTIMODAL_ROOT / "image-text/data",
            "audio-text": MULTIMODAL_ROOT / "audio-text/data",
            "video-text": MULTIMODAL_ROOT / "video-text/data",
            "v-jepa": VJEPA_DATA,
        }[self.name]
        if checkpoint is not None:
            command.extend(("--checkpoint", str(checkpoint)))
        if self.name == "v-jepa":
            command.extend(("--reference", str(VJEPA_REFERENCE)))
        command.extend(
            (
                "--data-directory",
                str(data),
                "--output",
                str(output),
                "--steps",
                str(STEPS),
                "--seed",
                str(seed),
            )
        )
        if self.name == "late-interaction":
            command.extend(
                (
                    "--representax-python",
                    python,
                    "--reference-python",
                    str(PYLATE_PYTHON),
                    "--hf-home",
                    "/raid/.cache/huggingface",
                    "--gpu",
                    str(gpus[0]),
                )
            )
        elif self.name == "outcome-reward":
            command.extend(
                (
                    "--trl-python",
                    python,
                    "--padding",
                    "static",
                    "--gpu",
                    str(gpus[0]),
                )
            )
        elif self.name == "process-reward":
            command.extend(
                (
                    "--representax-python",
                    python,
                    "--reference-python",
                    python,
                    "--hf-home",
                    "/raid/.cache/huggingface",
                    "--gpu",
                    str(gpus[0]),
                )
            )
        elif self.name == "audio-text":
            command.extend(
                (
                    "--batch-size",
                    "256",
                    "--representax-gpus",
                    str(gpus[0]),
                    "--representax-sharding",
                    "ddp",
                    "--reference-gpu",
                    str(gpus[0]),
                    "--continuous",
                )
            )
        elif self.name == "video-text":
            command.extend(("--batch-size", "128", "--gpu", str(gpus[0])))
        else:
            command.extend(("--gpu", str(gpus[0])))
        return command


RECIPES = (
    Recipe("dense-retrieval", 2),
    Recipe("semantic-similarity-mpnet-base", 2),
    Recipe("semantic-similarity-bert-base", 2),
    Recipe("pair-classification-mpnet-base", 2),
    Recipe("pair-classification-bert-base", 2),
    Recipe("cross-encoder", 1),
    Recipe("late-interaction", 1, loss_scale_comparable=False),
    Recipe("outcome-reward", 1),
    Recipe("process-reward", 1),
    Recipe("image-text", 1),
    Recipe("audio-text", 1, serial=True),
    Recipe("video-text", 1),
    Recipe("v-jepa", 1),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def _record_source() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    patch = _git("diff", "--binary", "HEAD", binary=True)
    assert isinstance(patch, bytes)
    source = OUTPUT_ROOT / "source"
    source.mkdir(exist_ok=True)
    (source / "working-tree.patch").write_bytes(patch)
    shutil.copy2(__file__, source / "run.py")
    _write_json(
        source / "provenance.json",
        {
            "representax_commit": str(_git("rev-parse", "HEAD")).strip(),
            "working_tree_patch_sha256": f"sha256:{hashlib.sha256(patch).hexdigest()}",
            "steps": STEPS,
            "seeds": SEEDS,
            "recipes": [recipe.name for recipe in RECIPES],
        },
    )


def _preserve_incomplete(path: Path) -> None:
    if not path.exists() or (path / "summary.json").is_file():
        return
    suffix = datetime.now().strftime("%Y%m%dT%H%M%S")
    path.rename(path.with_name(f"{path.name}-failed-{suffix}"))


def _run_one(recipe: Recipe, seed: int, gpus: tuple[int, ...]) -> dict[str, Any]:
    output = OUTPUT_ROOT / recipe.name / f"seed-{seed}"
    if (output / "summary.json").is_file():
        return {"recipe": recipe.name, "seed": seed, "status": "existing"}
    _preserve_incomplete(output)
    command = recipe.command(seed, gpus, output)
    rendered = tuple(map(str, command))
    for gpu in gpus:
        if str(gpu) not in rendered:
            raise AssertionError(
                f"{recipe.name} command omitted assigned GPU {gpu}: {command}"
            )
    log = OUTPUT_ROOT / "scheduler-logs" / recipe.name / f"seed-{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "REPRESENTAX_JAX_CACHE_DIR": str(OUTPUT_ROOT / "caches" / recipe.name),
            "REPRESENTAX_TORCHINDUCTOR_CACHE_DIR": str(
                OUTPUT_ROOT / "caches" / f"{recipe.name}-torchinductor"
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    status = "passed" if result.returncode == 0 else "failed"
    return {
        "recipe": recipe.name,
        "seed": seed,
        "status": status,
        "returncode": result.returncode,
        "gpus": gpus,
        "command": command,
        "log": str(log),
    }


def run(gpus: tuple[int, ...]) -> None:
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        raise ValueError("provide at least two distinct GPU indices")
    _record_source()
    pending = [(recipe, seed) for seed in SEEDS for recipe in RECIPES]
    free = list(gpus)
    active_recipes: dict[str, int] = {}
    futures: dict[Future[dict[str, Any]], tuple[Recipe, tuple[int, ...]]] = {}
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        while pending or futures:
            launched = True
            while launched:
                launched = False
                for index, (recipe, seed) in enumerate(pending):
                    cache_warmed = any(
                        (OUTPUT_ROOT / recipe.name / f"seed-{value}/summary.json").is_file()
                        for value in SEEDS
                    )
                    if (
                        active_recipes.get(recipe.name, 0) > 0
                        and (recipe.serial or not cache_warmed)
                    ) or recipe.gpu_count > len(free):
                        continue
                    assigned = tuple(free[: recipe.gpu_count])
                    del free[: recipe.gpu_count]
                    active_recipes[recipe.name] = active_recipes.get(recipe.name, 0) + 1
                    future = executor.submit(_run_one, recipe, seed, assigned)
                    futures[future] = (recipe, assigned)
                    pending.pop(index)
                    print(
                        f"launch {recipe.name} seed={seed} gpus={','.join(map(str, assigned))}",
                        flush=True,
                    )
                    launched = True
                    break
            if not futures:
                raise RuntimeError("no pending job fits the supplied GPU set")
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                recipe, assigned = futures.pop(future)
                free.extend(assigned)
                free.sort()
                active_recipes[recipe.name] -= 1
                if active_recipes[recipe.name] == 0:
                    del active_recipes[recipe.name]
                result = future.result()
                results.append(result)
                print(
                    f"{result['status']} {recipe.name} seed={result['seed']}",
                    flush=True,
                )
                _write_json(OUTPUT_ROOT / "scheduler-state.json", results)
    aggregate()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _representax_rows(
    summary: dict[str, Any], artifact_directory: Path
) -> list[dict[str, Any]]:
    report = summary["representax"]
    dense_metrics = artifact_directory / "runs/representax/metrics.jsonl"
    if dense_metrics.is_file():
        return [row for row in _metric_rows(dense_metrics) if row.get("event") == "training_step"]
    bundle = report.get("inference_bundle") or report.get("exported_bundle")
    if bundle:
        run = Path(bundle).parent
    else:
        command = summary.get("commands", {}).get("representax", [])
        try:
            run = Path(command[command.index("--run-directory") + 1])
        except (ValueError, IndexError) as error:
            raise ValueError("cannot locate Representax metric stream") from error
    metrics = run / "metrics.jsonl"
    if not metrics.is_file():
        candidates = list(run.rglob("metrics.jsonl"))
        if len(candidates) != 1:
            return []
        metrics = candidates[0]
    return [row for row in _metric_rows(metrics) if row.get("event") == "training_step"]


def _completion_throughput(
    summary: dict[str, Any], artifact_directory: Path
) -> tuple[float, list[float], float | None]:
    rows = _representax_rows(summary, artifact_directory)
    excluded = {1, STEPS // 2 + 1}
    durations = []
    examples = []
    data_wait = []
    report = summary["representax"]
    batch = float(
        report.get(
            "batch_size",
            report.get("global_batch_size", summary.get("configuration", {}).get("batch_size", 1)),
        )
    )
    for previous, current in zip(rows, rows[1:], strict=False):
        step = int(current.get("optimizer_step", current["iteration"]))
        if (
            step in excluded
            or "perf/compilation_and_first_step_seconds" in current["metrics"]
        ):
            continue
        start = datetime.fromisoformat(previous["timestamp"])
        end = datetime.fromisoformat(current["timestamp"])
        durations.append((end - start).total_seconds())
        examples.append(float(current["metrics"].get("perf/examples", batch)))
        data_wait.append(float(current["metrics"].get("perf/data_wait_seconds", 0.0)))
    if not durations:
        return float("nan"), [], None
    return sum(examples) / sum(durations), durations, sum(data_wait) / sum(durations)


def _reference(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for name in (
        "sentence-transformers",
        "sentence_transformers",
        "pylate",
        "trl",
        "facebookresearch-vjepa2",
        "stable-pretraining",
    ):
        if name in summary:
            return name, summary[name]
    raise KeyError("reference framework is absent")


def _reference_throughput(
    summary: dict[str, Any],
    reference: dict[str, Any],
    artifact_directory: Path,
) -> tuple[float, list[float]]:
    timings = reference.get("step_timings", [])
    if not timings:
        timings = reference.get("timing", {}).get("steps", [])
    dense_metrics = artifact_directory / "runs/sentence-transformers/metrics.jsonl"
    if not timings and dense_metrics.is_file():
        timings = [
            {
                "step": row.get("optimizer_step", row["iteration"]),
                "seconds": row["metrics"]["perf/step_seconds"],
            }
            for row in _metric_rows(dense_metrics)
            if row.get("event") == "training_step"
        ]
    excluded = {1, STEPS // 2 + 1}
    durations = [
        float(row["seconds"])
        for row in timings
        if int(row["step"]) not in excluded
    ]
    batch = float(
        reference.get(
            "batch_size",
            reference.get("global_batch_size", reference.get("examples_per_step", 1)),
        )
    )
    if durations:
        return batch * len(durations) / sum(durations), durations
    steady = reference.get("steady_state", {})
    value = steady.get("examples_per_second", reference.get("examples_per_second"))
    return float(value) if value is not None else float("nan"), []


def _losses(report: dict[str, Any]) -> list[float]:
    if "losses" in report:
        return [float(value) for value in report["losses"]][-STEPS:]
    history = report.get("training_metrics")
    if isinstance(history, list):
        return [
            float(row["loss"])
            for row in history
            if isinstance(row, dict) and row.get("loss") is not None
        ][-STEPS:]
    return []


def _reference_losses(
    report: dict[str, Any], artifact_directory: Path
) -> list[float]:
    values = _losses(report)
    if values:
        return values
    metrics = artifact_directory / "runs/sentence-transformers/metrics.jsonl"
    if metrics.is_file():
        return [
            float(row["metrics"]["train/loss"])
            for row in _metric_rows(metrics)
            if row.get("event") == "training_step"
            and "train/loss" in row["metrics"]
        ][-STEPS:]
    return []


def _representax_losses(
    summary: dict[str, Any], artifact_directory: Path
) -> list[float]:
    return [
        float(row["metrics"]["train/loss"])
        for row in _representax_rows(summary, artifact_directory)
        if "train/loss" in row["metrics"]
    ]


def _loss_comparison(native: list[float], reference: list[float]) -> dict[str, Any]:
    count = min(len(native), len(reference))
    if count == 0:
        return {
            "paired_steps": 0,
            "mean_absolute_difference": None,
            "final_difference": None,
            "direction_agrees": None,
        }
    native = native[-count:]
    reference = reference[-count:]
    return {
        "paired_steps": count,
        "mean_absolute_difference": statistics.fmean(
            abs(left - right) for left, right in zip(native, reference, strict=True)
        ),
        "final_difference": native[-1] - reference[-1],
        "direction_agrees": (native[-1] - native[0]) * (reference[-1] - reference[0])
        >= 0,
    }


def _median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _format_number(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _render_markdown(result: dict[str, Any]) -> str:
    rows = result["by_recipe"]
    completed = sum(row["completed_seeds"] for row in rows)
    expected = len(RECIPES) * len(SEEDS)
    lines = [
        "# Five-seed framework comparison",
        "",
        f"Status: {completed}/{expected} paired runs complete.",
        "",
        (
            f"Each recipe runs {len(SEEDS)} seeds x {STEPS} optimizer updates in "
            "each framework: 100 updates per framework and 200 paired updates "
            "per recipe. The complete sweep contains 130 framework runs and "
            "2,600 optimizer updates."
        ),
        "",
        (
            "Throughput is measured from one optimizer completion to the next, "
            "including input wait. Compiled-shape first uses and intervals that "
            "contain midpoint checkpoint or resume work are retained in raw logs "
            "but excluded from the steady-state statistic."
        ),
        "",
        (
            "| Recipe | Reference | Seeds | Representax ex/s | Reference ex/s | "
            "Median ratio [min, max] | Representax loss change | Reference loss "
            "change | Loss-direction agreement | Median step-loss difference | "
            "Representax input wait |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        loss_difference = (
            row["median_loss_mean_absolute_difference"]
            if row["loss_scale_comparable"]
            else None
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    row["recipe"],
                    row["reference"],
                    str(row["completed_seeds"]),
                    _format_number(
                        row["median_representax_examples_per_second"], digits=2
                    ),
                    _format_number(
                        row["median_reference_examples_per_second"], digits=2
                    ),
                    (
                        f"{row['median_throughput_ratio']:.3f}x "
                        f"[{row['minimum_throughput_ratio']:.3f}, "
                        f"{row['maximum_throughput_ratio']:.3f}]"
                    ),
                    _format_number(row["median_representax_loss_change"], digits=4),
                    _format_number(row["median_reference_loss_change"], digits=4),
                    (
                        "n/a"
                        if row["loss_direction_agreement_fraction"] is None
                        else f"{100 * row['loss_direction_agreement_fraction']:.0f}%"
                    ),
                    _format_number(loss_difference, digits=4),
                    f"{100 * row['median_representax_input_wait_fraction']:.1f}%",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            (
                "Loss change is final minus initial training loss over each "
                "20-update trace. Late-interaction losses use different framework "
                "reductions, so only direction and held-out metrics are comparable."
            ),
            "",
            (
                "These bounded traces measure steady execution and short loss "
                "dynamics. They do not establish converged held-out quality."
            ),
            "",
            "## Scope",
            "",
            (
                "This sweep covers the 13 implemented reusable paired runners. "
                "LeJEPA is omitted because its current one-step reference is an "
                "experiment-owned timm oracle using stable-pretraining components, "
                "not the external stable-pretraining training loop. "
                "Natural Questions and MIRACL are lifecycle checks without external "
                "framework pairs. The rejected BERT scaling toy is a separate study."
            ),
        )
    )
    if completed == expected and not result["failures"]:
        wins = sum(row["median_throughput_ratio"] > 1 for row in rows)
        lines.extend(
            (
                "",
                "## Decision",
                "",
                (
                    f"Representax is faster in {wins}/{len(rows)} median recipe "
                    "comparisons. These results can support a selective systems "
                    "paper, but not a universal speed claim. Promote only rows with "
                    "aligned loss dynamics and later held-out quality; retain losing "
                    "rows as explicit limitations."
                ),
            )
        )
    else:
        lines.extend(("", "## Decision", "", "Sweep incomplete; no decision yet."))
    return "\n".join(lines) + "\n"


def aggregate() -> None:
    rows = []
    failures = []
    for recipe in RECIPES:
        for seed in SEEDS:
            path = OUTPUT_ROOT / recipe.name / f"seed-{seed}/summary.json"
            if not path.is_file():
                failures.append({"recipe": recipe.name, "seed": seed, "reason": "missing"})
                continue
            try:
                summary = _read_json(path)
                reference_name, reference = _reference(summary)
                artifact_directory = path.parent
                native_rate, native_durations, native_input_wait = _completion_throughput(
                    summary, artifact_directory
                )
                reference_rate, reference_durations = _reference_throughput(
                    summary, reference, artifact_directory
                )
                native = summary["representax"]
                native_losses = _representax_losses(summary, artifact_directory)
                reference_losses = _reference_losses(reference, artifact_directory)
                rows.append(
                    {
                        "recipe": recipe.name,
                        "seed": seed,
                        "reference": reference_name,
                        "representax_examples_per_second": native_rate,
                        "reference_examples_per_second": reference_rate,
                        "throughput_ratio": native_rate / reference_rate,
                        "representax_step_seconds": native_durations,
                        "representax_input_wait_fraction": native_input_wait,
                        "reference_step_seconds": reference_durations,
                        "representax_losses": native_losses,
                        "reference_losses": reference_losses,
                        "loss_comparison": _loss_comparison(
                            native_losses, reference_losses
                        ),
                        "artifact": str(path),
                    }
                )
            except Exception as error:
                failures.append(
                    {"recipe": recipe.name, "seed": seed, "reason": repr(error)}
                )
    by_recipe = []
    for recipe in RECIPES:
        values = [row for row in rows if row["recipe"] == recipe.name]
        if not values:
            continue
        loss_values = [
            row["loss_comparison"]
            for row in values
            if row["loss_comparison"]["paired_steps"] == STEPS
        ]
        native_loss_changes = [
            row["representax_losses"][-1] - row["representax_losses"][0]
            for row in values
            if len(row["representax_losses"]) == STEPS
        ]
        reference_loss_changes = [
            row["reference_losses"][-1] - row["reference_losses"][0]
            for row in values
            if len(row["reference_losses"]) == STEPS
        ]
        ratios = [row["throughput_ratio"] for row in values]
        input_wait = [
            row["representax_input_wait_fraction"]
            for row in values
            if row["representax_input_wait_fraction"] is not None
        ]
        by_recipe.append(
            {
                "recipe": recipe.name,
                "reference": ", ".join(
                    sorted({row["reference"].replace("_", "-") for row in values})
                ),
                "loss_scale_comparable": recipe.loss_scale_comparable,
                "completed_seeds": len(values),
                "median_representax_examples_per_second": statistics.median(
                    row["representax_examples_per_second"] for row in values
                ),
                "median_reference_examples_per_second": statistics.median(
                    row["reference_examples_per_second"] for row in values
                ),
                "median_throughput_ratio": statistics.median(
                    ratios
                ),
                "minimum_throughput_ratio": min(ratios),
                "maximum_throughput_ratio": max(ratios),
                "throughput_ratio_median_absolute_deviation": (
                    _median_absolute_deviation(ratios)
                ),
                "median_representax_input_wait_fraction": (
                    statistics.median(input_wait) if input_wait else 0.0
                ),
                "completed_loss_seeds": len(loss_values),
                "median_representax_loss_change": (
                    statistics.median(native_loss_changes)
                    if native_loss_changes
                    else None
                ),
                "median_reference_loss_change": (
                    statistics.median(reference_loss_changes)
                    if reference_loss_changes
                    else None
                ),
                "median_loss_mean_absolute_difference": (
                    statistics.median(
                        row["mean_absolute_difference"] for row in loss_values
                    )
                    if loss_values
                    else None
                ),
                "median_final_loss_difference": (
                    statistics.median(row["final_difference"] for row in loss_values)
                    if loss_values
                    else None
                ),
                "loss_direction_agreement_fraction": (
                    statistics.fmean(row["direction_agrees"] for row in loss_values)
                    if loss_values
                    else None
                ),
            }
        )
    result = {
        "schema_version": "representax-five-seed-decision-sweep-v1",
        "steps_per_seed": STEPS,
        "seeds": SEEDS,
        "timing": (
            "optimizer-completion to optimizer-completion; includes input wait; "
            "excludes compiled-shape first uses and intervals containing midpoint "
            "checkpoint or resume work"
        ),
        "rows": rows,
        "by_recipe": by_recipe,
        "failures": failures,
    }
    _write_json(OUTPUT_ROOT / "summary.json", result)
    (OUTPUT_ROOT / "results.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    print(json.dumps(by_recipe, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--gpus", type=int, nargs="+", required=True)
    commands.add_parser("aggregate")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "run":
        run(tuple(arguments.gpus))
    else:
        aggregate()


if __name__ == "__main__":
    main()
