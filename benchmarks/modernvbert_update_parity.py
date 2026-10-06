#!/usr/bin/env python3
"""Compare one matched ModernVBERT optimizer update over every text parameter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

SCHEMA_VERSION = "representax-modernvbert-update-parity-v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--representax-weights", type=Path, required=True)
    parser.add_argument("--sentence-transformers-weights", type=Path, required=True)
    parser.add_argument("--representax-gradients", type=Path, required=True)
    parser.add_argument("--sentence-transformers-gradients", type=Path, required=True)
    parser.add_argument("--representax-report", type=Path, required=True)
    parser.add_argument("--sentence-transformers-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-delta-relative", type=float)
    parser.add_argument("--maximum-loss-absolute", type=float)
    parser.add_argument("--maximum-gradient-relative", type=float)
    parser.add_argument("--minimum-gradient-cosine", type=float)
    parser.add_argument("--maximum-update-relative", type=float)
    parser.add_argument("--minimum-update-cosine", type=float)
    return parser.parse_args()


def _report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"optimizer report must be an object: {path}")
    return value


def _require_matched_reports(native: dict[str, Any], upstream: dict[str, Any]) -> None:
    for report in (native, upstream):
        if report.get("status") != "completed" or report.get("oom") is not False:
            raise ValueError("optimizer-update worker did not complete")
        if report.get("optimizer_updates") != 1:
            raise ValueError("optimizer parity requires exactly one update")
    for name in (
        "batch_size",
        "checkpoint_revision",
        "chunk_size",
        "learning_rate",
        "max_gradient_norm",
        "seed",
        "sequence_length",
        "workload_fingerprints",
    ):
        if native.get(name) != upstream.get(name):
            raise ValueError(f"optimizer reports differ in {name!r}")
    for name in ("parameters", "compute", "objective", "float32_matmul"):
        if native["precision_policy"].get(name) != upstream["precision_policy"].get(
            name
        ):
            raise ValueError(f"optimizer precision policies differ in {name!r}")


def _tensor_metrics(
    original: np.ndarray | None,
    native: np.ndarray,
    upstream: np.ndarray,
) -> dict[str, float | int]:
    if native.shape != upstream.shape or (
        original is not None and original.shape != native.shape
    ):
        raise ValueError("matched optimizer tensors have different shapes")
    native_delta_squared = 0.0
    upstream_delta_squared = 0.0
    difference_squared = 0.0
    delta_inner_product = 0.0
    maximum_updated_absolute = 0.0
    maximum_delta_absolute = 0.0
    elements = int(native.size)
    original_flat = None if original is None else original.reshape(-1)
    native_flat = native.reshape(-1)
    upstream_flat = upstream.reshape(-1)
    for start in range(0, elements, 1_000_000):
        stop = min(start + 1_000_000, elements)
        native_delta = native_flat[start:stop].astype(np.float64)
        upstream_delta = upstream_flat[start:stop].astype(np.float64)
        if original_flat is not None:
            baseline = original_flat[start:stop].astype(np.float64)
            native_delta -= baseline
            upstream_delta -= baseline
        difference = native_delta - upstream_delta
        native_delta_squared += float(native_delta @ native_delta)
        upstream_delta_squared += float(upstream_delta @ upstream_delta)
        difference_squared += float(difference @ difference)
        delta_inner_product += float(native_delta @ upstream_delta)
        maximum_updated_absolute = max(
            maximum_updated_absolute,
            float(
                np.max(
                    np.abs(
                        native_flat[start:stop].astype(np.float64)
                        - upstream_flat[start:stop].astype(np.float64)
                    )
                )
            ),
        )
        maximum_delta_absolute = max(
            maximum_delta_absolute,
            float(np.max(np.abs(difference))),
        )
    native_norm = math.sqrt(native_delta_squared)
    upstream_norm = math.sqrt(upstream_delta_squared)
    difference_norm = math.sqrt(difference_squared)
    return {
        "elements": elements,
        "native_delta_l2": native_norm,
        "sentence_transformers_delta_l2": upstream_norm,
        "delta_difference_l2": difference_norm,
        "delta_relative_difference": difference_norm / max(upstream_norm, 1e-30),
        "delta_cosine": delta_inner_product / max(native_norm * upstream_norm, 1e-30),
        "maximum_updated_absolute_difference": maximum_updated_absolute,
        "maximum_delta_absolute_difference": maximum_delta_absolute,
    }


def _aggregate_metrics(
    tensors: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    native_squared = sum(
        float(value["native_delta_l2"]) ** 2 for value in tensors.values()
    )
    upstream_squared = sum(
        float(value["sentence_transformers_delta_l2"]) ** 2
        for value in tensors.values()
    )
    difference_squared = sum(
        float(value["delta_difference_l2"]) ** 2 for value in tensors.values()
    )
    inner_product = sum(
        float(value["delta_cosine"])
        * float(value["native_delta_l2"])
        * float(value["sentence_transformers_delta_l2"])
        for value in tensors.values()
    )
    native_norm = math.sqrt(native_squared)
    upstream_norm = math.sqrt(upstream_squared)
    difference_norm = math.sqrt(difference_squared)
    return {
        "native_l2": native_norm,
        "sentence_transformers_l2": upstream_norm,
        "difference_l2": difference_norm,
        "relative_difference": difference_norm / max(upstream_norm, 1e-30),
        "cosine": inner_product / max(native_norm * upstream_norm, 1e-30),
    }


def _matched_tensor_metrics(
    original_path: Path | None,
    native_path: Path,
    upstream_path: Path,
) -> dict[str, dict[str, float | int]]:
    with (
        safe_open(native_path, framework="np") as native,
        safe_open(upstream_path, framework="np") as upstream,
    ):
        native_names = set(native.keys())
        upstream_names = set(upstream.keys())
        if native_names != upstream_names:
            raise ValueError(
                "matched tensor names differ: "
                f"native-only={sorted(native_names - upstream_names)}, "
                f"upstream-only={sorted(upstream_names - native_names)}"
            )
        if original_path is None:
            return {
                name: _tensor_metrics(
                    None,
                    native.get_tensor(name),
                    upstream.get_tensor(name),
                )
                for name in sorted(native_names)
            }
        with safe_open(original_path, framework="np") as original:
            if not native_names.issubset(set(original.keys())):
                raise ValueError("matched tensors are absent from the checkpoint")
            return {
                name: _tensor_metrics(
                    original.get_tensor(name),
                    native.get_tensor(name),
                    upstream.get_tensor(name),
                )
                for name in sorted(native_names)
            }


def compare(arguments: argparse.Namespace) -> dict[str, Any]:
    native_report = _report(arguments.representax_report)
    upstream_report = _report(arguments.sentence_transformers_report)
    _require_matched_reports(native_report, upstream_report)
    checkpoint = arguments.checkpoint / "model.safetensors"
    tensors = _matched_tensor_metrics(
        checkpoint,
        arguments.representax_weights,
        arguments.sentence_transformers_weights,
    )
    gradients = _matched_tensor_metrics(
        None,
        arguments.representax_gradients,
        arguments.sentence_transformers_gradients,
    )
    changed = {
        name: metrics
        for name, metrics in tensors.items()
        if float(metrics["sentence_transformers_delta_l2"]) > 0.0
    }
    if not changed:
        raise ValueError("the upstream optimizer update changed no text weights")
    worst_name, worst = max(
        changed.items(),
        key=lambda item: float(item[1]["delta_relative_difference"]),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "batch_size": native_report["batch_size"],
            "sequence_length": native_report["sequence_length"],
            "chunk_size": native_report["chunk_size"],
            "seed": native_report["seed"],
            "learning_rate": native_report["learning_rate"],
            "max_gradient_norm": native_report["max_gradient_norm"],
            "workload_fingerprints": native_report["workload_fingerprints"],
        },
        "loss_absolute_difference": abs(
            float(native_report["losses"][0]) - float(upstream_report["losses"][0])
        ),
        "gradient_norm_relative_difference": abs(
            float(native_report["gradient_global_norm"])
            - float(upstream_report["gradient_global_norm"])
        )
        / max(abs(float(upstream_report["gradient_global_norm"])), 1e-30),
        "tensor_count": len(tensors),
        "changed_tensor_count": len(changed),
        "global_update": _aggregate_metrics(tensors),
        "global_gradient": _aggregate_metrics(gradients),
        "maximum_delta_relative_difference": float(worst["delta_relative_difference"]),
        "worst_tensor": worst_name,
        "worst_tensor_metrics": worst,
        "maximum_updated_absolute_difference": max(
            float(metrics["maximum_updated_absolute_difference"])
            for metrics in tensors.values()
        ),
        "maximum_delta_absolute_difference": max(
            float(metrics["maximum_delta_absolute_difference"])
            for metrics in tensors.values()
        ),
        "tensors": tensors,
        "gradients": gradients,
    }
    maximum = arguments.maximum_delta_relative
    if maximum is not None and result["maximum_delta_relative_difference"] > maximum:
        raise AssertionError(
            "optimizer delta relative difference exceeds acceptance threshold: "
            f"{result['maximum_delta_relative_difference']:.6g} > {maximum:.6g}"
        )
    gates = (
        (
            "loss absolute difference",
            result["loss_absolute_difference"],
            arguments.maximum_loss_absolute,
            "maximum",
        ),
        (
            "gradient relative difference",
            result["global_gradient"]["relative_difference"],
            arguments.maximum_gradient_relative,
            "maximum",
        ),
        (
            "gradient cosine",
            result["global_gradient"]["cosine"],
            arguments.minimum_gradient_cosine,
            "minimum",
        ),
        (
            "update relative difference",
            result["global_update"]["relative_difference"],
            arguments.maximum_update_relative,
            "maximum",
        ),
        (
            "update cosine",
            result["global_update"]["cosine"],
            arguments.minimum_update_cosine,
            "minimum",
        ),
    )
    for name, observed, threshold, direction in gates:
        if threshold is None:
            continue
        failed = (
            observed > threshold if direction == "maximum" else observed < threshold
        )
        if failed:
            raise AssertionError(
                f"{name} failed: observed={observed:.8g}, {direction}={threshold:.8g}"
            )
    return result


def main() -> None:
    arguments = _arguments()
    result = compare(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                name: value
                for name, value in result.items()
                if name not in {"tensors", "gradients"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
