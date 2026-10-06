"""Run the canonical evaluator adapters against their pinned upstream sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.core import EncoderMetadata, Modality, Route
from representax.data import build_dataset, mix
from representax.evaluation import (
    BANKING77_TEST,
    BANKING77_TRAIN,
    CIFAR100_TEST,
    CIFAR100_TRAIN,
    SPRINT_DUPLICATE_QUESTIONS,
    TWENTY_NEWSGROUPS,
    ClassificationProbeEvaluator,
    ClusteringEvaluator,
    JEPARepresentationEvaluator,
    PairClassificationEvaluator,
    clustering_evaluation_batches,
    clustering_samples,
    labeled_evaluation_batches,
    pair_classification_batches,
)
from representax.models.processing import Processor
from representax.train import EvaluationRunner


class Inputs(eqx.Module):
    values: jax.Array


class FrozenFeatures(eqx.Module):
    metadata: EncoderMetadata = eqx.field(static=True)

    def encode(self, inputs: Inputs, *, route=None, key=None):
        del route, key
        return inputs.values


MODEL = FrozenFeatures(
    EncoderMetadata(
        model_id="canonical-evaluator-source-acceptance",
        revision="v1",
        output_dimension=64,
        routes=frozenset({Route.GENERIC}),
        modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
    )
)


def _text_features(value: str) -> np.ndarray:
    digest = hashlib.shake_256(value.encode()).digest(64)
    values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    return (values - 127.5) / 127.5


def _image_features(value: Any) -> np.ndarray:
    pixels = np.asarray(value.convert("RGB"), dtype=np.float32) / 255.0
    height, width, _ = pixels.shape
    rows = np.array_split(np.arange(height), 4)
    columns = np.array_split(np.arange(width), 4)
    blocks = [
        pixels[np.ix_(row, column)].mean(axis=(0, 1))
        for row in rows
        for column in columns
    ]
    summary = np.concatenate(
        (*blocks, pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1)))
    )
    return np.pad(summary, (0, 64 - len(summary)))


def _features(values: Sequence[Any], **options: Any) -> Inputs:
    del options
    encoded = [
        _text_features(value) if isinstance(value, str) else _image_features(value)
        for value in values
    ]
    return Inputs(jnp.asarray(np.stack(encoded), dtype=jnp.float32))


PROCESSOR = Processor(
    process=_features,
    contract={
        "kind": "deterministic-source-acceptance-features",
        "dimension": 64,
        "scientific_quality_claim": False,
    },
)


def _dataset(source):
    return build_dataset(mix(source.source, shuffle=False))


def _digest_rows(records, fields: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for index in range(len(records)):
        row = records[index]
        for field in fields:
            value = row[field]
            if hasattr(value, "tobytes"):
                payload = value.tobytes()
            else:
                payload = json.dumps(value, sort_keys=True).encode()
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def _source_record(source, *, rows: int, digest: str) -> Mapping[str, Any]:
    return {
        "name": source.name,
        "uri": source.source.uri,
        "revision": source.source.revision,
        "split": source.source.split,
        "rows": rows,
        "expected_rows": source.expected_rows,
        "sha256": digest,
    }


def run(output: Path) -> None:
    started = time.perf_counter()
    print("[1/4] Sprint duplicate-question pair classification", flush=True)
    sprint = _dataset(SPRINT_DUPLICATE_QUESTIONS)
    sprint_row = sprint[0]
    sprint_count = len(sprint_row["labels"])
    sprint_result = EvaluationRunner(
        PairClassificationEvaluator(similarity_functions=("cosine",))
    ).run(
        MODEL,
        pair_classification_batches(
            sprint,
            processor=PROCESSOR,
            batch_size=1024,
        ),
    )

    print("[2/4] Banking77 linear probe", flush=True)
    banking_train = _dataset(BANKING77_TRAIN)
    banking_test = _dataset(BANKING77_TEST)
    banking_result = EvaluationRunner(
        ClassificationProbeEvaluator(
            inverse_regularization=(0.1, 1.0, 10.0),
            max_iterations=500,
            seed=0,
        )
    ).run(
        MODEL,
        labeled_evaluation_batches(
            train=banking_train,
            test=banking_test,
            processor=PROCESSOR,
            batch_size=512,
            validation_fraction=0.1,
            seed=0,
        ),
    )

    print("[3/4] Twenty Newsgroups clustering", flush=True)
    newsgroups = _dataset(TWENTY_NEWSGROUPS)
    clustering_metrics = []
    for index, sample in enumerate(clustering_samples(newsgroups)):
        result = EvaluationRunner(
            ClusteringEvaluator(
                name=f"clustering/sample_{index:02d}",
                batch_size=512,
                max_iterations=100,
                n_init=10,
                seed=0,
            )
        ).run(
            MODEL,
            clustering_evaluation_batches(
                sample,
                processor=PROCESSOR,
                batch_size=256,
            ),
        )
        clustering_metrics.append(dict(result.metrics))

    print("[4/4] CIFAR-100 JEPA transfer", flush=True)
    cifar_train = _dataset(CIFAR100_TRAIN)
    cifar_test = _dataset(CIFAR100_TEST)
    cifar_result = EvaluationRunner(
        JEPARepresentationEvaluator(
            inverse_regularization=(0.1, 1.0, 10.0),
            max_iterations=500,
            neighbors=20,
            query_batch_size=256,
            seed=0,
        )
    ).run(
        MODEL,
        labeled_evaluation_batches(
            train=cifar_train,
            test=cifar_test,
            processor=PROCESSOR,
            batch_size=512,
            example_field="img",
            label_field="fine_label",
            validation_fraction=0.1,
            seed=0,
        ),
    )

    artifact = {
        "schema_version": "representax-canonical-evaluator-acceptance-v1",
        "purpose": (
            "Adapter and evaluator acceptance only; deterministic frozen features "
            "are not a model-quality result."
        ),
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "elapsed_seconds": time.perf_counter() - started,
        "sources": [
            _source_record(
                SPRINT_DUPLICATE_QUESTIONS,
                rows=sprint_count,
                digest=hashlib.sha256(
                    json.dumps(
                        {
                            "sent1": sprint_row["sent1"],
                            "sent2": sprint_row["sent2"],
                            "labels": sprint_row["labels"],
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            ),
            _source_record(
                BANKING77_TRAIN,
                rows=len(banking_train),
                digest=_digest_rows(banking_train, ("text", "label")),
            ),
            _source_record(
                BANKING77_TEST,
                rows=len(banking_test),
                digest=_digest_rows(banking_test, ("text", "label")),
            ),
            _source_record(
                TWENTY_NEWSGROUPS,
                rows=len(newsgroups),
                digest=_digest_rows(newsgroups, ("sentences", "labels")),
            ),
            _source_record(
                CIFAR100_TRAIN,
                rows=len(cifar_train),
                digest=_digest_rows(cifar_train, ("fine_label",)),
            ),
            _source_record(
                CIFAR100_TEST,
                rows=len(cifar_test),
                digest=_digest_rows(cifar_test, ("fine_label",)),
            ),
        ],
        "metrics": {
            "sprint_duplicate_questions": dict(sprint_result.metrics),
            "banking77": dict(banking_result.metrics),
            "twenty_newsgroups": clustering_metrics,
            "cifar100_jepa_transfer": dict(cifar_result.metrics),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
