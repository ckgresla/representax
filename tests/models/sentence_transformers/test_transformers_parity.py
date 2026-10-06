"""Pinned full-string parity for the native dense sentence route."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest

from representax.integrations import (
    SENTENCE_TRANSFORMERS_ORACLE_VERSION,
    load_sentence_transformer,
)
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
)

pytestmark = pytest.mark.parity

_TEXTS = (
    "The weather is lovely today.",
    "It is raining.",
    "A bee lands on a flower.",
)


@dataclass(frozen=True, slots=True)
class _DenseParityCase:
    name: str
    checkpoint_environment: str
    model_id: str
    revision: str


_DENSE_PARITY_CASES = (
    _DenseParityCase(
        name="all-minilm-l6-v2",
        checkpoint_environment="REPRESENTAX_MINILM_CHECKPOINT",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    _DenseParityCase(
        name="all-mpnet-base-v2",
        checkpoint_environment="REPRESENTAX_MPNET_CHECKPOINT",
        model_id="sentence-transformers/all-mpnet-base-v2",
        revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
    ),
)


def _checkpoint(case: _DenseParityCase) -> Path:
    value = os.environ.get(case.checkpoint_environment)
    if value is None:
        pytest.skip(f"set {case.checkpoint_environment} for dense-route parity")
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"dense checkpoint not found: {path}")
    revisions = {
        path.name,
        *(tree.stem for tree in path.glob(".cache/huggingface/trees/*.json")),
    }
    if case.revision not in revisions:
        raise ValueError(
            f"dense parity requires {case.model_id} at revision {case.revision}; "
            f"found identities {sorted(revisions)}"
        )
    return path


def _upstream_python() -> str:
    return os.environ.get("REPRESENTAX_SENTENCE_TRANSFORMERS_PYTHON", sys.executable)


@pytest.mark.parametrize(
    "case",
    _DENSE_PARITY_CASES,
    ids=lambda case: case.name,
)
def test_dense_full_string_embedding_parity(case, tmp_path):
    checkpoint = _checkpoint(case)
    texts_path = tmp_path / "texts.json"
    upstream_path = tmp_path / "upstream.npy"
    metadata_path = tmp_path / "upstream.json"
    texts_path.write_text(json.dumps(_TEXTS) + "\n")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd())
    environment["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.sentence_transformers.transformers_oracle",
            "--checkpoint",
            str(checkpoint),
            "--texts",
            str(texts_path),
            "--output",
            str(upstream_path),
            "--metadata",
            str(metadata_path),
        ],
        check=True,
        cwd=Path.cwd(),
        env=environment,
        timeout=120,
    )

    model = load_sentence_transformer(checkpoint, local_files_only=True)
    with jax.default_matmul_precision("highest"):
        actual = model.embed(_TEXTS, batch_size=len(_TEXTS))
    expected = np.load(upstream_path)
    result = assert_numerically_equivalent(
        actual,
        expected,
        NumericalTolerance(
            absolute=2e-6,
            relative=2e-6,
            cosine=0.999999,
        ),
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["sentence_transformers"] == SENTENCE_TRANSFORMERS_ORACLE_VERSION
    print(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "checkpoint_revision": case.revision,
                "shape": actual.shape,
                "max_absolute": result.max_absolute,
                "relative_l2": result.relative_l2,
                "cosine": result.cosine,
                "oracle": metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )
