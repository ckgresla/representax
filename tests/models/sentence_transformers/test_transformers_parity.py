"""Pinned full-string parity for the native dense sentence route."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
_MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _checkpoint() -> Path:
    value = os.environ.get("REPRESENTAX_MINILM_CHECKPOINT")
    if value is None:
        pytest.skip("set REPRESENTAX_MINILM_CHECKPOINT for dense-route parity")
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"MiniLM checkpoint not found: {path}")
    revisions = {
        path.name,
        *(tree.stem for tree in path.glob(".cache/huggingface/trees/*.json")),
    }
    if _MINILM_REVISION not in revisions:
        raise ValueError(
            "dense parity requires sentence-transformers/all-MiniLM-L6-v2 at "
            f"revision {_MINILM_REVISION}; found identities {sorted(revisions)}"
        )
    return path


def _upstream_python() -> str:
    return os.environ.get("REPRESENTAX_SENTENCE_TRANSFORMERS_PYTHON", sys.executable)


def test_all_minilm_l6_v2_full_string_embedding_parity(tmp_path):
    checkpoint = _checkpoint()
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
                "checkpoint_revision": _MINILM_REVISION,
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
