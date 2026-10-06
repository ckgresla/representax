"""Real-checkpoint parity for scalar BERT sequence classification."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import score_logits
from representax.models.bert import load_bert_scorer


@pytest.mark.parity
def test_real_cross_encoder_logits_match_transformers() -> None:
    checkpoint = os.environ.get("REPRESENTAX_BERT_SCORER_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("set REPRESENTAX_BERT_SCORER_CHECKPOINT for physical parity")
    path = Path(checkpoint)
    model, processor = load_bert_scorer(
        path,
        local_files_only=True,
        compute_dtype=jnp.float32,
        sequence_length_buckets=(64,),
        rematerialization="none",
    )
    pairs = (
        ("what is a search engine?", "A search engine indexes web documents."),
        ("what is a search engine?", "Tennis is played with rackets."),
        ("how does photosynthesis work?", "Plants convert light into energy."),
    )
    native = np.asarray(score_logits(model, processor(pairs))).reshape(-1)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer is None:
        raise RuntimeError("checkpoint did not load a tokenizer")
    inputs = tokenizer(
        [query for query, _ in pairs],
        [document for _, document in pairs],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    reference = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True
    ).eval()
    with torch.no_grad():
        expected = reference(**inputs).logits.numpy().reshape(-1)
    jax.block_until_ready(native)
    np.testing.assert_allclose(native, expected, rtol=2e-4, atol=2e-5)
