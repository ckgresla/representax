"""NanoBEIR remains a pinned data recipe over generic IR evaluation."""

from examples.evaluation.nanobeir import (
    NANOBEIR_DATASET_ID,
    NANOBEIR_REVISION,
    nanobeir_source,
)


def test_nanobeir_sources_are_revision_pinned_without_intermediate_data() -> None:
    configuration = nanobeir_source("queries", dataset="NanoMSMARCO")

    assert len(configuration.sources) == 1
    source = configuration.sources[0]
    assert source.uri == f"hf://{NANOBEIR_DATASET_ID}"
    assert source.revision == NANOBEIR_REVISION
    assert source.subset == "queries"
    assert source.split == "NanoMSMARCO"
    assert configuration.shuffle is False
