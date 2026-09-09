"""Route independent held-out panels through Representax retrieval evaluation."""

import importlib
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp

from representax.core import Route
from representax.data import build_data_loader
from representax.data.media import decode_audio, decode_image, decode_video
from representax.evaluation import (
    InformationRetrievalEvaluator,
    retrieval_evaluation_batch,
)

DATA = importlib.import_module("experiments.14-text-to-any-modality.data")


class PanelBatch(eqx.Module):
    batch: object
    panel: str = eqx.field(static=True)

    @property
    def valid(self):
        return self.batch.valid


@dataclass(frozen=True)
class PanelEvaluator:
    evaluators: tuple
    name: str = "media-panels"

    def evaluate_batch(self, model, batch, *, key=None):
        evaluator = next(e for e in self.evaluators if e.name == batch.panel)
        return batch.panel, evaluator.evaluate_batch(model, batch.batch, key=key)

    def initialize(self):
        return {e.name: e.initialize() for e in self.evaluators}

    def accumulate(self, accumulator, output):
        name, value = output
        evaluator = next(e for e in self.evaluators if e.name == name)
        accumulator[name] = evaluator.accumulate(accumulator[name], value)
        return accumulator

    def finalize(self, accumulator):
        return {
            k: v
            for e in self.evaluators
            for k, v in e.finalize(accumulator[e.name]).items()
        }


def make_evaluator(config):
    return PanelEvaluator(
        tuple(
            InformationRetrievalEvaluator(**e.model_dump(exclude={"kind"}))
            for e in config.evaluators
        )
    )


class PanelCollator:
    def __init__(self, processor, name):
        self.processor, self.name = processor, name

    def __call__(self, rows):
        kind = rows[0]["kind"]
        if any(r["kind"] != kind for r in rows):
            raise ValueError("evaluation query/document boundary crossed")
        values = []
        for row in rows:
            modality = "text" if kind == "query" else row.get("modality", "text")
            if modality == "text":
                value = row["text"]
            elif modality == "image":
                value = {"image": decode_image(row["image"])}
            elif modality == "audio":
                value = {
                    "audio": decode_audio(DATA.audio_payload(row), max_seconds=10.0)
                }
            elif modality == "video":
                value = {"video": decode_video(row["video"], frames=8)}
            else:
                raise ValueError(f"unsupported modality: {modality}")
            values.append(value)
        inputs = self.processor(
            tuple(values), route=Route.QUERY if kind == "query" else Route.DOCUMENT
        )
        return PanelBatch(
            retrieval_evaluation_batch(
                inputs,
                jnp.asarray([r["identifier"] for r in rows], dtype=jnp.int32),
                kind=kind,
                valid=jnp.asarray([r.get("valid", True) for r in rows]),
            ),
            self.name,
        )


def evaluation_batches(config, processor):
    """Each panel/kind is a separate finite prefetched stream; never mixed corpora."""
    import grain

    for panel in config.data.distribution.sources:
        rows = DATA.read_rows(panel.uri)
        for kind in ("query", "document"):
            selected = [r for r in rows if r["kind"] == kind]
            remainder = len(selected) % config.batch_size
            if remainder:
                selected += [{**selected[-1], "valid": False}] * (
                    config.batch_size - remainder
                )
            loader = build_data_loader(
                grain.MapDataset.source(selected),
                batch_size=config.batch_size,
                batch_fn=PanelCollator(processor, panel.name),
                num_threads=config.data.num_threads,
                prefetch_buffer_size=config.data.prefetch_buffer_size,
                data_contract={"panel": panel.model_dump(), "kind": kind},
            )
            iterator = iter(loader)
            try:
                yield from iterator
            finally:
                iterator.close()
