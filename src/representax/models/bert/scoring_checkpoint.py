"""Hugging Face import and export for BERT sequence classifiers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from representax.integrations.huggingface import (
    load_hf_config,
    load_safetensor_subset,
)
from representax.models.components import AttentionImplementation, Linear
from representax.planning import RematerializationPolicy

from .checkpoint import BertCheckpointAdapter, bert_weight_names
from .config import BertConfig
from .scoring import BertScorer


def bert_scorer_weight_names(config: BertConfig) -> frozenset[str]:
    """Return the complete scalar ``BertForSequenceClassification`` state."""

    return frozenset(
        {
            *(f"bert.{name}" for name in bert_weight_names(config)),
            "classifier.weight",
            "classifier.bias",
        }
    )


def _classifier_dropout(config: Mapping[str, Any]) -> float:
    configured = config.get("classifier_dropout")
    return float(
        config.get("hidden_dropout_prob", 0.1) if configured is None else configured
    )


@dataclass(frozen=True, slots=True)
class BertScorerCheckpointAdapter:
    """Load and export scalar BERT sequence-classification checkpoints."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: BertConfig,
        state_dict: Mapping[str, Any],
        *,
        classifier_dropout_probability: float,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = "representax/bert-scorer",
        revision: str = "local",
    ) -> BertScorer:
        base_names = bert_weight_names(config)
        base_state = {name: state_dict[f"bert.{name}"] for name in base_names}
        backbone = BertCheckpointAdapter(
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        ).from_state_dict(
            config,
            base_state,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=model_id,
            revision=revision,
        )
        weight = jnp.asarray(state_dict["classifier.weight"], dtype=parameter_dtype)
        bias = jnp.asarray(state_dict["classifier.bias"], dtype=parameter_dtype)
        expected_weight = (1, config.hidden_size)
        if weight.shape != expected_weight or bias.shape != (1,):
            raise ValueError(
                "BERT scorer requires one scalar classification output; "
                f"received weight={weight.shape}, bias={bias.shape}"
            )
        if not 0.0 <= classifier_dropout_probability < 1.0:
            raise ValueError("classifier dropout probability must be in [0, 1)")
        return BertScorer(
            backbone=backbone,
            classifier=Linear(weight=weight, bias=bias),
            classifier_dropout_probability=classifier_dropout_probability,
        )

    def load(
        self,
        checkpoint: str | Path,
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> BertScorer:
        values = load_hf_config(checkpoint)
        architectures = tuple(values.get("architectures", ()))
        if architectures and "BertForSequenceClassification" not in architectures:
            raise ValueError(
                "BERT scorer requires a BertForSequenceClassification artifact"
            )
        config = BertConfig.from_hf_config(values)
        state = load_safetensor_subset(
            checkpoint,
            bert_scorer_weight_names(config),
            dtype=parameter_dtype,
        )
        return self.from_state_dict(
            config,
            state,
            classifier_dropout_probability=_classifier_dropout(values),
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=model_id
            or str(values.get("_name_or_path", Path(checkpoint).name)),
            revision=revision or str(values.get("_commit_hash", "local")),
        )

    def state_dict(self, model: BertScorer) -> dict[str, jax.Array]:
        base = BertCheckpointAdapter().state_dict(model.backbone)
        classifier = model.classifier.output_major()
        if classifier.bias is None:
            raise ValueError("BERT scorer classification head requires a bias")
        return {
            **{f"bert.{name}": value for name, value in base.items()},
            "classifier.weight": classifier.weight,
            "classifier.bias": classifier.bias,
        }

    def save(self, model: BertScorer, directory: str | Path) -> Path:
        """Export a Transformers-reloadable scalar classification artifact."""

        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        config = model.backbone.tower.config.to_hf_config()
        config.update(
            {
                "architectures": ["BertForSequenceClassification"],
                "classifier_dropout": model.classifier_dropout_probability,
                "id2label": {"0": "LABEL_0"},
                "label2id": {"LABEL_0": 0},
                "num_labels": 1,
                "sbert_ce_default_activation_function": (
                    "torch.nn.modules.linear.Identity"
                ),
            }
        )
        (target / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        save_file(
            {
                name: np.asarray(jax.device_get(value))
                for name, value in self.state_dict(model).items()
            },
            target / "model.safetensors",
        )
        return target


__all__ = ["BertScorerCheckpointAdapter", "bert_scorer_weight_names"]
