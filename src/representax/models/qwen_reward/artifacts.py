"""Hugging Face loading and export for native Qwen3 reward models."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.integrations.huggingface import (
    load_hf_config,
    load_safetensor_subset,
)
from representax.models.components import AttentionImplementation
from representax.models.qwen_reranker import (
    QwenRerankerCheckpointAdapter,
    qwen_reranker_weight_names,
)
from representax.planning import RematerializationPolicy

from .config import QWEN3_REWARD_0_6B_MODEL_ID, QwenRewardConfig
from .model import QwenRewardModel


@dataclass(frozen=True, slots=True)
class QwenRewardCheckpointAdapter:
    """Load a Qwen3 backbone, initialize/load its scalar head, and export it."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def load(
        self,
        checkpoint: str | Path,
        *,
        head_key: jax.Array | None = None,
        parameter_dtype: jnp.dtype = jnp.bfloat16,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        model_id: str = QWEN3_REWARD_0_6B_MODEL_ID,
        revision: str = "unknown",
    ) -> QwenRewardModel:
        root = Path(checkpoint)
        if head_key is None:
            head_key = jax.random.key(0)
        hf_config = load_hf_config(root)
        config = QwenRewardConfig.from_hf_config(hf_config)
        adapter = QwenRerankerCheckpointAdapter(
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )
        names = qwen_reranker_weight_names(config.backbone)
        exported_reward = "Qwen3ForSequenceClassification" in hf_config.get(
            "architectures", []
        )
        if exported_reward:
            names = names.union({"score.weight"})
        state = load_safetensor_subset(root, names, dtype=parameter_dtype)
        backbone = adapter.from_state_dict(
            config.backbone,
            state,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=model_id,
            revision=revision,
        )
        model = QwenRewardModel.from_backbone(
            backbone,
            key=head_key,
            model_id=model_id,
            revision=revision,
        )
        if exported_reward:
            score = state["score.weight"]
            if score.shape != (1, config.hidden_size):
                raise ValueError(
                    f"score.weight has shape {score.shape}; "
                    f"expected {(1, config.hidden_size)}"
                )
            model = eqx.tree_at(lambda value: value.score_head.weight, model, score)
        return model

    def state_dict(self, model: QwenRewardModel) -> dict[str, jax.Array]:
        state = QwenRerankerCheckpointAdapter(
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        ).state_dict(model.backbone)
        state["score.weight"] = model.score_head.weight
        return state

    def save(
        self,
        model: QwenRewardModel,
        directory: str | Path,
        *,
        source_checkpoint: str | Path,
    ) -> Path:
        from safetensors.numpy import save_file

        source = Path(source_checkpoint)
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.name == "model.safetensors" or path.name.startswith("model-"):
                continue
            if path.name in {"model.safetensors.index.json", "config.json"}:
                continue
            destination = target / path.name
            if path.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(path, destination)
            elif path.is_file():
                shutil.copy2(path, destination)
        config = load_hf_config(source)
        config.update(
            {
                "architectures": ["Qwen3ForSequenceClassification"],
                "num_labels": 1,
                "id2label": {"0": "LABEL_0"},
                "label2id": {"LABEL_0": 0},
                "pad_token_id": model.config.backbone.pad_token_id,
                "problem_type": "regression",
            }
        )
        (target / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        save_file(
            {
                name: np.array(value, copy=True)
                for name, value in self.state_dict(model).items()
            },
            target / "model.safetensors",
        )
        return target


__all__ = ["QwenRewardCheckpointAdapter"]
