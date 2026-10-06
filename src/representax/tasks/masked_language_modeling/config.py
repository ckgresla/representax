"""Serializable encoder MLM task and objective."""

from typing import Literal

from representax.tasks.config import LossConfig, TaskConfig


class MaskedLanguageModelConfig(TaskConfig):
    kind: Literal["masked_language_modeling"] = "masked_language_modeling"


class MaskedLanguageModelLossConfig(LossConfig):
    kind: Literal["masked_language_model_cross_entropy"] = (
        "masked_language_model_cross_entropy"
    )
