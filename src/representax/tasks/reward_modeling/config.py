"""Serializable reward-modeling task and objective configurations."""

from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeFloat

from representax.tasks.config import LossConfig, TaskConfig

RewardObjective = Literal["binary_cross_entropy", "mse"]


class PairwiseRewardConfig(TaskConfig):
    kind: Literal["pairwise_reward"] = "pairwise_reward"


class ListwiseRewardConfig(TaskConfig):
    kind: Literal["listwise_reward"] = "listwise_reward"


class PointwiseRewardConfig(TaskConfig):
    kind: Literal["pointwise_reward"] = "pointwise_reward"


class ProcessRewardConfig(TaskConfig):
    kind: Literal["process_reward"] = "process_reward"


class BradleyTerryConfig(LossConfig):
    kind: Literal["bradley_terry"] = "bradley_terry"
    center_rewards_coefficient: NonNegativeFloat | None = None


class PlackettLuceConfig(LossConfig):
    kind: Literal["plackett_luce"] = "plackett_luce"


class PointwiseRewardLossConfig(LossConfig):
    kind: Literal["pointwise_reward"] = "pointwise_reward"
    objective: RewardObjective = "mse"


class ProcessRewardLossConfig(LossConfig):
    kind: Literal["process_reward"] = "process_reward"
    objective: RewardObjective = "binary_cross_entropy"


__all__ = [
    "BradleyTerryConfig",
    "ListwiseRewardConfig",
    "PairwiseRewardConfig",
    "PlackettLuceConfig",
    "PointwiseRewardConfig",
    "PointwiseRewardLossConfig",
    "ProcessRewardConfig",
    "ProcessRewardLossConfig",
    "RewardObjective",
]
