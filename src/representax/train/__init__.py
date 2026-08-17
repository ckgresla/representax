"""Generic training state and compiled update construction."""

from representax.config import CheckpointConfig, LoggingConfig, MegaBatchMiningConfig

from .checkpoint import (
    CheckpointError,
    CheckpointManager,
    CheckpointRecord,
    CheckpointTicket,
    CheckpointWriteError,
    IncompleteCheckpointError,
    RestoredTrainingState,
    scientific_fingerprint,
    training_checkpointables,
    validate_complete_checkpoint,
)
from .config import build_loss_execution
from .execution import Direct, ExecutionContext, LossExecution
from .grad_cache import GradCache
from .logging import MetricRecord, Reporter, RunLogger
from .loop import TrainingRunResult, run_training
from .mega_batch import MegaBatchMining
from .optimizer import build_optimizer
from .sharding import DataParallel, build_data_parallel_train_step
from .state import StepMetrics, StepResult, TrainState
from .step import (
    TrainStep,
    build_train_step,
    init_train_state,
    make_train_state,
    tree_all_finite,
    tree_global_norm,
)

__all__ = [
    "CheckpointConfig",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointRecord",
    "CheckpointTicket",
    "CheckpointWriteError",
    "Direct",
    "DataParallel",
    "ExecutionContext",
    "GradCache",
    "IncompleteCheckpointError",
    "LossExecution",
    "MegaBatchMining",
    "MegaBatchMiningConfig",
    "MetricRecord",
    "Reporter",
    "RestoredTrainingState",
    "RunLogger",
    "StepMetrics",
    "StepResult",
    "TrainState",
    "TrainStep",
    "LoggingConfig",
    "TrainingRunResult",
    "build_train_step",
    "build_loss_execution",
    "build_data_parallel_train_step",
    "build_optimizer",
    "init_train_state",
    "make_train_state",
    "run_training",
    "scientific_fingerprint",
    "training_checkpointables",
    "tree_all_finite",
    "tree_global_norm",
    "validate_complete_checkpoint",
]
