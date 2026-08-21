"""Generic training state and compiled update construction."""

from representax.config import CheckpointConfig, LoggingConfig, MegaBatchMiningConfig
from representax.precision import PrecisionPolicy, resolve_precision_policy

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
from .evaluation import EvaluationResult, EvaluationRunner, evaluate
from .execution import Direct, ExecutionContext, LossExecution
from .grad_cache import GradCache
from .job import (
    JobRuntime,
    build_batches,
    build_collate,
    build_component,
    build_job_runtime,
    build_model,
    resolve_target,
    run_job,
)
from .logging import MetricRecord, Reporter, RunLogger
from .loop import TrainingRunResult, run_training
from .mega_batch import MegaBatchMining
from .optimizer import build_optimizer, build_schedule
from .sharding import (
    ShardingPlan,
    fsdp_partition_spec,
    parameter_specs_from_rules,
)
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
    "ShardingPlan",
    "ExecutionContext",
    "EvaluationResult",
    "EvaluationRunner",
    "GradCache",
    "IncompleteCheckpointError",
    "JobRuntime",
    "LossExecution",
    "MegaBatchMining",
    "MegaBatchMiningConfig",
    "PrecisionPolicy",
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
    "build_batches",
    "build_collate",
    "build_component",
    "build_job_runtime",
    "build_model",
    "build_optimizer",
    "build_schedule",
    "evaluate",
    "fsdp_partition_spec",
    "parameter_specs_from_rules",
    "init_train_state",
    "make_train_state",
    "run_training",
    "run_job",
    "resolve_target",
    "resolve_precision_policy",
    "scientific_fingerprint",
    "training_checkpointables",
    "tree_all_finite",
    "tree_global_norm",
    "validate_complete_checkpoint",
]
