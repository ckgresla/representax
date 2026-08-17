"""A Git-trackable, CLI-overridable Hydra-Zen training job."""

from hydra_zen import builds, make_config, store, zen

from representax.config import (
    BatchConfig,
    CheckpointConfig,
    ComponentConfig,
    DataConfig,
    GradCacheConfig,
    JobConfig,
    LoggingConfig,
    MeshConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from representax.data import mix, source
from representax.tasks.retrieval import MNRConfig, RetrievalConfig
from representax.train import run_job

# Hydra-Zen's overload does not expose positional arguments forwarded to a target.
Recipe = builds(  # ty: ignore[no-matching-overload]
    mix,
    source(
        "examples/data/toy.jsonl",
        revision="example-v1",
        map="examples.recipes.toy_components.to_features",
    ),
    weights=(1.0,),
    seed=17,
    populate_full_signature=True,
)
Data = builds(
    DataConfig,
    recipe=Recipe,
    collate=builds(
        ComponentConfig,
        target="examples.recipes.toy_components.collate",
    ),
)
Model = builds(
    ModelConfig,
    target="representax.models.DenseEncoder",
    parameters={"input_dimension": 8, "output_dimension": 4},
)
Task = builds(RetrievalConfig)
Loss = builds(
    MNRConfig,
    scale=20.0,
    symmetric=True,
    negative_scope="global",
)
Optimization = builds(
    OptimizationConfig,
    optimizer=builds(
        ComponentConfig,
        target="optax.adamw",
        parameters={"learning_rate": 0.001},
    ),
)
Training = builds(
    TrainingConfig,
    global_batch_size=4,
    max_steps=1,
    seed=17,
    mesh=builds(
        MeshConfig,
        axis_shapes=(1, 1),
        axis_names=("fsdp", "tensor"),
    ),
    batch=builds(
        BatchConfig,
        micro_batch_size=4,
        gradient_accumulation_steps=1,
    ),
    grad_cache=builds(GradCacheConfig, micro_batch_size=2),
)
Job = builds(
    JobConfig,
    name="toy-retrieval",
    model=Model,
    task=Task,
    loss=Loss,
    optimization=Optimization,
    data=Data,
    training=Training,
    logging=builds(LoggingConfig, console_every=1),
    checkpointing=builds(CheckpointConfig, every=1, keep=1),
)
Config = make_config(job=Job, run_directory="runs/toy-retrieval")
store(Config, name="toy_retrieval")


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(run_job).hydra_main(config_name="toy_retrieval", version_base="1.3")
