"""A Git-trackable, CLI-overridable Hydra-Zen job configuration."""

from hydra_zen import builds, make_config, store, zen

from representax.config import (
    BatchConfig,
    CheckpointConfig,
    ComponentConfig,
    JobConfig,
    LoggingConfig,
    MeshConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from representax.data import mix, source
from representax.tasks.retrieval import MNRConfig, RetrievalConfig


def to_features(record):
    """Example named mapper; real projects keep task mapping beside recipes."""

    return record


Data = builds(
    mix,
    source(
        "file://examples/data/toy.jsonl",
        revision="example-v1",
        map="examples.recipes.toy_retrieval.to_features",
    ),
    weights=(1.0,),
    seed=17,
    populate_full_signature=True,
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
    global_batch_size=32,
    max_steps=100,
    seed=17,
    mesh=builds(
        MeshConfig,
        axis_shapes=(1, 1),
        axis_names=("fsdp", "tensor"),
    ),
    batch=builds(
        BatchConfig,
        micro_batch_size=8,
        gradient_accumulation_steps=4,
    ),
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
    logging=builds(LoggingConfig, console_every=10),
    checkpointing=builds(CheckpointConfig, every=25, keep=3),
)
Config = make_config(job=Job)
store(Config, name="toy_retrieval")


def show(job: JobConfig) -> None:
    """Validate the composed user configuration before building JAX objects."""

    print(job.model_dump_json(indent=2))


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(show).hydra_main(config_name="toy_retrieval", version_base="1.3")
