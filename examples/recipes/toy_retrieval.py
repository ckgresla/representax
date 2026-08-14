"""A Git-trackable, CLI-overridable Hydra-Zen run configuration."""

from hydra_zen import builds, make_config, store, zen

from representax.config import (
    CheckpointConfig,
    ComponentConfig,
    ExecutionConfig,
    RunConfig,
    RuntimeConfig,
    ScientificConfig,
    TrainingConfig,
)
from representax.data import mix, source


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
Scientific = builds(
    ScientificConfig,
    task="retrieval/mnr",
    global_batch_size=32,
    max_steps=100,
    seed=17,
)
Execution = builds(
    ExecutionConfig,
    device_count=1,
    data_axis_size=1,
    per_device_batch_size=8,
    gradient_accumulation_steps=4,
)
Training = builds(
    TrainingConfig,
    scientific=Scientific,
    execution=Execution,
    runtime=builds(RuntimeConfig, console_every=10),
    checkpoint=builds(CheckpointConfig, every=25, keep=3),
)
Run = builds(
    RunConfig,
    name="toy-retrieval",
    model=builds(
        ComponentConfig,
        target="representax.models.DenseEncoder",
        parameters={"input_dimension": 8, "output_dimension": 4},
    ),
    optimizer=builds(
        ComponentConfig,
        target="optax.adamw",
        parameters={"learning_rate": 0.001},
    ),
    task=builds(
        ComponentConfig,
        target="representax.tasks.retrieval.MNRTask",
        parameters={"scale": 20.0, "symmetric": True},
    ),
    data=Data,
    training=Training,
)
Config = make_config(run=Run)
store(Config, name="toy_retrieval")


def show(run: RunConfig) -> None:
    """Validate the composed user configuration before building JAX objects."""

    print(run.model_dump_json(indent=2))


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(show).hydra_main(config_name="toy_retrieval", version_base="1.3")
