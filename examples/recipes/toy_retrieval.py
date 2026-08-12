"""A Git-trackable Hydra-Zen recipe for the executable reference model."""

from hydra_zen import builds, instantiate, make_config

from representax.config import RunRecipe
from representax.data import mix, source
from representax.models import DenseEncoder
from representax.planning import ExecutionPlan, ScientificSpec
from representax.tasks.retrieval import MNRTask


def to_features(record):
    """Example named mapper; real projects keep task mapping beside recipes."""

    return record


Data = builds(
    mix,
    source(
        "file://examples/data/toy.jsonl",
        revision="example-v1",
        map=to_features,
    ),
    weights=(1.0,),
    seed=17,
    populate_full_signature=True,
)
Task = builds(MNRTask, scale=20.0, symmetric=True)
Science = builds(
    ScientificSpec,
    task="retrieval/mnr",
    global_batch_size=32,
    max_steps=100,
    seed=17,
)
Execution = builds(
    ExecutionPlan,
    device_count=1,
    data_axis_size=1,
    per_device_batch_size=8,
    gradient_accumulation_steps=4,
)
Config = make_config(
    recipe=builds(
        RunRecipe,
        name="toy-retrieval",
        model=builds(
            DenseEncoder,
            input_dimension=8,
            output_dimension=4,
            zen_partial=True,
        ),
        task=Task,
        data=Data,
        science=Science,
        execution=Execution,
    )
)


if __name__ == "__main__":
    print(instantiate(Config.recipe))
