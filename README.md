# Representax

Representax is a native JAX and Equinox system for efficient, task-general
representation learning. Retrieval is the first working task; classification,
reward modeling, distillation, and self-supervised objectives are planned on
the same core boundary.

The project is pre-alpha. The current slice provides:

- an Equinox-native encoder protocol with typed routes;
- direct multiple-negatives ranking, including symmetric and Matryoshka modes;
- a compiled Optax training step with finite-update protection and metrics;
- lazy, source-neutral data recipe contracts designed for Grain;
- separate scientific and execution specifications for future planning; and
- explicit unit, runtime, parity, distributed, and performance test lanes.

## Principles

1. Native Equinox models are the supported execution path.
2. Upstream implementations are optional development-time parity oracles.
3. Scientific intent is separate from topology-dependent execution choices.
4. Data recipes point at immutable artifacts and map them lazily into task
   examples; they do not require proprietary materialized datasets.
5. Text, image, audio, video, and fused inputs must be supported without
   changing the training abstractions.

## Install

The base package does not install PyTorch or an accelerator-specific JAX wheel:

```bash
python -m pip install -e .
```

Choose the appropriate JAX accelerator installation separately. Optional
capabilities are grouped deliberately:

```bash
python -m pip install -e ".[config,data,hf]"
python -m pip install -e ".[test,parity]"  # development oracle only
```

## Encoding

The compiled primitive has one route-aware operation:

```python
import jax
import jax.numpy as jnp
import representax as rx

model = rx.models.DenseEncoder(8, 4, key=jax.random.key(0))
batch = jnp.ones((2, 8))

embeddings = rx.encode(model, batch, route=rx.Route.QUERY)
encode_documents = rx.bind(model, route=rx.Route.DOCUMENT)
document_embeddings = encode_documents(batch)
```

Host-side tokenization, media decoding, and batching will be exposed through a
higher-level `embed` API as production model integrations land.

## Versioned data recipes

Recipes are ordinary Python values that can be composed in Hydra-Zen config
files and reviewed in Git:

```python
from representax import data

recipe = data.mix(
    data.source(
        "hf://organization/dataset",
        revision="immutable-revision",
        map="my_project.mappers.to_retrieval_example",
    ),
    data.source(
        "s3://bucket/corpus/*.parquet",
        revision="version-id",
        map="my_project.mappers.to_retrieval_example",
    ),
    weights=(0.7, 0.3),
    seed=17,
)
```

The recipe records artifact identity, mapping code identity, and sampling
policy. Grain performs lazy mapping, deterministic mixing, shuffling, and
checkpointable iteration. A single source is the one-element form of the same
sampling policy.

## Tests

```bash
pytest
pytest tests/runtime -m runtime
pytest tests/parity -m parity
pytest tests/distributed -m distributed
pytest tests/performance -m performance
```

The performance lane reports measurements but contains no universal speed
threshold: results depend on hardware and compiler versions. Reproducible
benchmarks must record compile time separately from steady-state execution.

## Roadmap

[`todo.org`](todo.org) is the canonical project roadmap and shared source of
truth. It tracks the production encoder port, parity gates, GradCache,
distributed training, task-native audio/video, reward modeling, JEPA, Profilax,
and the systems-then-model research program.
