# Representax

Representax is a native JAX and Equinox system for efficient, task-general
representation learning. Retrieval is the first working task; classification,
reward modeling, distillation, and self-supervised objectives are planned on
the same core boundary.

The project is pre-alpha. The current slice provides:

- an Equinox-native encoder protocol with typed routes;
- a native ModernVBERT text-image encoder with bidirectional Hugging Face
  weight maps for every tensor used by its forward pass;
- direct multiple-negatives ranking, including symmetric and Matryoshka modes;
- an end-to-end Grain-to-compiled-step single-device trainer with local logs;
- lazy Grain recipes with built-in Hugging Face and local artifact resolvers;
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
python -m pip install -e ".[test,parity-modernvbert]"  # pinned model oracle
python -m pip install -e ".[test,parity-modernvbert,performance]"  # GPU gates
```

## Encoding

The compiled primitive has one route-aware operation:

```python
import jax
import jax.numpy as jnp
import representax as rpx

model = rpx.models.DenseEncoder(8, 4, key=jax.random.key(0))
batch = jnp.ones((2, 8))

embeddings = rpx.encode(model, batch, route=rpx.Route.QUERY)
encode_documents = rpx.bind(model, route=rpx.Route.DOCUMENT)
document_embeddings = encode_documents(batch)
```

Host-side tokenization, media decoding, and batching will be exposed through a
higher-level `embed` API as production model integrations land.

## ModernVBERT

The first production-family integration loads pinned Hugging Face safetensors
directly into native Equinox text and SigLIP vision towers:

```python
import jax.numpy as jnp
import representax as rpx

adapter = rpx.models.ModernVBERTCheckpointAdapter()
model = adapter.load("/path/to/modernvbert-checkpoint")
image_tokens = jnp.full((1, 64), 50407, dtype=jnp.int32)
input_ids = jnp.concatenate(
    (jnp.asarray([[1]]), image_tokens, jnp.asarray([[2]])), axis=1
)
batch = rpx.models.ModernVBERTBatch(
    input_ids=input_ids,
    attention_mask=jnp.ones_like(input_ids),
    pixel_values=jnp.ones((1, 1, 3, 512, 512), dtype=jnp.float32),
)
embeddings = rpx.encode(model, batch, route=rpx.Route.QUERY)
```

The real checkpoint uses 64 image tokens per processed 512x512 image. The
ordinary runtime needs `safetensors` but not PyTorch. An optional pinned
Transformers environment verifies vision features, fused representations, and
pixel gradients. Host-side Idefics3-compatible processing remains the next API
slice.

## Versioned data recipes

Recipes are ordinary Python values that can be composed in Hydra-Zen config
files and reviewed in Git:

```python
from representax import data

recipe = data.mix(
    data.source(
        "hf://organization/dataset",
        revision="immutable-revision",
        split="train",
        map="my_project.mappers.to_retrieval_example",
    ),
    data.source(
        "file:///data/corpus/train.parquet",
        map="my_project.mappers.to_retrieval_example",
    ),
    weights=(0.7, 0.3),
    seed=17,
)
dataset = data.build_grain_dataset(recipe)
```

The recipe records artifact identity, mapping code identity, and sampling
policy. Grain performs lazy mapping, deterministic mixing, shuffling, and
checkpointable iteration. A single source is the one-element form of the same
sampling policy. Built-in resolvers support revision-pinned Hugging Face splits
and local JSONL, Parquet, Arrow, or dataset directories. See
[the data contract](docs/data.md) for cache and extension behavior.

## Training

Application code imports concrete operations from their owning modules:

```python
import optax

from representax.data import build_grain_iterator
from representax.planning import ScientificSpec
from representax.train import (
    CheckpointConfig,
    build_train_step,
    make_train_state,
    run_training,
)

optimizer = optax.adamw(learning_rate=1e-3)
state = make_train_state(model, optimizer)
batches = build_grain_iterator(recipe, batch_size=32, batch_fn=collate)
result = run_training(
    state=state,
    step=build_train_step(task, optimizer),
    batches=batches,
    science=ScientificSpec(
        task="retrieval/mnr",
        global_batch_size=32,
        max_steps=10_000,
        seed=17,
    ),
    run_directory="runs/example",
    checkpoint=CheckpointConfig(every=1_000, keep=3),
)
```

The loop records every optimizer attempt in `metrics.jsonl`, lifecycle events
in `events.jsonl`, and final status in `run.json`. Checkpoints are written by
Orbax with at most one asynchronous save in flight. Recreate the same recipe,
model/state template, task/optimizer program, and batch source and pass
`resume=True` to continue from the latest complete checkpoint. See
[the training contract](docs/training.md).

## Tests

```bash
pytest
pytest -m runtime
pytest -m parity
pytest -m distributed
pytest -m performance
```

Tests live outside the package and mirror its model, task, data, and runtime
structure. Pytest markers select orthogonal runtime, parity, distributed, and
performance lanes. The default command runs fast, dependency-light tests.

Performance acceptance is evaluated against a matched upstream implementation
on pinned hardware. Compile time, steady-state work, and peak device memory are
measured separately; see [the test contract](docs/testing.md).

## Roadmap

[`todo.org`](todo.org) is the canonical project roadmap and shared source of
truth. It tracks the production encoder port, parity gates, GradCache,
distributed training, checkpoint/resume, task-native audio/video, reward
modeling, JEPA, Profilax, and the systems-then-model research program.
