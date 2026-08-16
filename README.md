<h1 align="center">Representax</h1>

Representax is a native JAX and Equinox system for efficient, task-general
representation learning. Retrieval is the first working task; classification,
reward modeling, distillation, and self-supervised objectives are planned on
the same core boundary.

The project is alpha. The current slice provides:

- an Equinox-native encoder protocol with typed routes;
- native BERT, MPNet, and ModernVBERT text-image encoders with direct Hugging
  Face safetensor loading and pinned numerical acceptance;
- a Torch-free dense Sentence Transformers module loader and fixed-shape host
  embedding API;
- direct and cached multiple-negatives ranking, including symmetric and
  Matryoshka modes;
- task-native labeled-pair cosine regression, contrastive and online mining,
  CoSENT, and AnglE objectives;
- explicit triplet learning plus all, hard, hard soft-margin, and semi-hard
  within-batch mining;
- an end-to-end Grain-to-compiled-step trainer with asynchronous reporting;
- lazy Grain recipes with built-in Hugging Face and local artifact resolvers;
- validated domain configs with annotated scientific and execution parameters; and
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

The base package installs the ordinary CPU JAX runtime, Grain data pipeline,
and safetensor checkpoint support. It never installs PyTorch:

```bash
python -m pip install representax
python -m pip install "representax[cuda13]"  # NVIDIA GPU, recommended
python -m pip install "representax[cuda12]"  # older NVIDIA drivers
```

For an editable source checkout, put `-e .` in place of `representax`.
Optional capabilities are grouped deliberately. Upstream parity oracles are
repository-only dependency groups rather than published package extras:

```bash
python -m pip install -e ".[config,hf]"
python -m pip install -e ".[config,hf,test,performance]" --group static
python -m pip install -e ".[test]" --group parity
python -m pip install -e ".[test]" --group parity-modernvbert
python -m pip install -e ".[test,performance]" --group parity-modernvbert
```

The v0 Hugging Face reference is pinned to Transformers 5.3.0. Its complete
architecture catalog is distinct from native support: BERT, MPNet, and
ModernVBERT are the currently verified native families. Repository-only
dense-route acceptance uses Sentence Transformers 5.6.1, the latest stable
multimodal release line.

See the [compatibility matrix](https://github.com/ckgresla/representax/blob/main/docs/compatibility.md) for the locally accepted
Python/JAX combinations and the distinction between CPU CI and accelerator
acceptance. Maintainers should follow the
[release procedure](https://github.com/ckgresla/representax/blob/main/docs/releasing.md)
for artifact inspection and trusted publication.

## Gotchas

### Pip-managed CUDA can be shadowed by `LD_LIBRARY_PATH`

The `cuda12` and `cuda13` extras use JAX's pip-managed NVIDIA libraries. A
shell-level `LD_LIBRARY_PATH` that points at another CUDA installation can take
precedence.

Did you see this?

```text
Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
RuntimeError: Unable to load cuSPARSE. Is it installed?
```

Or did `jax.devices()` return only `CpuDevice` despite a working NVIDIA driver?
Then compare the ordinary process with one that ignores `LD_LIBRARY_PATH`:

```bash
python -c 'import jax; print(jax.devices())'
env -u LD_LIBRARY_PATH python -c 'import jax; print(jax.devices())'
```

If the second command restores the GPU, launch Representax the same way or
unset the variable in that environment:

```bash
env -u LD_LIBRARY_PATH python train.py
# Or, for the current shell:
unset LD_LIBRARY_PATH
```

Do not remove a machine-wide CUDA configuration blindly: JAX's `*-local`
installations intentionally use a system toolkit. This advice applies to the
pip-managed extras documented above and follows JAX's
[NVIDIA installation guidance](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier).

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

For standard dense Sentence Transformers artifacts, the host API resolves the
static serialized module chain without importing PyTorch, tokenizes on the
host, and executes fixed-shape native batches:

```python
from representax.integrations import load_sentence_transformer

model = load_sentence_transformer(
    "sentence-transformers/all-mpnet-base-v2",
    revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
)
embeddings = model.embed(
    ["A small bee.", "A large flower."],
    batch_size=2,
)
similarities = model.similarity(embeddings, embeddings)
```

Install `representax[hf]` for Hub transport and tokenization. The native model
runtime itself requires neither Torch nor Sentence Transformers. The pinned
`all-MiniLM-L6-v2` and `all-mpnet-base-v2` acceptance cases verify the complete
text-to-normalized-embedding path against Sentence Transformers 5.6.1.

## Pairwise representation learning

Labeled pair objectives use one modality-neutral batch and keep semantic routes
on the task rather than inside the loss:

```python
import jax.numpy as jnp
from representax.tasks import build_task
from representax.tasks.pairwise import CoSENTConfig, PairwiseConfig, pairwise_batch

task = build_task(PairwiseConfig(), CoSENTConfig(scale=20.0))
batch = pairwise_batch(
    left=left_model_inputs,
    right=right_model_inputs,
    labels=jnp.asarray([1.0, 0.7, 0.0]),
)
```

The same task boundary supports text, image, audio, video, or fused model-native
payloads. See the
[Sentence Transformers capability ledger](docs/sentence-transformers-capabilities.md)
for exact native and remaining coverage.

## Triplet representation learning

Supplied triplets and class-labeled mining batches are distinct data contracts.
Both keep semantic routes on the task configuration and share native distance
primitives:

```python
import jax.numpy as jnp
from representax.tasks import build_task
from representax.tasks.triplet import (
    BatchTripletLossConfig,
    LabeledExamplesConfig,
    labeled_examples_batch,
)

task = build_task(
    LabeledExamplesConfig(),
    BatchTripletLossConfig(mining="semi_hard", margin=0.5),
)
batch = labeled_examples_batch(
    examples=model_inputs,
    labels=jnp.asarray([0, 0, 1, 1]),
)
```

Explicit triplets support cosine, Euclidean, and Manhattan distance. In-batch
mining supports cosine, Euclidean, and squared Euclidean distance, explicit
padding validity, and fixed-shape JIT compilation.

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
ordinary runtime includes `safetensors` but not PyTorch. A repository-only
pinned Transformers environment verifies vision features, fused
representations, and pixel gradients. Host-side Idefics3-compatible processing
remains the next API slice.

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
policy. A training iterator additionally fingerprints the resolved mapper and
resolver implementations, batch mapper, batching contract, and Grain version.
Grain performs lazy mapping, deterministic mixing, shuffling, and checkpointable
iteration. A single source is the one-element form of the same sampling policy.
Built-in resolvers support revision-pinned Hugging Face splits and local JSONL,
Parquet, Arrow, or dataset directories. See
[the data contract](https://github.com/ckgresla/representax/blob/main/docs/data.md)
for cache and extension behavior.

## Training

Application code imports concrete operations from their owning modules:

```python
from representax.config import (
    BatchConfig,
    CheckpointConfig,
    ComponentConfig,
    JobConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from representax.data import build_grain_iterator
from representax.tasks import build_task
from representax.tasks.retrieval import MNRConfig, RetrievalConfig
from representax.train import (
    build_optimizer,
    build_train_step,
    init_train_state,
    run_training,
)

job = JobConfig(
    name="example",
    model=ModelConfig(target="my_project.Model"),
    task=RetrievalConfig(),
    loss=MNRConfig(scale=20.0, symmetric=True, negative_scope="global"),
    optimization=OptimizationConfig(
        optimizer=ComponentConfig(
            target="optax.adamw",
            parameters={"learning_rate": 1e-3},
        ),
    ),
    data=recipe,
    training=TrainingConfig(
        global_batch_size=32,
        max_steps=10_000,
        seed=17,
        batch=BatchConfig(micro_batch_size=32),
    ),
    logging=LoggingConfig(console_every=100),
    checkpointing=CheckpointConfig(every=1_000, keep=3),
)
optimizer = build_optimizer(job.optimization)
task = build_task(job.task, job.loss)
state = init_train_state(model, optimizer)
batches = build_grain_iterator(job.data, batch_size=32, batch_fn=collate)
result = run_training(
    state=state,
    step=build_train_step(task, optimizer),
    batches=batches,
    job=job,
    run_directory="runs/example",
)
```

Array-facing APIs use `jaxtyping` to state dtype and symbolic shape contracts
directly on model forwards, tasks, losses, and compiled-step keys. Representax
does not install a runtime type-checking hook; explicit domain validation remains
responsible for semantic requirements that shapes and dtypes cannot express.

The loop records W&B-ready metric names such as `train/loss`, `valid/loss`, and
`perf/...` in `metrics.jsonl`, lifecycle events in `events.jsonl`, and final
status in `run.json`. A bounded reporter worker performs the device-to-host
metric transfer and fans the same ordered rows out to optional consumers without
placing a synchronization barrier in every training iteration. Checkpoints are
written by Orbax with at most one asynchronous save in flight. Recreate the same
recipe, model/state template, task/optimizer program, and batch source and pass
`resume=True` to continue from the latest complete checkpoint. See
[the training contract](https://github.com/ckgresla/representax/blob/main/docs/training.md).

## Tests

```bash
python scripts/check.py
pytest
pytest -m runtime
pytest -m parity
pytest -m distributed
pytest -m performance
```

The first command is the fast static gate: Ruff formatting and linting followed
by ty type checking. It does not import JAX or compile model programs.

Tests live outside the package and mirror its model, task, data, and runtime
structure. Pytest markers select orthogonal runtime, parity, distributed, and
performance lanes. The default command runs fast, dependency-light tests.

Performance acceptance is evaluated against a matched upstream implementation
on pinned hardware. Compile time, steady-state work, and peak device memory are
measured separately; see
[the test contract](https://github.com/ckgresla/representax/blob/main/docs/testing.md).

## Roadmap

[`todo.org`](https://github.com/ckgresla/representax/blob/main/todo.org) is the
canonical project roadmap and shared source of
truth. It tracks the production encoder port, parity gates, GradCache,
distributed training, checkpoint/resume, task-native audio/video, reward
modeling, JEPA, Profilax, and the systems-then-model research program.
