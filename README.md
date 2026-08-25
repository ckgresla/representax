<h1 align="center">Representax</h1>

Representax is a native JAX and Equinox system for efficient, task-general
representation learning. Retrieval is the first mature task; classification,
distillation, regularization, and denoising use the same core boundary, with
reward modeling and self-supervised objectives next.

The project is alpha. The current slice provides:

- an Equinox-native encoder protocol with typed routes;
- native BERT, MPNet, ModernVBERT text-image, pinned Jina v5 Omni Small text,
  Qwen3-VL text-image-video embedding/reranking, and Qwen2.5-Omni
  text-image-audio-video embedding models with direct Hugging Face safetensor
  loading and numerical acceptance;
- a Torch-free dense Sentence Transformers module loader and fixed-shape host
  embedding API;
- direct and cached multiple-negatives ranking, including symmetric and
  Matryoshka modes;
- task-native labeled-pair cosine regression, contrastive and online mining,
  CoSENT, and AnglE objectives;
- explicit triplet learning plus all, hard, hard soft-margin, and semi-hard
  within-batch mining;
- native scientific and execution contracts for all 29 Sentence Transformers
  5.6.1 dense loss classes, including Matryoshka/adaptive-layer modifiers,
  direct/cached GIST, contrastive tension, classification, orthogonal
  regularization, denoising, and bounded mega-batch mining;
- a Grain-to-compiled-step training loop with asynchronous reporting, exact
  checkpoint resume, configured validation, best-model selection, and atomic
  inference export;
- a typed evaluator protocol with deterministic, corpus-level embedding
  similarity metrics matching Sentence Transformers 5.6.1;
- lazy Grain distributions with built-in Hugging Face and local source resolvers;
- validated domain configs with annotated scientific and execution parameters; and
- explicit FP32 and BF16-mixed policies with FP32 master/Optax state;
- experimental native FP8 linear compute and packed INT4 LoRA training; and
- explicit unit, runtime, parity, distributed, and performance test lanes.

## Principles

1. Native Equinox models are the supported execution path.
2. Upstream implementations are optional development-time parity oracles.
3. Scientific intent is separate from topology-dependent execution choices.
4. Data distributions point at immutable sources and map them lazily into task
   examples; they do not require proprietary materialized datasets.
5. Text, image, audio, video, and their compositions must be supported without
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
python -m pip install -e ".[test]" --group parity-llava-next-legacy
python -m pip install -e ".[test,performance]" --group parity-modernvbert
```

The v0 Hugging Face reference is pinned to Transformers 5.3.0. Its complete
architecture catalog is distinct from native support: BERT, MPNet, and
ModernVBERT have the broadest current acceptance, Jina v5 Omni Small has a
pinned native text path, Qwen3-VL 2B has native text/image/video embedding and
reranking paths, Qwen2.5-Omni has a native text/image/audio/video embedding
path, and the Qwen2/Qwen2.5-VL, CLIP/BGE-VL, and LLaVA-NeXT families have native
text-image paths. Repository-only dense-route acceptance uses Sentence
Transformers 5.6.1, with a checkpoint-authored 5.4/5.5 oracle retained for
E5-V's legacy LLaVA-NeXT layout.

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

### CUDA 12 and CUDA 13 packages can overwrite one another

Keep the Torch parity runtime separate from the ordinary Representax runtime.
Both CUDA generations install files into the same `site-packages/nvidia`
namespace, so installing Torch's CUDA 13 packages into a JAX CUDA 12 environment
can leave a mixed cuDNN installation.

Did you see this?

```text
RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed:
CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

First inspect the environment rather than changing model code:

```bash
python -m pip freeze | rg '^nvidia-(cublas|cudnn)-cu(12|13)'
```

If both generations are present, recreate the dedicated training environment
with exactly one of `representax[cuda12]` or `representax[cuda13]`. Install the
repository-only Sentence Transformers/Torch parity group in a separate
environment. Reinstalling only one cuDNN wheel may appear to repair the current
process, but another package operation can overwrite the shared files again.

### A near-capacity sharded job can fragment the default GPU pool

First verify from the resolved `ShardingPlan` that the persistent model and
optimizer shards really fit on every device. Did you then see this from a
compiled step?

```text
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8.23GiB.
[executable_name='jit_mapped_train_step_body']
```

JAXlib's CUDA-async allocator can make a physically feasible, near-capacity
layout executable when the default BFC pool cannot provide one contiguous live
workspace. Select it before importing JAX:

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async python train.py
```

`TF_GPU_ALLOCATOR=cuda_malloc_async` is a TensorFlow control and does not select
the allocator in JAX. Treat CUDA async as a measured execution choice rather
than a blanket default: an impossible at-rest layout will still OOM, and memory
and throughput should be re-profiled for the actual job.

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

Jina v5 Omni Small has a separate, pinned text-only factory for training and
inference without placing its unused vision or audio towers:

```python
from representax.integrations import load_jina_v5_small_text_encoder

encoder = load_jina_v5_small_text_encoder(
    revision="12949877f0092093f366c6450340011320152a05",
)
```

Its optional checkpoint is not bundled and is separately licensed under CC
BY-NC 4.0. See [`THIRD_PARTY.md`](THIRD_PARTY.md) before using those weights.

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

The same task boundary supports text, image, audio, video, or composed
model-native payloads. See the
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

## Representation distillation

Teacher artifacts remain ordinary fixed-shape batch data rather than live
models hidden in a loss. Representax supports embedding matching, score-margin
regression, and distribution distillation through registered tasks:

```python
from representax.tasks import build_task
from representax.tasks.distillation import (
    DistributionDistillationConfig,
    DistributionKLLossConfig,
    distribution_distillation_batch,
)

task = build_task(
    DistributionDistillationConfig(),
    DistributionKLLossConfig(temperature=2.0),
)
batch = distribution_distillation_batch(
    query=query_model_inputs,
    candidates=(positive_model_inputs, negative_model_inputs),
    teacher_scores=teacher_scores,
)
```

Embedding targets may be broadcast across input columns or supplied per
column, with MSE, L2, or cosine matching. Learned student-to-teacher projections
belong in the model composition so optimizer-visible state is never concealed
inside a task.

## Loss composition and bounded execution

Loss modifiers are scientific job configuration, while GradCache and
mega-batch mining are execution configuration. For example, the same MNR
objective can be trained at several prefix dimensions with direct execution or
bounded encoder replay:

```python
from representax.tasks import build_task
from representax.tasks.modifiers import MatryoshkaModifierConfig
from representax.tasks.retrieval import MNRConfig, RetrievalConfig

task = build_task(
    RetrievalConfig(),
    MNRConfig(scale=20.0),
    modifiers=(
        MatryoshkaModifierConfig(
            dimensions=(768, 512, 256, 128, 64),
        ),
    ),
)
```

The pinned capability ledger records the exact class mapping and evidence.
Forty paired checks cover the inventory plus 39 same-tensor value-and-gradient
cases. A separate GPU lane measures compiled objective forward and
backward performance without turning uncontrolled timing noise into a
correctness failure.

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

## Versioned data distributions

Distributions are ordinary Python values that can be composed in Hydra-Zen
config files and reviewed in Git:

```python
from representax import data

distribution = data.mix(
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
dataset = data.build_dataset(distribution)
```

The distribution records source identity, mapping code identity, and sampling
policy. A data loader additionally fingerprints the resolved mapper and
resolver implementations, batch mapper, batching contract, and Grain version.
Grain performs lazy mapping, deterministic mixing, shuffling, and checkpointable
iteration. A single source is the one-element form of the same sampling policy.
Built-in resolvers support revision-pinned Hugging Face splits and local JSONL,
Parquet, Arrow, or dataset directories. Existing Grain datasets can enter the
lower-level loader directly. Task-specific samples compose atomic `Artifact`
leaves; model-specific preprocessing travels beside the Equinox model in a
`ModelBundle`, so a configured job constructs the model and processor once. See
[the data contract](https://github.com/ckgresla/representax/blob/main/docs/data.md)
for cache and extension behavior.

## Training

Application code imports concrete operations from their owning modules:

```python
from representax.config import (
    BatchConfig,
    CheckpointConfig,
    ComponentConfig,
    DataConfig,
    EvaluationConfig,
    ExportConfig,
    JobConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    PrecisionConfig,
    TrainingConfig,
)
from representax.data import mix, source
from representax.tasks.retrieval import MNRConfig, RetrievalConfig
from representax.train import run_job

train_data = DataConfig(
    distribution=mix(source("train.jsonl", map="my_project.to_retrieval_record")),
    collate=ComponentConfig(target="my_project.collate_retrieval"),
)
valid_data = DataConfig(
    distribution=mix(source("valid.jsonl", map="my_project.to_retrieval_record")),
    collate=ComponentConfig(target="my_project.collate_retrieval"),
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
    data=train_data,
    training=TrainingConfig(
        global_batch_size=32,
        max_steps=10_000,
        seed=17,
        batch=BatchConfig(micro_batch_size=32),
        precision=PrecisionConfig.bfloat16_mixed(),
    ),
    logging=LoggingConfig(console_every=100),
    checkpointing=CheckpointConfig(every=1_000, keep=3),
    evaluation=EvaluationConfig(
        data=valid_data,
        batch_size=32,
        every_steps=1_000,
        primary_metric="valid/loss",
    ),
    export=ExportConfig(selection="best"),
)
result = run_job(job, "runs/example")
```

`run_job` is the canonical configured boundary: it builds the Equinox model,
task and loss modifiers, Optax schedule/state, Grain sources, compiled execution
strategy, evaluator cache, Orbax lifecycle, and selected inference artifact.
The lower-level builders and `run_training` remain public for research programs
that need to assemble those pieces directly.

W&B is an optional reporter rather than a training dependency. Install
`representax[wandb]`, then configure the existing logging boundary:

```python
from representax.config import LoggingConfig, WandbConfig

logging = LoggingConfig(wandb=WandbConfig(project="representax"))
```

JSONL remains the local source of truth and the W&B client runs on the bounded
reporter worker.

Mixed precision keeps parameters, checkpoints, Optax state, gradients, and
objectives in FP32 while using transient BF16 parameter views and activations.
The same policy applies to in-training validation and composes with direct,
GradCache, DDP, FSDP, and custom-sharded execution. See the
[precision contract](docs/precision.md).

Packed four-bit base weights can instead train low-rank adapters while keeping
only the adapters in FP32 optimizer state:

```python
from representax.config import PrecisionConfig, QuantizedLoRAConfig, TrainingConfig

training = TrainingConfig(
    # ordinary batch, mesh, and lifecycle fields omitted
    adapter=QuantizedLoRAConfig(rank=8, alpha=16.0),
    precision=PrecisionConfig.bfloat16_mixed(),
)
```

This is weight-quantized adapter training with BF16 matrix compute, not a claim
of native INT4 training arithmetic. See the measured
[low-bit adapter contract](docs/adapters.md). Native FP8 matrix compute is also
available as an experimental policy; BF16 remains the recommended default.

Named DDP and FSDP are configuration presets, not separate trainers. Explicit
partition rules resolve through the same internal plan:

```python
from representax.config import FSDPConfig, MeshConfig

training = TrainingConfig(
    global_batch_size=32,
    max_steps=10_000,
    seed=17,
    mesh=MeshConfig(axis_shapes=(2,), axis_names=("data",)),
    sharding=FSDPConfig(
        data_axis="data",
    ),
    batch=BatchConfig(micro_batch_size=16),
)
```

Named DDP, FSDP, and custom partition rules are global JAX programs using the
same compiled step. The plan declares batch, parameter, Optax-state, and result
shardings. Shared model primitives annotate exact parameter-use and activation
layouts, while JAX derives the required communication and its autodiff
transpose. FSDP is therefore architecture-agnostic: there is no separate
trainer, per-model materializer, custom VJP, or manual gradient synchronization.
The acceptance profile records exact 2/4-device trajectories and a physical
1.9795B-parameter capacity run on two 24 GB GPUs. See
[`fsdp-annotations-20260820`](benchmarks/results/fsdp-annotations-20260820/README.org).

Array-facing APIs use `jaxtyping` to state dtype and symbolic shape contracts
directly on model forwards, tasks, losses, and compiled-step keys. Representax
does not install a runtime type-checking hook; explicit domain validation remains
responsible for semantic requirements that shapes and dtypes cannot express.

The loop records canonical metric names such as `train/loss`, `valid/loss`, and
`perf/...` in `metrics.jsonl`, lifecycle events in `events.jsonl`, and final
status in `run.json`. A bounded reporter worker performs the device-to-host
metric transfer and fans the same ordered rows out to optional consumers,
including W&B, without placing a synchronization barrier in every training
iteration. Checkpoints are
written by Orbax with at most one asynchronous save in flight. Recreate the same
distribution, model/state template, task/optimizer program, and batch source and pass
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
The broader Sentence Transformers dense-system audit and the definition of a
true dataset-on-disk to final-model benchmark are in
[the dense-system audit](https://github.com/ckgresla/representax/blob/main/docs/dense-system-audit.md).

## Roadmap

[`todo.org`](https://github.com/ckgresla/representax/blob/main/todo.org) is the
canonical project roadmap and shared source of
truth. It tracks the production encoder port, parity gates, GradCache,
distributed training, checkpoint/resume, task-native audio/video, reward
modeling, JEPA, Profilax, and the systems-then-model research program.
