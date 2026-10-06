# Testing

Representax uses pytest as its single test runner. Tests live outside `src/`
and mirror the package by model, task, data, planning, and training domain.
Markers describe how a test executes:

- unmarked tests are fast and dependency-light;
- `runtime` tests compile complete forward, backward, and update paths;
- `parity` tests compare against an optional upstream implementation;
- `distributed` tests require multiple devices or processes; and
- `performance` tests measure compilation, throughput, latency, or memory.

The default `pytest` command excludes the four environment-sensitive lanes.
Each lane can be selected across the mirrored tree with `pytest -m <marker>`.

## Task-loss acceptance

The canonical Sentence Transformers loss inventory is one discoverable file:

```bash
python -m pip install -e ".[test,performance]" --group parity
pytest -m parity tests/tasks/test_sentence_transformers_parity.py
pytest -m performance tests/tasks/test_sentence_transformers_parity.py -s
```

The parity lane generates shared NumPy tensors, passes the same values to JAX
and the pinned Sentence Transformers 5.6.1 class, then compares the scalar loss
and every representation gradient. An explicit coverage assertion keeps the 29
classes claimed as native synchronized with the paired cases. Cached MNR uses
each runtime's chunked score-row objective; end-to-end encoder replay and
distributed execution remain in the dedicated GradCache acceptance suites.

The performance lane compiles one native forward-and-backward program per
class, warms both runtimes, synchronizes every sample, and reports median
latency. It is intentionally separate from parity and emits performance
shortfalls as warnings. Small loss-only measurements show dispatch and fusion
quality; full encoder training remains the authoritative systems benchmark.

## GradCache training-step matrix

The pinned ModernVBERT GradCache matrix measures one complete device-side
optimizer step: token encoder forwards, pooling, cached MNR, representation
cotangents, rematerialized parameter-gradient replay, gradient metrics, and one
AdamW update. It deliberately excludes dataset opening, preprocessing,
reporting, checkpoint publication, and final export, so it is not described as
an end-to-end training benchmark.

Run the four matched points concurrently on two isolated GPUs:

```bash
python -m benchmarks.grad_cache_matrix \
  --checkpoint /immutable/path/to/modernvbert-snapshot \
  --output-directory /raid/representax/benchmarks/gradcache-st56 \
  --native-gpu 4 --upstream-gpu 5 \
  --batches 32 128 512 1024 \
  --source-commit "$(git rev-parse HEAD)"
```

The matrix runner requires Sentence Transformers 5.6.1, runs each matched pair
concurrently in fresh subprocesses, removes a potentially conflicting
`LD_LIBRARY_PATH`, disables the native persistent compilation cache for the
cold-start measurement, and writes raw logs, reports, and one validated
summary. Contract or numerical mismatches fail immediately. A speed shortfall
is recorded as a warning in the artifact; the controlled repository acceptance
test requires all four recorded points to beat the oracle.

To rerun the same gate through pytest, set the checkpoint and GPU pair:

```bash
export REPRESENTAX_MODERNVBERT_CHECKPOINT=/immutable/path/to/snapshot
export REPRESENTAX_GRAD_CACHE_PERFORMANCE_GPUS=4,5
pytest -q -s -m performance \
  tests/train/test_grad_cache_performance.py
```

The disk-to-final-model contract is broader and is specified separately in
[`dense-system-audit.md`](dense-system-audit.md).

## Static analysis

Run the import-free development gate before pytest:

```bash
python -m pip install -e ".[config,hf,test,performance]" --group static
python scripts/check.py
```

It checks formatting and lint rules with Ruff, then type-checks `src`, tests,
examples, and repository scripts with ty. The tools live in the repository-only
`static` dependency group. Optional PyTorch and Pillow imports are dynamic only
inside their upstream-oracle files; the production source receives no relaxed
type-checking rules.

Ordinary CI runs the fast suite on CPU across Python 3.11, 3.12, and 3.13.  It
runs the static gate once on Python 3.13, the generic training/runtime tree once
on Python 3.13, exercises the two-
and four-device distributed semantics with four virtual CPU devices, and builds
and installs the wheel in a fresh CPU-only context.  Model-family runtime,
upstream parity, and matched systems measurements remain their explicit lanes
rather than multiplying expensive accelerator work across the Python matrix.

The distributed GradCache gate should run with four visible devices so its
two- and four-device cases share one test invocation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_DEFAULT_MATMUL_PRECISION=highest \
pytest -m distributed tests/train/test_distributed_grad_cache.py
```

It compares both replicated-data-parallel updates with a one-device global
batch oracle, including task metrics, gradient and update norms, model and
optimizer state, and a cross-device hard-negative canary.

The process-boundary acceptance is one parent job that launches two coordinated
JAX workers, each with two of GPUs 0–3:

```bash
python benchmarks/launch_multiprocess_grad_cache_modernvbert.py \
  --checkpoint /immutable/path/to/modernvbert-snapshot \
  --output-directory benchmarks/results/multiprocess-grad-cache-modernvbert
```

Each worker receives only its process-local token rows, retains global relation
columns, and compares the same four-device update with a full-batch one-device
oracle. Passing this on one machine validates process boundaries and
collectives. A physical multi-host claim additionally requires a run over a
routable coordinator plus transport, checkpoint-publication, and failure
evidence.

## Checkpoint acceptance

Checkpoint mechanics have fast controlled tests for one-in-flight asynchronous
publication, backpressure, background failure propagation, incomplete markers,
and byte-accurate log rollback. The `runtime` lane then uses real Orbax and
Grain to require latest-N retention, semantics, data-contract, and PyTree checks,
mixed-dtype restoration, and exact equality between uninterrupted and resumed
model, optimizer, loss, batch position, and random-key trajectories. Snapshot
and backpressure latency must remain separate lifecycle events rather than
being reported as compiled-step throughput.

The compiled-training fixture uses eight batches of 16 continuous paired views
with independent nuisance features and requires the learned retrieval loss to
fall by more than half. Batch-two fixtures are reserved for focused host-loop
mechanics where accelerator behavior is not under test.

## Model-family acceptance

Every supported model family owns a paired upstream reference under its test
directory. Native and upstream programs must use the same:

- immutable checkpoint and configuration;
- explicitly selected native and upstream Python environments;
- tokenized or decoded inputs and static shapes;
- parameter, compute, accumulation, and output dtypes;
- attention masks, pooling, normalization, loss, and optimizer semantics; and
- device class and relevant kernel or compiler settings.

The required layout is:

```text
src/representax/models/<family>/
tests/models/<family>/
├── test_model.py                 # native configuration and numerical units
├── test_train_step.py            # compiled task integration when trainable
├── test_transformers_parity.py   # model-specific numerical contract
├── transformers_oracle.py        # optional upstream reference generation
└── performance_probe.py          # isolated native/upstream programs
```

Checkpoint-backed model packages are discovered automatically. The fast
acceptance test fails if any lacks a registration or required test surface in
`tests/models/acceptance.py`; each registration is also parameterized into the
general systems gate in `tests/models/test_implementations.py`.

Acceptance has two consecutive gates:

1. **Numerical equivalence.** Compare layer or final outputs, pooled
   representations, input and parameter gradients, objective values, and one
   optimizer update with explicit absolute, relative, and cosine tolerances.
2. **Systems improvement.** After equivalence passes, compare compilation,
   synchronized steady-state latency or throughput, and peak device memory.
   Representax must perform equivalent scientific work faster. Peak memory is
   always recorded; a model case may enforce a stable controlled threshold or
   emit an explicit warning while an observed regression is still noisy or
   awaiting a kernel-level optimization.

A performance result is not valid if it changes batch semantics, negative
population, sequence or media shape, precision, rematerialization policy, or
the requested outputs and gradients.

## Measurement protocol

Pytest orchestrates performance comparisons, but each runtime executes in a
fresh subprocess. This isolates process-local compilation state and memory
allocators. A fresh process does not isolate JAX's persistent compilation
cache: cold-start comparisons must also disable that cache or use a unique,
empty cache directory per probe.

Every comparison must:

1. record hardware, software versions, Git revision, checkpoint revision, and
   a workload fingerprint;
2. disable broad device-memory preallocation when measuring resident memory;
3. record persistent-cache policy and measure import or initialization and
   compile-plus-first-execution separately;
4. warm up the exact compiled program before steady-state samples;
5. explicitly synchronize device results around every timed region;
6. report a distribution over repeated samples, not one observation;
7. capture both runtime allocator peaks and an external device-memory peak when
   the platform exposes them; and
8. emit a machine-readable artifact containing raw samples and derived ratios.

The report also computes the number of steady-state steps required to amortize
any native compilation and initialization disadvantage. Steady-state speed and
numerical equivalence are acceptance gates. Memory is never hidden: allocator
and end-to-end process peaks, ratios, and warnings remain in every artifact.
Cold start and break-even are always visible and must be considered when
selecting a serving or training runtime.

Ordinary CI checks numerical behavior. Relative speed and memory assertions run
on a controlled scheduled hardware lane, where thresholds can be tied to a
specific accelerator and software matrix rather than ambient shared runners.

Run every registered model comparison with:

```bash
python -m pip install -e ".[test,performance]" --group parity-modernvbert
export REPRESENTAX_MODERNVBERT_TRANSFORMERS_PYTHON=/path/to/tf53/bin/python
pytest -m performance tests/models
```

The dense Sentence Transformers cases use the repository-only `parity` group
and immutable local snapshots:

```bash
python -m pip install -e ".[test,performance]" --group parity
export REPRESENTAX_SENTENCE_TRANSFORMERS_PYTHON=/path/to/parity/bin/python
export REPRESENTAX_MINILM_CHECKPOINT=/path/to/all-MiniLM-L6-v2
export REPRESENTAX_MPNET_CHECKPOINT=/path/to/all-mpnet-base-v2
pytest -m parity tests/models/sentence_transformers
pytest -m performance tests/models/test_implementations.py
```

Native MPNet parameter, input-gradient, optimizer-update, and export parity use
the same Transformers 5.3.0 environment through
`REPRESENTAX_MPNET_TRANSFORMERS_PYTHON`.

LLaVA-NeXT acceptance uses current Sentence Transformers 5.6.1 for the four
BGE checkpoints. E5-V records Sentence Transformers 5.4.0 and Transformers
5.5.0 in its own artifact metadata, so its upstream oracle lives in a separate
repository-only environment:

```bash
python -m pip install -e ".[test,performance]" --group parity-llava-next-legacy
export REPRESENTAX_E5_SENTENCE_TRANSFORMERS_PYTHON=/path/to/legacy/bin/python
export REPRESENTAX_E5_SENTENCE_TRANSFORMERS_VERSION=5.4.0
```

This split is intentional. Sentence Transformers 5.6.1 delegates E5-V to the
new Transformers 5.6 base `LlavaNextModel`, whose renamed vision prefix leaves
all 24 source vision layers missing and randomly initialized. The legacy oracle
loads all 686 encoder tensors; Representax's native loader accepts both layouts
without executing repository code.

The pinned Jina v5 Omni Small text path has its own BF16 forward gate. The
checkpoint is optional, non-redistributed test data and remains governed by its
CC BY-NC 4.0 artifact license:

```bash
export REPRESENTAX_JINA_V5_SMALL_CHECKPOINT=/path/to/pinned/snapshot
export REPRESENTAX_JINA_V5_SMALL_ORACLE=/path/to/jina-v5-small-text-st56.npz
python -m tests.models.jina_v5.transformers_oracle \
  "$REPRESENTAX_JINA_V5_SMALL_CHECKPOINT" \
  "$REPRESENTAX_JINA_V5_SMALL_ORACLE"
pytest -q -m parity tests/models/jina_v5/test_transformers_parity.py
```

The complete raw-JSONL-to-fresh-reload comparison for MiniLM and Jina Small is
replayed with `python -m benchmarks.dense_end_to_end`; the reviewed baseline is
in [`benchmarks/results/dense-e2e-20260817`](../benchmarks/results/dense-e2e-20260817/README.org).

Each comparison also checks its final output numerically before applying the
speed and memory thresholds. A faster program doing different work cannot pass.
