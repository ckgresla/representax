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
Grain to require latest-N retention, science and PyTree compatibility checks,
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
python -m pip install -e ".[test,parity-modernvbert,performance]"
export REPRESENTAX_MODERNVBERT_TRANSFORMERS_PYTHON=/path/to/tf53/bin/python
pytest -m performance tests/models
```

Each comparison also checks its final output numerically before applying the
speed and memory thresholds. A faster program doing different work cannot pass.
