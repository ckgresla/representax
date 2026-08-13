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
fresh subprocess. This prevents JAX and PyTorch compilation caches and memory
allocators from contaminating one another.

Every comparison must:

1. record hardware, software versions, Git revision, checkpoint revision, and
   a workload fingerprint;
2. disable broad device-memory preallocation when measuring resident memory;
3. measure import or initialization and compile-plus-first-execution separately;
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
