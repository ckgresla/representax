# Compatibility

Representax is pre-release. The base distribution is pure Python and installs
the CPU JAX runtime; accelerator-specific JAX wheels remain an explicit user
choice.

## Pre-release CPU matrix

The `0.0.1` wheel built on 2026-08-14 was installed outside the source checkout
in fresh environments and completed one compiled Dense encoder, MNR, and Optax
update:

| Python | Resolved JAX | Backend | Result |
|---|---|---|---|
| 3.11.15 | 0.10.2 | CPU | passed |
| 3.12.12 | 0.11.0 | CPU | passed |
| 3.13.14 | 0.11.0 | CPU | passed |

The same artifacts pass `twine check`. This is local pre-release evidence, not
a substitute for the public CI and published-package readback gates.

## Test lanes

- Fast source tests run on CPU across Python 3.11, 3.12, and 3.13.
- Generic trainer, Grain, GradCache, and Orbax runtime tests run on the primary
  Python version.
- Two- and four-device sharding semantics run with virtual CPU devices in CI.
- Model parity and matched throughput/memory gates remain explicit accelerator
  lanes with pinned upstream environments.

The base dependency contract intentionally excludes PyTorch, Transformers,
CUDA-specific JAX wheels, model checkpoints, and datasets. Optional extras add
data, Hugging Face, parity, and performance tooling without changing the base
import namespace.
