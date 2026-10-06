# Compatibility

Representax is alpha. The base distribution is pure Python and installs the CPU
JAX runtime, Grain, and safetensors; accelerator-specific JAX wheels remain an
explicit user choice.

## Alpha CPU matrix

The `0.0.1` wheel built on 2026-08-14 was installed outside the source checkout
in fresh environments and completed one compiled Dense encoder, MNR, and Optax
update:

| Python | Resolved JAX | Backend | Result |
|---|---|---|---|
| 3.11.15 | 0.10.2 | CPU | passed |
| 3.12.12 | 0.11.0 | CPU | passed |
| 3.13.14 | 0.11.0 | CPU | passed |

The same artifacts pass `twine check`. This is local release-candidate evidence,
not a substitute for the public CI and published-package readback gates.

## Test lanes

- Fast source tests run on CPU across Python 3.11, 3.12, and 3.13.
- Generic trainer, Grain, GradCache, and Orbax runtime tests run on the primary
  Python version.
- Two- and four-device sharding semantics run with virtual CPU devices in CI.
- Model parity and matched throughput/memory gates remain explicit accelerator
  lanes with pinned upstream environments.

The base dependency contract intentionally excludes PyTorch, Transformers,
CUDA-specific JAX wheels, model checkpoints, and datasets. The optional `hf`
extra adds Hugging Face datasets, Hub, tokenizer, and pinned Transformers
interoperability; repository-only parity groups and the optional performance
extra do not change the base import namespace.

The base install is the CPU path. The `cuda13` and `cuda12` extras select JAX's
official pip-managed NVIDIA runtimes; they do not depend on a system CUDA
toolkit. CUDA 13 is the preferred v0 GPU path and requires a sufficiently recent
NVIDIA driver. TPU remains an explicit future compatibility gate.

## Alpha wheel acceptance

The exact rebuilt `0.0.1` wheel was installed outside the source checkout on
2026-08-14 in two new Python 3.13.14 environments:

| Install | Resolved runtime | Hardware | Result |
|---|---|---|---|
| base wheel | JAX/JAXlib 0.11.0, no CUDA or NVIDIA packages | one CPU device | passed |
| `wheel[cuda13]` | JAX/JAXlib and CUDA 13 plugin 0.11.0 | RTX 4090, driver 580.159.03 | passed |

Both environments compiled and accepted the same Dense/MNR/Optax update and
reported loss `2.3512461185455322`. The final dependency-boundary rebuild also
resolved Grain 0.2.18 and safetensors 0.8.0 in both fresh environments. The GPU
environment resolved JAX's pip-managed CUDA, cuDNN, NCCL, and related NVIDIA
packages without using the system CUDA toolkit. A third fresh install accepted
the `hf` extra with Datasets, Hub, PyArrow, tokenizers, and Transformers 5.3.0
while keeping PyTorch absent.
