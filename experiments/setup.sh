#!/usr/bin/env bash
set -euo pipefail

experiments_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${experiments_dir}/.." && pwd)"
environment_dir="${REPRESENTAX_EXPERIMENT_ENV:-${experiments_dir}/.venv}"
uv_command="${UV:-uv}"
setup_gpu="${REPRESENTAX_SETUP_GPU:-0}"

if ! command -v "${uv_command}" >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "an NVIDIA driver is required; a system CUDA toolkit is not" >&2
  exit 2
fi
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
if (( ${driver_version%%.*} < 525 )); then
  echo "NVIDIA driver 525 or newer is required; found ${driver_version}" >&2
  exit 2
fi

UV_PROJECT_ENVIRONMENT="${environment_dir}" \
  "${uv_command}" sync --project "${experiments_dir}" --locked

(
  cd "${repository_root}"
  "${environment_dir}/bin/python" - <<'PY'
from experiments.preflights.provenance import installed_reference_provenance

evidence = installed_reference_provenance("sentence-transformers")
print(
    "installed "
    f"{evidence['distribution']}=={evidence['installed_version']} "
    f"from upstream commit {evidence['commit']}"
)
PY
)

CUDA_VISIBLE_DEVICES="${setup_gpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${environment_dir}/bin/python" - <<'PY'
import jax
import jax.numpy as jnp

value = (jnp.ones((32, 32)) @ jnp.ones((32, 32))).block_until_ready()
print(f"JAX {jax.__version__}: {jax.devices()[0]}, smoke={float(value[0, 0])}")
PY

CUDA_VISIBLE_DEVICES="${setup_gpu}" "${environment_dir}/bin/python" - <<'PY'
import torch

value = torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")
print(
    f"PyTorch {torch.__version__}: {torch.cuda.get_device_name()}, "
    f"smoke={float(value[0, 0])}"
)
PY
