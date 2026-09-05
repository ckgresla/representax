#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
torch_project_dir="${project_dir}/torch-xla"
uv_command="${UV:-uv}"
jax_environment="${REPRESENTAX_TPU_JAX_ENV:-${project_dir}/.venv-jax}"
torch_environment="${REPRESENTAX_TPU_TORCH_ENV:-${project_dir}/.venv-torch-xla}"

if ! command -v "${uv_command}" >/dev/null 2>&1; then
  if [[ ${uv_command} == "uv" && -x "${HOME}/.local/bin/uv" ]]; then
    uv_command="${HOME}/.local/bin/uv"
  else
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 2
  fi
fi

UV_PROJECT_ENVIRONMENT="${jax_environment}" \
  "${uv_command}" sync \
    --project "${project_dir}" \
    --locked \
    --python 3.13

UV_PROJECT_ENVIRONMENT="${torch_environment}" \
  "${uv_command}" sync \
    --project "${torch_project_dir}" \
    --locked \
    --python /usr/bin/python3.10

"${jax_environment}/bin/python" -c \
  'import jax; print(f"JAX {jax.__version__}")'
PJRT_DEVICE=TPU "${torch_environment}/bin/python" -c \
  'import torch, torch_xla; print(f"PyTorch {torch.__version__}, PyTorch/XLA {torch_xla.__version__}")'
