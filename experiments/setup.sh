#!/usr/bin/env bash
set -euo pipefail

experiments_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${experiments_dir}/.." && pwd)"
environment_dir="${REPRESENTAX_EXPERIMENT_ENV:-${experiments_dir}/.venv}"
late_interaction_environment_dir="${REPRESENTAX_LATE_INTERACTION_ENV:-${experiments_dir}/.venv-late-interaction}"
reference_root="${REPRESENTAX_REFERENCE_ROOT:-${experiments_dir}/.references}"
uv_command="${UV:-uv}"
setup_gpu="${REPRESENTAX_SETUP_GPU:-0}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "$2" >&2
    exit 2
  fi
}

checkout_reference() {
  local name="$1"
  local repository="$2"
  local commit="$3"
  local directory="$4"

  if [[ ! -e "${directory}" ]]; then
    git clone --filter=blob:none "${repository}" "${directory}"
  elif [[ ! -d "${directory}/.git" ]]; then
    echo "reference path is not a Git checkout: ${directory}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${directory}" status --porcelain)" ]]; then
    echo "reference checkout has local changes: ${directory}" >&2
    exit 2
  fi

  if [[ "$(git -C "${directory}" rev-parse HEAD)" != "${commit}" ]]; then
    git -C "${directory}" fetch --quiet --depth 1 origin "${commit}"
    git -C "${directory}" checkout --quiet --detach "${commit}"
  fi
  echo "checked out ${name} at ${commit}"
}

require_command "${uv_command}" \
  "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
require_command git "git is required to materialize pinned reference sources"
require_command python "python is required to read the reference manifest"
require_command nvidia-smi \
  "an NVIDIA driver is required; a system CUDA toolkit is not"

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
if (( ${driver_version%%.*} < 525 )); then
  echo "NVIDIA driver 525 or newer is required; found ${driver_version}" >&2
  exit 2
fi

mkdir -p "${reference_root}"
while IFS=$'\t' read -r name repository commit directory; do
  checkout_reference \
    "${name}" "${repository}" "${commit}" "${reference_root}/${directory}"
done < <(
  python - "${repository_root}/benchmarks/configs/paper-references-v1.json" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
directories = {
    "lejepa-paper": "lejepa",
    "stable-pretraining": "stable-pretraining",
    "vjepa2": "vjepa2",
}
for name, directory in directories.items():
    reference = document["references"][name]
    print(name, reference["repository"], reference["commit"], directory, sep="\t")
PY
)

UV_PROJECT_ENVIRONMENT="${environment_dir}" \
  "${uv_command}" sync \
    --project "${experiments_dir}" \
    --locked \
    --no-default-groups \
    --group main
UV_PROJECT_ENVIRONMENT="${late_interaction_environment_dir}" \
  "${uv_command}" sync \
    --project "${experiments_dir}" \
    --locked \
    --no-default-groups \
    --group late-interaction

(
  cd "${repository_root}"
  REPRESENTAX_REFERENCE_ROOT="${reference_root}" \
    "${environment_dir}/bin/python" - <<'PY'
import importlib
import os
import sys
from importlib.metadata import version
from pathlib import Path

from experiments.preflights.provenance import (
    checkout_reference_provenance,
    installed_reference_provenance,
)

root = Path(os.environ["REPRESENTAX_REFERENCE_ROOT"])
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
for name in ("sentence-transformers", "trl", "stable-pretraining"):
    evidence = installed_reference_provenance(name)
    print(f"installed {evidence['distribution']}=={evidence['installed_version']}")
print(f"installed lejepa=={version('lejepa')}")

for name, directory in {
    "lejepa-paper": "lejepa",
    "stable-pretraining": "stable-pretraining",
    "vjepa2": "vjepa2",
}.items():
    evidence = checkout_reference_provenance(name, root / directory)
    print(f"source {name} at {evidence['commit']}")

importlib.import_module("sentence_transformers")
trl = importlib.import_module("trl")
assert trl.RewardConfig and trl.RewardTrainer
prm = importlib.import_module("trl.experimental.prm")
assert prm.PRMConfig and prm.PRMTrainer
importlib.import_module("lejepa")
importlib.import_module("stable_pretraining")
sys.path.insert(0, str(root / "vjepa2"))
importlib.import_module("app.vjepa_2_1.models.vision_transformer")
importlib.import_module("app.vjepa_2_1.models.predictor")
print("main reference imports passed")
PY
)

(
  cd "${repository_root}"
  "${late_interaction_environment_dir}/bin/python" - <<'PY'
from importlib import import_module
from importlib.metadata import version

from experiments.preflights.provenance import installed_reference_provenance

evidence = installed_reference_provenance("pylate")
assert version("sentence-transformers") == "5.3.0"
assert version("transformers") == "5.3.0"
for module in ("pylate.losses", "pylate.models", "pylate.scores", "pylate.utils"):
    import_module(module)
print(
    f"installed {evidence['distribution']}=={evidence['installed_version']} "
    "with sentence-transformers==5.3.0 and transformers==5.3.0"
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

for python in \
  "${environment_dir}/bin/python" \
  "${late_interaction_environment_dir}/bin/python"; do
  CUDA_VISIBLE_DEVICES="${setup_gpu}" "${python}" - <<'PY'
import torch

value = torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")
print(
    f"PyTorch {torch.__version__}: {torch.cuda.get_device_name()}, "
    f"smoke={float(value[0, 0])}"
)
PY
done
