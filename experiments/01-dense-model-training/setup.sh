#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${experiment_dir}/../.." && pwd)"
environment_dir="${REPRESENTAX_EXPERIMENT_ENV:-${experiment_dir}/.venv}"
bootstrap_python="${PYTHON:-python3}"
accelerator_extra="${REPRESENTAX_JAX_EXTRA:-cuda12}"

case "${accelerator_extra}" in
  cuda12|cuda13) ;;
  *)
    echo "REPRESENTAX_JAX_EXTRA must be cuda12 or cuda13" >&2
    exit 2
    ;;
esac

"${bootstrap_python}" -m venv "${environment_dir}"
"${environment_dir}/bin/python" -m pip install --upgrade pip
(
  cd "${repository_root}"
  "${environment_dir}/bin/python" -m pip install \
    -e ".[config,hf,performance,wandb,${accelerator_extra}]" \
    --group parity
)

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
