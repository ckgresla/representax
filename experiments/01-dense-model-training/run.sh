#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiments_dir="$(cd "${experiment_dir}/.." && pwd)"
environment_dir="${REPRESENTAX_EXPERIMENT_ENV:-${experiments_dir}/.venv}"
python="${REPRESENTAX_EXPERIMENT_PYTHON:-${environment_dir}/bin/python}"
seeds=(7 42 773)

usage() {
  cat <<EOF
Usage: $0 -g GPU_IDS

GPU_IDS is a comma-separated list of 2, 4, or 6 indices.
EOF
}

gpu_ids=""
while getopts ":g:h" option; do
  case "${option}" in
    g) gpu_ids="${OPTARG}" ;;
    h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done
shift "$((OPTIND - 1))"
if [[ -z "${gpu_ids}" ]] || (( $# )); then
  usage >&2
  exit 2
fi
if [[ ! -x "${python}" ]]; then
  echo "experiment Python not found at ${python}; run ${experiments_dir}/setup.sh" >&2
  exit 2
fi

IFS=',' read -r -a gpus <<< "${gpu_ids}"
if (( ${#gpus[@]} != 2 && ${#gpus[@]} != 4 && ${#gpus[@]} != 6 )); then
  usage >&2
  exit 2
fi
pair_count=$(( ${#gpus[@]} / 2 ))
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]] || [[ -n "${seen_gpus[${gpu}]:-}" ]]; then
    echo "GPU indices must be unique non-negative integers" >&2
    exit 2
  fi
  seen_gpus["${gpu}"]=1
done

for ((offset = 0; offset < ${#seeds[@]}; offset += pair_count)); do
  pids=()
  for ((pair = 0; pair < pair_count && offset + pair < ${#seeds[@]}; pair++)); do
    seed="${seeds[offset + pair]}"
    first_gpu="${gpus[2 * pair]}"
    second_gpu="${gpus[2 * pair + 1]}"
    "${python}" "${experiment_dir}/run.py" run \
      --seed "${seed}" \
      --gpus "${first_gpu}" "${second_gpu}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "one or more paired runs failed; aggregate was not written" >&2
    exit 1
  fi
done

"${python}" "${experiment_dir}/run.py" aggregate
