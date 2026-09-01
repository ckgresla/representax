#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
environment_dir="${REPRESENTAX_EXPERIMENT_ENV:-${experiment_dir}/.venv}"
python="${REPRESENTAX_EXPERIMENT_PYTHON:-${environment_dir}/bin/python}"
seeds=(7 42 773)

usage() {
  echo "usage: $0 GPU0 GPU1 [GPU2 GPU3 [GPU4 GPU5]]" >&2
  echo "provide one GPU pair per concurrent seed; remaining seeds run in later waves" >&2
}

if [[ ! -x "${python}" ]]; then
  echo "experiment Python not found at ${python}; run ${experiment_dir}/setup.sh" >&2
  exit 2
fi
if (( $# != 2 && $# != 4 && $# != 6 )); then
  usage
  exit 2
fi

gpus=("$@")
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
