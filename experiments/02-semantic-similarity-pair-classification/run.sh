#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiments_dir="$(cd "${experiment_dir}/.." && pwd)"
environment_dir="${REPRESENTAX_EXPERIMENT_ENV:-${experiments_dir}/.venv}"
python="${REPRESENTAX_EXPERIMENT_PYTHON:-${environment_dir}/bin/python}"
artifact_root="${REPRESENTAX_PAPER_ROOT:-/raid/representax-paper}/02-semantic-similarity-pair-classification"

usage() {
  cat <<EOF
Usage: $0 -g GPU_IDS

GPU_IDS is an even-length comma-separated list of 2, 4, or 6 unique indices.
Each pair runs Representax and Sentence Transformers for one cell.
Reference timing is then repeated in framework-only waves to avoid XLA compile contention.
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
if (( ${#gpus[@]} < 2 || ${#gpus[@]} > 6 || ${#gpus[@]} % 2 )); then
  usage >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]] || [[ -n "${seen_gpus[${gpu}]:-}" ]]; then
    echo "GPU indices must be unique non-negative integers" >&2
    exit 2
  fi
  seen_gpus["${gpu}"]=1
done

"${python}" "${experiment_dir}/run.py" --artifact-root "${artifact_root}" prepare

jobs=()
for workload in semantic-similarity pair-classification; do
  for model in mpnet-base bert-base; do
    for seed in 7 42 773; do
      jobs+=("${workload}:${model}:${seed}")
    done
  done
done

pair_count=$(( ${#gpus[@]} / 2 ))
for ((offset = 0; offset < ${#jobs[@]}; offset += pair_count)); do
  pids=()
  for ((pair = 0; pair < pair_count && offset + pair < ${#jobs[@]}; pair++)); do
    IFS=':' read -r workload model seed <<< "${jobs[offset + pair]}"
    summary="${artifact_root}/runs/${workload}/${model}/seed-${seed}/summary.json"
    if [[ -f "${summary}" ]]; then
      echo "already complete: ${workload}/${model}/seed-${seed}"
      continue
    fi
    "${python}" "${experiment_dir}/run.py" \
      --artifact-root "${artifact_root}" \
      run \
      --workload "${workload}" \
      --model "${model}" \
      --seed "${seed}" \
      --gpus "${gpus[2 * pair]}" "${gpus[2 * pair + 1]}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "one or more Experiment 02 runs failed; aggregate was not written" >&2
    exit 1
  fi
done

for ((offset = 0; offset < ${#jobs[@]}; offset += ${#gpus[@]})); do
  pids=()
  for ((slot = 0; slot < ${#gpus[@]} && offset + slot < ${#jobs[@]}; slot++)); do
    IFS=':' read -r workload model seed <<< "${jobs[offset + slot]}"
    report="${artifact_root}/timing-validation/${workload}/${model}/seed-${seed}/sentence-transformers.json"
    if [[ -f "${report}" ]]; then
      echo "timing already complete: ${workload}/${model}/seed-${seed}"
      continue
    fi
    "${python}" "${experiment_dir}/run.py" \
      --artifact-root "${artifact_root}" \
      reference-timing \
      --workload "${workload}" \
      --model "${model}" \
      --seed "${seed}" \
      --gpu "${gpus[slot]}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "one or more isolated reference timing runs failed" >&2
    exit 1
  fi
done

"${python}" "${experiment_dir}/run.py" --artifact-root "${artifact_root}" aggregate
