#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --gpus 0,1[,2,3,4,5] [--asset-root PATH] [--output-root PATH]" >&2
}

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$experiment_dir/../.." && pwd)
python=${REPRESENTAX_EXPERIMENT_PYTHON:-$repository_root/experiments/.venv/bin/python}
late_python=${REPRESENTAX_LATE_INTERACTION_PYTHON:-$repository_root/experiments/.venv-late-interaction/bin/python}
asset_root=/raid/representax-paper-assets
output_root=/raid/representax-paper/10-cross-accelerator-framework-comparison/gpu-rtx4090
steps=22
gpu_list=

while (( $# )); do
  case $1 in
    --gpus) gpu_list=${2:?--gpus requires a comma-separated list}; shift 2 ;;
    --asset-root) asset_root=${2:?--asset-root requires a path}; shift 2 ;;
    --output-root) output_root=${2:?--output-root requires a path}; shift 2 ;;
    --steps) steps=${2:?--steps requires an integer}; shift 2 ;;
    --help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -x $python && -x $late_python ]] || {
  echo "missing experiment environments; run experiments/setup.sh first" >&2
  exit 2
}
IFS=, read -r -a gpus <<<"$gpu_list"
if (( ${#gpus[@]} < 2 || ${#gpus[@]} % 2 != 0 )); then
  echo "--gpus requires an even number of GPU indices" >&2
  exit 2
fi
for gpu in "${gpus[@]}"; do
  [[ $gpu =~ ^[0-9]+$ ]] || { echo "invalid GPU index: $gpu" >&2; exit 2; }
done
[[ $steps =~ ^[0-9]+$ && $steps -gt 2 ]] || {
  echo "--steps must be an integer greater than two" >&2
  exit 2
}

readonly seeds=(7 42 773 1234 2026)
readonly parallel_recipes=(
  dense-retrieval
  semantic-similarity-mpnet-base
  semantic-similarity-bert-base
  pair-classification-mpnet-base
  pair-classification-bert-base
  cross-encoder
  late-interaction
  outcome-reward
  process-reward
  image-text
)
readonly isolated_recipes=(audio-text video-text v-jepa)
readonly pair_count=$((${#gpus[@]} / 2))
readonly cache_root="$output_root/caches"
readonly inductor_root="${output_root}-torchinductor"
mkdir -p "$output_root/orchestrator" "$cache_root" "$inductor_root/orchestrator"

variant() {
  local recipe=$1
  local framework=$2
  if [[ $framework == representax ]]; then
    case $recipe in
      dense-retrieval|late-interaction|image-text|audio-text|video-text)
        echo representax-local
        return
        ;;
    esac
  fi
  echo "$framework"
}

prune_run_artifacts() {
  local output=$1
  local path
  local -a paths=()
  while IFS= read -r -d '' path; do
    paths+=("$path")
  done < <(
    find "$output" \
      \( -type d \( -name checkpoints -o -name final-model -o -name flat-index \) \
      -o -type f -name final-model.pt \) -prune -print0
  )
  (( ${#paths[@]} )) || return
  printf '%s\n' "${paths[@]}" >"$output/pruned-artifacts.txt"
  for path in "${paths[@]}"; do
    find "$path" -xdev -depth -delete
  done
}

run_one() {
  local recipe=$1
  local seed=$2
  local framework=$3
  local gpu=$4
  local root=${5:-$output_root}
  local compile=${6:-false}
  local output="$root/seed-$seed/$recipe/$(variant "$recipe" "$framework")"
  local -a command=(
    "$python" "$experiment_dir/run.py" suite
    --framework "$framework"
    --asset-root "$asset_root"
    --output "$root/seed-$seed"
    --seed "$seed"
    --steps "$steps"
    --platform gpu
    --gpu "$gpu"
    --jax-python "$python"
    --reference-python "$python"
    --late-reference-python "$late_python"
    --recipe "$recipe"
  )
  [[ $compile == true ]] && command+=(--torch-compile)

  if [[ -f $output/run.json ]] && grep -q '"status": "completed"' "$output/run.json"; then
    prune_run_artifacts "$output"
    printf '[%s] complete; skipping %s seed=%s framework=%s\n' \
      "$(date -Is)" "$recipe" "$seed" "$framework"
    return
  fi
  if [[ -e $output ]]; then
    local partial="$output.partial-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$output" "$partial"
    prune_run_artifacts "$partial"
  fi

  printf '[%s] gpu=%s recipe=%s seed=%s framework=%s compile=%s\n' \
    "$(date -Is)" "$gpu" "$recipe" "$seed" "$framework" "$compile"
  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  JAX_COMPILATION_CACHE_DIR="$cache_root/jax/gpu-$gpu/$recipe" \
  TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor/gpu-$gpu/$recipe" \
    "${command[@]}"
  prune_run_artifacts "$output"
}

run_pair() {
  local recipe=$1
  local seed=$2
  local native_gpu=$3
  local reference_gpu=$4
  local native_log="$output_root/orchestrator/$recipe-seed-$seed-representax.log"
  local reference_log="$output_root/orchestrator/$recipe-seed-$seed-reference.log"
  local native_status=0
  local reference_status=0

  run_one "$recipe" "$seed" representax "$native_gpu" >"$native_log" 2>&1 &
  local native_pid=$!
  run_one "$recipe" "$seed" reference "$reference_gpu" >"$reference_log" 2>&1 &
  local reference_pid=$!
  wait "$native_pid" || native_status=$?
  wait "$reference_pid" || reference_status=$?
  if (( native_status != 0 || reference_status != 0 )); then
    echo "failed pair: recipe=$recipe seed=$seed; see $native_log and $reference_log" >&2
    return 1
  fi
}

run_lane() {
  local lane=$1
  local native_gpu=${gpus[$((lane * 2))]}
  local reference_gpu=${gpus[$((lane * 2 + 1))]}
  local index=0
  for recipe in "${parallel_recipes[@]}"; do
    for seed in "${seeds[@]}"; do
      if (( index % pair_count == lane )); then
        run_pair "$recipe" "$seed" "$native_gpu" "$reference_gpu"
      fi
      ((index += 1))
    done
  done
}

declare -a lane_pids=()
for ((lane = 0; lane < pair_count; lane += 1)); do
  run_lane "$lane" &
  lane_pids+=("$!")
done
for pid in "${lane_pids[@]}"; do
  wait "$pid"
done

# These recipes load large media tensors or 3B-parameter models into host RAM.
# One matched pair at a time avoids the host-memory failures seen in earlier runs.
for recipe in "${isolated_recipes[@]}"; do
  for seed in "${seeds[@]}"; do
    run_pair "$recipe" "$seed" "${gpus[0]}" "${gpus[1]}"
  done
done

declare -a inductor_pids=()
for index in "${!seeds[@]}"; do
  seed=${seeds[$index]}
  gpu=${gpus[$((index % ${#gpus[@]}))]}
  log="$inductor_root/orchestrator/dense-retrieval-seed-$seed-reference.log"
  run_one dense-retrieval "$seed" reference "$gpu" "$inductor_root" true >"$log" 2>&1 &
  inductor_pids+=("$!")
done
for pid in "${inductor_pids[@]}"; do
  wait "$pid"
done

"$python" "$experiment_dir/run.py" aggregate --input "$output_root" --output "$output_root"
"$python" "$experiment_dir/run.py" aggregate --input "$inductor_root" --output "$inductor_root"
