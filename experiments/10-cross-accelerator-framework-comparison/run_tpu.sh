#!/usr/bin/env bash
set -euo pipefail

: "${TPU_PROJECT:?set TPU_PROJECT to the Google Cloud project ID}"
: "${TPU_NAME:?set TPU_NAME to the TPU VM name}"

TPU_ZONE=${TPU_ZONE:-us-central1-a}
CLOUDSDK_CONFIG=${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}
REMOTE_REPOSITORY=${REMOTE_REPOSITORY:-/home/ckg/representax}
REMOTE_ASSET_ROOT=${REMOTE_ASSET_ROOT:-/home/ckg/representax-paper-assets}
REMOTE_OUTPUT_ROOT=${REMOTE_OUTPUT_ROOT:-/home/ckg/representax-paper-results/10-cross-accelerator-framework-comparison/tpu-v5e-16}
STEPS=${STEPS:-22}

readonly recipes=(
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
  audio-text
  video-text
  v-jepa
)
readonly seeds=(7 42 773 1234 2026)
readonly frameworks=(representax reference)

run_one() {
  local recipe=$1
  local seed=$2
  local framework=$3
  local variant=$framework
  local output
  local command
  local status_command
  local status_output
  local response_count
  local complete_count
  local summary_count

  if [[ $framework == representax ]]; then
    case $recipe in
      dense-retrieval|late-interaction|image-text|audio-text|video-text)
        variant=representax-local
        ;;
    esac
  fi
  output="$REMOTE_OUTPUT_ROOT/seed-$seed/$recipe/$variant"

  printf -v status_command \
    'if grep -q %q %q 2>/dev/null; then if [[ -f %q ]]; then echo COMPLETE_SUMMARY; else echo COMPLETE; fi; else echo INCOMPLETE; fi' \
    '"status": "completed"' \
    "$output/run.json" \
    "$output/summary.json"
  status_output=$(
    CLOUDSDK_CONFIG=$CLOUDSDK_CONFIG \
      gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
        --project="$TPU_PROJECT" \
        --zone="$TPU_ZONE" \
        --worker=all \
        --batch-size=4 \
        --command="$status_command" 2>/dev/null
  )
  response_count=$(grep -Ec '^(COMPLETE|COMPLETE_SUMMARY|INCOMPLETE)$' <<<"$status_output" || true)
  complete_count=$(grep -Ec '^COMPLETE(_SUMMARY)?$' <<<"$status_output" || true)
  summary_count=$(grep -c '^COMPLETE_SUMMARY$' <<<"$status_output" || true)
  if (( response_count != 4 )); then
    echo "could not read run status from all four TPU workers" >&2
    return 1
  fi
  if (( complete_count == 4 && summary_count == 1 )); then
    printf '[%s] skipping completed recipe=%s seed=%s framework=%s\n' \
      "$(date -Is)" "$recipe" "$seed" "$framework"
    return
  fi
  if (( complete_count != 0 || summary_count != 0 )); then
    printf '[%s] archiving partial recipe=%s seed=%s framework=%s\n' \
      "$(date -Is)" "$recipe" "$seed" "$framework"
  fi
  printf -v command \
    'if [[ -e %q ]]; then mv %q %q; fi' \
    "$output" \
    "$output" \
    "$output.partial-$(date -u +%Y%m%dT%H%M%SZ)"
  CLOUDSDK_CONFIG=$CLOUDSDK_CONFIG gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$TPU_PROJECT" \
    --zone="$TPU_ZONE" \
    --worker=all \
    --batch-size=4 \
    --command="$command"

  printf -v command \
    'cd %q && JAX_COMPILATION_CACHE_DIR=%q %q %q suite --framework %q --asset-root %q --output %q --seed %q --steps %q --platform tpu --recipe %q' \
    "$REMOTE_REPOSITORY" \
    "$REMOTE_ASSET_ROOT/jax-cache" \
    "$REMOTE_REPOSITORY/experiments/tpu/.venv-jax/bin/python" \
    "$REMOTE_REPOSITORY/experiments/10-cross-accelerator-framework-comparison/run.py" \
    "$framework" \
    "$REMOTE_ASSET_ROOT" \
    "$REMOTE_OUTPUT_ROOT/seed-$seed" \
    "$seed" \
    "$STEPS" \
    "$recipe"

  printf '\n[%s] recipe=%s seed=%s framework=%s\n' \
    "$(date -Is)" "$recipe" "$seed" "$framework"
  CLOUDSDK_CONFIG=$CLOUDSDK_CONFIG gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$TPU_PROJECT" \
    --zone="$TPU_ZONE" \
    --worker=all \
    --batch-size=4 \
    --command="$command"
}

for recipe in "${recipes[@]}"; do
  for seed in "${seeds[@]}"; do
    for framework in "${frameworks[@]}"; do
      run_one "$recipe" "$seed" "$framework"
    done
  done
done
