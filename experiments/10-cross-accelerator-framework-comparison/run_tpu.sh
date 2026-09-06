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
  local command

  printf -v command \
    'cd %q && %q %q suite --framework %q --asset-root %q --output %q --seed %q --steps %q --platform tpu --recipe %q' \
    "$REMOTE_REPOSITORY" \
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
