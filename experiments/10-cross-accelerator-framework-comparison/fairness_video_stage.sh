#!/usr/bin/env bash
# Restore only the frozen environment and assets needed by the final references.
set -euo pipefail
readonly revision=cdcb1047a4fb269fc994b110ff532ae1e38b3b72
readonly repo="$HOME/representax-fairness-cdcb104"
readonly assets="$HOME/representax-paper-assets"
readonly bucket=gs://representax-paper-assets-project-3fed1b4b
export PATH="$HOME:$HOME/.local/bin:$PATH"

git clone --filter=blob:none --no-checkout --single-branch \
  --branch codex/fairness-tpu-memory-20260916 \
  https://github.com/ckgresla/representax.git "$repo"
git -C "$repo" checkout --detach "$revision"
git -C "$repo" diff --exit-code
ln -s "$repo" "$HOME/representax-audio-rerun"

UV_PROJECT_ENVIRONMENT="$repo/experiments/tpu/.venv-jax" \
  uv sync --project "$repo/experiments/tpu" --locked --python 3.13
UV_PROJECT_ENVIRONMENT="$repo/experiments/tpu/.venv-torch-xla" \
  uv sync --project "$repo/experiments/tpu/torch-xla" \
    --locked --python /usr/bin/python3.10

mkdir -p "$assets/video-data" "$assets/omni-3b"
gcloud storage rsync --recursive "$bucket/data/video-text" "$assets/video-data"
gcloud storage rsync --recursive \
  "$bucket/models/LCO-Embedding-Omni-3B-2605/5f6b5329da5141367da30e06a9826d1322d6c9b2" \
  "$assets/omni-3b"
git -C "$repo" rev-parse HEAD
git -C "$repo" status --porcelain
