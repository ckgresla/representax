#!/usr/bin/env bash
# One invocation per paired cell, sent to all four TPU workers together.
set -euo pipefail
readonly repo="$HOME/representax-fairness-cdcb104"
readonly environments="$HOME/representax-audio-rerun/experiments/tpu"
readonly output="$HOME/representax-fairness-results"
readonly deadline=1789622967 # 2026-09-17 05:29:27 UTC
readonly recipe=${1:?recipe}
readonly framework=${2:?framework}
readonly seed=${3:?seed}
readonly steps=${4:-22}
readonly phase=${5:-paired}
test "$(date -u +%s)" -lt "$((deadline - 300))"
case "$recipe" in
  audio-text|late-interaction) scope=global ;;
  outcome-reward|process-reward|video-text) scope=local ;;
  *) echo "Recipe not approved for this rerun: $recipe" >&2; exit 2 ;;
esac
export PYTHONPATH="$repo/src:$repo"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export JAX_COMPILATION_CACHE_DIR="$HOME/representax-paper-assets/jax-cache-fairness"
export XLA_PERSISTENT_CACHE_PATH="$HOME/representax-paper-assets/torch-cache-fairness/$recipe"
export XLA_PERSISTENT_CACHE_READ_ONLY=0
unset XLA_USE_BF16 XLA_DOWNCAST_BF16
cd "$repo"
mkdir -p "$output"
timeout --signal=TERM --kill-after=30s 2400 \
  "$environments/.venv-jax/bin/python" \
  experiments/10-cross-accelerator-framework-comparison/run.py suite \
  --platform tpu --recipe "$recipe" --negative-scope "$scope" \
  --asset-root "$HOME/representax-paper-assets" \
  --output "$output/$phase/seed-$seed" --framework "$framework" --seed "$seed" \
  --steps "$steps" --jax-python "$environments/.venv-jax/bin/python" \
  --reference-python "$environments/.venv-torch-xla/bin/python" \
  --late-reference-python "$environments/.venv-torch-xla-late/bin/python"
