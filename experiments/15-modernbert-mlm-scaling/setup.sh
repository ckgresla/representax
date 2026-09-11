#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v uv >/dev/null || { echo 'Install uv before running this setup.' >&2; exit 2; }
command -v nvidia-smi >/dev/null
# Reuse the shared lock and editable checkout; no reference-framework environments.
uv sync --project "${root}/experiments" --locked --no-default-groups --group native
