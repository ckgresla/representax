#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 GCS_BUCKET LOCAL_ASSET_ROOT" >&2
  exit 2
fi

bucket=${1%/}
asset_root=$2
experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$experiment_dir/../.." && pwd)
reference="$repository_root/experiments/.references/vjepa2"
python=${REPRESENTAX_EXPERIMENT_PYTHON:-$repository_root/experiments/.venv/bin/python}

command -v gcloud >/dev/null 2>&1 || {
  echo "gcloud is required to stage the immutable paper assets" >&2
  exit 2
}
[[ -d $reference ]] || {
  echo "missing pinned V-JEPA checkout; run experiments/setup.sh first" >&2
  exit 2
}
[[ -x $python ]] || {
  echo "missing experiment environment; run experiments/setup.sh first" >&2
  exit 2
}
[[ $(git -C "$reference" rev-parse HEAD) == 204698b45b3712590f06245fbfba32d3be539812 ]] || {
  echo "the V-JEPA checkout is not at the paper-pinned commit" >&2
  exit 2
}

mkdir -p "$asset_root"

stage() {
  local source=$1
  local destination=$2
  printf '[%s] staging %s\n' "$(date -Is)" "$destination"
  gcloud storage rsync --recursive "$bucket/$source" "$asset_root/$destination"
}

stage models/all-mpnet-base-v2/e8c3b32edf5434bc2275fc9bab85f82640a19130 all-mpnet-base-v2
stage models/bert-base-uncased/86b5e0934494bd15c9632b12f734a8a67f723594 bert-base-source
stage models/ms-marco-MiniLM-L6-v2/233902d25c440f23af6f7d6e94d2946bac0bee0a cross-checkpoint
stage models/GTE-ModernColBERT-v1/cbbe53366e564450558f5e639dd499171f127538 late-checkpoint
stage models/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca qwen3-0.6b
stage models/clip-ViT-B-32/327ab6726d33c0e22f920c83f2ff9e4bd38ca37f clip-vit-b-32
stage models/LCO-Embedding-Omni-3B-2605/5f6b5329da5141367da30e06a9826d1322d6c9b2 omni-3b

stage data/dense-msmarco-unique-v1 dense-msmarco-unique-v1
stage data/semantic-pair pairs
stage data/cross-encoder cross-data
stage data/late-interaction late-data
stage data/outcome-reward outcome-data
stage data/process-reward process-data
stage data/image-text image-data
stage data/audio-text audio-data
stage data/video-text video-data
stage data/vjepa vjepa-data-2816

if [[ ! -f $asset_root/vjepa-data-2816/official-initialization.npz ]]; then
  "$python" -m experiments.preflights.vjepa convert-checkpoint \
    --input "$asset_root/vjepa-data-2816/official-initialization.pth.tar" \
    --output "$asset_root/vjepa-data-2816/official-initialization.npz"
fi
VJEPA_DATA="$asset_root/vjepa-data-2816" "$python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

data = Path(os.environ["VJEPA_DATA"])
source = data / "official-initialization.pth.tar"
converted = data / "official-initialization.npz"
manifest_path = data / "manifest.json"
manifest = json.loads(manifest_path.read_text())

def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


source_hash = sha256(source)
if manifest["files"][source.name] != source_hash:
    raise RuntimeError("staged V-JEPA initialization does not match its manifest")
manifest["files"][converted.name] = sha256(converted)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

BERT_SOURCE="$asset_root/bert-base-source" BERT_OUTPUT="$asset_root/bert-base" \
  "$python" - <<'PY'
import os
import shutil
from pathlib import Path

from sentence_transformers import SentenceTransformer

source = Path(os.environ["BERT_SOURCE"])
output = Path(os.environ["BERT_OUTPUT"])
temporary = output.with_name(output.name + ".tmp")
shutil.rmtree(temporary, ignore_errors=True)
model = SentenceTransformer(str(source), device="cpu", local_files_only=True)
model.save(str(temporary))
shutil.rmtree(output, ignore_errors=True)
temporary.rename(output)
if not (output / "modules.json").is_file():
    raise RuntimeError("prepared BERT sentence-transformer bundle has no modules.json")
PY

if [[ ! -e $asset_root/vjepa2-reference ]]; then
  ln -s "$reference" "$asset_root/vjepa2-reference"
fi
[[ $(git -C "$asset_root/vjepa2-reference" rev-parse HEAD) == 204698b45b3712590f06245fbfba32d3be539812 ]]
