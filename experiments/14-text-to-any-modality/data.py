"""Cache experiment 14 media and select homogeneous source batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"audio": "audiocaps-train", "video": "msrvtt-train"}


def map_image(row):
    from representax.data.media import decode_image

    return {"query": row["caption"], "positive": {"image": decode_image(row["image"])}}


def map_audio(row, *, seconds: float):
    from representax.data.media import decode_audio

    return {
        "query": row["caption"],
        "positive": {"audio": decode_audio(row["audio"], max_seconds=seconds)},
    }


def map_video(row, *, frames: int):
    from representax.data.media import decode_video

    return {
        "query": row["caption"],
        "positive": {"video": decode_video(row["video"], frames=frames)},
    }


def integration_sources(directory: Path, *, assets: Path) -> dict[str, str]:
    """Write tiny metadata-only integration inputs; never a convergence dataset.

    Image/video rows use distinct media and captions; all source media stays in
    its existing cache. Audio reads a full original training shard directly.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (ROOT / "benchmarks/configs/paper-multimodal-jepa-v1.json").read_text()
    )["datasets"]
    snapshots = {}
    for name, key in SOURCES.items():
        spec = manifest[key]
        snapshots[name] = Path(
            snapshot_download(
                spec["repo_id"],
                repo_type="dataset",
                revision=spec["revision"],
                local_files_only=True,
            )
        )

    def write(name, rows):
        path = directory / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return str(path)

    selected = []
    seen_ids, seen_captions = set(), set()
    image_root = assets / "image-text-convergence"
    with (image_root / "train.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["image_id"] in seen_ids or row["caption"] in seen_captions:
                continue
            seen_ids.add(row["image_id"])
            seen_captions.add(row["caption"])
            row["image"] = str(image_root / row["image"])
            selected.append(row)
            if len(selected) == 8:
                break
    if len(selected) != 8:
        raise ValueError("integration check requires eight distinct COCO images")
    paths = {"image": write("image", selected)}

    selected = []
    seen_ids, seen_captions = set(), set()
    for row in json.loads((snapshots["video"] / "msrvtt_train_7k.json").read_text()):
        caption = row["caption"][0]
        if row["video_id"] in seen_ids or caption in seen_captions:
            continue
        seen_ids.add(row["video_id"])
        seen_captions.add(caption)
        selected.append(
            {
                "video_id": row["video_id"],
                "caption": caption,
                "video": str(snapshots["video"] / "raw_videos" / row["video"]),
            }
        )
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("integration check requires eight distinct MSR-VTT videos")
    paths["video"] = write("video", selected)
    audio_path = sorted((snapshots["audio"] / "data").glob("train-*.parquet"))[0]
    audio_table = pq.read_table(
        audio_path, columns=["youtube_id", "start_time", "caption"]
    )
    audio_rows = audio_table.to_pylist()
    if len({(r["youtube_id"], r["start_time"]) for r in audio_rows}) != len(audio_rows):
        raise ValueError("integration audio shard has duplicate clips")
    if len({r["caption"] for r in audio_rows}) != len(audio_rows):
        raise ValueError("integration audio shard has duplicate captions")
    paths["audio"] = str(audio_path)
    text_file = assets / "dense-msmarco-full-unique/seed-7.parquet"
    text_rows = next(pq.ParquetFile(text_file).iter_batches(batch_size=8)).to_pylist()
    paths["text"] = write("text", text_rows)
    (directory / "sources.json").write_text(
        json.dumps(
            {
                "purpose": "integration-only; not the nine-run training recipe",
                "sources": paths,
                "dataset_pins": {
                    key: manifest[key]
                    for key in ("coco2017-train", "audiocaps-train", "msrvtt-train")
                },
                "text_provenance": json.loads(
                    (text_file.parent / "manifest.json").read_text()
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return paths


def download(source: str, *, cache_dir: Path | None = None) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    manifest = json.loads(
        (ROOT / "benchmarks/configs/paper-multimodal-jepa-v1.json").read_text()
    )
    spec = manifest["datasets"][SOURCES[source]]
    path = snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        revision=spec["revision"],
        cache_dir=cache_dir,
        max_workers=2,
    )
    return {
        "source": source,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "snapshot": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=tuple(SOURCES), nargs="+", default=list(SOURCES)
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Default: the shared HF Hub cache"
    )
    args = parser.parse_args()
    for source in args.source:
        print(json.dumps(download(source, cache_dir=args.cache_dir)), flush=True)


if __name__ == "__main__":
    main()
