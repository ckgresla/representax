"""Cache experiment 14 media and select homogeneous source batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
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
        "positive": {"audio": decode_audio(audio_payload(row), max_seconds=seconds)},
    }


@lru_cache(maxsize=4)
def audio_shard(path):
    from representax.data.resolvers import ParquetSource

    return ParquetSource.open(Path(path), cache_size=1)


def audio_payload(row):
    if "audio" in row:
        return row["audio"]
    return audio_shard(row["audio_parquet"])[row["audio_row"]]["audio"]


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def unique_batch_order(rows, *, seed, batch_size):
    """Seeded full-corpus order, deferring same-media/caption collisions."""
    from collections import deque

    import numpy as np

    pending = deque(int(i) for i in np.random.default_rng(seed).permutation(len(rows)))
    result = []
    while pending:
        batch, media, captions = [], set(), set()
        for _ in range(len(pending)):
            index = pending.popleft()
            row = rows[index]
            if row["media_id"] in media or row["caption"] in captions:
                pending.append(index)
                continue
            batch.append(row)
            media.add(row["media_id"])
            captions.add(row["caption"])
            if len(batch) == batch_size:
                break
        if len(batch) < batch_size:
            # Retain the small tail in the manifest; Grain drops incomplete batches.
            result.extend(batch)
            result.extend(rows[i] for i in pending)
            break
        result.extend(batch)

    def unique(batch):
        return len({r["media_id"] for r in batch}) == len(batch) and len(
            {r["caption"] for r in batch}
        ) == len(batch)

    # A deferred tail may need exchanges with earlier, already complete batches.
    for start in range(0, len(result) - batch_size + 1, batch_size):
        for offset in range(batch_size):
            position = start + offset
            row = result[position]
            others = result[start:position] + result[position + 1 : start + batch_size]
            if all(
                row["media_id"] != r["media_id"] and row["caption"] != r["caption"]
                for r in others
            ):
                continue
            for candidate in range(start):
                other_start = candidate // batch_size * batch_size
                other_batch = result[other_start : other_start + batch_size].copy()
                other_batch[candidate - other_start] = row
                if unique(others + [result[candidate]]) and unique(other_batch):
                    result[position], result[candidate] = result[candidate], row
                    break
        if not unique(result[start : start + batch_size]):
            raise ValueError("could not construct duplicate-free complete batches")
    for start in range(0, len(result) - batch_size + 1, batch_size):
        batch = result[start : start + batch_size]
        if (
            len({r["media_id"] for r in batch}) != batch_size
            or len({r["caption"] for r in batch}) != batch_size
        ):
            raise ValueError("could not construct duplicate-free complete batches")
    return result


def prepare_training(directory: Path, *, assets: Path, seeds=(7, 42, 773)):
    """Index cached media without copying audio/video/image payloads."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    directory.mkdir(parents=True, exist_ok=False)
    pins = json.loads(
        (ROOT / "benchmarks/configs/paper-multimodal-jepa-v1.json").read_text()
    )["datasets"]
    snapshots = {
        name: Path(
            snapshot_download(
                pins[key]["repo_id"],
                repo_type="dataset",
                revision=pins[key]["revision"],
                local_files_only=True,
            )
        )
        for name, key in SOURCES.items()
    }
    image_root = assets / "image-text-convergence"
    images = [
        {
            **row,
            "media_id": str(row["image_id"]),
            "image": str(image_root / row["image"]),
        }
        for row in read_rows(image_root / "train.jsonl")
    ]
    video_root = snapshots["video"]
    video_train = json.loads((video_root / "msrvtt_train_7k.json").read_text())
    video_test = json.loads((video_root / "msrvtt_test_1k.json").read_text())
    assert not (
        {r["video_id"] for r in video_train} & {r["video_id"] for r in video_test}
    )
    videos = [
        {
            "media_id": r["video_id"],
            "caption": caption,
            "video": str(video_root / "raw_videos" / r["video"]),
        }
        for r in video_train
        for caption in dict.fromkeys(r["caption"])
    ]

    def audio_rows(split):
        rows = []
        for path in sorted((snapshots["audio"] / "data").glob(f"{split}-*.parquet")):
            metadata = pq.read_table(
                path, columns=["youtube_id", "start_time", "caption"]
            )
            rows.extend(
                {
                    **r,
                    "media_id": f"{r['youtube_id']}:{r['start_time']}",
                    "audio_parquet": str(path),
                    "audio_row": index,
                }
                for index, r in enumerate(metadata.to_pylist())
            )
        if not rows:
            raise ValueError(f"empty AudioCaps {split}")
        return rows

    audio_train, audio_test = audio_rows("train"), audio_rows("test")
    assert not (
        {r["media_id"] for r in audio_train} & {r["media_id"] for r in audio_test}
    )
    training = {"image": images, "audio": audio_train, "video": videos}
    manifest = {"dataset_pins": pins, "training": {}, "seeds": {}, "evaluation": {}}
    for name, rows in training.items():
        manifest["training"][name] = {
            "pairs": len(rows),
            "media": len({r["media_id"] for r in rows}),
        }
    for seed in seeds:
        sources = {}
        for index, (name, rows) in enumerate(training.items()):
            path = directory / f"{name}-seed-{seed}.jsonl"
            write_rows(path, unique_batch_order(rows, seed=seed + index, batch_size=32))
            sources[name] = str(path)
        sources["text"] = str(assets / f"dense-msmarco-full-unique/seed-{seed}.parquet")
        if not Path(sources["text"]).is_file():
            raise FileNotFoundError(sources["text"])
        manifest["seeds"][str(seed)] = sources
    manifest["text_provenance"] = json.loads(
        (assets / "dense-msmarco-full-unique/manifest.json").read_text()
    )
    manifest["image_provenance"] = json.loads(
        (image_root / "manifest.json").read_text()
    )

    def paired_panel(name, rows, modality):
        ids = {
            r["media_id"]: i
            for i, r in enumerate({r["media_id"]: r for r in rows}.values())
        }
        documents = {r["media_id"]: r for r in rows}
        records = [
            {"kind": "query", "identifier": i, "text": r["caption"]}
            for i, r in enumerate(rows)
        ]
        records += [
            {**r, "kind": "document", "identifier": ids[mid], "modality": modality}
            for mid, r in documents.items()
        ]
        path = directory / f"evaluation-{name}.jsonl"
        write_rows(path, records)
        manifest["evaluation"][name] = {
            "path": str(path),
            "queries": len(rows),
            "documents": len(documents),
            "relevant_documents": {i: [ids[r["media_id"]]] for i, r in enumerate(rows)},
        }

    paired_panel("audiocaps", audio_test, "audio")
    paired_panel(
        "msrvtt",
        [
            {
                "media_id": r["video_id"],
                "caption": r["caption"],
                "video": str(video_root / "raw_videos" / r["video"]),
            }
            for r in video_test
        ],
        "video",
    )
    flickr = [r for r in read_rows(image_root / "evaluation.jsonl") if r["valid"]]
    for row in flickr:
        if row["kind"] == "document":
            row.update(image=str(image_root / row["image"]), modality="image")
    path = directory / "evaluation-flickr30k.jsonl"
    write_rows(path, flickr)
    image_manifest = manifest["image_provenance"]
    manifest["evaluation"]["flickr30k"] = {
        "path": str(path),
        "relevant_documents": image_manifest["relevant_documents"],
        "queries": sum(r["kind"] == "query" for r in flickr),
        "documents": sum(r["kind"] == "document" for r in flickr),
    }
    text_root = assets / "late-data"
    path = directory / "evaluation-nanomsmarco.jsonl"
    text_rows = [
        {**r, "kind": kind, "modality": "text"}
        for filename, kind in (("queries.jsonl", "query"), ("corpus.jsonl", "document"))
        for r in read_rows(text_root / filename)
    ]
    write_rows(path, text_rows)
    manifest["evaluation"]["nanomsmarco"] = {
        "path": str(path),
        "relevant_documents": json.loads((text_root / "qrels.json").read_text()),
        "provenance": json.loads((text_root / "manifest.json").read_text()),
        "queries": sum(r["kind"] == "query" for r in text_rows),
        "documents": sum(r["kind"] == "document" for r in text_rows),
    }
    manifest["file_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob("*.jsonl")
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def map_video(row, *, frames: int):
    from representax.data.media import decode_video

    return {
        "query": row["caption"],
        "positive": {"video": decode_video(row["video"], frames=frames)},
    }


def integration_sources(
    directory: Path, *, assets: Path, sample_count: int = 8
) -> dict[str, str]:
    """Write tiny metadata-only integration inputs; never a convergence dataset.

    Image/video rows use distinct media and captions; all source media stays in
    its existing cache. Audio reads a full original training shard directly.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
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
            if len(selected) == sample_count:
                break
    if len(selected) != sample_count:
        raise ValueError(
            f"integration check requires {sample_count} distinct COCO images"
        )
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
        if len(selected) == sample_count:
            break
    if len(selected) != sample_count:
        raise ValueError(
            f"integration check requires {sample_count} distinct MSR-VTT videos"
        )
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
    text_rows = next(
        pq.ParquetFile(text_file).iter_batches(batch_size=sample_count)
    ).to_pylist()
    if (
        len(text_rows) != sample_count
        or len({row["query"] for row in text_rows}) != sample_count
    ):
        raise ValueError(
            f"integration check requires {sample_count} distinct text queries"
        )
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
