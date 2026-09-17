"""Freeze duplicate-free pairs from the already pinned MS MARCO source."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.template / "manifest.json").read_text())
    queries, positives, selected = set(), set(), []
    paths = []
    for path in sorted(args.source.glob("*.parquet")):
        paths.append({"path": str(path), "sha256": sha(path)})
        for batch in pq.ParquetFile(path).iter_batches(columns=["query", "positive"]):
            for row in batch.to_pylist():
                query, positive = row["query"], row["positive"]
                if not query.strip() or not positive.strip():
                    continue
                if query in queries or positive in positives:
                    continue
                queries.add(query)
                positives.add(positive)
                selected.append({"query": query, "positive": positive})
                if len(selected) == 512 * 22:
                    break
            if len(selected) == 512 * 22:
                break
        if len(selected) == 512 * 22:
            break
    if len(selected) != 512 * 22:
        raise ValueError("Insufficient unique pairs in the pinned source")
    target = args.output / "train.jsonl"
    with target.open("w") as stream:
        for row in selected:
            stream.write(json.dumps(row) + "\n")
    for name in ("corpus.jsonl", "queries.jsonl", "qrels.json"):
        shutil.copy2(args.template / name, args.output / name)
    manifest.pop("schema_version", None)
    manifest["training"] = {
        "path": target.name, "rows": len(selected), "sha256": sha(target),
        "sources": paths, "duplicate_queries": 0, "duplicate_positives": 0,
        "selection": "source order, first unique query and unique positive; no shuffling",
        "global_batch_size": 512,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["training"], indent=2))


if __name__ == "__main__":
    main()
