# Data artifacts and recipes

Representax separates three concerns:

1. an artifact resolver exposes immutable upstream records through random
   access;
2. user-owned mapping code converts each record into the example contract of a
   task; and
3. Grain applies lazy mapping, deterministic shuffle, and mixture sampling.

The library does not define a materialized Representax dataset format. Mapping
code runs only when Grain accesses a record.

## Hugging Face datasets

An `hf://` URI identifies a dataset repository. Its revision and split must be
explicit; `subset` selects a named dataset configuration when required.

```python
from representax import data

source = data.source(
    "hf://organization/dataset",
    revision="0123456789abcdef",
    split="train",
    subset="english",
    map="my_project.mappers.to_example",
)
dataset = data.build_grain_dataset(data.mix(source, seed=17))
```

The resolver delegates downloading, validation, and the memory-mapped Arrow
cache to Hugging Face Datasets. Streaming `IterableDataset` sources are not yet
part of this random-access `MapDataset` boundary.

## Local artifacts

Plain paths and `file://` URIs support:

- `.jsonl` and `.ndjson` through the Hugging Face JSON loader;
- `.parquet` through the Hugging Face Parquet loader;
- `.arrow` through direct memory mapping with `Dataset.from_file`; and
- local Hugging Face dataset directories.

JSONL and Parquet may be prepared in the standard Hugging Face Arrow cache.
Representax does not create another converted copy or require users to publish
a framework-specific intermediate dataset.

## Mapping code

The `map` field stores a dotted path to a named, importable Python callable.
Passing the callable itself records the same stable path. Anonymous lambdas and
local nested functions are rejected because another process cannot import
them. Building the Grain dataset imports the mapper automatically; an explicit
mapper registry can override it for dependency injection and testing.

Recipes execute Python imports and should therefore be reviewed like other
source code rather than loaded from untrusted parties.

## Extension points

`build_grain_dataset(..., resolvers={"scheme": resolver})` adds or replaces a
resolver. A resolver accepts an `ArtifactSpec` and returns a random-access
source implementing `__len__` and `__getitem__`. S3, streaming sources, media
decoding, shape-aware batching, and distributed iterator state are deliberately
left to later scoped work.
