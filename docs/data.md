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
Install `representax[hf]` to use this resolver and the built-in local
JSONL/Parquet/Arrow resolvers backed by Hugging Face Datasets.

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

## Training iterators

`build_grain_iterator` adds static batching and Grain's native threaded
prefetch to a recipe. The user-provided `batch_fn` owns collation into the
task's model-ready batch type; the trainer does not impose a generic example
schema.

```python
from representax.data import build_grain_iterator

batches = build_grain_iterator(
    recipe,
    batch_size=32,
    batch_fn=collate_retrieval_examples,
    drop_remainder=True,
    num_threads=16,
    prefetch_buffer_size=2,
)
```

Dropping the remainder is the default because a stable batch shape avoids an
otherwise-surprising final compilation. Exhausting a finite iterator before
`ScientificConfig.max_steps` is a training failure, not silent early completion.
The returned source exposes its exact `global_batch_size`, allowing the trainer
to reject a mismatch with `ScientificConfig` before creating a run.

It also exposes a complete `data_contract` and `data_fingerprint`. The contract
contains the artifact recipe and revisions, declared mapper paths, digests of
the resolved mapper and resolver modules, the batch mapper implementation,
batch size and remainder policy, and the Grain version. Checkpoint resume
requires an exact fingerprint match, so edited preprocessing cannot interpret
an old cursor under a different record stream.

The Grain iterator implements `get_state` and `set_state`; Representax stores
that native state in each training checkpoint. Restore seeks directly to the
next record instead of iterating through or repeating earlier preprocessing,
including when the input pipeline had prefetched ahead.

## Extension points

`build_grain_dataset(..., resolvers={"scheme": resolver})` adds or replaces a
resolver. A resolver accepts an `ArtifactSource` and returns a random-access
source implementing `__len__` and `__getitem__`. S3, streaming sources, media
decoding, shape-aware batching, and distributed iterator state are deliberately
left to later scoped work.
