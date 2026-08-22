# Data sources, artifacts, samples, and loaders

Representax uses Grain as its dataset and iterator implementation. It does not
define or materialize a framework-specific dataset format. The public concepts
describe reproducible configuration and the boundary between raw records and a
model-ready task batch.

## Sources and distributions

A `DataSourceConfig` identifies one immutable dataset source and the importable
mapping code that converts each row into a task sample. `hf://` sources require
an explicit revision and split. Local paths and `file://` sources support JSONL,
Parquet, Arrow, and Hugging Face dataset directories.

`DataDistributionConfig` is the sampling policy over those sources. A single
dataset is the one-source case; there is no separate recipe type.

```python
from representax import data

distribution = data.mix(
    data.source(
        "hf://organization/dataset",
        revision="0123456789abcdef",
        split="train",
        map="my_project.mappers.to_sample",
        name="remote",
    ),
    data.source(
        "file:///data/local.jsonl",
        map="my_project.mappers.to_sample",
        name="local",
    ),
    weights=(0.8, 0.2),
    seed=17,
)
dataset = data.build_dataset(distribution)
```

`build_dataset` returns a native Grain `MapDataset`. Grain performs lazy
mapping, deterministic shuffle, and weighted mixing directly.

## Artifacts and samples

An `Artifact` is one raw model-input leaf. It contains either inline data or an
immutable lazy URI reference:

```python
from dataclasses import dataclass

from representax import data
from representax.core import Modality


@dataclass(frozen=True)
class RetrievalSample:
    query: object
    document: object


sample = RetrievalSample(
    query={"instruction": data.Artifact.text("find the matching image")},
    document=data.Artifact.ref(
        Modality.IMAGE,
        uri="s3://images/00042.jpg",
        revision="dataset-v2",
        etag='"immutable-object-etag"',
        metadata={"width": 1024, "height": 768},
    ),
)
```

Lazy references preserve the identity and selectors needed to avoid an
intermediate media dataset: URI, revision or ETag, optional archive member,
an optional `[start, stop)` byte range, checksum, and probe metadata. The
checksum is over the selected bytes returned to the decoder. Local files,
ZIP/TAR members, and HTTP(S) ranges use the built-in `read_artifact()` boundary;
additional URI schemes supply one byte reader:

```python
payload = data.read_artifact(
    artifact,
    readers={"s3": read_s3_object_range},
)
```

HTTP range reads fail closed if the server ignores the requested range, and
ETag/checksum mismatches abort preprocessing. Remote archives similarly require
an indexed scheme reader rather than silently downloading the whole shard.

A sample is one task-specific training unit. Representax deliberately does not
impose a universal `Sample` base class: retrieval, classification, reward, and
JEPA samples should use fields natural to their scientific contracts. A sample
may contain one artifact or a tree of named artifacts. Text-image or other
multimodal fusion is composition of atomic artifacts, not a `fused` modality.

## Models and processors

A model loader returns the native Equinox model and its host-side processor:

```python
from representax.core import Route, encode
from representax.models import SentenceEncoder

model, processor = SentenceEncoder.load_from_hf(checkpoint)
model_inputs = processor(samples, route=Route.QUERY, seed=17)
representations = encode(model, model_inputs, route=Route.QUERY)
```

The processor travels with the model checkpoint in the same spirit as Hugging
Face processors, but it is native Representax code. It owns tokenization,
media selection and decoding, model-specific normalization, padding, special
tokens, and construction of the model's fixed-shape Equinox batch. It stays in
the host data path and never enters `jax.jit`.

Every processor implements `data_contract()` and admits a finite set of
model-native shapes. Data-dependent work happens first; the result is then
padded, cropped, sampled, or resampled into one admitted bucket. Consequently,
JAX can compile only the finite set of signatures declared by the processor,
rather than encountering an unbounded stream of media-dependent shapes. The
contract is included in the loader fingerprint, so changing bucket policy makes
an old data cursor incompatible with resume.

For text models, bucket selection occurs after route prompts and special tokens
are applied. Existing behavior remains one fixed maximum unless the model
configuration opts into several lengths:

```python
model = ModelConfig(
    target="representax.models.sentence:SentenceEncoder.load_from_hf",
    parameters={
        "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "<immutable-revision>",
        "sequence_length_buckets": [128, 256, 512],
    },
)
```

The sentence processor tokenizes each batch once, selects the smallest
admissible length, and pads the NumPy output directly. Inputs beyond the largest
bucket are deterministically truncated there. Image, audio, and video processors
use the same admission primitive for shapes such as `(height, width)`, samples,
or `(frames, height, width)`. The shared helper rejects incomparable choices—for
example, more frames versus higher resolution—because that tradeoff belongs to
the model-specific processor rather than a generic memory heuristic.

Media processors use one callable-based boundary rather than a framework of
adapter classes:

```python
from representax.models import make_image_processor

processor = make_image_processor(
    admitted_shapes=((224, 224), (336, 336)),
    probe=probe_image_requirement,
    prepare=decode_resize_normalize_and_pad,
    batch_builder=ImageBatch,
    configuration={
        "resize": "shortest-edge-center-crop",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
)
```

`probe(artifact, route=...)` inspects immutable metadata without decoding and
returns the shape required by that model policy. Representax selects one bucket
for the complete batch. Only then does
`prepare(artifact, bucket=..., route=..., rng=...)` resolve bytes, decode,
select content, transform it, and emit fixed-shape NumPy leaves. Finally,
`batch_builder(**stacked_arrays)` constructs the model-native JAX batch.

The audio and video constructors use the same execution path, with conventional
shape meanings `(samples,)` and `(frames, height, width)`. A supplied seed is
split deterministically per artifact for reproducible audio windows or video
frame selection. Processor contracts fingerprint the finite shapes, callable
implementations, and the complete JSON preprocessing configuration. The actual
resize, normalization, resampling, tiling, and sampling policy remains with the
model integration where it can match that model's saved artifacts exactly.

Task collators may accept an injected `processor` constructor argument. This lets
them apply the model processor to the artifact fields and then assemble labels,
relations, masks, or other task-owned batch state without duplicating processor
configuration in `DataConfig`. For example, the native Sentence Transformers
loader returns one model/processor pair, and the retrieval collator reuses that
processor rather than loading a second tokenizer:

```python
from representax.config import ComponentConfig, DataConfig, ModelConfig

model = ModelConfig(
    target="representax.models.sentence:SentenceEncoder.load_from_hf",
    parameters={
        "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "<immutable-revision>",
    },
)
training_data = DataConfig(
    distribution=distribution,
    collate=ComponentConfig(
        target="representax.tasks.retrieval.RetrievalCollator",
        parameters={"query_field": "query", "document_field": "positive"},
    ),
)
```

The job builder calls `load_model()` once, passes the model through
`prepare_model()`, and injects the processor only when the collator declares a
`processor` parameter.

## Data loaders

`build_data_loader` applies Grain's native batching and read prefetch and returns
a thin iterable carrying the exact batch-size and reproducibility contracts the
trainer needs:

```python
loader = data.build_data_loader(
    distribution,
    batch_size=32,
    batch_fn=collate_samples,
    drop_remainder=True,
    num_threads=16,
    prefetch_buffer_size=2,
)
```

Advanced Python callers can start from an existing Grain dataset directly:

```python
loader = data.build_data_loader(
    grain_dataset,
    batch_size=32,
    batch_fn=collate_samples,
    data_contract={"name": "project-stream", "revision": "v3"},
)
```

The explicit contract is required because Representax cannot infer the source,
mapping, or preprocessing semantics of an arbitrary live Python object. A
configured distribution records these automatically.

The loader fingerprint includes source revisions, distribution and sampling
policy, mapper/resolver implementations, batch mapper, batch size, remainder
policy, and Grain version. Grain's native `get_state` and `set_state` are stored
in each training checkpoint, so resume seeks directly to the next logical
batch instead of replaying preprocessing.

## Extension boundary

Additional URI schemes plug in as source resolvers returning a random-access
object accepted by `grain.MapDataset.source`. Model-specific media processing
belongs to the model-associated processor rather than the source resolver. The
source supplies rows and lazy references; the processor determines how those
artifacts become model inputs.
