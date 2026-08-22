# Multimodal Sentence Transformers checkpoint panel

This is Representax's minimum external checkpoint panel for multimodal model
coverage. It snapshots the models linked by Hugging Face's April 2026
[Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers)
release. Presence here means **acceptance target**, not native support. The
Qwen3-VL, Qwen2.5-Omni, CLIP, and BGE-VL CLIP rows identify checkpoints that
have completed native acceptance; the remaining rows are still targets.

Representax must not implement one bespoke wrapper per checkpoint. A checkpoint
resolves through:

1. a static Sentence Transformers module graph;
2. one reviewed native Equinox backbone family;
3. shared pooling, normalization, projection, routing, or scoring modules;
4. a pinned Hugging Face processor recipe producing the typed model-ready ABI;
5. checkpoint-specific numerical, gradient, export, and systems evidence.

Torch, Transformers forwards, and repository remote code may be installed in
the parity environment as upstream oracles. They are not the Representax
training runtime and never turn a catalogue entry into a support claim.

## Shared graph forms

The linked artifacts reduce to six composition forms:

- dense embedding: `Transformer -> Pooling -> Normalize`;
- direct embedding: a native backbone that already returns
  `sentence_embedding`, followed by optional projection or normalization;
- generative reranking: `Transformer -> LogitScore`;
- feature reranking: `Transformer -> Pooling -> Dense`; and
- legacy `CLIPModel`; and
- routed encoding: modality-specific branches plus shared projection,
  normalization, or pooling.

`load_sentence_transformer_graph()` parses these forms, their modality methods,
structured-message inputs, and serialized Router branches without importing
PyTorch or repository Python. A custom module can therefore be inventoried but
cannot execute. Graph recognition is not a support claim: model-family forward
code remains independently reviewed, registered, and native.

The graph contract is checked against pinned upstream JSON metadata:

| Checkpoint revision | Parsed graph |
|---|---|
| `Qwen/Qwen3-VL-Embedding-2B@9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | dense embedding; text, image, video, and structured messages |
| `Qwen/Qwen3-VL-Reranker-2B@4bd860ac4f15ad1897a214615cccc700f8f71818` | generative reranker over the same input forms |
| `sentence-transformers/clip-ViT-B-32@327ab6726d33c0e22f920c83f2ff9e4bd38ca37f` | legacy CLIP; text and image |
| `BAAI/BGE-VL-base@cc4c733ed997dbee4ac70ccffb911e70c9c24b93` | custom direct embedding; text, image, and composed image-plus-text |

## Official linked variants

| Native family target | Embedding checkpoints | Reranker checkpoints | Modalities |
|---|---|---|---|
| Qwen3-VL | `Qwen/Qwen3-VL-Embedding-2B`, `Qwen/Qwen3-VL-Embedding-8B`, `eagerworks/eager-embed-v1` | `Qwen/Qwen3-VL-Reranker-2B`, `Qwen/Qwen3-VL-Reranker-8B` | text, image, video |
| Qwen2.5-Omni | `LCO-Embedding/LCO-Embedding-Omni-3B`, `LCO-Embedding/LCO-Embedding-Omni-7B`, `Haon-Chen/e5-omni-3B`, `Haon-Chen/e5-omni-7B` | — | text, image, audio, video |
| Llama Nemotron VL | `nvidia/llama-nemotron-embed-vl-1b-v2` | `nvidia/llama-nemotron-rerank-vl-1b-v2` | text, image |
| NVIDIA Omni Embed | `nvidia/omni-embed-nemotron-3b` | — | text, image |
| BidirLM Omni | `BidirLM/BidirLM-Omni-2.5B-Embedding` | — | text, image, audio |
| CLIP and BGE-VL CLIP | `BAAI/BGE-VL-base`, `BAAI/BGE-VL-large`, `sentence-transformers/clip-ViT-L-14`, `sentence-transformers/clip-ViT-B-16`, `sentence-transformers/clip-ViT-B-32` | — | text, image |
| Multilingual CLIP-aligned text tower | `sentence-transformers/clip-ViT-B-32-multilingual-v1` | — | text |
| LLaVA-NeXT retrieval | `BAAI/BGE-VL-MLLM-S1`, `BAAI/BGE-VL-MLLM-S2`, `BAAI/BGE-VL-v1.5-zs`, `BAAI/BGE-VL-v1.5-mmeb`, `royokong/e5-v` | — | text, image |
| Qwen2.5-VL retrieval | `BAAI/BGE-VL-Screenshot` | — | text, image |
| Nomic multimodal | `nomic-ai/nomic-embed-multimodal-3b`, `nomic-ai/nomic-embed-multimodal-7b` | — | text, image |
| Qwen2-VL ranking | — | `jinaai/jina-reranker-m0` | text, image |
| Qwen3 text ranking | — | `Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen3-Reranker-4B`, `Qwen/Qwen3-Reranker-8B`, `ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b` | text |
| Qwen2 text ranking | — | `mixedbread-ai/mxbai-rerank-base-v2`, `mixedbread-ai/mxbai-rerank-large-v2` | text |

## Native Qwen3-VL 2B usage

Both accepted 2B artifacts are Apache-2.0. Loading resolves one immutable
Hugging Face snapshot, constructs the native Equinox model once, and returns
the model-associated processor that turns text, image, video, or their
composition into finite-shape native batches:

```python
from representax.core import Route
from representax.models import Qwen3VLEncoder, Qwen3VLReranker

encoder, processor = Qwen3VLEncoder.load_from_hf(
    "Qwen/Qwen3-VL-Embedding-2B",
    revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
)
batch = processor(
    [{"text": "A tabby cat", "image": image}],
    route=Route.DOCUMENT,
)
representations = encoder.encode(batch, route=Route.DOCUMENT)

reranker, pair_processor = Qwen3VLReranker.load_from_hf(
    "Qwen/Qwen3-VL-Reranker-2B",
    revision="4bd860ac4f15ad1897a214615cccc700f8f71818",
)
pair_batch = pair_processor(
    [{"query": "Which animal?", "document": {"image": image}}],
    route=Route.DOCUMENT,
)
scores = reranker.score(pair_batch)
```

The processor uses the checkpoint's Hugging Face tokenizer and media artifacts,
so it requires `representax[hf]`; it does not require Sentence Transformers or
PyTorch. Host preprocessing admits only configured sequence and patch buckets
before the compiled step. The JAX forward therefore receives clean,
fixed-shape arrays and does not decode media or trigger shape-dependent Python
work.

The shared family has 625 tensors and 2,127,532,032 parameters. Tiny models pass
same-tensor forward, input-gradient, parameter-gradient, and AdamW-update gates
against Transformers 5.3.0. On the real 2B embedding artifact, an FP32
image/text forward reaches 0.9999998 hidden-state cosine and 0.9999999 final
embedding cosine. BF16 throughput probes are useful implementation evidence,
but do not substitute for the matched dataset-to-trained-model jobs required
for a paper claim.

The same forward accepts
`eagerworks/eager-embed-v1@51dfdee0d1d1067afe00d816dca2cd72a02f6bec`.
Its processor preserves the checkpoint's instruction-free user message, left
padding, generation prompt, and terminal token. The real BF16 acceptance gate
reaches 0.99984 cosine for text and 0.99861 for image-plus-text against the
pinned upstream runtime, then completes three finite INT4-LoRA updates. Native
and Hugging Face exports both reload without loading the source checkpoint.

## Native Qwen2.5-Omni usage

The Qwen2.5-Omni family composes text, image, audio, and video without defining
an artificial fused modality. Its model-associated processor owns the exact
chat template, tokenizer, image normalization, video frame packing, Whisper
features, placeholder expansion, multimodal positions, and finite buckets:

```python
from representax.core import Route
from representax.models import Qwen2_5OmniEncoder

model, processor = Qwen2_5OmniEncoder.load_from_hf(
    "LCO-Embedding/LCO-Embedding-Omni-3B-2605",
    revision="5f6b5329da5141367da30e06a9826d1322d6c9b2",
)
batch = processor(
    [{"text": "What is happening?", "video": video, "audio": waveform}],
    route=Route.QUERY,
)
representations = model.encode(batch, route=Route.QUERY)
```

The tiny FP32 oracle matches Transformers 5.6 within `7.2e-7` for hidden
states, `1.1e-6` for image-input gradients, and `5.9e-10` for audio-input
gradients; its three-step AdamW trajectory finishes within `1.2e-5` maximum
parameter error and its native export reloads in Transformers. A deterministic
real processor case matches all 358 token IDs and 100 valid audio frames
exactly. PIL versus Torchvision bicubic video resizing accounts for a bounded
`5.5e-5` relative pixel L2 difference. The full LCO 3B checkpoint executes
native BF16 inference and three generic packed-INT4 LoRA updates on one 24 GB
GPU.

The family also accepts
`nvidia/omni-embed-nemotron-3b@865db1bb57e369a85357cf114cbd6b3c5322d19d`
with its query/document prefixes and masked-mean pooling. Real query and
document embeddings reach at least 0.99935 cosine against Sentence
Transformers 5.6. The pinned runtime currently executes causal text attention
because its custom mask override no longer intercepts the Transformers 5.6
forward; Representax preserves that behavior by default and separately tests
an explicit bidirectional mode against NVIDIA's intended layer contract.

## Native CLIP and BGE-VL usage

One native CLIP family handles ordinary Hugging Face `CLIPModel` checkpoints,
the legacy Sentence Transformers `0_CLIPModel/` layout, and BGE-VL's additive
late-fusion graph. The loader reads the outer Sentence Transformers modules to
preserve whether the checkpoint normalizes its projected representation:

```python
from representax.core import Route
from representax.models import CLIPEncoder

model, processor = CLIPEncoder.load_from_hf(
    "BAAI/BGE-VL-base",
    revision="cc4c733ed997dbee4ac70ccffb911e70c9c24b93",
)
batch = processor(
    [{"image": image, "text": "make the background dark"}],
    route=Route.QUERY,
)
representations = model.encode(batch, route=Route.QUERY)
```

Raw strings and image objects remain convenient inputs, while the same
processor also consumes Representax `Artifact.text(...)` and image artifacts
from a Grain data distribution without constructing an intermediate dataset.
It emits only fixed `input_ids`, masks, and pixel arrays before the compiled
step. Standard CLIP patch projection is expressed as an equivalent nonoverlap
patch extraction plus GEMM rather than a cuDNN convolution.

The tiny FP32 oracle matches Transformers 5.6 within `6.5e-7` for text/image
and composed outputs, `8.9e-9` for image-input gradients, `3.1e-6` for all
parameter gradients, and `1.8e-7` after three matched AdamW updates. Both an
ordinary checkpoint and the legacy nested layout round-trip through the same
398-tensor adapter. On the real 149.6M-parameter BGE-VL base artifact,
tokenizer/image preprocessing is exact and Sentence Transformers cosine is at
least `0.99999958` for text, image, and composed inputs. The real 151.3M
CLIP-B/32 artifact reaches at least `0.99999976` cosine for text and image.
BGE-VL base also completes three compiled BF16-compute/FP32-master AdamW
updates with losses `0.25018 -> 0.14765 -> 0.05490`, followed by exact native
export/reload. Hugging Face export preserves either the root or
`0_CLIPModel/` source layout and its tokenizer/image assets. In the matched
batch-16, sequence-77 FP32 text-forward gate, Representax sustains 3,495
examples/s versus 2,404 for Transformers 5.6 (`1.454x` throughput), with
`9.24e-7` maximum output error and `1.84e-6` relative L2. Its 15.54-second
first compile amortizes after roughly 8,250 repeated steps in this deliberately
small inference workload.

The similarly named `clip-ViT-B-32-multilingual-v1` is not another CLIP
dual-encoder checkpoint: it is a text-only DistilBERT projection trained into
CLIP space. It remains a separate text-family target rather than being falsely
claimed by this implementation.

The Hugging Face article currently lists 20 multimodal embedders, four
multimodal rerankers, six text rerankers, and four legacy CLIP variants. Some
entries use integration PR revisions or repository remote code. Before an
implementation lands, its target must be replaced with an immutable commit SHA,
its actual model and processor layouts must be audited, and its weight license
must be recorded. Jina's reranker is CC BY-NC 4.0 and remains an external oracle
unless its terms are compatible with the intended use.

## Implementation order

1. **Qwen3-VL 2B embedder and reranker — accepted.** One backbone proves
   image/video preprocessing, dense pooling, and tied-token scoring; the 8B
   variants test configuration scaling and FSDP capacity rather than a new
   forward.
2. **Qwen2.5-Omni 3B — accepted.** LCO supplies the first linked
   text/image/audio/video case; the pinned LCO 7B and E5 3B/7B configs reuse
   the same family.
3. **BGE-VL base/large and legacy CLIP — accepted.** These provide inexpensive
   image-text training and processor parity gates suitable for frequent runs.
4. **The remaining architecture families — next.** Add them by shared forward family,
   prioritizing checkpoints that create a new modality, reranking method,
   scaling result, or legally clean comparison.

## Per-checkpoint acceptance

A checkpoint becomes supported only after all applicable gates pass:

- immutable revision, license record, configuration, processor, and tensor-name
  inventory;
- processor parity from the same source records through the model-ready static
  batch ABI;
- same-tensor forward, input-gradient, parameter-gradient, and optimizer-update
  comparisons against the pinned upstream oracle;
- native and source-compatible export with fresh reload;
- isolated compilation, sustained throughput, feasible-workload, live-memory,
  and process-memory evidence; and
- one dataset-to-trained-model job using the same scientific configuration as
  its Sentence Transformers reference where a paired comparison is claimed.

The first paper panel may select a representative subset, but the entire linked
list remains the minimum model-integration backlog.
