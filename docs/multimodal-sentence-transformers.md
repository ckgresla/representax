# Multimodal Sentence Transformers checkpoint panel

This is Representax's minimum external checkpoint panel for multimodal model
coverage. It snapshots the models linked by Hugging Face's April 2026
[Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers)
release. Presence here means **acceptance target**, not native support;
Qwen3-VL 2B embedding/reranking and Qwen2.5-Omni LCO 3B are the first entries
to complete their native acceptance slices.

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
| CLIP and BGE-VL CLIP | `BAAI/BGE-VL-base`, `BAAI/BGE-VL-large`, `sentence-transformers/clip-ViT-L-14`, `sentence-transformers/clip-ViT-B-16`, `sentence-transformers/clip-ViT-B-32`, `sentence-transformers/clip-ViT-B-32-multilingual-v1` | — | text, image |
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
3. **BGE-VL base/large and legacy CLIP — next.** These provide inexpensive image-text
   training and processor parity gates suitable for ordinary CI artifacts.
4. **The remaining architecture families.** Add them by shared forward family,
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
