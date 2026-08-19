# Multimodal Sentence Transformers checkpoint panel

This is Representax's minimum external checkpoint panel for multimodal model
coverage. It snapshots the models linked by Hugging Face's April 2026
[Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers)
release. Presence here means **acceptance target**, not native support.

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

The linked artifacts primarily use four composition forms:

- dense embedding: `Transformer -> Pooling -> Normalize`;
- generative reranking: `Transformer -> LogitScore`;
- feature reranking: `Transformer -> Pooling -> Dense`; and
- dual-tower or routed encoding: modality-specific encoders plus a projection
  into one shared representation space.

Legacy Sentence Transformers CLIP checkpoints use their `CLIPModel` module.
These forms should extend the existing static module-graph loader; model-family
forward code remains independently reviewed and native.

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

The Hugging Face article currently lists 20 multimodal embedders, four
multimodal rerankers, six text rerankers, and four legacy CLIP variants. Some
entries use integration PR revisions or repository remote code. Before an
implementation lands, its target must be replaced with an immutable commit SHA,
its actual model and processor layouts must be audited, and its weight license
must be recorded. Jina's reranker is CC BY-NC 4.0 and remains an external oracle
unless its terms are compatible with the intended use.

## Implementation order

1. **Qwen3-VL 2B embedder and reranker.** One backbone proves image/video
   preprocessing, dense pooling, and `LogitScore`; the 8B variants then test
   configuration scaling and FSDP capacity rather than a new forward.
2. **Qwen2.5-Omni 3B.** LCO supplies the first linked text/image/audio/video
   acceptance case; the 7B and E5 variants should reuse the same family.
3. **BGE-VL base/large and legacy CLIP.** These provide inexpensive image-text
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
