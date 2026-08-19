# Dense representation-system audit

Representax pins Sentence Transformers 5.6.1 as a repository-only behavioral
oracle. Matching its 29 dense losses closes the objective inventory, not the
dense training system. This audit separates capabilities worth expressing in
native JAX abstractions from PyTorch- or Trainer-specific API surface that
Representax should not copy.

## Current boundary

The reviewed 5.6.1 dense surface contains ten task-specific evaluator classes
(`EmbeddingSimilarity`, `BinaryClassification`, `LabelAccuracy`, `Triplet`,
`InformationRetrieval`, `Reranking`, `ParaphraseMining`, `Translation`, `MSE`,
and `NanoBEIR`), an MSE dataframe variant, and sequential composition. It has
four single-dataset batch
policies (ordinary, no-duplicates, hashed no-duplicates, group-by-label), and
two multi-dataset policies (proportional and round-robin). Its `encode` path
also exposes prompts, sentence or token outputs, dimension truncation,
float/int8/uint8/binary output, normalization, chunking, and multi-device
workers. The trainer adds per-dataset loss routing, validation/best-model
selection, callbacks/reporters, checkpoint resume, final save, model cards, and
Hub publication. Those are the concrete upstream capabilities assessed below.

| Capability | Representax status | Decision |
|---|---|---|
| Dense loss functions | Complete | Preserve the task-native registry and paired numerical gates. |
| BERT/MPNet sentence module chains | Partial | Extend native model coverage; keep static, reviewed module graphs. |
| String-to-embedding inference | Partial | Add typed query/document helpers, truncation, quantized outputs, token outputs, and multi-device execution. |
| Local and Hugging Face data artifacts | Partial | Keep Grain recipes; add streaming/cloud resolvers and a first-class preprocessing cache contract. |
| Prompt-aware collation | Partial | Move prompt and route selection into reproducible mapper/collator configuration shared by training and inference. |
| Batch sampling | Missing | Add default, no-duplicate, hashed no-duplicate, and group-by-label policies. |
| Multi-source sampling | Partial | Grain provides weighted mixing; add explicit proportional and round-robin policies plus per-source task/loss routing. |
| Generic training loop | Complete core | One validated `JobConfig` constructs data, model, task, optimizer/schedule, compiled updates, evaluation, checkpointing, and final export. Extend its sampler and routing inventory without adding a second trainer. |
| Evaluators and model selection | Partial | Shared offline/in-training loss and embedding-similarity evaluation, validation cadence, primary metrics, and Orbax best-checkpoint selection are complete; add the remaining released dense evaluator inventory. |
| Offline hard-negative mining | Missing | Add a source-neutral mining transform that emits a new recipe/artifact manifest without hiding data provenance. |
| Final dense artifact export | Partial | Atomic native export and source-compatible Hugging Face export/reload are complete; add prompt/truncation metadata, model cards, and optional Hub publication outside the training core. |
| Reporters and lifecycle hooks | Partial | Keep namespaced asynchronous metrics; add W&B and a small typed lifecycle protocol rather than copying Trainer callbacks. |
| Distributed execution | Single-host complete core | Named DDP/FSDP and custom partition rules share one configured runtime; exact 2/4-device updates, Orbax restore, StableHLO, bucketed DDP gradients, model/layer FSDP materialization, physical memory placement, and NCCL execution are accepted. Complete the larger-workload frontier and defer physical multi-host acceptance until hardware is available. |

## What to add first

1. **Complete the trainer input policies.** Add prompt-aware model-ready mapping,
   batch-sampler policies, and per-source task/loss routing to the accepted
   `JobConfig`-driven runtime.
2. **Complete the evaluator inventory.** Implement embedding similarity,
   binary classification, label accuracy, triplet, information retrieval,
   reranking, paraphrase mining, translation, MSE, NanoBEIR, and sequential
   composition through one evaluator protocol. Validation metrics should use
   `valid/...` names and drive best-checkpoint selection explicitly.
3. **Complete portable-model metadata.** Extend the accepted atomic native and
   Hugging Face export path with prompts, truncation metadata, model cards, and
   optional Hub publication.
4. **Complete the dense inference surface.** Add typed `embed_query` and
   `embed_document` functions, truncation, float/int8/uint8/binary output
   policies, token embeddings, semantic search, and multi-device encoding.
5. **Add data curation utilities.** Implement offline hard-negative mining and
   the planned online JEST selector as reproducible transforms over artifact
   recipes, not opaque mutations of datasets.
6. **Close systems acceptance.** Measure disk-to-final-model wall time, peak
   host/device memory, compile phases, throughput, restart cost, final quality,
   and single-device/device-scaled behavior against the pinned oracle.

Weighted-layer pooling, static embeddings, Router-style heterogeneous towers,
and additional dense module types should be added when a model or task requires
them. They are useful composition primitives, but they should not precede the
trainer, evaluator, and final-artifact path.

## What not to copy

Representax should not reproduce `SentenceTransformerTrainer`, legacy `fit`,
Accelerate/DeepSpeed/FSDP configuration, PyTorch tensor-conversion switches, or
ONNX/OpenVINO export APIs. Their scientific capabilities should map onto Grain,
JAX sharding, Equinox/Optax, Orbax, and portable JAX export mechanisms. The
public API should remain smaller and typed even when the behavior is equivalent.

## Disk-to-final-model benchmark contract

“End to end” means process start with a pinned dataset and initial checkpoint
already present on local disk, through durable publication of the final model.
The timer includes:

1. imports, configuration validation, and model/tokenizer loading;
2. opening raw JSONL/Parquet/Arrow input and executing preprocessing;
3. sampling, collation, host-to-device placement, compilation, and every update;
4. scheduled validation and metric reporting;
5. final asynchronous-checkpoint synchronization;
6. final model export, manifest publication, and reload verification.

The timer stops only after a fresh process can load the final artifact and
reproduce its recorded validation embeddings. Network download time is excluded
by pinning and checksumming both source data and the initial checkpoint before
either arm starts.

The first cold baseline is recorded in
[`benchmarks/results/dense-e2e-20260817`](../benchmarks/results/dense-e2e-20260817/README.org).
MiniLM and Jina Small both satisfy initial-quality and reload-equivalence gates,
but neither yet beats the pinned oracle on complete cold wall time or memory.
Compilation and duplicate source-checkpoint reconstruction during native reload
are measured regressions, so this systems item remains open.

Both frameworks must consume the same ordered example identifiers, tokenizer
revision, prompts/routes, global batch, objective, optimizer and schedule,
precision, number of updates, validation set, and initial weights. Use a
non-zero learning rate and compare final validation quality and parameter/update
statistics in addition to wall time. Record process startup, preprocessing,
compile/first-step, steady training, evaluation, checkpoint/export, total wall
time, host memory, allocator memory, and process-visible device memory as
separate fields.

Two results are useful and must not be conflated:

- **cold preprocessing:** only raw source artifacts are present;
- **warm preprocessing:** a fingerprint-matched preprocessing cache may be
  reused, while compiled-program cache state is reported independently.

The existing ModernVBERT GradCache matrix is a device-side optimizer-step
benchmark. It validates encoder/loss/backward/update systems performance, but it
is not this disk-to-final-model benchmark.
