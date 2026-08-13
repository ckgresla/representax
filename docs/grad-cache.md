# Exact GradCache execution

GradCache is an execution schedule for a representation objective. It is not a
second loss. In Representax, direct and cached MNR both evaluate `MNRTask` and
`mnr_loss_terms`; the execution choice only determines which encoder
activations survive until the backward pass.

```python
from representax.tasks.retrieval import MNRTask
from representax.train import GradCache, build_train_step

step = build_train_step(
    MNRTask(scale=20.0, symmetric=True),
    optimizer,
    execution=GradCache(
        query_chunk_size=16,
        document_chunk_size=8,
    ),
)
```

The logical batch and negative population do not change. The query and document
chunk sizes are topology-dependent execution choices.

## Algorithm

For each query and document role, the native JAX path:

1. pads the execution batch to a static number of chunks;
2. encodes those chunks with `lax.scan`;
3. wraps the scan body in `jax.checkpoint` with `nothing_saveable`;
4. evaluates the ordinary full-batch MNR function from the representations; and
5. lets JAX's transposition replay one encoder chunk at a time during backward.

Only the compact representations remain live across the forward scan. Explicit
per-role and per-chunk PRNG keys make a replay of a stochastic encoder use the
same random values as its graphless forward. The resulting parameter gradient
then enters the same clipping, finite-check, Optax update, and logging path used
by direct training.

This is the compact JAX form of the three-stage algorithm in Gao et al.:
graphless representations, representation cotangents from the full contrastive
loss, then encoder replay with those cotangents. It avoids PyTorch backward
hooks and avoids maintaining a second cached MNR formula.

## Reference and performance contract

The correctness oracle is direct Representax MNR. Tests compare the loss,
metrics, parameter gradients through their global norm and clipped update,
optimizer state, and updated parameters. Separate replay tests cover stochastic
encoders and partial padded chunks.

The external reference and target to beat is Sentence Transformers'
`CachedMultipleNegativesRankingLoss`, pinned by version or Git revision in each
benchmark artifact. A matched comparison must use the same checkpoint, inputs,
pooling, scale, precision, optimizer, logical batch, negative population, and
encoder chunk geometry. It reports compile-plus-first execution separately from
warmup and steady-state measurements, along with the OOM boundary, throughput,
framework allocator peak, and process GPU-memory peak.

Performance regressions warn rather than masquerading as numerical failures.
The roadmap item is not complete, however, until Representax fits every matched
batch the pinned reference fits and has better steady-state throughput and peak
memory on the primary workload.

## Pallas boundary

Pallas is not used to orchestrate encoder replay: an arbitrary Equinox encoder
forward and backward remain ordinary JAX. If profiling shows that the
representation objective becomes material at very large logical batches, a
later tiled custom VJP may implement normalized similarity, streaming
log-sum-exp, and representation cotangents in Pallas. That kernel must retain a
native-JAX fallback and match the canonical MNR oracle before it is benchmarked.

## Sources

- [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983)
- [Sentence Transformers GradCache engine](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/base/losses/gradcache.py)
- [Sentence Transformers cached MNR](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/losses/cached_multiple_negatives_ranking.py)
- [Original GradCache JAX transform](https://github.com/luyug/GradCache/blob/main/src/grad_cache/cachex/functional.py)
