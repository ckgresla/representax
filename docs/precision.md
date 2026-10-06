# Mixed precision

`TrainingConfig.precision` is an execution policy, not a scientific model
change. The default is FP32 throughout. `PrecisionConfig.bfloat16_mixed()` uses
the conventional training split:

| boundary | dtype |
|---|---|
| persistent parameters and checkpoints | FP32 |
| Optax state | FP32 |
| forward parameter view and activations | BF16 |
| matrix operands | BF16 |
| representation and GradCache accumulation | FP32 |
| task losses and metrics | FP32 |

```python
from representax.config import PrecisionConfig, TrainingConfig

training = TrainingConfig(
    # scientific and other execution fields omitted
    precision=PrecisionConfig.bfloat16_mixed(),
)
```

The compiled step retains one FP32 master model. A transient BF16 view is made
inside the model-use boundary, rather than before the task or before GradCache.
This placement matters: GradCache's replay `lax.scan` therefore accumulates
cotangents into FP32 master leaves instead of carrying a BF16 parameter-gradient
buffer. `representax.core.encode` casts only model inputs; labels, relation
weights, teacher targets, and other task-owned floating data are not silently
downcast. Encoder outputs cross back to FP32 before the objective.

Native encoder families may retain an inference-time `compute_dtype` default.
The active training policy overrides it while the compiled step is traced, so
an FP32 checkpoint can be trained with BF16 compute without reconstructing the
model. Shared linear heads and reconstruction parameters use the same active
policy outside the encoder entry point. A custom task that bypasses both the
shared `encode` contract and shared numerical components owns its precision
boundary explicitly.

FSDP stores FP32 parameter and Optax shards. Conversion occurs before each
shared parameter-use materialization, so the forward StableHLO requests BF16
replication while autodiff returns FP32 master gradients. Backend optimization
may rewrite the physical collective for a particular device; accelerator
acceptance therefore inspects the compiled HLO in addition to the portable
StableHLO and measures live memory and throughput.

BF16 is the recommended mixed mode. It has FP32-like exponent range and does not
require dynamic loss scaling.

## FP8 matrix compute

`PrecisionConfig.float8_mixed()` is an experimental matrix policy. Parameters,
FSDP communication, residuals, normalization, softmax, and nonlinearities remain
BF16; only owned linear products use dynamically scaled FP8 operands with FP32
accumulation. The custom VJP uses the conventional hybrid format: E4M3 operands
in the forward program and E5M2 output cotangents in the backward program. FP32
masters, Optax state, final gradients, representations, and losses are unchanged.

On RTX 4090, optimized HLO proves native `__cublas$lt$matmul$f8` lowering. An
isolated 4096-cubed GEMM is 1.37x faster than BF16, but the complete 149M
ModernVBERT DDP job is 34.89 examples/s versus BF16's 35.91 examples/s because
per-use scaling offsets the raw GEMM gain at this size. Under FSDP, the SPMD
partitioner moves weight `amax` computations before gathers and introduces 28
additional scale AllReduces; throughput is therefore only 6.40 examples/s.

FP8 is supported as an evidence-backed experimental policy, not the default or
a performance claim. The next optimization is persistent delayed-scale state
with coalesced scale updates, followed by the same numerical, HLO, memory, and
physical-throughput gates. MXFP8 and NVFP4 block-scaled hardware paths are not
claimed on Ada GPUs.

Four-bit frozen-weight adapter training is documented separately in
[Low-bit adapters](adapters.md). Quantized optimizer state and selective
per-parameter compute exceptions remain future policies.

## Vectorization audit

The precision slice also audited Python loops that participate in numerical
programs. Homogeneous distribution-distillation candidate scoring now uses one
`jax.vmap`. The following loops remain deliberately:

- constructor and checkpoint mapping loops do not enter compiled programs;
- pooling modes and postprocessors are heterogeneous static compositions;
- guided negatives and distillation payload columns may be different PyTrees or
  routes, so stacking them would weaken their contract;
- Matryoshka dimensions produce different output shapes; and
- the legacy vision-layer tuple remains a future stack-and-`lax.scan` model
  refactor, while BERT, MPNet, ModernVBERT text, and Jina already scan their
  homogeneous layer stacks.

`vmap` is used where it expresses a batch axis. Existing batched matrix,
attention, and loss primitives are not wrapped merely to make the source look
more vectorized.
