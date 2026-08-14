# Activation rematerialization

Activation rematerialization is an execution choice. It changes which forward
intermediates reverse-mode autodiff retains, but it does not change model
parameters, the forward function, the task, or the logical batch.

Representax exposes three stable values through `ExecutionConfig` and the native
ModernVBERT constructors and checkpoint adapters:

- `"none"` leaves the scanned layer body uncheckpointed;
- `"selective"` applies JAX's transformer-oriented
  `dots_with_no_batch_dims_saveable` checkpoint policy; and
- `"full"` applies `nothing_saveable`, replaying all layer intermediates in
  backward.

The public contract deliberately does not accept arbitrary JAX checkpoint
policy callables. JAX documents the custom callable protocol as internal, while
the named policies are its supported surface. See the official
[gradient-checkpointing guide](https://docs.jax.dev/en/latest/gradient-checkpointing.html).

```python
from representax.models.modernvbert import ModernVBERTTextCheckpointAdapter
from representax.planning import ExecutionConfig

plan = ExecutionConfig(
    device_count=1,
    data_axis_size=1,
    per_device_batch_size=8,
    gradient_accumulation_steps=1,
    rematerialization="full",
)
model = ModernVBERTTextCheckpointAdapter().load(
    checkpoint,
    rematerialization=plan.rematerialization,
)
```

## Default

`"full"` is the default. On an RTX 4090, direct FP32 ModernVBERT training at
sequence length 512 showed that `"selective"` improved matched batch-8
throughput by 5.9%, but increased allocator peak from 4.259 to 8.129 GiB and
process peak from 8.402 to 16.402 GiB. Selective execution completed batch 16
but OOMed at batch 32; full execution completed batch 32 at a 7.590 GiB
allocator peak.

The complete protocol and raw results are in
[`modernvbert-rematerialization-20260813`](../benchmarks/results/modernvbert-rematerialization-20260813/README.org).
This default is measured, not universal: a topology-aware planner may select a
different policy when its memory budget and workload shape justify it.
