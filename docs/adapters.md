# Low-bit adapters

`QuantizedLoRAConfig` enables adapter-only training from one ordinary native
model recipe:

```python
from representax.config import QuantizedLoRAConfig, TrainingConfig

training = TrainingConfig(
    # other scientific and execution fields omitted
    adapter=QuantizedLoRAConfig(
        rank=8,
        alpha=16.0,
        target_pattern=r"layers.*(attention|mlp)",
    ),
)
```

Every matching native `Linear` is transformed structurally, including stacked
linear leaves consumed by scanned transformer layers. Base weights use signed,
per-output-row symmetric INT4 quantization and two nibbles per `uint8`; BF16
scale bit patterns and optional bias bit patterns remain integer leaves. They
are therefore frozen naturally and never receive gradients or Optax state.
Only FP32 `lora_a` and `lora_b` leaves are selected by the trainer.

The packed weight is dequantized at the projection-use boundary and the matrix
product runs in the configured compute dtype—normally BF16. This is QLoRA-style
weight-quantized adapter training, **not** native INT4 gradient arithmetic. FSDP
shards the packed integer state and gathers the compact representation before
dequantization; DDP simply replicates it. The same canonical train step handles
both strategies through its general trainable-parameter filter.

Native inference bundles preserve the compact model exactly. Optional Hugging
Face export merges the trained low-rank update into an ordinary FP32 `Linear`
weight and runs the existing exact checkpoint reload gate. No original full
linear weight is retained in the native bundle.

## Physical acceptance

The matched ModernVBERT point uses two RTX 4090s, global batch 8, sequence 512,
exact GradCache chunks of 2, full activation rematerialization, rank 8, alpha
16, one retained warmup update, and three measured updates.

| model state | strategy | examples/s | median step | peak live JAX/device |
|---|---:|---:|---:|---:|
| full BF16 training | DDP | 35.91 | 0.223 s | 3.872 GiB |
| INT4 base + LoRA-8 | DDP | 39.15 | 0.204 s | 0.507 GiB |
| full BF16 training | FSDP | 9.93 | 0.806 s | 2.052 GiB |
| INT4 base + LoRA-8 | FSDP | 30.30 | 0.264 s | 0.410 GiB |

The transformed model has 1,689,600 trainable adapter parameters. Its four
stacked packed linear groups occupy 55,148,544 bytes plus 270,336 bytes of BF16
scale bit patterns. DDP is preferable for this now-small state on the measured
PCIe node; FSDP exists for larger capacity frontiers and correctly shards both
packed base and adapter leaves.

These short synthetic-token jobs establish runtime, layout, and capacity
behavior—not downstream quality. Two-/four-device DDP/FSDP tests separately
match ten unsharded optimizer updates within reduction-order tolerance.
