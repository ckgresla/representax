# V-JEPA 2.1 reference contract

Representax treats V-JEPA 2.1 as a distinct, reference-matched task rather than
as an option on LeJEPA. This document freezes the scientific oracle used by the
implementation and parity tests.

## Frozen sources

- Paper: [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised
  Learning](https://arxiv.org/abs/2603.14482), especially equations 1--4.
- Code: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
  commit `204698b45b3712590f06245fbfba32d3be539812` (MIT).
- Paper recipe: `configs/train_2_1/vitb16/pretrain-256px-16f.yaml` at that
  commit.

## Equation-to-code map

| Scientific operation | Paper | Official implementation |
|---|---|---|
| Predict masked target features | Eq. 1 | `app/vjepa_2_1/train.py::loss_fn`, called with `z_pred`, `h`, and `masks_pred` |
| Predict visible/context features | Eq. 2 | the same `loss_fn`, called with `z_context`, `h`, and `masks_enc` |
| Distance-weight context loss | Eq. 3 | `app/vjepa_2_1/models/utils/masks_dist.py::compute_mask_distance` |
| Dense objective | Eq. 4 | `loss_pred + lambda_value * loss_context` in `app/vjepa_2_1/train.py` |
| Deep supervision | Sec. 3.2 | four normalized encoder levels in `models/vision_transformer.py`, fused and projected by `models/predictor.py` |
| Stop-gradient target | Sec. 3.1 | target forward under `torch.no_grad()` in `app/vjepa_2_1/train.py` |
| EMA target update | Sec. 3.1 | post-optimizer `target = m * target + (1-m) * encoder` in `app/vjepa_2_1/train.py` |
| Image/video tokenization | Sec. 3.3 | `PatchEmbed` and `PatchEmbed3D` in `models/utils/patch_embed.py` |
| Multi-block masks | Sec. 3.1 | `src/masks/multiseq_multiblock3d.py::MaskCollator` |

## Frozen base recipe

The reference ViT-B recipe uses 256-pixel crops, 16-frame clips, 16-pixel
patches, two-frame tubelets, BF16 compute, RoPE, modality embeddings, a
12-layer 384-wide predictor, dense prediction, L1 losses, distance-weighted
context prediction with `lambda=0.5`, and an EMA momentum of `0.99925`. Its
four encoder supervision depths are 2, 5, 8, and 11 (zero-indexed).

## Representax acceptance

Reference equivalence means more than implementing a masked video loss. On
identical small tensors and converted weights, tests must match:

1. image and video patch tokens plus selected context/target indices;
2. the four normalized target features and predictor outputs;
3. target and context loss numerators, valid-token denominators, distance
   weights, and total loss;
4. online-model gradients and one optimizer update; and
5. the post-update EMA target parameters.

The native implementation then has to pass a multi-step trajectory, clean
checkpoint/export reload, and 1/2/4-device DDP/FSDP execution. Performance is
measured only after all semantic gates pass.

## Native boundary

`load_vjepa2_1()` is the configuration-facing factory. It constructs the model
and its image or video processor once; `VJEPA2_1Collator` adds the task-owned
multiblock masks. A complete official training checkpoint can be converted with
`VJEPA2_1Model.load_from_reference()` or `load_reference_checkpoint()` without
making Torch a Representax runtime dependency.

The processor retains floating-point pixels through the official random crop,
shared-frame transform, horizontal flip, ImageNet normalization, and bilinear
half-pixel resize. The sampler preserves the official mask distribution and
minimum-valid-token truncation while emitting finite padded tensors and explicit
validity masks for JAX compilation.

Compact acceptance evidence is recorded in
`benchmarks/results/vjepa2-1-acceptance-20260825/`.
