"""Regenerate the frozen V-JEPA 2.1 PyTorch oracle fixture.

This program is repository-only parity tooling. It must be run against the
official facebookresearch/vjepa2 checkout frozen in the adjacent JSON manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


def run(reference: Path, output: Path) -> None:
    sys.path.insert(0, str(reference))
    import torch
    import torch.nn.functional as functional
    from app.vjepa_2_1.models.predictor import VisionTransformerPredictor
    from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
    from app.vjepa_2_1.models.vision_transformer import VisionTransformer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)

    def norm(size):
        return torch.nn.LayerNorm(size, eps=1e-6)

    encoder = VisionTransformer(
        img_size=(8, 8),
        patch_size=4,
        num_frames=4,
        tubelet_size=2,
        in_chans=3,
        embed_dim=12,
        depth=12,
        num_heads=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=norm,
        use_rope=True,
        interpolate_rope=False,
        img_temporal_dim_size=1,
        modality_embedding=True,
        n_output_distillation=4,
        use_sdpa=False,
    )
    predictor = VisionTransformerPredictor(
        img_size=(8, 8),
        patch_size=4,
        num_frames=4,
        tubelet_size=2,
        embed_dim=12,
        predictor_embed_dim=12,
        out_embed_dim=12,
        depth=4,
        num_heads=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=norm,
        use_rope=True,
        interpolate_rope=False,
        img_temporal_dim_size=1,
        modality_embedding=True,
        n_output_distillation=4,
        use_sdpa=False,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        return_all_tokens=True,
    )
    target = copy.deepcopy(encoder)
    encoder.eval()
    predictor.eval()
    target.eval()

    pixels = torch.randn(2, 3, 4, 8, 8)
    context_ids = torch.tensor([[0, 1, 4, 5], [0, 2, 4, 6]])
    target_ids = torch.tensor([[2, 3, 6, 7], [1, 3, 5, 7]])
    image_pixels = torch.randn(2, 3, 1, 8, 8)
    image_context_ids = torch.tensor([[0, 1], [0, 2]])
    image_target_ids = torch.tensor([[2, 3], [1, 3]])
    raw_resize_frames = torch.arange(2 * 5 * 7 * 3, dtype=torch.float32).reshape(
        2, 5, 7, 3
    )
    resized_frames = functional.interpolate(
        raw_resize_frames.permute(3, 0, 1, 2),
        size=(8, 8),
        mode="bilinear",
        align_corners=False,
    ).permute(1, 2, 3, 0)

    def objective():
        with torch.no_grad():
            target_features = target(pixels, training=True)
            target_features = torch.cat(
                tuple(
                    functional.layer_norm(
                        target_features[..., start : start + 12],
                        (12,),
                        eps=1e-6,
                    )
                    for start in range(0, 48, 12)
                ),
                dim=-1,
            )
        context_features = encoder(pixels, masks=context_ids, training=True)
        predicted_target, predicted_context = predictor(
            context_features,
            context_ids,
            target_ids,
            mod="video",
        )
        target_for_prediction = torch.stack(
            tuple(target_features[index, target_ids[index]] for index in range(2))
        )
        target_for_context = torch.stack(
            tuple(target_features[index, context_ids[index]] for index in range(2))
        )
        distance = compute_mask_distance(
            [[target_ids]],
            [[context_ids]],
            grid_size=2,
            offset_context_loss=False,
        )[0][0]
        prediction_loss = torch.mean(
            torch.abs(predicted_target - target_for_prediction)
        )
        context_loss = torch.mean(
            torch.abs(predicted_context - target_for_context)
            * (1.0 / distance.unsqueeze(-1))
        )
        return (
            prediction_loss + 0.5 * context_loss,
            target_features,
            context_features,
            predicted_target,
            predicted_context,
            distance,
            prediction_loss,
            context_loss,
        )

    initial_encoder = {
        name: value.detach().cpu().numpy().copy()
        for name, value in encoder.state_dict().items()
    }
    initial_predictor = {
        name: value.detach().cpu().numpy().copy()
        for name, value in predictor.state_dict().items()
    }
    image_encoder = copy.deepcopy(encoder)
    image_predictor = copy.deepcopy(predictor)
    optimizer = torch.optim.AdamW(
        (*encoder.parameters(), *predictor.parameters()),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    loss, target_features, context_features, pred, context, distance, lp, lc = (
        objective()
    )
    loss.backward()
    gradients = {
        "gradient_encoder_patch": encoder.patch_embed.proj.weight.grad,
        "gradient_encoder_qkv0": encoder.blocks[0].attn.qkv.weight.grad,
        "gradient_predictor_qkv0": predictor.predictor_blocks[0].attn.qkv.weight.grad,
        "gradient_predictor_projection": predictor.predictor_proj.weight.grad,
    }
    optimizer.step()
    with torch.no_grad():
        for online, target_parameter in zip(
            encoder.parameters(), target.parameters(), strict=True
        ):
            target_parameter.mul_(0.9).add_(online, alpha=0.1)
        image_context_features = image_encoder(
            image_pixels,
            masks=image_context_ids,
            training=True,
        )
        image_predicted_target, image_predicted_context = image_predictor(
            image_context_features,
            image_context_ids,
            image_target_ids,
            mod="image",
        )

    arrays = {
        "pixels": pixels,
        "context_ids": context_ids,
        "target_ids": target_ids,
        "image_pixels": image_pixels[:, :, 0],
        "image_context_ids": image_context_ids,
        "image_target_ids": image_target_ids,
        "image_context_features": image_context_features,
        "image_predicted_target": image_predicted_target,
        "image_predicted_context": image_predicted_context,
        "raw_resize_frames": raw_resize_frames,
        "resized_frames": resized_frames,
        "target_features": target_features,
        "context_features": context_features,
        "predicted_target": pred,
        "predicted_context": context,
        "distance": distance,
        "prediction_loss": lp,
        "context_loss": lc,
        "loss": loss,
        **gradients,
        "updated_encoder_patch": encoder.patch_embed.proj.weight,
        "updated_predictor_projection": predictor.predictor_proj.weight,
        "updated_target_patch": target.patch_embed.proj.weight,
    }
    arrays.update(
        {f"encoder::{name}": value for name, value in initial_encoder.items()}
    )
    arrays.update(
        {f"predictor::{name}": value for name, value in initial_predictor.items()}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{
            name: np.asarray(
                value.detach().cpu() if hasattr(value, "detach") else value
            )
            for name, value in arrays.items()
        },
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": "representax-vjepa2-1-oracle-v1",
                "reference_repository": "facebookresearch/vjepa2",
                "reference_commit": "204698b45b3712590f06245fbfba32d3be539812",
                "paper": "arXiv:2603.14482",
                "torch": torch.__version__,
                "purpose": "Tiny exact forward/loss/gradient/update/EMA oracle",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.output)


if __name__ == "__main__":
    main()
