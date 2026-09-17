"""Small CPU diagnostics against the installed historical reference packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch


def cached_loss() -> dict:
    from pylate.losses import CachedContrastive

    model = torch.nn.Identity()
    model.do_query_expansion = False
    objective = CachedContrastive(
        model, mini_batch_size=2, score_mini_batch_size=2, temperature=0.2
    )
    torch.manual_seed(71)
    originals = [torch.randn(5, 3, 4), torch.randn(5, 4, 4)]
    masks = [torch.ones(5, x.shape[1], dtype=torch.bool) for x in originals]

    def leaves():
        return [[x.detach().clone().requires_grad_() for x in value.split(2)]
                for value in originals]

    direct = leaves()
    direct_loss = objective.calculate_loss(direct, masks, with_backward=False)
    direct_loss.backward()
    cached = leaves()
    cached_loss = objective.calculate_loss_and_cache_gradients(cached, masks)
    direct_grad = torch.cat([x.grad.flatten() for group in direct for x in group])
    cached_grad = torch.cat([x.grad.flatten() for group in cached for x in group])
    ratio = float(torch.dot(cached_grad, direct_grad) / direct_grad.square().sum())
    relative_error = float((cached_grad - 5 * direct_grad).abs().max())
    normalized_error = float((cached_grad / cached_grad.norm()
                              - direct_grad / direct_grad.norm()).abs().max())
    assert torch.allclose(direct_loss, cached_loss)
    assert torch.allclose(cached_grad, 5 * direct_grad, rtol=1e-5, atol=1e-5)
    return {
        "pylate_version": importlib.metadata.version("pylate"),
        "batch_size": 5, "partial_final_chunk": True,
        "direct_loss": float(direct_loss.detach()),
        "cached_reported_loss": float(cached_loss.detach()),
        "cached_to_mean_gradient_ratio": ratio,
        "maximum_error_from_batch_scaled_gradient": relative_error,
        "direct_gradient_norm": float(direct_grad.norm()),
        "cached_gradient_norm": float(cached_grad.norm()),
        "unit_normalized_gradient_maximum_difference": normalized_error,
        "limit": "Embedding-gradient diagnostic, not historical model update parity.",
    }


def adam_dtypes() -> dict:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    parameter.grad = torch.tensor([0.1, -0.2], dtype=torch.bfloat16)
    optimizer.step()
    return {
        "torch_version": torch.__version__,
        "parameter_dtype": str(parameter.dtype),
        "optimizer_tensor_dtypes": {
            name: str(value.dtype) for name, value in optimizer.state[parameter].items()
            if isinstance(value, torch.Tensor)
        },
        "limit": "CPU AdamW dtype behavior; historical full-model dtypes require loader inspection.",
    }


def lora_dtypes() -> dict:
    from peft import LoraConfig
    from transformers import Qwen3Config, Qwen3Model

    model = Qwen3Model(Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
    )).to(torch.bfloat16)
    model.add_adapter(LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"]))
    return {
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        "trainable_parameters": {name: str(p.dtype) for name, p in model.named_parameters()
                                 if p.requires_grad},
        "limit": "Tiny CPU model using the same Transformers add_adapter API; real TPU audio separately instrumented.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    diagnostics = {"cached-loss": cached_loss, "adam-dtypes": adam_dtypes,
                   "lora-dtypes": lora_dtypes}
    parser.add_argument("diagnostic", choices=diagnostics)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnostics[args.diagnostic]()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
