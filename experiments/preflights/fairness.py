"""Shared initialization for the corrected, historical paired controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class XlaMixedPrecisionTrainer:
    """Request BF16 compute explicitly without downcasting FP32 optimizer state."""

    def autocast_smart_context_manager(self, cache_enabled=True):
        import torch
        from experiments.preflights.accelerator import torch_is_tpu

        if torch_is_tpu():
            return torch.autocast("xla", dtype=torch.bfloat16,
                                  cache_enabled=cache_enabled)
        return super().autocast_smart_context_manager(cache_enabled=cache_enabled)


def enable_xla_mixed_precision_attention():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def mixed_sdpa(module, query, key, value, *args, **kwargs):
        # Qwen's FP32 RMSNorm weights promote Q/K; CUDA autocast handles this
        # at SDPA, whereas XLA requires explicitly matching V's compute dtype.
        return sdpa_attention_forward(module, query.to(value.dtype),
                                      key.to(value.dtype), value, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("sdpa", mixed_sdpa)


def scalar_head(checkpoint: str | Path, seed: int) -> np.ndarray:
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    generator = np.random.default_rng(seed)
    return (generator.standard_normal((1, config["hidden_size"]))
            * config.get("initializer_range", 0.02)).astype(np.float32)


def load_reward_model(model_name_or_path, *, head_seed=0, **kwargs):
    import equinox as eqx
    import jax.numpy as jnp
    from representax.models.qwen_reward import load_qwen_reward_model

    model, processor = load_qwen_reward_model(
        model_name_or_path, head_seed=head_seed, **kwargs
    )
    values = jnp.asarray(scalar_head(model_name_or_path, head_seed),
                         dtype=model.score_head.weight.dtype)
    if model.score_head.weight_layout == "input_output":
        values = values.T
    if values.shape != model.score_head.weight.shape:
        raise ValueError("Unexpected native scalar-head layout")
    return eqx.tree_at(lambda value: value.score_head.weight, model, values), processor


def initialize_torch_reward(model, checkpoint, seed, *, round_bfloat16=False):
    import torch

    head = torch.from_numpy(scalar_head(checkpoint, seed))
    if round_bfloat16:
        head = head.to(torch.bfloat16).float()
    model.float()
    if model.score.weight.shape != head.shape:
        raise ValueError("Reference must expose exactly one scalar score")
    with torch.no_grad():
        model.score.weight.copy_(head)
    digest = hashlib.sha256(head.numpy().tobytes()).hexdigest()
    print(json.dumps({"initial_scalar_head_sha256": digest,
                      "seed": seed, "master_dtype": "float32"}), flush=True)
    return model


def initialize_torch_lora(model, seed):
    import torch
    from experiments.preflights.accelerator import torch_device, torch_synchronize, torch_is_tpu, torch_world_size

    generator = torch.Generator(device="cpu").manual_seed(seed)
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_A." in name:
            initial = torch.randn(parameter.shape, generator=generator,
                                  dtype=torch.float32) * parameter.shape[-1] ** -0.5
        elif ".lora_B." in name:
            initial = torch.zeros(parameter.shape, dtype=torch.float32)
        else:
            raise ValueError(f"Unexpected trainable adapter parameter: {name}")
        digest.update(name.encode())
        digest.update(initial.numpy().tobytes())
        parameter.data = initial.to(parameter.device)
    torch_synchronize()
    if torch_is_tpu():
        import torch_xla.core.xla_model as xm
        fingerprints = xm.all_gather(torch.tensor(
            list(digest.digest()), dtype=torch.int32, device=torch_device()
        )).cpu().reshape(torch_world_size(), 32)
        if not torch.equal(fingerprints, fingerprints[0].expand_as(fingerprints)):
            raise RuntimeError("Adapter initialization differs between TPU ranks")
    return digest.hexdigest()


def assert_torch_replicas_equal(model):
    import torch
    import torch_xla.core.xla_model as xm
    from experiments.preflights.accelerator import torch_device, torch_synchronize, torch_world_size

    torch_synchronize()
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    fingerprints = xm.all_gather(torch.tensor(
        list(digest.digest()), dtype=torch.int32, device=torch_device()
    )).cpu().reshape(torch_world_size(), 32)
    if not torch.equal(fingerprints, fingerprints[0].expand_as(fingerprints)):
        raise RuntimeError("Trained parameters differ between TPU ranks")
    return digest.hexdigest()
