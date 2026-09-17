"""Shared initialization for the corrected, historical paired controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def xla_all_gather_with_grad(value):
    """Gather candidates, summing remote-query contributions in backward."""
    import torch
    import torch_xla.core.xla_model as xm
    from experiments.preflights.accelerator import torch_rank

    class Gather(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            ctx.start = torch_rank() * tensor.shape[0]
            ctx.count = tensor.shape[0]
            return xm.all_gather(tensor, dim=0, pin_layout=False)

        @staticmethod
        def backward(ctx, gradient):
            total = xm.all_reduce("sum", gradient, pin_layout=False)
            return total.narrow(0, ctx.start, ctx.count)

    return Gather.apply(value)


def prepare_late_checkpoint(source: Path, destination: Path) -> Path:
    """Share the recipe's truncation limits without modifying upstream weights."""
    from experiments.preflights.late_interaction import frozen_contract

    contract = frozen_contract()
    metadata = "config_sentence_transformers.json"
    original = (source / metadata).read_bytes()
    effective = json.loads(original)
    effective.update(query_length=contract.maximum_query_length,
                     document_length=contract.maximum_document_length)
    if destination.exists():
        if json.loads((destination / metadata).read_text()) != effective:
            raise RuntimeError("Prepared checkpoint limits differ from this recipe")
        return destination
    destination.mkdir()
    for path in source.iterdir():
        if path.name != metadata:
            (destination / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (destination / metadata).write_text(json.dumps(effective, indent=2) + "\n")
    (destination / "preparation.json").write_text(json.dumps({
        "source": str(source), "source_metadata_sha256": hashlib.sha256(original).hexdigest(),
        "query_length": contract.maximum_query_length,
        "document_length": contract.maximum_document_length,
        "weights": "unchanged symlinks to the pinned checkpoint",
    }, indent=2) + "\n")
    return destination


class XlaGradientSynchronization:
    """Reduce functional gradient outputs before clipping in PJRT Trainer runs."""

    def train(self, *args, **kwargs):
        from experiments.preflights.accelerator import torch_is_tpu

        if torch_is_tpu():
            digest = assert_torch_replicas_equal(self.model)
            print(json.dumps({"initial_parameter_sha256": digest}), flush=True)
        return super().train(*args, **kwargs)

    def _clip_grad_norm(self, model):
        import torch
        from experiments.preflights.accelerator import torch_is_tpu, torch_synchronize, torch_world_size

        if not torch_is_tpu():
            return super()._clip_grad_norm(model)
        import torch_xla.core.xla_model as xm

        parameters = [p for p in model.parameters() if p.grad is not None]
        for parameter in parameters:
            parameter.grad = xm.all_reduce(
                "sum", parameter.grad, scale=1.0 / torch_world_size(), pin_layout=False
            )
        # Materialize reduced gradients before clipping, without a full-model buffer.
        torch_synchronize()
        self.accelerator.gradient_state.is_xla_gradients_synced = True
        return torch.nn.utils.clip_grad_norm_(parameters, self.args.max_grad_norm)


class XlaMixedPrecisionTrainer(XlaGradientSynchronization):
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
