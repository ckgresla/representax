"""Check corrected PyLate global loss/gradients/update on all sixteen TPU ranks."""

import json
import os
from pathlib import Path


def worker(_index):
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    from pylate import losses
    from experiments.preflights.late_interaction import _pylate_loss

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch_xla._XLAC._xla_set_mat_mul_precision("highest")
    dist.init_process_group("xla", init_method="xla://")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 16

    class Encoder(torch.nn.Module):
        skiplist = []
        do_query_expansion = False

        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(7, 5, bias=False)

        def forward(self, features):
            return {"token_embeddings": self.projection(features["values"])}

    weight = torch.sin(torch.arange(35, dtype=torch.float32).reshape(5, 7)) * .2
    queries = torch.sin(torch.arange(48 * 2 * 7, dtype=torch.float32).reshape(48, 2, 7) * .31)
    documents = torch.cos(torch.arange(48 * 3 * 7, dtype=torch.float32).reshape(48, 3, 7) * .17)
    fixture = os.environ.get("AUDIT_LATE_FIXTURE", "rank-two")
    if fixture == "full-rank":
        generator = torch.Generator().manual_seed(71)
        weight = torch.randn(5, 7, generator=generator) * .2
        queries = torch.randn(48, 2, 7, generator=generator)
        documents = torch.randn(48, 3, 7, generator=generator)

    def features(values):
        return {"values": values, "input_ids": torch.ones(values.shape[:2], dtype=torch.int64, device=values.device),
                "attention_mask": torch.ones(values.shape[:2], dtype=torch.int64, device=values.device)}

    expected = Encoder()
    expected.projection.weight.data.copy_(weight)
    target = losses.Contrastive(expected, temperature=.02)([features(queries), features(documents)])
    target.backward()
    expected_gradient = expected.projection.weight.grad.clone()
    optimizer = torch.optim.AdamW(expected.parameters(), lr=2e-5, weight_decay=0)
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.)
    optimizer.step()

    model = Encoder()
    model.projection.weight.data.copy_(weight)
    model.to(xm.xla_device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    full = Encoder()
    full.projection.weight.data.copy_(weight)
    full.to(xm.xla_device())
    full_loss = losses.Contrastive(full, temperature=.02)([
        features(queries.to(xm.xla_device())), features(documents.to(xm.xla_device()))])
    full_loss.backward()
    full_gradient = full.projection.weight.grad.detach().clone()
    full_optimizer = torch.optim.AdamW(full.parameters(), lr=2e-5, weight_decay=0)
    torch.nn.utils.clip_grad_norm_(full.parameters(), 1.)
    full_optimizer.step()
    selection = slice(rank * 3, (rank + 1) * 3)
    loss = _pylate_loss(losses, model, "tpu")([
        features(queries[selection].to(xm.xla_device())),
        features(documents[selection].to(xm.xla_device())),
    ])
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad = xm.all_reduce("sum", parameter.grad,
                                           scale=1 / world, pin_layout=True)
    gradient = model.projection.weight.grad.detach().clone()
    mean_loss = xm.all_reduce(xm.REDUCE_SUM, loss.detach(), scale=1 / world,
                              pin_layout=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    diagnostic = Path.home() / "representax-fairness-results" / f"late-pinned-oracle-{fixture}"
    diagnostic.mkdir(parents=True, exist_ok=True)
    (diagnostic / f"rank-{rank}.hlo").write_text(torch_xla._XLAC._get_xla_tensors_hlo([mean_loss, gradient, full_loss, full_gradient]))
    torch_xla.sync(wait=True)
    actual_loss, actual_gradient = mean_loss.cpu(), gradient.cpu()
    actual_weight = model.projection.weight.detach().cpu()
    print(json.dumps({"rank": rank, "actual_loss": actual_loss.item(),
        "target_loss": target.item(), "gradient_max_error": (actual_gradient - expected_gradient).abs().max().item(),
        "full_xla_loss": full_loss.cpu().item(),
        "full_xla_gradient_error": (actual_gradient - full_gradient.cpu()).abs().max().item(),
        "gradient_relative_error": ((actual_gradient - expected_gradient).norm() / expected_gradient.norm()).item(),
        "weight_max_error": (actual_weight - expected.projection.weight.detach()).abs().max().item()}), flush=True)
    # The oracle for distributed semantics uses the same arithmetic backend.
    # Keep CPU errors visible rather than treating them as distributed errors.
    torch.testing.assert_close(actual_loss, full_loss.detach().cpu(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_gradient, full_gradient.cpu(), rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(actual_weight, full.projection.weight.detach().cpu(), rtol=2e-5, atol=2e-6)
    result = {"rank": rank, "status": "passed", "oracle": "single-device XLA global batch",
              "loss": actual_loss.item(), "cpu_loss": target.item(),
              "cpu_gradient_max_error": (actual_gradient - expected_gradient).abs().max().item(),
              "gradient_max_error": (actual_gradient - full_gradient.cpu()).abs().max().item(),
              "update_max_error": (actual_weight - full.projection.weight.detach().cpu()).abs().max().item()}
    output = diagnostic
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker)
