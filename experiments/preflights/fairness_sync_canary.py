"""Isolate XLA gradient reduction from model/loss and dataloader code."""

import json
from pathlib import Path


def worker(_index):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from accelerate import Accelerator

    accelerator = Accelerator()
    rank, world = xr.global_ordinal(), xr.world_size()
    values = torch.tensor([float(rank)], device=accelerator.device)
    xm.all_reduce("sum", [values], scale=1/world)
    torch_xla.sync(wait=True)
    print("BASIC_REDUCTION", rank, values.cpu().tolist(), flush=True)
    module = torch.nn.Linear(1, 1, bias=False)
    module.weight.data.fill_(1.)
    optimizer = torch.optim.SGD(module.parameters(), lr=.01)
    module, optimizer = accelerator.prepare(module, optimizer)
    torch_xla.sync(wait=True)
    value = torch.tensor([[float(rank + 1)]], device=accelerator.device)
    accelerator.backward(module(value).sum())
    torch_xla.sync(wait=True)
    norm = accelerator.clip_grad_norm_(module.parameters(), 100.)
    torch_xla.sync(wait=True)
    actual_norm, actual_gradient = norm.cpu().tolist(), module.weight.grad.cpu().tolist()
    print("STOCK_REDUCTION", rank, actual_norm, actual_gradient, flush=True)
    module.weight.grad = value
    module.weight.grad = xm.all_reduce("sum", module.weight.grad, scale=1/world,
                                       groups=[list(range(world))], pin_layout=False)
    torch_xla.sync(wait=True)
    corrected_gradient = module.weight.grad.cpu().tolist()
    print("FUNCTIONAL_REDUCTION", rank, corrected_gradient, flush=True)
    output = Path.home() / "representax-fairness-results" / "minimal-gradient-sync"
    output.mkdir(parents=True, exist_ok=True)
    report = {"rank": rank, "world": world, "basic": values.cpu().tolist(),
              "stock_norm": actual_norm, "stock_gradient": actual_gradient,
              "functional_gradient": corrected_gradient}
    (output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    import torch_xla
    torch_xla.launch(worker)
