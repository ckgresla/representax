"""Verify physical master/state precision, not only Torch's logical dtype."""

import json
import os
from pathlib import Path


def worker(_index):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from accelerate import Accelerator

    before = os.environ.get("AUDIT_ALLOCATE_BEFORE_ACCELERATE", "0") == "1"
    parameter = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
    if before:
        parameter = torch.nn.Parameter(parameter.to(xm.xla_device()))
    accelerator = Accelerator(mixed_precision="bf16")
    if not before:
        parameter = torch.nn.Parameter(parameter.to(xm.xla_device()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5, weight_decay=0)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    tensors = [parameter, state["exp_avg"], state["exp_avg_sq"]]
    linear = torch.nn.Linear(8, 8, bias=False).to(xm.xla_device())
    inputs = torch.ones(2, 8, device=xm.xla_device())
    raw_output = linear(inputs)
    with torch.autocast("xla", dtype=torch.bfloat16):
        mixed_output = linear(inputs)
    tensors.extend([raw_output, mixed_output])
    destination = Path.home() / "representax-fairness-results" / f"precision-before-{before}"
    destination.mkdir(parents=True, exist_ok=True)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo(tensors)
    (destination / f"rank-{xr.global_ordinal()}.hlo").write_text(hlo)
    torch_xla.sync(wait=True)
    result = {"rank": xr.global_ordinal(), "allocate_before_accelerate": before,
              "logical_dtypes": [str(x.dtype) for x in tensors],
              "parameter": parameter.detach().cpu().tolist(),
              "first_moment": state["exp_avg"].cpu().tolist(),
              "second_moment": state["exp_avg_sq"].cpu().tolist(),
              "unwrapped_forward_dtype": str(raw_output.dtype),
              "autocast_forward_dtype": str(mixed_output.dtype),
              "native_amp": accelerator.native_amp,
              "xla_use_bf16": os.environ.get("XLA_USE_BF16")}
    (destination / f"rank-{xr.global_ordinal()}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker)
