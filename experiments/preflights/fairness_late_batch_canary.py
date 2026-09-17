"""Trace the real prepared TPU PyLate reader without performing training."""

import json
import os
from pathlib import Path


def worker(_index):
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.distributed.xla_backend
    from pylate import models, utils
    from sentence_transformers import SentenceTransformerTrainer
    from experiments.preflights.accelerator import torch_device, torch_rank, torch_world_size
    from experiments.preflights.late_interaction import (
        _pylate_loss, _reference_arguments, _reference_dataset,
    )
    from pylate import losses

    torch.set_num_threads(1)
    dist.init_process_group("xla", init_method="xla://")
    rank, world = torch_rank(), torch_world_size()
    assert (dist.get_rank(), dist.get_world_size()) == (rank, world)
    assets = Path.home() / "representax-paper-assets"
    seed = int(os.environ.get("AUDIT_SEED", "7"))
    destination = Path.home() / "representax-fairness-results" / f"late-batch-trace-{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    data = assets / "late-fair-20260916/train.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    positions = {row["query"]: index for index, row in enumerate(rows)}
    assert len(positions) == len(rows)
    seen = []
    model = models.ColBERT(str(assets / "late-checkpoint"),
                           device=torch_device(), local_files_only=True)
    model.query_length, model.document_length = 32, 256
    collator = utils.ColBERTCollator(model.tokenize)

    class TraceCollator:
        valid_label_columns = collator.valid_label_columns

        def __call__(self, features):
            seen.append([positions[row["query"]] for row in features])
            return collator(features)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=_reference_arguments(destination / f"process-{rank}", max_steps=22,
                                  save_steps=11, seed=seed, save=False, platform="tpu"),
        train_dataset=_reference_dataset(data, rows=512 * 22),
        loss=_pylate_loss(losses, model, "tpu"), data_collator=TraceCollator(),
    )
    loader = trainer.get_train_dataloader()
    iterator = iter(loader)
    for _ in range(2):
        next(iterator)
        torch_xla.sync(wait=True)
    result = {"rank": rank, "world_size": world, "seed": seed,
              "accelerator_rank": trainer.accelerator.process_index,
              "accelerator_world_size": trainer.accelerator.num_processes,
              "batches": seen[:2]}
    (destination / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker)
