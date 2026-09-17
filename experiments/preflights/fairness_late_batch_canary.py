"""Trace the real prepared TPU PyLate reader without performing training."""

import json
import os
from pathlib import Path


def worker(_index):
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    from pylate import models, utils
    from sentence_transformers import SentenceTransformerTrainer
    from experiments.preflights.accelerator import torch_device, torch_rank, torch_world_size
    from experiments.preflights.late_interaction import (
        _pylate_loss, _reference_arguments, _reference_dataset,
    )
    from pylate import losses
    from experiments.preflights.fairness import XlaMixedPrecisionTrainer
    from transformers.trainer_pt_utils import nested_gather

    torch.set_num_threads(1)
    dist.init_process_group("xla", init_method="xla://")
    rank, world = torch_rank(), torch_world_size()
    assert (dist.get_rank(), dist.get_world_size()) == (rank, world)
    assets = Path.home() / "representax-paper-assets"
    seed = int(os.environ.get("AUDIT_SEED", "7"))
    destination = Path.home() / "representax-fairness-results" / f"late-training-entry-{seed}"
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

    expected = model.tokenize([row["query"] for row in rows], is_query=True, pad=True)
    token_positions = {tuple(ids.tolist()): i for i, ids in enumerate(expected["input_ids"])}

    class Captured(Exception):
        pass

    class EntryTrainer(XlaMixedPrecisionTrainer, SentenceTransformerTrainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            features, _ = self.collect_features(inputs)
            actual = [token_positions.get(tuple(ids.tolist()), -1)
                      for ids in features[0]["input_ids"].cpu()]
            with torch.no_grad(), self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            mean = xm.all_reduce("sum", loss.detach(), scale=1 / world, pin_layout=False)
            torch_xla.sync(wait=True)
            logged = nested_gather(loss.detach(), self.args.parallel_mode).mean()
            result = {"rank": rank, "world_size": world, "seed": seed,
                      "actual_rows": actual, "collator_batches": seen[:2],
                      "local_loss": loss.cpu().item(), "global_loss": mean.cpu().item(),
                      "logged_loss": logged.cpu().item(),
                      "accelerator_rank": self.accelerator.process_index,
                      "accelerator_world_size": self.accelerator.num_processes}
            (destination / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            raise Captured()

    trainer = EntryTrainer(
        model=model,
        args=_reference_arguments(destination / f"process-{rank}", max_steps=22,
                                  save_steps=11, seed=seed, save=False, platform="tpu"),
        train_dataset=_reference_dataset(data, rows=512 * 22),
        loss=_pylate_loss(losses, model, "tpu"), data_collator=TraceCollator(),
    )
    try:
        trainer.train()
    except Captured:
        pass
    else:
        raise RuntimeError("Did not reach the first training batch")
    dist.destroy_process_group()


if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker)
