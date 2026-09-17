"""Capture one real TPU PyLate optimizer step for a scoring audit."""

import json
import os
from pathlib import Path


def worker(_index):
    import numpy as np
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
    from transformers import TrainerCallback
    from experiments.preflights.timing import CudaStepTimer

    torch.set_num_threads(1)
    dist.init_process_group("xla", init_method="xla://")
    rank, world = torch_rank(), torch_world_size()
    assert (dist.get_rank(), dist.get_world_size()) == (rank, world)
    assets = Path.home() / "representax-paper-assets"
    seed = int(os.environ.get("AUDIT_SEED", "7"))
    destination = Path.home() / "representax-fairness-results" / f"late-forward-boundary-{seed}"
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
    encoded = []
    model.register_forward_hook(lambda _model, _inputs, output:
                                encoded.append(output["token_embeddings"].detach().clone()))
    criterion = _pylate_loss(losses, model, "tpu")
    original_score = criterion.score_metric
    score_chunks = []
    scoring_queries = []
    score_arguments = []

    def capture_score(queries, documents, **kwargs):
        scores = original_score(queries, documents, **kwargs)
        score_chunks.append(scores.detach().clone())
        scoring_queries.append(queries.detach().clone())
        if not score_arguments:
            score_arguments.append((documents.detach().clone(),
                                    kwargs["documents_mask"].detach().clone()))
        return scores

    criterion.score_metric = capture_score

    class Captured(Exception):
        pass

    class EntryTrainer(XlaMixedPrecisionTrainer, SentenceTransformerTrainer):
        def compute_loss(self, model, inputs, **kwargs):
            self.probe_features, _ = self.collect_features(inputs)
            loss = super().compute_loss(model, inputs, **kwargs)
            self.probe_loss = loss.detach().clone()
            torch_xla.sync(wait=True)
            if os.environ.get("AUDIT_CAPTURE_BEFORE_BACKWARD") == "1":
                CaptureCallback().on_step_end(self.args, self.state, self.control)
            return loss

    class CaptureCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            loss = trainer.probe_loss
            mean = xm.all_reduce("sum", loss.detach(), scale=1 / world, pin_layout=False)
            torch_xla.sync(wait=True)
            logged = nested_gather(loss.detach(), args.parallel_mode).mean()
            actual = [token_positions.get(tuple(ids.tolist()), -1)
                      for ids in trainer.probe_features[0]["input_ids"].cpu()]
            result = {"rank": rank, "world_size": world, "seed": seed,
                      "actual_rows": actual, "collator_batches": seen[:2],
                      "local_loss": loss.cpu().item(), "global_loss": mean.cpu().item(),
                      "logged_loss": logged.cpu().item(),
                      "accelerator_rank": trainer.accelerator.process_index,
                      "accelerator_world_size": trainer.accelerator.num_processes}
            arrays = {f"{i}_{key}": value.detach().cpu().numpy()
                      for i, features in enumerate(trainer.probe_features)
                      for key, value in features.items() if key in {"input_ids", "attention_mask"}}
            arrays.update({f"encoded_{i}": value.float().cpu().numpy()
                           for i, value in enumerate(encoded)})
            arrays["scores"] = torch.cat(score_chunks).float().cpu().numpy()
            arrays["scoring_queries"] = torch.cat(scoring_queries).float().cpu().numpy()
            arrays["gathered_documents"] = score_arguments[0][0].float().cpu().numpy()
            arrays["gathered_document_masks"] = score_arguments[0][1].cpu().numpy()
            import hashlib
            docs = arrays["gathered_documents"]
            masks = arrays["gathered_document_masks"]
            norms = np.linalg.norm(docs, axis=-1)[masks.astype(bool)]
            result["gathered_documents_sha256"] = hashlib.sha256(docs.tobytes()).hexdigest()
            result["gathered_valid_norm_range"] = [float(norms.min()), float(norms.max())]
            result["before_backward"] = os.environ.get("AUDIT_CAPTURE_BEFORE_BACKWARD") == "1"
            np.savez_compressed(destination / f"rank-{rank}.npz", **arrays)
            (destination / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            raise Captured()

    trainer = EntryTrainer(
        model=model,
        args=_reference_arguments(destination / f"process-{rank}", max_steps=22,
                                  save_steps=11, seed=seed, save=False, platform="tpu"),
        train_dataset=_reference_dataset(data, rows=512 * 22),
        loss=criterion, data_collator=TraceCollator(),
        callbacks=[CudaStepTimer().callback(), CaptureCallback()],
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
