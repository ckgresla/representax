"""Inspect scoring precision and local loss means on one real global batch."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from pylate import losses, models
from pylate.losses.contrastive import extract_skiplist_mask
from sentence_transformers.util import batch_to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    rows = [json.loads(line) for line in args.data.read_text().splitlines()][:512]
    model = models.ColBERT(str(args.checkpoint), device="cuda", local_files_only=True)
    model.query_length, model.document_length = 32, 256
    model.eval()
    representations, features = [], []
    with torch.no_grad():
        for name, is_query in (("query", True), ("positive", False)):
            blocks, inputs = [], []
            for start in range(0, len(rows), 32):
                batch = batch_to_device(model.tokenize(
                    [row[name] for row in rows[start:start + 32]],
                    is_query=is_query, pad=True), model.device)
                inputs.append({key: value.clone() for key, value in batch.items()})
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    blocks.append(torch.nn.functional.normalize(
                        model(batch)["token_embeddings"], p=2, dim=-1))
            representations.append(torch.cat(blocks))
            features.append({key: torch.cat([batch[key] for batch in inputs])
                             for key in inputs[0]})
        masks = extract_skiplist_mask(features, model.skiplist)
        criterion = losses.Contrastive(model, temperature=0.02)
        reports = {}
        for mode in ("float32", "bfloat16"):
            values = []
            dtype = getattr(torch, mode)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode == "bfloat16"):
                for start in range(0, len(rows), 8):
                    scores = criterion.score_metric(
                        representations[0][start:start + 8].to(dtype),
                        representations[1].to(dtype).unsqueeze(1),
                        queries_mask=None if model.do_query_expansion else masks[0][start:start + 8],
                        documents_mask=masks[1].unsqueeze(1),
                    )
                    values.append(torch.nn.functional.cross_entropy(
                        scores / 0.02, torch.arange(start, start + 8, device="cuda"),
                        reduction="none").float())
            per_query = torch.cat(values)
            reports[mode] = {"global_mean": per_query.mean().item(),
                             "rank_means": per_query.reshape(16, 32).mean(1).cpu().tolist()}
    result = {"training_data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "encoder_compute": "bfloat16", "batch_size": len(rows),
              "query_shape": list(representations[0].shape),
              "document_shape": list(representations[1].shape), "scores": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
