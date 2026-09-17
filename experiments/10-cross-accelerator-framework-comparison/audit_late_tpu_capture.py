"""Compare saved real TPU embeddings, masks and scores with a GPU reconstruction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from pylate import models
from pylate.losses.contrastive import extract_skiplist_mask
from pylate.scores import ColBERTScores
from sentence_transformers.util import batch_to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=list(range(16)))
    parser.add_argument("--capture-name", default="late-score-trace-7")
    parser.add_argument("--assert-gather", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    paths = {int(p.stem.split("-")[1]): p for p in args.captures.glob(
        f"worker-*/{args.capture_name}/rank-*.npz")}
    assert set(args.ranks) <= set(paths)
    model = models.ColBERT(str(args.checkpoint), device="cuda", local_files_only=True)
    model.query_length, model.document_length = 32, 256
    model.eval()
    rows = [json.loads(line) for line in args.data.read_text().splitlines()][:512]
    reports = []
    scorer = ColBERTScores()
    with torch.no_grad():
        for rank in args.ranks:
            with np.load(paths[rank]) as capture:
                saved = {key: torch.from_numpy(capture[key]).to("cuda") for key in capture.files}
            report = {"rank": rank}
            features = []
            for i, (column, is_query) in enumerate((("query", True), ("positive", False))):
                tokens = batch_to_device(model.tokenize(
                    [row[column] for row in rows[rank * 32:(rank + 1) * 32]],
                    is_query=is_query, pad=True), model.device)
                features.append({key: value.clone() for key, value in tokens.items()})
                report[f"{column}_ids_equal"] = torch.equal(tokens["input_ids"], saved[f"{i}_input_ids"])
                report[f"{column}_mask_equal"] = torch.equal(tokens["attention_mask"], saved[f"{i}_attention_mask"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actual = model(tokens)["token_embeddings"].float()
                expected = saved[f"encoded_{i}"]
                report[f"{column}_embedding_max_error"] = (actual - expected).abs().max().item()
                cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
                report[f"{column}_embedding_min_cosine"] = cosine.min().item()
            documents = saved["gathered_documents"]
            masks = saved["gathered_document_masks"]
            report["gathered_documents_sha256"] = hashlib.sha256(
                documents.cpu().numpy().tobytes()).hexdigest()
            norms = documents.norm(dim=-1)[masks.bool()]
            report["gathered_valid_norm_range"] = [norms.min().item(), norms.max().item()]
            local_documents = torch.nn.functional.normalize(saved["encoded_1"], dim=-1)
            report["gathered_local_document_max_error"] = (
                documents[rank * 32:(rank + 1) * 32, 0] - local_documents).abs().max().item()
            query_mask, expected_mask = extract_skiplist_mask(features, model.skiplist)
            report["gathered_local_mask_equal"] = torch.equal(
                masks[rank * 32:(rank + 1) * 32, 0], expected_mask)
            queries = saved.get("scoring_queries")
            if queries is None:
                queries = torch.nn.functional.normalize(saved["encoded_0"], dim=-1)
            reproduced = torch.cat([scorer(queries[i:i + 8], documents,
                                           queries_mask=None if model.do_query_expansion else query_mask[i:i + 8],
                                           documents_mask=masks, backend="torch")
                                    for i in range(0, 32, 8)])
            scores = saved["scores"]
            report["score_max_error"] = (reproduced - scores).abs().max().item()
            labels = torch.arange(rank * 32, (rank + 1) * 32, device="cuda")
            for name, values in (("captured", scores), ("recomputed", reproduced)):
                losses = torch.nn.functional.cross_entropy(values / 0.02, labels, reduction="none")
                report[f"{name}_loss"] = losses.mean().item()
                worst = int(losses.argmax())
                report[f"{name}_worst"] = {
                    "row": rank * 32 + worst, "loss": losses[worst].item(),
                    "positive_score": values[worst, labels[worst]].item(),
                    "top_document": int(values[worst].argmax()),
                    "top_score": values[worst].max().item(),
                }
            reports.append(report)
            print(json.dumps(report), flush=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    if args.assert_gather:
        assert set(args.ranks) == set(range(16))
        assert len({row["gathered_documents_sha256"] for row in reports}) == 1
        assert all(row["gathered_local_mask_equal"] and
                   row["gathered_local_document_max_error"] < .004 and
                   .98 < row["gathered_valid_norm_range"][0] <=
                   row["gathered_valid_norm_range"][1] < 1.02
                   for row in reports)


if __name__ == "__main__":
    main()
