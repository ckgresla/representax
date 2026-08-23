"""Generate a deterministic Transformers 5.6 DistilBERT oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    import torch
    import transformers
    from transformers import DistilBertConfig, DistilBertModel

    if transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "DistilBERT parity requires transformers==5.6.0; "
            f"found {transformers.__version__}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(157)
    config = DistilBertConfig(
        vocab_size=31,
        dim=12,
        hidden_dim=24,
        n_layers=2,
        n_heads=3,
        max_position_embeddings=16,
        dropout=0.1,
        attention_dropout=0.1,
        pad_token_id=0,
    )
    model = DistilBertModel(config).eval()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(arguments.output_directory, safe_serialization=True)

    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    inputs_embeds = model.embeddings.word_embeddings(input_ids).detach()
    inputs_embeds.requires_grad_(True)
    embedded_hidden = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )[0]
    objective = torch.linspace(-0.5, 0.5, config.dim)
    input_loss = torch.sum(embedded_hidden * objective)
    (input_gradient,) = torch.autograd.grad(input_loss, inputs_embeds)

    model.zero_grad(set_to_none=True)
    training_hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    parameter_loss = torch.sum(training_hidden * objective)
    parameter_loss.backward()
    parameter_gradients = {
        "parameter_gradient__" + name: value.grad.detach().numpy()
        for name, value in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    optimizer.step()
    updated_parameters = {
        "updated_parameter__" + name: value.detach().numpy()
        for name, value in model.named_parameters()
    }
    np.savez(
        arguments.output_directory / "oracle.npz",
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        hidden=hidden.numpy(),
        inputs_embeds=inputs_embeds.detach().numpy(),
        embedded_hidden=embedded_hidden.detach().numpy(),
        input_gradient=input_gradient.detach().numpy(),
        parameter_loss=parameter_loss.detach().numpy(),
        **parameter_gradients,
        **updated_parameters,
    )


if __name__ == "__main__":
    main()
