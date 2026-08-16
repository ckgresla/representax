"""Generate a deterministic Transformers 5.3 MPNet checkpoint and oracle."""

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
    from transformers import MPNetConfig, MPNetModel

    if transformers.__version__ != "5.3.0":
        raise RuntimeError(
            "MPNet parity requires transformers==5.3.0; "
            f"found {transformers.__version__}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(19)
    config = MPNetConfig(
        vocab_size=31,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        max_position_embeddings=16,
        relative_attention_num_buckets=32,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    model = MPNetModel(config).eval()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(arguments.output_directory, safe_serialization=True)

    input_ids = torch.tensor([[0, 4, 5, 2, 1], [0, 6, 2, 1, 1]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    inputs_embeds = model.embeddings.word_embeddings(input_ids).detach()
    inputs_embeds.requires_grad_(True)
    embedded_output = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )
    objective = torch.linspace(-0.5, 0.5, config.hidden_size)
    loss = torch.sum(embedded_output.last_hidden_state * objective)
    (input_gradient,) = torch.autograd.grad(loss, inputs_embeds)

    model.zero_grad(set_to_none=True)
    training_output = model(input_ids=input_ids, attention_mask=attention_mask)
    parameter_loss = torch.sum(training_output.last_hidden_state * objective)
    parameter_loss = parameter_loss + torch.sum(
        training_output.pooler_output * objective
    )
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
        hidden=output.last_hidden_state.detach().numpy(),
        pooler=output.pooler_output.detach().numpy(),
        inputs_embeds=inputs_embeds.detach().numpy(),
        embedded_hidden=embedded_output.last_hidden_state.detach().numpy(),
        input_gradient=input_gradient.detach().numpy(),
        parameter_loss=parameter_loss.detach().numpy(),
        **parameter_gradients,
        **updated_parameters,
    )


if __name__ == "__main__":
    main()
