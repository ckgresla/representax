"""Generate a pinned PyTorch/Transformers ModernVBERT parity oracle.

This module belongs to the optional parity environment.  Ordinary Representax
training never imports PyTorch or Transformers model definitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.models.upstream import configure_torch_float32_highest, transformers_tacet


def generate_oracle(
    checkpoint: str | Path,
    output: str | Path,
    texts: list[str],
    *,
    device: str = "cpu",
) -> None:
    import torch
    import transformers
    from transformers import Idefics3Processor, ModernVBertModel

    if transformers.__version__ != "5.3.0":
        raise RuntimeError(
            "ModernVBERT parity requires transformers==5.3.0; "
            f"found {transformers.__version__}"
        )
    configure_torch_float32_highest()
    checkpoint = Path(checkpoint)
    with transformers_tacet():
        processor = Idefics3Processor.from_pretrained(checkpoint, local_files_only=True)
        model = ModernVBertModel.from_pretrained(
            checkpoint,
            dtype=torch.float32,
            local_files_only=True,
        ).to(device)
    model.eval()
    encoded = processor(
        text=texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=7999,
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)

    inputs_embeds = (
        model.get_input_embeddings()(encoded["input_ids"]).detach().requires_grad_(True)
    )
    gradient_hidden = model.text_model(
        inputs_embeds=inputs_embeds,
        attention_mask=encoded["attention_mask"],
    ).last_hidden_state
    gradient_mask = encoded["attention_mask"].unsqueeze(-1).to(gradient_hidden.dtype)
    gradient_pool = (gradient_hidden * gradient_mask).sum(1) / (
        gradient_mask.sum(1).clamp_min(1)
    )
    gradient_pool = torch.nn.functional.normalize(gradient_pool.float(), p=2, dim=-1)
    objective = torch.linspace(-0.5, 0.5, gradient_pool.shape[-1], device=device)
    torch.sum(gradient_pool * objective).backward()

    np.savez(
        output,
        input_ids=encoded["input_ids"].cpu().numpy(),
        attention_mask=encoded["attention_mask"].cpu().numpy(),
        hidden=hidden.float().cpu().numpy(),
        pooled=pooled.float().cpu().numpy(),
        inputs_embeds=inputs_embeds.detach().float().cpu().numpy(),
        input_grads=inputs_embeds.grad.float().cpu().numpy(),
        transformers=np.asarray(transformers.__version__),
        dtype=np.asarray("float32"),
        device=np.asarray(str(device)),
    )


def generate_multimodal_oracle(
    checkpoint: str | Path,
    image: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    split_image: bool = False,
) -> None:
    """Generate processed vision, multimodal, and pixel-gradient references."""

    import torch
    import transformers
    from PIL import Image
    from transformers import Idefics3Processor, ModernVBertModel

    if transformers.__version__ != "5.3.0":
        raise RuntimeError(
            "ModernVBERT parity requires transformers==5.3.0; "
            f"found {transformers.__version__}"
        )
    configure_torch_float32_highest()
    checkpoint = Path(checkpoint)
    with transformers_tacet():
        processor = Idefics3Processor.from_pretrained(checkpoint, local_files_only=True)
        model = ModernVBertModel.from_pretrained(
            checkpoint,
            dtype=torch.float32,
            local_files_only=True,
        ).to(device)
    model.eval()
    encoded = processor(
        text=[
            "<|begin_of_text|>User:<image>Describe the image."
            "<end_of_utterance>\nAssistant:"
        ],
        images=[Image.open(image).convert("RGB")],
        images_kwargs={"do_image_splitting": split_image},
        return_tensors="pt",
        padding="longest",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    pixels = encoded["pixel_values"].detach().requires_grad_(True)
    image_features = model.get_image_features(
        pixels,
        encoded.get("pixel_attention_mask"),
    ).pooler_output
    hidden = model(
        **{name: value for name, value in encoded.items() if name != "pixel_values"},
        pixel_values=pixels,
    ).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
    objective = torch.linspace(-1.0, 1.0, pooled.shape[-1], device=device)[None]
    pixel_gradient = torch.autograd.grad((pooled * objective).sum(), pixels)[0]

    arrays = {
        name: value.detach().cpu().numpy()
        for name, value in encoded.items()
        if hasattr(value, "detach")
    }
    arrays.update(
        image_features=image_features.detach().float().cpu().numpy(),
        pooled=pooled.detach().float().cpu().numpy(),
        pixel_grad=pixel_gradient.detach().float().cpu().numpy(),
        transformers=np.asarray(transformers.__version__),
        dtype=np.asarray("float32"),
        device=np.asarray(str(device)),
        split_image=np.asarray(split_image),
    )
    np.savez(output, **arrays)
