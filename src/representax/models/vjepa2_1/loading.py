"""One-shot native V-JEPA 2.1 model and processor construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from representax.core import Modality
from representax.models.components import AttentionImplementation
from representax.models.processing import Processor
from representax.planning import RematerializationPolicy

from .config import VJEPA2_1Config
from .model import VJEPA2_1Model
from .processing import make_vjepa2_1_processor
from .reference import load_reference_checkpoint


def load_vjepa2_1(
    *,
    key: PRNGKeyArray,
    config: VJEPA2_1Config | Mapping[str, Any] | None = None,
    modality: Modality | str = Modality.VIDEO,
    checkpoint: str | Path | None = None,
    training: bool = True,
    dtype: jnp.dtype = jnp.float32,
    implementation: AttentionImplementation = "xla",
    rematerialization: RematerializationPolicy = "full",
) -> tuple[VJEPA2_1Model, Processor]:
    """Construct the native model and its image/video processor exactly once."""

    architecture = (
        VJEPA2_1Config() if config is None else VJEPA2_1Config.model_validate(config)
    )
    model = VJEPA2_1Model.init(
        architecture,
        key=key,
        dtype=dtype,
        implementation=implementation,
        rematerialization=rematerialization,
    )
    if checkpoint is not None:
        model = load_reference_checkpoint(model, checkpoint)
    processor = make_vjepa2_1_processor(
        architecture,
        modality=modality,
        training=training,
    )
    return model, processor


__all__ = ["load_vjepa2_1"]
