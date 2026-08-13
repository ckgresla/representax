"""Utilities confined to optional upstream model-reference environments."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager


def configure_torch_float32_highest() -> None:
    """Use full FP32 for Torch GEMMs and cuDNN convolutions.

    ``torch.set_float32_matmul_precision("highest")`` does not disable cuDNN's
    independent TF32 policy. Model parity therefore sets all three controls
    explicitly before loading or executing an upstream vision model.
    """

    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def transformers_tacet() -> Iterator[None]:
    """Temporarily silence known Transformers reference-runtime chatter."""

    import transformers

    previous = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"transformers(?:\..*)?")
            yield
    finally:
        transformers.logging.set_verbosity(previous)
