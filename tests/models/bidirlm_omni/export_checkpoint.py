"""Export the pinned native BidirLM Omni checkpoint for upstream reload."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp

from representax.models.bidirlm_omni import BidirLMOmniCheckpointAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    adapter = BidirLMOmniCheckpointAdapter(rematerialization="none")
    model = adapter.load(
        arguments.checkpoint,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
    )
    adapter.save(
        model,
        arguments.output,
        source_checkpoint=arguments.checkpoint,
    )


if __name__ == "__main__":
    main()
