"""Shared Pydantic boundary for declarative Representax configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenConfig(BaseModel):
    """Immutable, closed configuration accepted from Python or Hydra-Zen."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


__all__ = ["FrozenConfig"]
