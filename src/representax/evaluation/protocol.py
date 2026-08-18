"""Typed boundary between compiled evaluation and host-side reduction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeVar

import equinox as eqx
from jaxtyping import PRNGKeyArray

_BatchOutput = TypeVar("_BatchOutput")
_Accumulator = TypeVar("_Accumulator")


class Evaluator(Protocol[_BatchOutput, _Accumulator]):
    """A compiled per-batch computation plus an exact host reducer.

    ``evaluate_batch`` is the only method traced by JAX. Accumulator construction,
    updates, and final metric naming remain ordinary Python so evaluators may
    perform corpus-level reductions without retaining device buffers.
    """

    @property
    def name(self) -> str: ...

    def evaluate_batch(
        self,
        model: eqx.Module,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> _BatchOutput: ...

    def initialize(self) -> _Accumulator: ...

    def accumulate(
        self,
        accumulator: _Accumulator,
        output: _BatchOutput,
    ) -> _Accumulator: ...

    def finalize(self, accumulator: _Accumulator) -> Mapping[str, float]: ...


__all__ = ["Evaluator"]
