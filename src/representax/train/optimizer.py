"""Construct Optax transformations from declarative configuration."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import optax

from representax.config import ComponentConfig, OptimizationConfig


def _resolve_optimizer_factory(target: str) -> Callable[..., Any]:
    module_name, separator, attribute_name = target.rpartition(".")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "optimizer target must be a dotted import path such as 'optax.adamw'"
        )

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            raise ImportError(
                f"could not import optimizer module {module_name!r}"
            ) from error
        raise

    try:
        factory = getattr(module, attribute_name)
    except AttributeError as error:
        raise AttributeError(
            f"optimizer module {module_name!r} has no attribute {attribute_name!r}"
        ) from error
    if not callable(factory):
        raise TypeError(f"optimizer target {target!r} is not callable")
    return factory


def build_schedule(config: ComponentConfig | None) -> Callable[[Any], Any] | None:
    """Build an optional Optax-compatible scalar schedule."""

    if config is None:
        return None
    factory = _resolve_optimizer_factory(config.target)
    schedule = factory(**config.parameters)
    if not callable(schedule):
        raise TypeError(f"schedule target {config.target!r} must return a callable")
    return schedule


def build_optimizer(
    config: OptimizationConfig,
) -> optax.GradientTransformationExtraArgs:
    """Build the configured Optax transformation from a trusted recipe.

    Importing a dotted target executes its module, so callers should only build
    optimizer configurations from recipes they trust.
    """

    component = config.optimizer
    factory = _resolve_optimizer_factory(component.target)
    parameters: dict[str, Any] = dict(component.parameters)
    schedule = build_schedule(config.schedule)
    if schedule is not None:
        if config.schedule_parameter in parameters:
            raise ValueError(
                f"optimizer parameters already define {config.schedule_parameter!r}"
            )
        parameters[config.schedule_parameter] = schedule
    optimizer = factory(**parameters)
    if not isinstance(
        optimizer,
        (optax.GradientTransformation, optax.GradientTransformationExtraArgs),
    ):
        raise TypeError(
            f"optimizer target {component.target!r} must return an Optax "
            "GradientTransformation"
        )
    return optimizer


__all__ = ["build_optimizer", "build_schedule"]
