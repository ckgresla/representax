"""Reproducible experiment programs kept outside the library package."""

from __future__ import annotations

import sys

from experiments import preflights as _preflights

# Saved run manifests may still contain the former module prefix.
sys.modules.setdefault(f"{__name__}.paper", _preflights)
