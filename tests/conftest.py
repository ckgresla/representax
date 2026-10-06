"""Repository-wide numerical policy for correctness and parity tests."""

from __future__ import annotations

import os

# Correctness gates should not depend on the invoking shell's accelerator
# defaults. Individual mixed-precision tests still enter explicit policies.
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
