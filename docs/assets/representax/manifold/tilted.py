"""Italic 3D prefix with uniform midtone colors and the JAX-like TAX mark."""

from dataclasses import replace

from generate import HERE
from midtones import DEEP_PREFIX, UNIFORM_PREFIX, generate_variants
from unified import CONFIG

SHALLOW = replace(UNIFORM_PREFIX, shear=CONFIG.shear)
DEEP = replace(DEEP_PREFIX, shear=CONFIG.shear)
VARIANTS = (
    ("01-italic-25", "01  Italic + shallow 3D | uniform 25% tint", 0.25, SHALLOW),
    ("02-italic-40", "02  Italic + shallow 3D | uniform 40% tint", 0.40, SHALLOW),
    ("03-italic-deep-25", "03  Italic + stronger 3D | uniform 25% tint", 0.25, DEEP),
    ("04-italic-deep-40", "04  Italic + stronger 3D | uniform 40% tint", 0.40, DEEP),
)


if __name__ == "__main__":
    generate_variants(HERE / "tilted-wordmarks", VARIANTS, extra_sources=("tilted.py",))
