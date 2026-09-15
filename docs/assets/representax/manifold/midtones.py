"""Uniform-color wordmarks comparing shallow and stronger upright depth."""

import hashlib
import json
from dataclasses import replace

import matplotlib
import numpy as np
from generate import HERE
from PIL import Image, ImageDraw, ImageFont
from tax import FONT
from upright import PREFIX, render

# Use TAX's side-face direction, but less depth to preserve the small counters.
UNIFORM_PREFIX = replace(PREFIX, uniform_surface=True)
DEEP_PREFIX = replace(UNIFORM_PREFIX, depth_x=0.42, depth_y=-0.72, gap=1.05)
VARIANTS = (
    ("01-shallow-25", "01  Shallow upright | uniform 25% tint", 0.25, UNIFORM_PREFIX),
    ("02-shallow-40", "02  Shallow upright | uniform 40% tint", 0.40, UNIFORM_PREFIX),
    ("03-dimensional-25", "03  Stronger 3D | uniform 25% tint", 0.25, DEEP_PREFIX),
    ("04-dimensional-40", "04  Stronger 3D | uniform 40% tint", 0.40, DEEP_PREFIX),
)


def generate_variants(output, variants=VARIANTS, extra_sources=()):
    output.mkdir(parents=True, exist_ok=True)
    records = {
        name: render(output, name, opacity, opacity, prefix)
        for name, _, opacity, prefix in variants
    }
    sheet = Image.new("RGB", (3000, 2200), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT), 32)
    for i, (name, label, _, _) in enumerate(variants):
        draw.text((85, 35 + i * 550), label, font=font, fill="black")
        with Image.open(output / f"{name}.png") as image:
            image = image.convert("RGB")
            image.thumbnail((2830, 445), Image.Resampling.LANCZOS)
            sheet.paste(image, ((3000 - image.width) // 2, 85 + i * 550))
    sheet.save(output / "comparison.png")
    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in (
            "midtones.py",
            "upright.py",
            "unified.py",
            "tax.py",
            "saturated.py",
            "lattice.py",
            "generate.py",
            *extra_sources,
        )
    }
    records["packages"] = {
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
    }
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    generate_variants(HERE / "midtone-wordmarks")
