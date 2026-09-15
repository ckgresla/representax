"""Optically joined wordmarks with a modestly wider interlocking TAX."""

import hashlib
import json
from dataclasses import replace

import matplotlib
import numpy as np
from generate import HERE
from lattice import lattice
from PIL import Image, ImageDraw, ImageFont
from saturated import CONFIG as TAX_CONFIG
from scipy.spatial import ConvexHull
from tax import FONT, TAX_FACES
from tilted import SHALLOW
from unified import layout, project
from upright import placement, render

VARIANTS = tuple(
    (
        f"extended-{extra}-{opacity}",
        f"TAX {extra}% wider | uniform {opacity}% tint",
        1 + extra / 100,
        opacity / 100,
    )
    for extra in (15, 30)
    for opacity in (20, 30, 40)
)
VISIBLE_GAP = 0.1
HIRES_PREFIX = replace(SHALLOW, width_px=11200, diameter_px=6.4)


def section_extent(polygon, heights):
    start, end = polygon, np.roll(polygon, -1, axis=0)
    dy = end[:, 1] - start[:, 1]
    cross = (heights[:, None] >= np.minimum(start[:, 1], end[:, 1])) & (
        heights[:, None] < np.maximum(start[:, 1], end[:, 1])
    )
    fraction = np.divide(
        heights[:, None] - start[:, 1],
        dy,
        out=np.zeros((len(heights), len(start))),
        where=dy != 0,
    )
    x = start[:, 0] + fraction * (end[:, 0] - start[:, 0])
    return np.where(cross, x, np.inf).min(axis=1), np.where(cross, x, -np.inf).max(
        axis=1
    )


def fitted_join(tax_width, visible_gap=VISIBLE_GAP):
    geometry = lattice(TAX_CONFIG, TAX_FACES)
    xy = geometry[0].copy()
    anchor = xy[:, 0].min()
    xy[:, 0] = anchor + tax_width * (xy[:, 0] - anchor)
    mapping, _ = placement((xy, *geometry[1:]), SHALLOW)
    _, offset, rings = layout(SHALLOW)[-1]
    front = project(rings[0], offset, SHALLOW)
    cloud = mapping.transform(
        np.concatenate((front, front + (SHALLOW.depth_x, SHALLOW.depth_y)))
    )
    # N's right exterior is convex; counters do not affect this envelope.
    outline = cloud[ConvexHull(cloud).vertices]
    heights = np.linspace(xy[:, 1].min() + 1e-7, xy[:, 1].max() - 1e-7, 4096)
    _, prefix_right = section_extent(outline, heights)
    mark_left = np.full(len(heights), np.inf)
    for _, _, _, vertices in TAX_FACES:
        polygon = (np.array(vertices, dtype=float) - (126, 72)) * (0.01, -0.01)
        polygon[:, 0] = anchor + tax_width * (polygon[:, 0] - anchor)
        mark_left = np.minimum(mark_left, section_extent(polygon, heights)[0])
    clearance = mark_left - prefix_right
    shift = clearance[np.isfinite(clearance)].min() - visible_gap
    return 0.09 - float(shift)


def main():
    output = HERE / "extended-wordmarks"
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, _, width, opacity in VARIANTS:
        records[name] = render(
            output,
            name,
            opacity,
            opacity,
            HIRES_PREFIX,
            tax_width=width,
            join_gap=fitted_join(width),
        )
        records[name]["minimum_horizontal_outline_gap"] = VISIBLE_GAP
    font = ImageFont.truetype(str(FONT), 64)
    for extra in (15, 30):
        sheet = Image.new("RGB", (6000, 3300), "white")
        draw = ImageDraw.Draw(sheet)
        variants = [v for v in VARIANTS if v[0].startswith(f"extended-{extra}-")]
        for i, (name, label, _, _) in enumerate(variants):
            draw.text((170, 70 + i * 1100), label, font=font, fill="black")
            with Image.open(output / f"{name}.png") as image:
                image = image.convert("RGB")
                image.thumbnail((5660, 890), Image.Resampling.LANCZOS)
                sheet.paste(image, ((6000 - image.width) // 2, 170 + i * 1100))
        sheet.save(output / f"comparison-{extra}.png")
    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in (
            "extended.py",
            "tilted.py",
            "midtones.py",
            "upright.py",
            "unified.py",
            "tax.py",
            "saturated.py",
            "lattice.py",
            "generate.py",
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
    main()
