"""Thicken the original A's left arm outwards, without adding a top face."""

import hashlib
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from edge_dots import render
from generate import HERE
from matplotlib.font_manager import FontProperties
from PIL import Image
from structural import PREFIX, TAX_WIDTH, fill_regions, orient, scatter
from tax import FONT, TAX_FACES, save_figure
from wider_a import DETAIL_LENGTH, DETAIL_SPACING, LONG_SPACING, adaptive_nodes

VARIANTS = (("original", 0), ("arm-7", 7), ("arm-14", 14))


def arm_faces(extra):
    if extra < 0:
        raise ValueError("arm expansion must be nonnegative")
    faces = []
    for name, group, shade, vertices in TAX_FACES:
        vertices = np.array(vertices, dtype=float)
        if name == "A left":
            vertices[:2, 0] -= extra
        elif name == "A left cap":
            vertices[[0, 3], 0] -= extra
        faces.append((name, group, shade, vertices))
    return faces


def arm_regions(extra):
    anchor = (28 - 126) / 100
    regions = []
    for _, group, _, vertices in arm_faces(extra):
        polygon = (vertices - (126, 72)) * (0.01, -0.01)
        polygon[:, 0] = anchor + TAX_WIDTH * (polygon[:, 0] - anchor)
        regions.append(([orient(polygon)], group))
    return regions


def main():
    output = HERE / "a-arm-wordmarks"
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, extra in VARIANTS:
        records[name] = render(
            output,
            name,
            0.2,
            0.9,
            regions=arm_regions(extra),
            node_sampler=adaptive_nodes,
        )
        records[name].pop("edge_spacing")
        records[name].update(
            arm_extra_units=extra,
            detail_spacing=DETAIL_SPACING,
            long_spacing=LONG_SPACING,
            detail_length=DETAIL_LENGTH,
            faces=[(n, g, s, v.tolist()) for n, g, s, v in arm_faces(extra)],
        )
        print(
            name,
            records[name]["prefix_points"],
            records[name]["tax_points"],
            flush=True,
        )

    fig = plt.figure(figsize=(27, 8), dpi=200)
    palette = np.array([matplotlib.colors.to_rgb(color) for color in PREFIX.palette])
    bounds = np.concatenate([ring for rings, _ in arm_regions(0) for ring in rings])
    limits = (bounds[:, 0].min() - 0.12, bounds[:, 0].max() + 0.12)
    for i, (_, extra) in enumerate(VARIANTS):
        label = ("Original", "Left arm 1.5x", "Left arm 2x")[i]
        fig.text(
            i / 3 + 0.018,
            0.91,
            label,
            fontproperties=FontProperties(fname=FONT, size=20),
        )
        ax = fig.add_axes((i / 3 + 0.008, 0.06, 1 / 3 - 0.016, 0.79))
        ax.set(xlim=limits, ylim=(-0.86, 0.86), aspect="equal")
        ax.set_axis_off()
        regions = arm_regions(extra)
        fill_regions(ax, regions, palette, opacity=0.2)
        points, groups = adaptive_nodes(regions)
        scatter(ax, points, palette[groups], diameter=7, opacity=0.9)
    save_figure(fig, output, "comparison")

    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in (
            "a_arm.py",
            "wider_a.py",
            "edge_dots.py",
            "structural.py",
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


if __name__ == "__main__":
    main()
