"""Light uniform fills and seam-defining dot chains on the existing geometry."""

import hashlib
import json
from dataclasses import asdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from extended import fitted_join
from generate import HERE
from PIL import Image, ImageDraw, ImageFont
from structural import (
    PREFIX,
    TAX_WIDTH,
    fill_regions,
    prefix_regions,
    scatter,
    tax_regions,
    visible_nodes,
)
from tax import FONT, save_figure
from unified import layout
from upright import placement

EDGE_SPACING = 0.055
VARIANTS = tuple(
    (f"fill-{fill}-points-{points}", fill / 100, points / 100)
    for fill in (20, 30)
    for points in (80, 90)
)


def render(
    output, name, fill_opacity, point_opacity, *, regions=None, node_sampler=None
):
    regions = tax_regions() if regions is None else regions
    if node_sampler is None:

        def node_sampler(faces, scale=1):
            return visible_nodes(faces, EDGE_SPACING / scale)

    bounds = np.concatenate([ring for rings, _ in regions for ring in rings])
    mapping, (left, right, bottom, top) = placement(
        (bounds,), PREFIX, fitted_join(TAX_WIDTH)
    )
    palette = np.array([matplotlib.colors.to_rgb(color) for color in PREFIX.palette])
    tax_points, groups = node_sampler(regions)
    fig = plt.figure(
        figsize=(
            PREFIX.width_px / 200,
            PREFIX.width_px / 200 * (top - bottom) / (right - left),
        ),
        dpi=200,
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(left, right), ylim=(bottom, top), aspect="equal")
    ax.set_axis_off()
    fill_regions(ax, regions, palette, opacity=fill_opacity)
    scatter(ax, tax_points, palette[groups], opacity=point_opacity)
    prefix_points = []
    for index, (_, offset, rings) in enumerate(layout(PREFIX)):
        letter = prefix_regions(rings, offset)
        fill_regions(
            ax, letter, np.zeros((1, 3)), mapping + ax.transData, opacity=fill_opacity
        )
        points, _ = node_sampler(letter, scale=mapping.get_matrix()[0, 0])
        scatter(
            ax,
            points,
            np.zeros((len(points), 3)),
            transform=mapping + ax.transData,
            opacity=point_opacity,
        )
        prefix_points.extend((index, *point) for point in mapping.transform(points))
    plt.rcParams["svg.hashsalt"] = "representax-edge-dots"
    save_figure(fig, output, name)
    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(
        xlim=(bounds[:, 0].min() - 0.14, bounds[:, 0].max() + 0.14),
        ylim=(bounds[:, 1].min() - 0.14, bounds[:, 1].max() + 0.14),
        aspect="equal",
    )
    ax.set_axis_off()
    fill_regions(ax, regions, palette, opacity=fill_opacity)
    scatter(ax, tax_points, palette[groups], opacity=point_opacity)
    save_figure(fig, output, f"{name}-tax")
    np.savetxt(
        output / f"{name}-prefix.csv",
        prefix_points,
        delimiter=",",
        header="letter,x,y",
        comments="",
        fmt="%.10f",
    )
    np.savetxt(
        output / f"{name}-tax.csv",
        np.column_stack((tax_points, groups)),
        delimiter=",",
        header="x,y,color_group",
        comments="",
        fmt="%.10f",
    )
    return {
        "fill_opacity": fill_opacity,
        "point_opacity": point_opacity,
        "edge_spacing": EDGE_SPACING,
        "prefix_points": len(prefix_points),
        "tax_points": len(tax_points),
        "prefix": asdict(PREFIX),
        "tax_width": TAX_WIDTH,
        "prefix_transform": mapping.get_matrix().tolist(),
    }


def main():
    output = HERE / "edge-dot-wordmarks"
    output.mkdir(parents=True, exist_ok=True)
    records = {
        name: render(output, name, fill, points) for name, fill, points in VARIANTS
    }
    sheet = Image.new("RGB", (6000, 4400), "white")
    details = Image.new("RGB", (4400, 3200), "white")
    font = ImageFont.truetype(str(FONT), 64)
    detail_font = ImageFont.truetype(str(FONT), 46)
    for i, (name, fill, points) in enumerate(VARIANTS):
        label = f"{fill:.0%} uniform fill | {points:.0%} points"
        ImageDraw.Draw(sheet).text((170, 70 + i * 1100), label, font=font, fill="black")
        with Image.open(output / f"{name}.png") as image:
            image = image.convert("RGB")
            image.thumbnail((5660, 890), Image.Resampling.LANCZOS)
            sheet.paste(image, ((6000 - image.width) // 2, 170 + i * 1100))
        x, y = (i % 2) * 2200, (i // 2) * 1600
        ImageDraw.Draw(details).text(
            (x + 100, y + 70), label, font=detail_font, fill="black"
        )
        with Image.open(output / f"{name}-tax.png") as image:
            image = image.convert("RGB")
            image.thumbnail((2100, 1350), Image.Resampling.LANCZOS)
            details.paste(image, (x + (2200 - image.width) // 2, y + 190))
    sheet.save(output / "comparison.png")
    details.save(output / "tax-comparison.png")
    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in (
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
    print(
        json.dumps(
            {
                name: (records[name]["prefix_points"], records[name]["tax_points"])
                for name, _, _ in VARIANTS
            }
        )
    )


if __name__ == "__main__":
    main()
