"""Condensed, fixed-advance prefix with the original interlocking TAX mark."""

import hashlib
import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
from a_arm import arm_regions
from extended import fitted_join, section_extent
from generate import HERE
from matplotlib.colors import to_rgb
from matplotlib.font_manager import findfont
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw, ImageFont
from structural import PREFIX, TAX_WIDTH, fill_regions, prefix_regions, scatter
from tax import save_figure
from unified import layout, rings_for
from upright import placement
from wider_a import adaptive_nodes

VARIANTS = (("current", None), ("mono-80", 0.8), ("mono-65", 0.65))
CELL_WIDTH = 4.2
CELL_GAP = 0.6
JOIN_GAP = 0.1


def mono_layout(width):
    advance = CELL_WIDTH * width + CELL_GAP
    letters = []
    for index, letter in enumerate(PREFIX.text):
        rings = rings_for(letter)
        glyph_width = np.ptp(rings[0][:, 0])
        for ring in rings:
            ring[:, 0] = (ring[:, 0] + (CELL_WIDTH - glyph_width) / 2) * width
        letters.append((letter, index * advance, rings))
    return letters


def composition(width, arm_extra):
    mark = arm_regions(arm_extra)
    bounds = np.concatenate([ring for rings, _ in mark for ring in rings])
    mapping, _ = placement((bounds,), PREFIX, fitted_join(TAX_WIDTH))
    letters = layout(PREFIX) if width is None else mono_layout(width)
    prefix = [prefix_regions(rings, offset) for _, offset, rings in letters]
    if width is not None:
        scale = mapping.get_matrix()[0, 0]
        y = mapping.get_matrix()[1, 2]
        mapping = Affine2D().scale(scale).translate(0, y)
        heights = np.linspace(
            bounds[:, 1].min() + 1e-7, bounds[:, 1].max() - 1e-7, 4096
        )
        right = np.max(
            [
                section_extent(mapping.transform(rings[0]), heights)[1]
                for letter in prefix
                for rings, _ in letter
            ],
            axis=0,
        )
        left = np.min(
            [section_extent(rings[0], heights)[0] for rings, _ in mark], axis=0
        )
        clearance = left - right
        shift = clearance[np.isfinite(clearance)].min() - JOIN_GAP
        mapping.translate(shift, 0)
    cloud = np.concatenate(
        [
            mapping.transform(ring)
            for letter in prefix
            for rings, _ in letter
            for ring in rings
        ]
        + [bounds]
    )
    limits = (
        cloud[:, 0].min() - 0.2,
        cloud[:, 0].max() + 0.2,
        bounds[:, 1].min() - 0.24,
        bounds[:, 1].max() + 0.24,
    )
    return prefix, mark, mapping, limits


def render(
    output, name, width, pixels_per_unit, *, arm_extra, fill_opacity, point_opacity
):
    prefix, mark, mapping, (left, right, bottom, top) = composition(width, arm_extra)
    dpi = 200
    fig = plt.figure(
        figsize=(
            (right - left) * pixels_per_unit / dpi,
            (top - bottom) * pixels_per_unit / dpi,
        ),
        dpi=dpi,
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(left, right), ylim=(bottom, top), aspect="equal")
    ax.set_axis_off()
    palette = np.array([to_rgb(color) for color in PREFIX.palette])
    fill_regions(ax, mark, palette, opacity=fill_opacity)
    mark_points, groups = adaptive_nodes(mark)
    scatter(ax, mark_points, palette[groups], opacity=point_opacity)
    prefix_points = []
    for index, letter in enumerate(prefix):
        fill_regions(
            ax, letter, np.zeros((1, 3)), mapping + ax.transData, opacity=fill_opacity
        )
        points, _ = adaptive_nodes(letter, scale=mapping.get_matrix()[0, 0])
        scatter(
            ax,
            points,
            np.zeros((len(points), 3)),
            transform=mapping + ax.transData,
            opacity=point_opacity,
        )
        prefix_points.extend((index, *point) for point in mapping.transform(points))
    plt.rcParams["svg.hashsalt"] = "representax-condensed-mono"
    save_figure(fig, output, name)
    bounds = np.concatenate([ring for rings, _ in mark for ring in rings])
    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(
        xlim=(bounds[:, 0].min() - 0.14, bounds[:, 0].max() + 0.14),
        ylim=(bounds[:, 1].min() - 0.14, bounds[:, 1].max() + 0.14),
        aspect="equal",
    )
    ax.set_axis_off()
    fill_regions(ax, mark, palette, opacity=fill_opacity)
    scatter(ax, mark_points, palette[groups], opacity=point_opacity)
    plt.rcParams["svg.hashsalt"] = "representax-edge-dots"
    save_figure(fig, output, f"{name}-tax")
    np.savetxt(
        output / f"{name}-tax.csv",
        np.column_stack((mark_points, groups)),
        delimiter=",",
        header="x,y,color_group",
        comments="",
        fmt="%.10f",
    )
    np.savetxt(
        output / f"{name}-prefix.csv",
        prefix_points,
        delimiter=",",
        header="letter,x,y",
        comments="",
        fmt="%.10f",
    )
    return {
        "fill_opacity": fill_opacity,
        "point_opacity": point_opacity,
        "glyph_width_scale": width,
        "cell_advance": None if width is None else CELL_WIDTH * width + CELL_GAP,
        "cell_width": CELL_WIDTH,
        "cell_gap": CELL_GAP,
        "join_gap": JOIN_GAP,
        "width": right - left,
        "prefix_points": len(prefix_points),
        "tax_points": len(mark_points),
        "prefix": asdict(PREFIX),
        "tax_width": TAX_WIDTH,
        "prefix_transform": mapping.get_matrix().tolist(),
    }


def main():
    from logo import ARM_EXTRA, FILL_OPACITY, POINT_OPACITY

    output = HERE / "condensed-wordmarks"
    output.mkdir(exist_ok=True)
    limits = composition(None, ARM_EXTRA)[3]
    pixels_per_unit = PREFIX.width_px / (limits[1] - limits[0])
    records = {
        name: render(
            output,
            name,
            width,
            pixels_per_unit,
            arm_extra=ARM_EXTRA,
            fill_opacity=FILL_OPACITY,
            point_opacity=POINT_OPACITY,
        )
        for name, width in VARIANTS
    }
    sheet = Image.new("RGB", (3000, 1830), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(findfont("DejaVu Sans"), 34)
    for index, (name, width) in enumerate(VARIANTS):
        reduction = 1 - records[name]["width"] / records["current"]["width"]
        label = (
            "Current"
            if width is None
            else f"{name[-2:]}% glyph width / fixed advance / "
            f"{reduction:.0%} shorter overall"
        )
        draw.text((80, 35 + index * 610), label, font=font, fill="#303030")
        with Image.open(output / f"{name}.png") as image:
            scale = 2840 / PREFIX.width_px
            preview = image.convert("RGB").resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
            sheet.paste(preview, (80, 120 + index * 610))
    sheet.save(output / "comparison.png")
    records["source_sha256"] = hashlib.sha256(
        (HERE / "condensed.py").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
