"""Upright geometric REPRESEN with the original JAX-like TAX faces."""

import hashlib
import json
from dataclasses import asdict, replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from generate import HERE
from lattice import canvas, lattice
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw, ImageFont
from saturated import CONFIG as TAX_CONFIG
from tax import FONT, TAX_FACES, save_figure
from unified import CONFIG, draw_letters, layout, project

PREFIX = replace(CONFIG, text="REPRESEN", shear=0, grid_step=0.15)
VARIANTS = (
    ("01-equal-opacity", "01  Upright + JAX-like TAX | 10% faces throughout", 0.1, 0.1),
    (
        "02-stronger-tax",
        "02  Upright black + stronger TAX | 100% black / 65% color",
        1,
        0.65,
    ),
)


def placement(geometry, prefix=PREFIX, join_gap=0.09):
    low, high = geometry[0].min(axis=0), geometry[0].max(axis=0)
    scale = (high[1] - low[1]) / (prefix.cap_height - prefix.depth_y)
    width = (
        max(
            project(rings[0], offset, prefix)[:, 0].max()
            for _, offset, rings in layout(prefix)
        )
        + prefix.depth_x
    )
    x = low[0] - join_gap - width * scale
    y = low[1] - prefix.depth_y * scale
    return Affine2D().scale(scale).translate(x, y), (
        x - 0.2,
        high[0] + 0.2,
        low[1] - 0.24,
        high[1] + 0.24,
    )


def render(
    output,
    name,
    prefix_opacity,
    tax_opacity,
    prefix=PREFIX,
    *,
    tax_width=1.0,
    join_gap=0.09,
):
    config = replace(
        TAX_CONFIG, fill_opacity=tax_opacity, diameter_px=prefix.diameter_px
    )
    faces = (
        tuple((name, group, 1.0, vertices) for name, group, _, vertices in TAX_FACES)
        if prefix.uniform_surface
        else TAX_FACES
    )
    geometry = lattice(config, faces)
    xy = geometry[0].copy()
    xy[:, 0] = xy[:, 0].min() + tax_width * (xy[:, 0] - xy[:, 0].min())
    geometry = (xy, *geometry[1:])
    mapping, (left, right, bottom, top) = placement(geometry, prefix, join_gap)
    fig, _, _ = canvas(config, *geometry, patches=faces)
    ax = fig.axes[0]
    fig.set_size_inches(
        prefix.width_px / fig.dpi,
        prefix.width_px / fig.dpi * (top - bottom) / (right - left),
    )
    ax.set(xlim=(left, right), ylim=(bottom, top), aspect="equal")
    points = draw_letters(ax, prefix_opacity, prefix, mapping + ax.transData)
    points[:, 1:] = mapping.transform(points[:, 1:])
    plt.rcParams["svg.hashsalt"] = "representax-upright-tax"
    save_figure(fig, output, name)
    np.savetxt(
        output / f"{name}-prefix-points.csv",
        points,
        delimiter=",",
        header="letter_index,x,y",
        comments="",
        fmt="%.10f",
    )
    np.savetxt(
        output / f"{name}-tax-points.csv",
        geometry[0],
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.10f",
    )
    return {
        "prefix": asdict(prefix),
        "prefix_opacity": prefix_opacity,
        "tax": asdict(config),
        "tax_faces": faces,
        "tax_width": tax_width,
        "join_gap": join_gap,
        "prefix_transform": mapping.get_matrix().tolist(),
    }


def main():
    output = HERE / "upright-wordmarks"
    output.mkdir(parents=True, exist_ok=True)
    records = {
        name: render(output, name, prefix_opacity, tax_opacity)
        for name, _, prefix_opacity, tax_opacity in VARIANTS
    }
    sheet = Image.new("RGB", (2800, 1200), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT), 30)
    for i, (name, label, _, _) in enumerate(VARIANTS):
        draw.text((85, 60 + i * 590), label, font=font, fill="black")
        with Image.open(output / f"{name}.png") as image:
            image = image.convert("RGB")
            image.thumbnail((2660, 450), Image.Resampling.LANCZOS)
            sheet.paste(image, ((2800 - image.width) // 2, 135 + i * 590))
    sheet.save(output / "comparison.png")
    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in (
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
