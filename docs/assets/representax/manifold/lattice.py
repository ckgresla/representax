"""Regular face vertices and faint interpolated color on the analytic mark."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from generate import HERE, PALETTE, PATCHES, embed, project
from matplotlib.collections import TriMesh
from matplotlib.colors import to_rgb
from matplotlib.tri import Triangulation
from PIL import Image


@dataclass(frozen=True)
class LatticeConfig:
    divisions: int = 8
    deformation: float = 0.7
    fill_opacity: float = 0.09
    diameter_px: float = 3.2
    size_px: int = 2800
    yaw_degrees: float = -12.0
    pitch_degrees: float = 14.0
    palette: tuple[str, str, str] = PALETTE
    point_opacity: float = 0.96
    gradient_floor: float = 0.35


def face_labels(xy, patches=PATCHES):
    """Boundary-inclusive membership in convex face domains, in draw order."""
    labels = np.full(len(xy), -1, dtype=int)
    for i, (_, _, _, vertices) in enumerate(patches):
        vertices = (np.array(vertices) - (126, 72)) * (0.01, -0.01)
        edges = np.roll(vertices, -1, axis=0) - vertices
        delta = xy[:, None, :] - vertices
        cross = edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
        inside = np.all(cross >= -1e-10, axis=1) | np.all(cross <= 1e-10, axis=1)
        labels[inside] = i
    return labels


def lattice(config, patches=PATCHES):
    if config.divisions < 2 or config.divisions % 2:
        raise ValueError("divisions must be a positive even integer of at least two")
    step = 28 / config.divisions
    rows = 6 * config.divisions + 1
    columns = 12 * config.divisions + 1
    i, j = np.meshgrid(np.arange(columns) - 3 * config.divisions, np.arange(rows))
    # The three edge directions are those of the original oblique face grid.
    drawing = np.column_stack(
        ((i + j / 2).ravel() * step, j.ravel() * 24 / config.divisions)
    )
    xy = (drawing - (126, 72)) * (0.01, -0.01)
    ids = np.arange(len(xy)).reshape(rows, columns)
    a, b = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel()
    c, d = ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
    triangles = np.concatenate((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    faces = face_labels(xy[triangles].mean(axis=1), patches)
    triangles, faces = triangles[faces >= 0], faces[faces >= 0]
    used, inverse = np.unique(triangles, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    # Shared vertices are drawn once, with the foremost incident face's color.
    labels = np.full(len(used), -1, dtype=int)
    np.maximum.at(labels, triangles.ravel(), np.repeat(faces, 3))
    return xy[used], triangles, faces, labels


def face_color(face, palette=PALETTE, patches=PATCHES):
    _, group, shade, _ = patches[face]
    return np.clip(np.array(to_rgb(palette[group])) * shade, 0, 1)


def canvas(config, xy, triangles, faces, labels, patches=PATCHES):
    xyz, _ = embed(xy, config.deformation)
    screen = project(xyz, config)
    dpi = 200
    fig = plt.figure(figsize=(config.size_px / dpi, config.size_px / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(-1.60, 1.60), ylim=(-1.60, 1.60), aspect="equal")
    ax.set_axis_off()
    for face in np.unique(faces):
        cells = triangles[faces == face]
        members = np.unique(cells)
        height = xy[members, 1]
        t = np.clip((xy[:, 1] - height.min()) / max(np.ptp(height), 1e-9), 0, 1)
        alpha = config.fill_opacity * (
            config.gradient_floor + (1 - config.gradient_floor) * t
        )
        # Precompose against the white page: translucent triangle edges otherwise
        # double-blend in Gouraud renderers and introduce an unwanted wire grid.
        tint = 1 - alpha[:, None] * (1 - face_color(face, config.palette, patches))
        rgba = np.column_stack((tint, np.ones(len(xy))))
        mesh = TriMesh(
            Triangulation(screen[:, 0], screen[:, 1], triangles=cells),
            facecolors=rgba,
            edgecolors="none",
            linewidths=0,
        )
        ax.add_collection(mesh)
    colors = np.array([face_color(face, config.palette, patches) for face in labels])
    ax.scatter(
        screen[:, 0],
        screen[:, 1],
        s=(config.diameter_px * 72 / dpi) ** 2,
        c=colors,
        alpha=config.point_opacity,
        linewidths=0,
        edgecolors="none",
        zorder=3,
    )
    return fig, xyz, screen


def save_mark(output, name, config, geometry, patches=PATCHES):
    xy, triangles, faces, labels = geometry
    fig, xyz, screen = canvas(config, *geometry, patches=patches)
    plt.rcParams["svg.hashsalt"] = "representax-lattice"
    for suffix in ("png", "pdf", "svg"):
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if suffix == "pdf"
            else {"Date": None}
            if suffix == "svg"
            else None
        )
        fig.savefig(output / f"{name}.{suffix}", facecolor="white", metadata=metadata)
    plt.close(fig)
    np.savetxt(
        output / f"{name}-vertices.csv",
        np.column_stack((xy, xyz, screen[:, :2], labels)),
        delimiter=",",
        header="source_x,source_y,x,y,z,screen_x,screen_y,face",
        comments="",
        fmt="%.10f",
    )
    np.savetxt(
        output / f"{name}-faces.csv",
        np.column_stack((triangles, faces)),
        delimiter=",",
        header="vertex_0,vertex_1,vertex_2,face",
        comments="",
        fmt="%d",
    )
    return {"config": asdict(config), "vertices": len(xy), "triangles": len(triangles)}


def save_animation(output, config, geometry):
    cfg = replace(
        config, size_px=960, diameter_px=config.diameter_px * 960 / config.size_px
    )
    frames = []
    for phase in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        amount = config.deformation * (0.5 - 0.5 * np.cos(phase))
        fig, _, _ = canvas(replace(cfg, deformation=amount), *geometry)
        fig.canvas.draw()
        frames.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
        )
        plt.close(fig)
    # A fixed palette prevents subtle face gradients changing color per frame.
    palette = frames[32].quantize(colors=256)
    frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    frames[0].save(
        output / "lattice.gif",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )


def save_comparison(output, variants):
    from PIL import ImageDraw, ImageFont

    sheet = Image.new("RGB", (2400, 2100), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(
        str(HERE.parents[3] / "paper/assets/InterVariable.ttf"), 28
    )
    small = ImageFont.truetype(
        str(HERE.parents[3] / "paper/assets/InterVariable.ttf"), 22
    )
    for i, (name, title, cfg) in enumerate(variants):
        x, y = (i % 2) * 1200, (i // 2) * 1050
        with Image.open(output / f"{name}.png") as image:
            # Crop page margins only; keep the complete object and its point size.
            preview = image.convert("RGB").crop((100, 520, 2700, 2240))
            preview = preview.resize((1100, 728), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x + 50, y + 165))
        draw.text((x + 75, y + 75), title, fill="#303a46", font=font)
        count = len(lattice(cfg)[0])
        detail = (
            f"{count:,} vertices | bend {cfg.deformation:g} | "
            f"tint {100 * cfg.fill_opacity:g}% max"
        )
        draw.text((x + 75, y + 120), detail, fill="#68717c", font=small)
    sheet.save(output / "comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "lattice-variants")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = LatticeConfig(fill_opacity=0.08)
    variants = (
        ("01-geometric", "01  Geometric", replace(config, deformation=0)),
        ("02-manifold", "02  Gentle manifold", config),
        ("03-whisper", "03  Lighter surface", replace(config, fill_opacity=0.035)),
        ("04-open", "04  Open lattice", replace(config, divisions=4)),
    )
    records = {}
    for name, _, cfg in variants:
        records[name] = save_mark(args.output, name, cfg, lattice(cfg))
    save_comparison(args.output, variants)
    if args.animate:
        save_animation(args.output, config, lattice(config))
    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("lattice.py", "generate.py")
    }
    records["packages"] = {
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
    }
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
