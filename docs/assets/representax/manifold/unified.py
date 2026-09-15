"""One analytic, oblique display alphabet for the entire Representax wordmark."""

import hashlib
import json
from dataclasses import asdict, dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from generate import HERE
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from PIL import Image, ImageDraw, ImageFont
from saturated import CONFIG as COLOR_CONFIG
from tax import save_figure

# All glyphs use a six-unit cap height and roughly one-unit strokes. Rings are
# explicit polygons, not text from a second typeface or traced raster artwork.
BOWL = ((0.95, 3.95), (2.6, 3.95), (3.05, 4.4), (3.05, 4.6), (2.6, 5.05), (0.95, 5.05))
GLYPHS = {
    "R": (
        (
            (0, 0),
            (0, 6),
            (3, 6),
            (4, 5),
            (4, 4),
            (3, 3),
            (2.6, 3),
            (4.2, 0),
            (3.1, 0),
            (1.5, 3),
            (0.95, 3),
            (0.95, 0),
        ),
        (BOWL,),
    ),
    "E": (
        (
            (0, 0),
            (0, 6),
            (4, 6),
            (4, 5.05),
            (0.95, 5.05),
            (0.95, 3.55),
            (3.4, 3.55),
            (3.4, 2.6),
            (0.95, 2.6),
            (0.95, 0.95),
            (4, 0.95),
            (4, 0),
        ),
        (),
    ),
    "P": (
        ((0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0.95, 3), (0.95, 0)),
        (BOWL,),
    ),
    "S": (
        (
            (0, 0),
            (3, 0),
            (4, 1),
            (4, 3),
            (3, 4),
            (1, 4),
            (1, 5),
            (4, 5),
            (4, 6),
            (1, 6),
            (0, 5),
            (0, 3.8),
            (1, 2.8),
            (3, 2.8),
            (3, 1),
            (0, 1),
        ),
        (),
    ),
    "N": (
        (
            (0, 0),
            (0, 6),
            (1, 6),
            (3.1, 1.9),
            (3.1, 6),
            (4.1, 6),
            (4.1, 0),
            (3.1, 0),
            (1, 4.1),
            (1, 0),
        ),
        (),
    ),
    "T": (
        ((0, 5), (0, 6), (4.4, 6), (4.4, 5), (2.7, 5), (2.7, 0), (1.7, 0), (1.7, 5)),
        (),
    ),
    "A": (
        (
            (0, 0),
            (1.8, 6),
            (2.9, 6),
            (4.7, 0),
            (3.65, 0),
            (3.18, 1.7),
            (1.52, 1.7),
            (1.05, 0),
        ),
        (((1.65, 2.55), (2.35, 5.05), (3.05, 2.55)),),
    ),
    "X": (
        (
            (0, 0),
            (1.65, 3),
            (0, 6),
            (1.1, 6),
            (2.25, 3.9),
            (3.4, 6),
            (4.5, 6),
            (2.85, 3),
            (4.5, 0),
            (3.4, 0),
            (2.25, 2.1),
            (1.1, 0),
        ),
        (),
    ),
}


@dataclass(frozen=True)
class Config:
    text: str = "REPRESENTAX"
    black_prefix_length: int = 8
    cap_height: float = 6
    gap: float = 0.85
    shear: float = 1 / np.sqrt(3)
    depth_x: float = 0.22
    depth_y: float = -0.22 * np.sqrt(3)
    grid_step: float = 0.24
    diameter_px: float = 3.2
    width_px: int = 5600
    face_opacity: float = 0.1
    point_opacity: float = 0.9
    palette: tuple[str, str, str] = COLOR_CONFIG.palette
    uniform_surface: bool = False


CONFIG = Config()


def area(ring):
    return (
        np.sum(
            ring[:, 0] * np.roll(ring[:, 1], -1) - ring[:, 1] * np.roll(ring[:, 0], -1)
        )
        / 2
    )


def rings_for(letter):
    outer, holes = GLYPHS[letter]
    rings = []
    for i, vertices in enumerate((outer, *holes)):
        ring = np.array(vertices, dtype=float)
        # Nonzero winding: counterclockwise outer, clockwise holes.
        if (area(ring) > 0) != (i == 0):
            ring = ring[::-1]
        rings.append(ring)
    return rings


def project(xy, offset, config=CONFIG):
    xy = np.array(xy, dtype=float, copy=True)
    xy[:, 0] += config.shear * xy[:, 1] + offset
    return xy


def closed_path(ring):
    vertices = np.concatenate((ring, ring[:1]))
    return Path(
        vertices, [Path.MOVETO, *([Path.LINETO] * (len(ring) - 1)), Path.CLOSEPOLY]
    )


def layout(config=CONFIG):
    offset = 0
    letters = []
    for letter in config.text:
        rings = rings_for(letter)
        letters.append((letter, offset, rings))
        offset += np.ptp(rings[0][:, 0]) + config.gap
    return letters


def front_points(rings, config=CONFIG):
    width = np.max(rings[0][:, 0])
    x, y = np.meshgrid(
        np.arange(0, width + 1e-9, config.grid_step),
        np.arange(0, config.cap_height + 1e-9, config.grid_step),
    )
    xy = np.column_stack((x.ravel(), y.ravel()))
    keep = closed_path(rings[0]).contains_points(xy, radius=1e-9)
    for ring in rings[1:]:
        keep &= ~closed_path(ring[::-1]).contains_points(xy, radius=-1e-9)
    return xy[keep]


def even_surface_points(rings, config=CONFIG):
    """Keep a consistent projected dot spacing without stacking edge samples."""
    screen = [project(ring, 0, config) for ring in rings]
    depth = np.array((config.depth_x, config.depth_y))
    candidates = [project(front_points(rings, config), 0, config)]
    for ring in screen:
        for start, end in zip(ring, np.roll(ring, -1, axis=0), strict=True):
            edge = end - start
            length = np.linalg.norm(edge)
            thickness = np.array((edge[1], -edge[0])) @ depth / length
            if thickness <= 0:
                continue
            along = np.arange(config.grid_step / 2, length, config.grid_step) / length
            across = (
                np.arange(config.grid_step / 2, thickness, config.grid_step) / thickness
            )
            points = (
                start + along[:, None, None] * edge + across[None, :, None] * depth
            ).reshape(-1, 2)
            hidden = closed_path(screen[0]).contains_points(points, radius=1e-9)
            for hole in screen[1:]:
                hidden &= ~closed_path(hole[::-1]).contains_points(points, radius=-1e-9)
            candidates.append(points[~hidden])
    # Front-grid vertices win at shared edges; side samples cannot crowd them.
    distance = 0.8 * config.grid_step
    bins = {}
    kept = []
    for point in np.concatenate(candidates):
        cell = tuple(np.floor(point / distance).astype(int))
        neighbors = (
            p
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for p in bins.get((cell[0] + dx, cell[1] + dy), ())
        )
        if any(np.linalg.norm(point - p) < distance - 1e-10 for p in neighbors):
            continue
        bins.setdefault(cell, []).append(point)
        kept.append(point)
    return np.array(kept)


def draw_letters(ax, black_fill, config=CONFIG, transform=None):
    letters = layout(config)
    depth = np.array((config.depth_x, config.depth_y))
    transform = ax.transData if transform is None else transform
    dpi = ax.figure.dpi
    points_record = []
    colors = [np.array(matplotlib.colors.to_rgb(c)) for c in config.palette]
    for index, (_letter, offset, rings) in enumerate(letters):
        black = index < config.black_prefix_length
        color = np.zeros(3) if black else colors[index - config.black_prefix_length]
        opacity = black_fill if black else config.face_opacity
        screen_rings = [project(ring, offset, config) for ring in rings]
        side_points = []
        for ring in screen_rings:
            for start, end in zip(ring, np.roll(ring, -1, axis=0), strict=True):
                edge = end - start
                normal = np.array((edge[1], -edge[0]))
                if normal @ depth <= 1e-10:
                    continue
                quad = np.array((start, end, end + depth, start + depth))
                side = (
                    color
                    if config.uniform_surface
                    else np.full(3, 0.23)
                    if black
                    else color * 0.72
                )
                ax.add_patch(
                    Polygon(
                        quad,
                        facecolor=1 - opacity * (1 - side),
                        edgecolor="none",
                        zorder=index * 3,
                        transform=transform,
                    )
                )
                if config.uniform_surface:
                    continue
                n = max(2, int(np.ceil(np.linalg.norm(edge) / config.grid_step)))
                along = np.linspace(0, 1, n + 1)
                across = np.linspace(0, 1, 5)
                side_points.append(
                    (
                        start
                        + along[:, None, None] * edge
                        + across[None, :, None] * depth
                    ).reshape(-1, 2)
                )
        if side_points:
            sides = np.unique(np.round(np.concatenate(side_points), 10), axis=0)
            ax.scatter(
                sides[:, 0],
                sides[:, 1],
                s=(config.diameter_px * 72 / dpi) ** 2,
                color=color,
                alpha=config.point_opacity,
                linewidths=0,
                edgecolors="none",
                zorder=index * 3 + 0.5,
                transform=transform,
            )
        compound = Path.make_compound_path(
            *[closed_path(ring) for ring in screen_rings]
        )
        ax.add_patch(
            PathPatch(
                compound,
                facecolor=1 - opacity * (1 - color),
                edgecolor="none",
                zorder=index * 3 + 1,
                transform=transform,
            )
        )
        points = (
            even_surface_points(rings, config) + (offset, 0)
            if config.uniform_surface
            else project(front_points(rings, config), offset, config)
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=(config.diameter_px * 72 / dpi) ** 2,
            color=color,
            alpha=config.point_opacity,
            linewidths=0,
            edgecolors="none",
            zorder=index * 3 + 2,
            transform=transform,
        )
        if side_points:
            points = np.concatenate((points, sides))
        points = np.unique(np.round(points, 10), axis=0)
        points_record.extend((index, *p) for p in points)
    return np.array(points_record)


def render(output, name, black_fill, config=CONFIG):
    letters = layout(config)
    xmin = -0.9
    xmax = (
        max(
            project(rings[0], offset, config)[:, 0].max()
            for _, offset, rings in letters
        )
        + config.depth_x
        + 0.9
    )
    ymin, ymax = config.depth_y - 1.1, config.cap_height + 1.1
    dpi = 200
    fig = plt.figure(
        figsize=(
            config.width_px / dpi,
            config.width_px / dpi * (ymax - ymin) / (xmax - xmin),
        ),
        dpi=dpi,
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
    ax.set_axis_off()
    points_record = draw_letters(ax, black_fill, config)
    plt.rcParams["svg.hashsalt"] = "representax-unified-wordmark"
    save_figure(fig, output, name)
    np.savetxt(
        output / f"{name}-points.csv",
        np.array(points_record),
        delimiter=",",
        header="letter_index,x,y",
        comments="",
        fmt="%.10f",
    )
    return {
        "config": asdict(config),
        "black_face_opacity": black_fill,
        "vertices": len(points_record),
    }


def main():
    output = HERE / "unified-wordmarks"
    output.mkdir(parents=True, exist_ok=True)
    records = {
        "unified-black": render(output, "unified-black", 1),
        "unified-lattice": render(output, "unified-lattice", 0.1),
    }
    sheet = Image.new("RGB", (2800, 1200), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(
        str(HERE.parents[3] / "paper/assets/InterVariable.ttf"), 30
    )
    for i, (name, label) in enumerate(
        (
            ("unified-black", "01  Black geometric prefix"),
            ("unified-lattice", "02  One lattice treatment throughout"),
        )
    ):
        draw.text((85, 65 + i * 590), label, font=font, fill="black")
        with Image.open(output / f"{name}.png") as image:
            image = image.convert("RGB")
            image.thumbnail((2660, 440), Image.Resampling.LANCZOS)
            sheet.paste(image, ((2800 - image.width) // 2, 150 + i * 590))
    sheet.save(output / "comparison.png")
    records["glyphs"] = GLYPHS
    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in (
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
                name: record
                for name, record in records.items()
                if name.startswith("unified-")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
