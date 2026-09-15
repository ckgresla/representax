"""Open the A counter and concentrate edge dots around geometric landmarks."""

import hashlib
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from edge_dots import render
from generate import HERE
from matplotlib.font_manager import FontProperties
from PIL import Image
from structural import (
    PREFIX,
    TAX_WIDTH,
    cross,
    edges,
    fill_regions,
    inside,
    intersections,
    orient,
    scatter,
)
from tax import FONT, TAX_FACES, save_figure

DETAIL_SPACING = 0.045
LONG_SPACING = 0.14
DETAIL_LENGTH = 0.22
VARIANTS = (("current-adaptive", 0), ("wider-a-14", 14), ("wider-a-28", 28))


def trim_bar_at_a(vertices):
    """End the rear blue face under A's left leg, before its open counter."""
    normal = np.array((1, 7 / 12))
    clipped = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        a, b = start @ normal - 168, end @ normal - 168
        if a <= 1e-10:
            clipped.append(start)
        if (a < -1e-10 and b > 1e-10) or (b < -1e-10 and a > 1e-10):
            clipped.append(start + a / (a - b) * (end - start))
    return np.array(clipped)


def wider_faces(extra):
    if extra < 0:
        raise ValueError("A expansion must be nonnegative")
    faces = []
    for name, group, shade, vertices in TAX_FACES:
        vertices = np.array(vertices, dtype=float)
        if extra and name in ("T bar front", "T bar side"):
            vertices = trim_bar_at_a(vertices)
        elif name.startswith("X "):
            vertices[:, 0] += extra
        elif name == "A right" and extra:
            # Keep both leg slopes; extend the apex into a short horizontal crown.
            vertices = np.array(
                (
                    (140, 24),
                    (168 + extra, 24),
                    (238 + extra, 144),
                    (210 + extra, 144),
                    (147 + extra, 36),
                    (147, 36),
                ),
                dtype=float,
            )
        elif name == "A bridge":
            vertices[2:, 0] += extra
        elif name == "A bridge side":
            vertices[1:3, 0] += extra
        faces.append((name, group, shade, vertices))
        if name == "A right" and extra:
            faces.append(
                (
                    "A crown top",
                    1,
                    1.0,
                    np.array(
                        ((140, 24), (154, 0), (154 + extra, 0), (140 + extra, 24)),
                        dtype=float,
                    ),
                )
            )
    return faces


def wider_regions(extra):
    anchor = (28 - 126) / 100
    regions = []
    for _, group, _, vertices in wider_faces(extra):
        polygon = (vertices - (126, 72)) * (0.01, -0.01)
        polygon[:, 0] = anchor + TAX_WIDTH * (polygon[:, 0] - anchor)
        regions.append(([orient(polygon)], group))
    return regions


def visible_segments(regions):
    all_edges = edges([ring for rings, _ in regions for ring in rings])
    landmarks = np.unique(
        np.round(
            np.concatenate((all_edges[0], intersections(all_edges, all_edges))), 10
        ),
        axis=0,
    )
    segments = {}
    for i, (rings, group) in enumerate(regions):
        for start, end in zip(*edges(rings), strict=True):
            direction = end - start
            length = np.linalg.norm(direction)
            if length < 1e-10:
                continue
            fraction = (landmarks - start) @ direction / length**2
            on_edge = np.abs(cross(landmarks - start, direction)) < 1e-8 * length
            on_edge &= (fraction >= -1e-8) & (fraction <= 1 + 1e-8)
            cuts = np.unique(np.round(np.clip(fraction[on_edge], 0, 1), 9))
            for low, high in zip(cuts[:-1], cuts[1:], strict=True):
                midpoint = start + (low + high) / 2 * direction
                if any(
                    inside(midpoint[None, :], later)[0] for later, _ in regions[i + 1 :]
                ):
                    continue
                a, b = (tuple(np.round(start + t * direction, 8)) for t in (low, high))
                segments[tuple(sorted((a, b)))] = group
    return segments


def edge_fractions(length):
    if length <= DETAIL_LENGTH:
        return np.linspace(0, 1, max(1, int(np.ceil(length / DETAIL_SPACING))) + 1)
    shoulder = DETAIL_SPACING / length
    intervals = max(1, int(np.ceil((length - 2 * DETAIL_SPACING) / LONG_SPACING)))
    return np.r_[0, np.linspace(shoulder, 1 - shoulder, intervals + 1), 1]


def adaptive_nodes(regions, scale=1):
    nodes = {}
    for (a, b), group in visible_segments(regions).items():
        start, end = np.array(a), np.array(b)
        for t in edge_fractions(np.linalg.norm(end - start) * scale):
            key = tuple(np.round(start + t * (end - start), 8))
            nodes[key] = max(group, nodes.get(key, group))
    return np.array(list(nodes)), np.array(list(nodes.values()), dtype=int)


def main():
    output = HERE / "wider-a-wordmarks"
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, extra in VARIANTS:
        records[name] = render(
            output,
            name,
            0.2,
            0.9,
            regions=wider_regions(extra),
            node_sampler=adaptive_nodes,
        )
        records[name].pop("edge_spacing")
        records[name].update(
            a_extra_units=extra,
            detail_spacing=DETAIL_SPACING,
            long_spacing=LONG_SPACING,
            detail_length=DETAIL_LENGTH,
            faces=[(n, g, s, v.tolist()) for n, g, s, v in wider_faces(extra)],
        )
        print(
            name,
            records[name]["prefix_points"],
            records[name]["tax_points"],
            flush=True,
        )

    fig = plt.figure(figsize=(27, 8), dpi=200)
    palette = np.array([matplotlib.colors.to_rgb(color) for color in PREFIX.palette])
    widest = np.concatenate([ring for rings, _ in wider_regions(28) for ring in rings])
    span = np.ptp(widest[:, 0]) + 0.24
    for i, (_name, extra) in enumerate(VARIANTS):
        label = ("Original A", "Wider A", "Widest A")[i]
        fig.text(
            i / 3 + 0.018,
            0.91,
            label,
            fontproperties=FontProperties(fname=FONT, size=20),
        )
        regions = wider_regions(extra)
        bounds = np.concatenate([ring for rings, _ in regions for ring in rings])
        center = (bounds[:, 0].min() + bounds[:, 0].max()) / 2
        ax = fig.add_axes((i / 3 + 0.008, 0.06, 1 / 3 - 0.016, 0.79))
        ax.set(
            xlim=(center - span / 2, center + span / 2),
            ylim=(-0.86, 0.86),
            aspect="equal",
        )
        ax.set_axis_off()
        fill_regions(ax, regions, palette, opacity=0.2)
        points, groups = adaptive_nodes(regions)
        scatter(ax, points, palette[groups], diameter=7, opacity=0.9)
    save_figure(fig, output, "comparison")
    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in (
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
