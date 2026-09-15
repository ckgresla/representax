"""Uniform seamless fills, with dots at visible face corners and junctions."""

import hashlib
import json
from dataclasses import asdict, replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from extended import HIRES_PREFIX, fitted_join
from generate import HERE
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from tax import TAX_FACES, save_figure
from unified import area, closed_path, layout, project
from upright import placement

PREFIX = replace(HIRES_PREFIX, diameter_px=10)
OPACITY = 0.4
TAX_WIDTH = 1.15


def orient(vertices, outer=True):
    vertices = np.asarray(vertices, dtype=float)
    return vertices if (area(vertices) > 0) == outer else vertices[::-1]


def inside(points, rings):
    result = closed_path(rings[0]).contains_points(points, radius=-1e-9)
    for ring in rings[1:]:
        result &= ~closed_path(ring[::-1]).contains_points(points, radius=1e-9)
    return result


def edges(rings):
    return np.concatenate(rings), np.concatenate(
        [np.roll(ring, -1, axis=0) for ring in rings]
    )


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def intersections(first, second):
    a, b = first
    c, d = second
    r, s = b - a, d - c
    delta = c[None, :, :] - a[:, None, :]
    denominator = cross(r[:, None, :], s[None, :, :])
    t = np.divide(
        cross(delta, s[None, :, :]),
        denominator,
        out=np.full(denominator.shape, np.inf),
        where=np.abs(denominator) > 1e-12,
    )
    u = np.divide(
        cross(delta, r[:, None, :]),
        denominator,
        out=np.full(denominator.shape, np.inf),
        where=np.abs(denominator) > 1e-12,
    )
    good = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    row, column = np.nonzero(good)
    return a[row] + t[row, column, None] * r[row]


def edge_samples(rings, all_edges, spacing):
    if spacing <= 0:
        raise ValueError("edge spacing must be positive")
    samples = []
    for start, end in zip(*edges(rings), strict=True):
        direction = end - start
        length = np.linalg.norm(direction)
        hits = intersections((start[None, :], end[None, :]), all_edges)
        cuts = np.unique(
            np.round(
                np.clip(np.r_[0, 1, (hits - start) @ direction / length**2], 0, 1), 10
            )
        )
        for low, high in zip(cuts[:-1], cuts[1:], strict=True):
            count = max(1, int(np.ceil((high - low) * length / spacing)))
            t = np.linspace(low, high, count + 1)[1:-1]
            samples.extend(start + t[:, None] * direction)
    return np.array(samples).reshape(-1, 2)


def visible_nodes(regions, edge_spacing=None):
    """Corners and junctions, optionally joined by regularly spaced edge dots."""
    all_edges = edges([ring for rings, _ in regions for ring in rings])
    nodes = {}
    for i, (rings, group) in enumerate(regions):
        candidates = np.concatenate((*rings, intersections(edges(rings), all_edges)))
        if edge_spacing is not None:
            candidates = np.concatenate(
                (candidates, edge_samples(rings, all_edges, edge_spacing))
            )
        visible = np.ones(len(candidates), dtype=bool)
        for later, _ in regions[i + 1 :]:
            visible &= ~inside(candidates, later)
        for point in candidates[visible]:
            nodes[tuple(np.round(point, 10))] = group
    points = np.array(list(nodes))
    return points, np.array(list(nodes.values()), dtype=int)


def prefix_regions(rings, offset):
    front = [project(ring, offset, PREFIX) for ring in rings]
    depth = np.array((PREFIX.depth_x, PREFIX.depth_y))
    regions = []
    for ring in front:
        for start, end in zip(ring, np.roll(ring, -1, axis=0), strict=True):
            edge = end - start
            if np.array((edge[1], -edge[0])) @ depth > 1e-10:
                regions.append(([orient((start, end, end + depth, start + depth))], 0))
    return [*regions, (front, 0)]


def tax_regions():
    regions = []
    anchor = (28 - 126) / 100
    for _, group, _, vertices in TAX_FACES:
        polygon = (np.array(vertices, dtype=float) - (126, 72)) * (0.01, -0.01)
        polygon[:, 0] = anchor + TAX_WIDTH * (polygon[:, 0] - anchor)
        regions.append(([orient(polygon)], group))
    return regions


def fill_regions(ax, regions, palette, transform=None, opacity=OPACITY):
    transform = ax.transData if transform is None else transform
    for group in sorted({group for _, group in regions}):
        rings = [ring for region, color in regions if color == group for ring in region]
        # A single nonzero-winding path per color removes internal face seams.
        path = Path.make_compound_path(*[closed_path(ring) for ring in rings])
        ax.add_patch(
            PathPatch(
                path,
                transform=transform,
                facecolor=1 - opacity * (1 - palette[group]),
                edgecolor="none",
                linewidth=0,
                zorder=1 + group * 0.1,
            )
        )


def scatter(
    ax, points, colors, diameter=PREFIX.diameter_px, transform=None, opacity=0.9
):
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=(diameter * 72 / ax.figure.dpi) ** 2,
        c=colors,
        alpha=opacity,
        linewidths=0,
        edgecolors="none",
        zorder=10,
        transform=ax.transData if transform is None else transform,
    )


def main():
    output = HERE / "structural-wordmarks"
    output.mkdir(parents=True, exist_ok=True)
    regions = tax_regions()
    bounds = np.concatenate([ring for rings, _ in regions for ring in rings])
    mapping, (left, right, bottom, top) = placement(
        (bounds,), PREFIX, fitted_join(TAX_WIDTH)
    )
    palette = np.array([matplotlib.colors.to_rgb(color) for color in PREFIX.palette])
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
    fill_regions(ax, regions, palette)
    tax_points, groups = visible_nodes(regions)
    scatter(ax, tax_points, palette[groups])
    prefix_points = []
    for index, (_, offset, rings) in enumerate(layout(PREFIX)):
        letter = prefix_regions(rings, offset)
        fill_regions(ax, letter, np.zeros((1, 3)), mapping + ax.transData)
        points, _ = visible_nodes(letter)
        scatter(
            ax, points, np.zeros((len(points), 3)), transform=mapping + ax.transData
        )
        prefix_points.extend((index, *point) for point in mapping.transform(points))
    plt.rcParams["svg.hashsalt"] = "representax-structural-nodes"
    save_figure(fig, output, "representax-40")
    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(
        xlim=(bounds[:, 0].min() - 0.14, bounds[:, 0].max() + 0.14),
        ylim=(bounds[:, 1].min() - 0.14, bounds[:, 1].max() + 0.14),
        aspect="equal",
    )
    ax.set_axis_off()
    fill_regions(ax, regions, palette)
    scatter(ax, tax_points, palette[groups], diameter=10)
    save_figure(fig, output, "tax-detail")
    np.savetxt(
        output / "prefix-nodes.csv",
        prefix_points,
        delimiter=",",
        header="letter,x,y",
        comments="",
        fmt="%.10f",
    )
    np.savetxt(
        output / "tax-nodes.csv",
        np.column_stack((tax_points, groups)),
        delimiter=",",
        header="x,y,color_group",
        comments="",
        fmt="%.10f",
    )
    records = {
        "prefix": asdict(PREFIX),
        "tax_width": TAX_WIDTH,
        "opacity": OPACITY,
        "prefix_nodes": len(prefix_points),
        "tax_nodes": len(tax_points),
        "selection": "visible polygon vertices and exact face-edge intersections",
        "prefix_transform": mapping.get_matrix().tolist(),
        "tax_faces": TAX_FACES,
        "sources": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in (
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
        },
        "packages": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            {
                "prefix_nodes": len(prefix_points),
                "tax_nodes": len(tax_points),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
