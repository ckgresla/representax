"""An analytic, sparse representation of JAX-inspired geometry. CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.path import Path as Polygon
from PIL import Image
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
# The paper's named sky, mint, and periwinkle palette.
PALETTE = ("#3f78b5", "#4f8a67", "#8278b8")

# Analytic face domains in an oblique drawing plane, not a raster-derived mask.
# Later patches have priority at overlaps, so a location has one face label.
PATCHES = (
    ("J stem", 0, 1.00, ((56, 120), (126, 0), (154, 0), (84, 120))),
    ("J hook", 0, 1.00, ((0, 120), (28, 72), (56, 72), (28, 120))),
    ("J foot", 0, 1.00, ((28, 120), (42, 96), (70, 96), (56, 120))),
    ("J inner side", 0, 0.74, ((56, 72), (70, 96), (42, 96))),
    ("J front", 0, 0.74, ((0, 120), (84, 120), (98, 144), (14, 144))),
    ("A left", 1, 0.87, ((84, 120), (140, 24), (154, 24), (98, 120))),
    ("A left cap", 1, 0.74, ((84, 120), (98, 120), (112, 144), (98, 144))),
    ("A right", 1, 0.82, ((140, 24), (168, 24), (238, 144), (210, 144))),
    ("A bridge side", 1, 0.74, ((112, 96), (182, 96), (196, 120), (126, 120))),
    ("A bridge", 1, 1.05, ((112, 96), (126, 72), (168, 72), (182, 96))),
    ("X rising", 2, 1.13, ((140, 120), (210, 0), (238, 0), (168, 120))),
    ("X far side", 2, 0.80, ((168, 120), (238, 0), (252, 24), (182, 144))),
    ("X cap", 2, 1.13, ((140, 24), (154, 0), (182, 0), (168, 24))),
    ("X falling", 2, 0.88, ((168, 24), (182, 0), (252, 120), (238, 144))),
    ("X foot", 2, 0.74, ((140, 120), (168, 120), (182, 144), (154, 144))),
)


@dataclass(frozen=True)
class Config:
    seed: int = 7
    points: int = 900
    deformation: float = 1.3
    normal_scatter: float = 0.009
    diameter_px: float = 3.6
    size_px: int = 2800
    yaw_degrees: float = -12.0
    pitch_degrees: float = 14.0


def domain_labels(xy):
    labels = np.full(len(xy), -1, dtype=int)
    for i, (_, _, _, vertices) in enumerate(PATCHES):
        vertices = np.array(vertices, dtype=float)
        vertices[:, 0] = (vertices[:, 0] - 126) / 100
        vertices[:, 1] = (72 - vertices[:, 1]) / 100
        labels[Polygon(vertices).contains_points(xy)] = i
    return labels


def sample_domain(config):
    if not 1 <= config.points <= 4000:
        raise ValueError("Choose between 1 and 4000 points for this sparse mark")
    candidates = qmc.Sobol(2, scramble=True, seed=config.seed).random_base2(15)
    candidates = candidates * (2.52, 1.44) - (1.26, 0.72)
    labels = domain_labels(candidates)
    candidates, labels = candidates[labels >= 0], labels[labels >= 0]
    # Greedy maximin sampling spreads points without outlining edges or a grid.
    distance = np.full(len(candidates), np.inf)
    selected = []
    index = int(np.argmin(np.sum(candidates**2, axis=1)))
    for _ in range(config.points):
        selected.append(index)
        distance = np.minimum(
            distance, np.sum((candidates - candidates[index]) ** 2, axis=1)
        )
        index = int(np.argmax(distance))
    return candidates[selected], labels[selected]


def embed(xy, amount):
    """Two invertible planar shears, lifted onto a smooth graph in R^3."""
    x, y = xy.T
    u = x + 0.10 * amount * np.sin(2 * y)
    v = y + 0.16 * amount * np.sin(2 * u)
    w = amount * (0.22 * np.sin(1.8 * u) + 0.12 * np.cos(3 * v))
    du_dy = 0.20 * amount * np.cos(2 * y)
    dv_dx = 0.32 * amount * np.cos(2 * u)
    dv_dy = 1 + dv_dx * du_dy
    dw_du = 0.396 * amount * np.cos(1.8 * u)
    dw_dv = -0.36 * amount * np.sin(3 * v)
    tangent_x = np.column_stack((np.ones_like(x), dv_dx, dw_du + dw_dv * dv_dx))
    tangent_y = np.column_stack((du_dy, dv_dy, dw_du * du_dy + dw_dv * dv_dy))
    normals = np.cross(tangent_x, tangent_y)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return np.column_stack((u, v, w)), normals


def cloud(config, xy, labels):
    xyz, normals = embed(xy, config.deformation)
    noise = np.random.default_rng(config.seed).normal(size=(len(xy), 1))
    noise = np.clip(noise, -2.5, 2.5)
    xyz += config.normal_scatter * config.deformation * noise * normals
    colors = np.array(
        [
            np.clip(np.array(to_rgb(PALETTE[PATCHES[i][1]])) * PATCHES[i][2], 0, 1)
            for i in labels
        ]
    )
    return xyz, colors


def project(xyz, config):
    yaw, pitch = np.deg2rad((config.yaw_degrees, config.pitch_degrees))
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rx = np.array(((1, 0, 0), (0, cp, -sp), (0, sp, cp)))
    return xyz @ ry.T @ rx.T


def canvas(config, xyz, colors):
    projected = project(xyz, config)
    order = np.argsort(projected[:, 2], kind="stable")
    dpi = 200
    fig = plt.figure(figsize=(config.size_px / dpi, config.size_px / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(-1.64, 1.64), ylim=(-1.64, 1.64), aspect="equal")
    ax.set_axis_off()
    ax.scatter(
        projected[order, 0],
        projected[order, 1],
        s=(config.diameter_px * 72 / dpi) ** 2,
        c=colors[order],
        edgecolors="none",
        alpha=0.94,
        linewidths=0,
    )
    return fig, ax


def save_mark(output, name, config, xy, labels):
    xyz, colors = cloud(config, xy, labels)
    fig, _ = canvas(config, xyz, colors)
    plt.rcParams["svg.hashsalt"] = "representax-manifold"
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
        output / f"{name}-points.csv",
        np.column_stack((xy, xyz, project(xyz, config)[:, :2], labels, colors)),
        delimiter=",",
        header="source_x,source_y,x,y,z,screen_x,screen_y,face,r,g,b",
        comments="",
        fmt="%.10f",
    )
    return {
        "config": asdict(config),
        "points_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
    }


def save_animation(output, config, xy, labels):
    config = replace(
        config, size_px=960, diameter_px=config.diameter_px * 960 / config.size_px
    )
    frames = []
    for phase in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        amount = config.deformation * (0.5 - 0.5 * np.cos(phase))
        frame_config = replace(config, deformation=amount)
        xyz, colors = cloud(frame_config, xy, labels)
        fig, _ = canvas(frame_config, xyz, colors)
        fig.canvas.draw()
        frames.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
        )
        plt.close(fig)
    frames[0].save(
        output / "manifold.gif",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "rendered")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = Config()
    xy, labels = sample_domain(config)
    records = {}
    for name, cfg in (
        ("manifold", config),
        ("manifold-airy", replace(config, points=480)),
        ("geometry-control", replace(config, deformation=0)),
    ):
        records[name] = save_mark(
            args.output, name, cfg, xy[: cfg.points], labels[: cfg.points]
        )
    if args.animate:
        save_animation(args.output, config, xy, labels)
    import scipy

    records["generator_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    records["packages"] = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
    }
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
