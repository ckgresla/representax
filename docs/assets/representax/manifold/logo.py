"""Export the selected Representax logo without regenerating design studies."""

import hashlib
import json

import matplotlib
import numpy as np
from a_arm import arm_faces
from condensed import composition, render
from generate import HERE
from PIL import Image
from structural import PREFIX
from wider_a import DETAIL_LENGTH, DETAIL_SPACING, LONG_SPACING

ARM_EXTRA = 7
FILL_OPACITY = 0.3
POINT_OPACITY = 0.9
PREFIX_WIDTH = 0.8
WEB_WIDTH = 2400
SOURCES = (
    "logo.py",
    "condensed.py",
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


def main():
    output = HERE.parent
    limits = composition(None, ARM_EXTRA)[3]
    pixels_per_unit = PREFIX.width_px / (limits[1] - limits[0])
    record = render(
        output,
        "representax",
        PREFIX_WIDTH,
        pixels_per_unit,
        arm_extra=ARM_EXTRA,
        fill_opacity=FILL_OPACITY,
        point_opacity=POINT_OPACITY,
    )
    for suffix in ("png", "svg", "pdf", "csv"):
        (output / f"representax-tax.{suffix}").replace(
            output / f"representax-mark.{suffix}"
        )
    for name in ("representax.svg", "representax-mark.svg"):
        path = output / name
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
    with Image.open(output / "representax.png") as source:
        image = source.convert("RGB")
        image.resize(
            (WEB_WIDTH, round(image.height * WEB_WIDTH / image.width)),
            Image.Resampling.LANCZOS,
        ).save(output / "representax-web.png")
    record.update(
        selected_variant=(
            "80% monospace prefix; original A with 1.5x left-arm thickness, "
            "no added top"
        ),
        arm_extra_units=ARM_EXTRA,
        background="white; face colors are uniform tints precomposed against white",
        detail_spacing=DETAIL_SPACING,
        long_spacing=LONG_SPACING,
        detail_length=DETAIL_LENGTH,
        faces=[(n, g, s, v.tolist()) for n, g, s, v in arm_faces(ARM_EXTRA)],
        sources={
            f"manifold/{name}": hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        packages={
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": Image.__version__,
        },
    )
    artifacts = [
        output / f"{stem}.{suffix}"
        for stem in ("representax", "representax-mark")
        for suffix in ("png", "svg", "pdf")
    ]
    artifacts += [
        output / name
        for name in (
            "representax-web.png",
            "representax-prefix.csv",
            "representax-mark.csv",
        )
    ]
    record["artifacts"] = {
        path.name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in artifacts
    }
    for path in artifacts:
        if path.suffix == ".png":
            with Image.open(path) as image:
                record["artifacts"][path.name]["pixels"] = list(image.size)
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "prefix_points": record["prefix_points"],
                "mark_points": record["tax_points"],
                "artifacts": record["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
