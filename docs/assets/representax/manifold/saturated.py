"""Level isometric geometry, saturated vertices, and ten-percent face fills."""

import hashlib
import json
from dataclasses import asdict

import matplotlib
import numpy as np
from generate import HERE
from lattice import LatticeConfig, lattice, save_mark
from PIL import Image

CONFIG = LatticeConfig(
    deformation=0,
    yaw_degrees=0,
    pitch_degrees=0,
    fill_opacity=0.1,
    palette=("#0066FF", "#00B88A", "#B000FF"),
    point_opacity=0.9,
    gradient_floor=1,
)


def main():
    output = HERE / "straight-10"
    output.mkdir(parents=True, exist_ok=True)
    geometry = lattice(CONFIG)
    records = {"straight-10": save_mark(output, "straight-10", CONFIG, geometry)}
    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("saturated.py", "lattice.py", "generate.py")
    }
    records["packages"] = {
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
    }
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps({"directory": str(output), "base_config": asdict(CONFIG)}, indent=2)
    )


if __name__ == "__main__":
    main()
