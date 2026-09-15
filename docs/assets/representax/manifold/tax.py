"""A T-led mark and two Representax wordmark compositions."""

import hashlib
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from generate import HERE, PATCHES
from lattice import canvas, lattice, save_mark
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw, ImageFont
from saturated import CONFIG

T_FACES = (
    ("T stem", 0, 1.00, ((28, 120), (84, 24), (112, 24), (56, 120))),
    ("T stem side", 0, 0.78, ((56, 120), (112, 24), (126, 48), (70, 144))),
    ("T foot", 0, 0.74, ((28, 120), (56, 120), (70, 144), (42, 144))),
    ("T bar front", 0, 0.82, ((56, 24), (140, 24), (154, 48), (70, 48))),
    ("T bar side", 0, 0.74, ((140, 24), (154, 0), (168, 24), (154, 48))),
    ("T bar", 0, 1.00, ((56, 24), (70, 0), (154, 0), (140, 24))),
)
TAX_FACES = T_FACES + tuple(face for face in PATCHES if not face[0].startswith("J "))
FONT = HERE.parents[3] / "paper/assets/InterVariable.ttf"


def word_path(text, height):
    path = TextPath((0, 0), text, size=1, prop=FontProperties(fname=FONT))
    box = path.get_extents()
    path = (
        Affine2D()
        .translate(-box.x0, -box.y0)
        .scale(height / box.height)
        .transform_path(path)
    )
    return path, path.get_extents()


def save_figure(fig, output, name):
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


def save_wordmarks(output, geometry):
    xy = geometry[0]
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    fig, _, _ = canvas(CONFIG, *geometry, patches=TAX_FACES)
    ax = fig.axes[0]
    prefix, box = word_path("represen", 1.14)
    text_x = x0 - 0.18 - box.width
    ax.add_patch(
        PathPatch(
            prefix,
            transform=Affine2D().translate(text_x, y0) + ax.transData,
            facecolor="#000000",
            edgecolor="none",
        )
    )
    left, right = text_x - 0.35, x1 + 0.35
    height = 2.5
    fig.set_size_inches(18, 18 * height / (right - left))
    ax.set(xlim=(left, right), ylim=(-height / 2, height / 2), aspect="equal")
    save_figure(fig, output, "representax-integrated")

    fig, _, _ = canvas(CONFIG, *geometry, patches=TAX_FACES)
    ax = fig.axes[0]
    name, box = word_path("representax", 0.31)
    text_x = (x0 + x1 - box.width) / 2
    ax.add_patch(
        PathPatch(
            name,
            transform=Affine2D().translate(text_x, y0 - 0.57) + ax.transData,
            facecolor="#000000",
            edgecolor="none",
        )
    )
    center = (x0 + x1) / 2
    ax.set(xlim=(center - 1.6, center + 1.6), ylim=(-1.9, 1.3), aspect="equal")
    save_figure(fig, output, "representax-stacked")


def comparison(output):
    sheet = Image.new("RGB", (2400, 1900), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT), 30)
    draw.text((100, 65), "01  Integrated: represen + TAX", font=font, fill="#394659")
    with Image.open(output / "representax-integrated.png") as image:
        image = image.convert("RGB")
        image.thumbnail((2200, 680), Image.Resampling.LANCZOS)
        sheet.paste(image, ((2400 - image.width) // 2, 180))
    draw.text((100, 880), "02  Mark + full name", font=font, fill="#394659")
    with Image.open(output / "representax-stacked.png") as image:
        image = image.convert("RGB")
        image.thumbnail((950, 950), Image.Resampling.LANCZOS)
        sheet.paste(image, ((2400 - image.width) // 2, 910))
    sheet.save(output / "comparison.png")


def main():
    output = HERE / "tax-wordmarks"
    output.mkdir(parents=True, exist_ok=True)
    geometry = lattice(CONFIG, TAX_FACES)
    records = {"tax": save_mark(output, "tax", CONFIG, geometry, patches=TAX_FACES)}
    save_wordmarks(output, geometry)
    comparison(output)
    records["faces"] = TAX_FACES
    records["sources"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("tax.py", "saturated.py", "lattice.py", "generate.py")
    }
    records["font_sha256"] = hashlib.sha256(FONT.read_bytes()).hexdigest()
    records["packages"] = {
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
    }
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
