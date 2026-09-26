"""Figure 1: Representax at a glance, following Section 3.

One job configuration (left) is partitioned into scientific and execution
parameters. Each partition drives its own lane:

  scientific parameters  ->  the four core abstractions   (3.1, 3.2)
  execution parameters   ->  execution strategies           (3.4)

and both lanes converge on the shared lifecycle, run_job (3.3). A research
change lives entirely in the scientific lane.

Shapes are rectangles with slightly rounded corners. Text is measured against
its container; the script refuses to export overlapping or overflowing labels.

    python3 figure1.py        # writes figures/overview.{pdf,png}
"""


import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
COLORS = {
    "sky": "#3f78b5", "rose": "#c85f65", "mint": "#4f8a67",
    "periwinkle": "#8278b8", "apricot": "#cc7a3a", "slate": "#68717c",
}
INK, GREY, RULE = "#24272b", COLORS["slate"], "#d7dbe0"
RADIUS = .3  # a tenth-inch unit is 7.2 pt: corners of about 2 pt

INTER = HERE / "assets/InterVariable.ttf"
font_manager.fontManager.addfont(INTER)
SANS = FontProperties(fname=INTER).get_name()


def semibold():
    """Matplotlib ignores variable-font weights, so cut Inter's SemiBold instance."""
    path = Path(tempfile.mkdtemp()) / "Inter-SemiBold.ttf"
    font = instantiateVariableFont(TTFont(INTER), {"wght": 600, "opsz": 14},
                                   updateFontNames=True)
    font.save(path)
    return path


SEMIBOLD = semibold()
plt.rcParams.update({"font.family": SANS, "text.color": INK,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def tint(color, amount):
    return tuple(1 - amount * (1 - np.array(matplotlib.colors.to_rgb(color))))


def shade(color, amount):
    return tuple(np.array(matplotlib.colors.to_rgb(color)) * (1 - amount))


class Canvas:
    """Figure in tenth-inch units, origin at the lower left."""

    def __init__(self, width, height):
        self.fig = plt.figure(figsize=(width / 10, height / 10))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, width), ylim=(0, height))
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.problems = []

    def text(self, x, y, s, size=8.6, color=INK, weight=None, **kw):
        kw.setdefault("va", "center")
        font = FontProperties(fname=SEMIBOLD if weight else INTER, size=size)
        return self.ax.text(x, y, s, fontproperties=font, color=color, zorder=5, **kw)

    def rows(self, artist):
        box = artist.get_window_extent(self.renderer)
        inverse = self.ax.transData.inverted()
        return inverse.transform((0, box.y0))[1], inverse.transform((0, box.y1))[1]

    def inside(self, artist, bottom, top):
        y0, y1 = self.rows(artist)
        if y0 < bottom - 1e-6 or y1 > top + 1e-6:
            self.problems.append(f"{artist.get_text()!r} leaves {bottom:.2f}..{top:.2f}")
        return artist

    def span(self, artist):
        box = artist.get_window_extent(self.renderer)
        inverse = self.ax.transData.inverted()
        return inverse.transform((box.x0, 0))[0], inverse.transform((box.x1, 0))[0]

    def within(self, artist, left, right):
        x0, x1 = self.span(artist)
        if x0 < left - 1e-6 or x1 > right + 1e-6:
            self.problems.append(f"{artist.get_text()!r} exceeds {left:.2f}..{right:.2f}")
        return artist

    def apart(self, first, second, space=1.2):
        if self.span(first)[1] + space > self.span(second)[0]:
            self.problems.append(f"{first.get_text()!r} meets {second.get_text()!r}")

    def box(self, x, y, w, h, face="white", edge="none", lw=.7):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={RADIUS}",
            facecolor=face, edgecolor=edge, lw=lw, zorder=1))

    def arrow(self, a, b, color=GREY, rad=0.0):
        self.ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>,head_length=3.0,head_width=1.6", color=color,
            lw=.8, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", zorder=2))

    def line(self, xs, ys, color=GREY):
        self.ax.plot(xs, ys, color=color, lw=.8, solid_capstyle="round", zorder=2)

    def save(self, name):
        if self.problems:
            raise ValueError("layout problems:\n" + "\n".join(self.problems))
        for ext in ("pdf", "png"):
            self.fig.savefig(HERE / "figures" / f"{name}.{ext}", facecolor="white",
                             dpi=300, bbox_inches="tight", pad_inches=.05,
                             metadata={"CreationDate": None} if ext == "pdf" else None)
        plt.close(self.fig)


ABSTRACTIONS = (  # JobConfig field -> core abstraction (Section 3.1)
    ("model", "Models & processors", "sky"),
    ("task", "Learning tasks", "periwinkle"),
    ("data", "Data distributions", "mint"),
    ("evaluation", "Evaluators", "rose"),
)
LIFECYCLE = ("train", "evaluate", "checkpoint", "export")


def overview():
    W = 72
    y_exe, h_exe = .4, 2.8  # execution lane
    lane_gap = 2.2
    tile_h, tile_gap = 2.6, .55
    y_sci = y_exe + h_exe + lane_gap  # scientific lane
    h_sci = 4 * tile_h + 3 * tile_gap
    head_h = 4.4
    top = y_sci + h_sci + head_h
    H = top + .4
    c = Canvas(W, H)

    cw, mw, lw, gap, pad = 14.0, 27.3, 21.5, 4.6, .9
    mx = cw + gap  # middle column
    lx = mx + mw + gap  # lifecycle column
    title_y, rule_y = top - 1.35, top - 2.4
    y_sci_mid, y_exe_mid = y_sci + h_sci / 2, y_exe + h_exe / 2
    centres = [y_sci + (3 - i) * (tile_h + tile_gap) + tile_h / 2 for i in range(4)]

    # Job configuration: scientific rows map onto the abstractions by colour.
    c.box(0, y_exe, cw, top - y_exe, edge=RULE)
    c.text(pad, title_y, "Job configuration", size=9.4, weight=600)
    c.line([pad, cw - pad], [rule_y, rule_y], RULE)
    c.text(pad, top - 3.45, "Scientific", size=8.6, weight=600)
    for (field, _, colour), y in zip(ABSTRACTIONS, centres):
        c.within(c.text(pad + .6, y, field, size=8.6), pad, cw - pad)
        c.arrow((cw + .6, y), (mx - .6, y), COLORS[colour])
    cut = y_sci - lane_gap / 2
    c.line([pad, cw - pad], [cut, cut], RULE)
    c.text(pad, y_exe_mid, "Execution", size=8.6, weight=600, color=GREY)
    c.arrow((cw + .6, y_exe_mid), (mx - .6, y_exe_mid))

    # Core abstractions, joined into one path.
    c.text(mx, title_y, "Core abstractions", size=9.4, weight=600)
    bus = mx + mw + 1.8
    for (_, name, colour), y in zip(ABSTRACTIONS, centres):
        c.box(mx, y - tile_h / 2, mw, tile_h, face=tint(COLORS[colour], .26))
        c.within(c.text(mx + pad, y, name, size=9.0, weight=600,
                        color=shade(COLORS[colour], .38)), mx + pad, mx + mw - pad)
        c.line([mx + mw + .3, bus], [y, y])
    c.line([bus, bus], [centres[0], centres[-1]])
    c.arrow((bus, y_sci_mid), (lx - .6, y_sci_mid))

    # Execution strategies: a device mesh, and how work is chunked over it.
    c.box(mx, y_exe, mw, h_exe, edge=RULE)
    for r in range(2):
        for k in range(4):
            c.box(mx + pad + .62 * k, y_exe_mid - .62 + .62 * r, .5, .5, face=tint(GREY, .55))
    c.within(c.text(mx + pad + 3.4, y_exe_mid, "sharding · gradient caching · prefetch",
                    size=7.9, color=GREY), mx + pad + 3.4, mx + mw - pad)
    c.arrow((mx + mw + .6, y_exe_mid), (lx - .6, y_exe_mid))

    # Shared lifecycle: train, evaluate, checkpoint, repeat; export at the end.
    c.box(lx, y_exe, lw, top - y_exe, face="#eef0f2")
    c.text(lx + pad, title_y, "Shared lifecycle", size=9.4, weight=600)
    c.line([lx + pad, lx + lw - pad], [rule_y, rule_y], RULE)
    xc, xr = lx + 8.0, lx + 15.6  # chain centre; return path
    ys = [top - 4.55 - 4.2 * i for i in range(4)]
    for s, y in zip(LIFECYCLE, ys):
        c.text(xc, y, s, size=9.0, ha="center")
    for y0, y1 in zip(ys, ys[1:]):
        c.arrow((xc, y0 - .95), (xc, y1 + .95))
    c.line([xc + 4.3, xr, xr], [ys[2], ys[2], ys[0]])
    c.arrow((xr, ys[0]), (xc + 2.6, ys[0]))
    c.save("overview")


if __name__ == "__main__":
    overview()
