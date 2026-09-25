"""Figure 1: where a research change enters Representax, and what surrounds it.

The drawing borrows the logo's language (docs/assets/representax): a 30 degree
italic shear, faces filled with a 30% tint and no outline, and small dots
along each edge, clustered at the corners. Scientific components are tinted
faces; execution settings are dots alone.

    python3 figure1.py        # writes figures/overview.{pdf,png}
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
LOGO = HERE.parent / "docs/assets/representax/representax.svg"
COLORS = {
    "sky": "#3f78b5", "rose": "#c85f65", "mint": "#4f8a67",
    "periwinkle": "#8278b8", "apricot": "#cc7a3a", "slate": "#68717c",
}
INK, RULE = "#24272b", "#d7dbe0"
SHEAR = np.tan(np.radians(30))          # the logo's italic angle
FILL, DOT_ALPHA = 0.30, 0.9              # the logo's face tint and dot opacity

font_manager.fontManager.addfont(HERE / "assets/InterVariable.ttf")
SANS = font_manager.FontProperties(fname=HERE / "assets/InterVariable.ttf").get_name()
MONO = "Noto Sans Mono"
plt.rcParams.update({"font.family": SANS, "font.size": 9, "text.color": INK,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def tint(color, amount=FILL):
    """Precomposite a colour over white, as the logo does."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(1 - amount * (1 - rgb))


def face(x, y, w, h, shear=SHEAR):
    """Sheared parallelogram with lower-left corner (x, y)."""
    s = shear * h
    return np.array([(x, y), (x + w, y), (x + w + s, y + h), (x + s, y + h)])


def edge_dots(poly, near=.42, far=2.2, count=2):
    """Dots at each corner, `count` close neighbours, sparse along the middle."""
    points = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        length = np.hypot(*(b - a))
        u = (b - a) / length
        ts = [0.0] + [near * k for k in range(1, count + 1)]
        ts += [length - near * k for k in range(1, count + 1)]
        span = length - 2 * near * (count + 1)
        if span > far:
            n = int(span // far)
            ts += list(near * (count + 1) + span * np.arange(1, n + 1) / (n + 1))
        points += [a + t * u for t in ts if 0 <= t <= length]
    return np.array(points)


class Canvas:
    """Figure in tenth-inch units, origin at the lower left."""

    def __init__(self, width, height):
        self.fig = plt.figure(figsize=(width / 10, height / 10))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
        self.ax.axis("off")

    def text(self, x, y, s, size=8.6, color=INK, weight=None, mono=False, **kw):
        kw.setdefault("va", "center")
        return self.ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                            family=MONO if mono else SANS, **kw)

    def poly(self, vertices, color, z=1):
        self.ax.add_patch(Polygon(vertices, closed=True, facecolor=color,
                                  edgecolor="none", zorder=z))

    def tile(self, vertices, color, filled=True, dot=1.5, **spacing):
        """A logo face: tinted fill without outline, dotted edges."""
        if filled:
            self.poly(vertices, tint(color))
        pts = edge_dots(vertices, **spacing)
        self.ax.scatter(pts[:, 0], pts[:, 1], s=dot, color=color, alpha=DOT_ALPHA,
                        lw=0, zorder=2)

    def wordmark(self, x, y, height, recolor):
        """Draw the logo SVG as vector paths with its lower-left corner at (x, y)."""
        svg = "{http://www.w3.org/2000/svg}"
        root = ET.parse(LOGO).getroot()
        faces, dots = [], []
        for p in root.iter(svg + "path"):
            fill = re.findall(r"fill: ?(#[0-9a-f]{6})", p.attrib.get("style", ""))
            if "id" not in p.attrib and fill and fill[0] != "#ffffff":
                faces.append((re.findall(r"([MLz])([^MLz]*)", p.attrib["d"]), fill[0]))
        for u in root.iter(svg + "use"):
            fill = re.findall(r"fill: ?(#[0-9a-f]{6})", u.attrib.get("style", ""))
            dots.append((float(u.attrib["x"]), float(u.attrib["y"]),
                         fill[0] if fill else "#000000"))
        corners = [tuple(map(float, n.split())) for cmds, _ in faces
                   for cmd, n in cmds if cmd != "z"]
        xy = np.array(corners + [(a, b) for a, b, _ in dots])
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        k = height / (hi[1] - lo[1])

        def place(px, py):
            return x + (px - lo[0]) * k, y + (hi[1] - py) * k

        for cmds, fill in faces:
            verts, codes = [], []
            for cmd, nums in cmds:
                if cmd == "z":
                    verts.append((0, 0))
                    codes.append(MPath.CLOSEPOLY)
                else:
                    verts.append(place(*map(float, nums.split())))
                    codes.append(MPath.MOVETO if cmd == "M" else MPath.LINETO)
            self.ax.add_patch(PathPatch(MPath(verts, codes), facecolor=recolor(fill, True),
                                        edgecolor="none", zorder=3))
        px, py = place(np.array([d[0] for d in dots]), np.array([d[1] for d in dots]))
        self.ax.scatter(px, py, s=.45, lw=0, alpha=DOT_ALPHA, zorder=4,
                        color=[recolor(d[2], False) for d in dots])
        return (hi[0] - lo[0]) * k

    def save(self, name):
        for ext in ("pdf", "png"):
            self.fig.savefig(HERE / "figures" / f"{name}.{ext}", facecolor="white",
                             dpi=300, bbox_inches="tight", pad_inches=.04,
                             metadata={"CreationDate": None} if ext == "pdf" else None)
        plt.close(self.fig)


LOGO_COLORS = {"#b2d1ff": "sky", "#0066ff": "sky", "#b2eadc": "mint",
               "#00b88a": "mint", "#e7b2ff": "periwinkle", "#b000ff": "periwinkle"}


def paper_colors(svg_color, is_face):
    """Map the logo's blue, green and violet onto the paper palette."""
    name = LOGO_COLORS.get(svg_color)
    if name is None:  # the grey REPRESEN prefix
        return tint(INK) if is_face else INK
    return tint(COLORS[name]) if is_face else COLORS[name]


CONFIG = (  # line, role, component colour; "+" marks the research change
    ("job = JobConfig(", None, None),
    ("  data=mix(image, audio, video),", "sci", "mint"),
    ("  model=jina_v5_omni_nano(),", "sci", "sky"),
    ("  task=RetrievalConfig(),", "sci", "periwinkle"),
    ("  loss=MNRConfig(symmetric=True),", "sci", "periwinkle"),
    ("+ loss_modifiers=(Matryoshka(…),),", "change", "periwinkle"),
    ("  evaluation=EvaluationConfig(…),", "sci", "rose"),
    ("  training=TrainingConfig(", None, None),
    ("    global_batch_size=32,", "sci", None),
    ("    grad_cache=GradCacheConfig(…),", "exe", None),
    ("    precision=bfloat16_mixed(),", "exe", None),
    ("  ),", None, None),
    (")", None, None),
    ("run_job(job)", "run", None),
)


def overview():
    W, H = 72, 31.5
    c = Canvas(W, H)
    grey = COLORS["slate"]
    top = H - 3.4

    # Configuration card.
    cx, cw, lead = .3, 25.6, 1.72
    ch = lead * len(CONFIG) + 1.6
    c.ax.add_patch(FancyBboxPatch((cx, top - ch), cw, ch,
                                  boxstyle="round,pad=0,rounding_size=.7",
                                  facecolor="#f6f7f9", edgecolor="none", zorder=0))
    c.text(cx, H - 1.4, "Experiment", size=10.5, weight=600)
    c.text(cx + cw, H - 1.4, "abridged JobConfig", size=8.2, color=grey, ha="right")
    for i, (line, role, colour) in enumerate(CONFIG):
        y = top - 1.2 - lead * i
        if role == "change":
            c.poly(face(cx + .2, y - lead / 2, cw - .4, lead, shear=0),
                   tint(COLORS["periwinkle"], .22))
            c.text(cx + .45, y, "+", size=8.6, color=COLORS["periwinkle"], weight=700,
                   mono=True)
        if role in ("sci", "change"):
            c.poly(face(cx + 1.05, y - .62, .34, 1.24),
                   COLORS[colour] if colour else INK, z=2)
        elif role == "exe":
            for k in (-.5, 0, .5):
                c.ax.scatter(cx + 1.4 + SHEAR * k, y + k, s=1.6, color=grey, lw=0,
                             alpha=DOT_ALPHA, zorder=2)
        c.text(cx + 1.9, y, line.lstrip("+"), size=8.1, mono=True,
               color=grey if role == "exe" else INK,
               weight=600 if role == "run" else None)

    # Library stack, headed by the wordmark.
    sx, sw = 29.4, 42.3
    c.wordmark(sx, H - 2.35, 1.9, paper_colors)

    def header(y, name, note):
        c.text(sx, y, name, size=9.4, weight=600)
        c.text(sx + sw, y, note, size=8.0, color=grey, ha="right")

    # Research interfaces: scientific, tinted faces.
    h1, g = 5.2, .9
    y1 = top - 1.4 - h1
    header(top - .15, "Research interfaces", "scientific · fingerprinted")
    w1 = (sw - SHEAR * h1 - 3 * g) / 4
    tiles = (("Data", "Grain sources", "mint"), ("Models", "Equinox modules", "sky"),
             ("Tasks", "MNR loss", "periwinkle"), ("Evaluation", "corpus metrics", "rose"))
    for i, (name, body, colour) in enumerate(tiles):
        x = sx + i * (w1 + g)
        c.tile(face(x, y1, w1, h1), COLORS[colour])
        c.text(x + SHEAR * 3.6 + .9, y1 + 3.6, name, size=9.4, weight=600)
        c.text(x + SHEAR * 1.6 + .9, y1 + 1.6, body, size=8.1)
    # The research change: a modifier wrapped around the task.
    x = sx + 2 * (w1 + g)
    bx, by, bw, bh = x + w1 - 4.9, y1 - 1.05, 6.1, 1.9
    c.poly(face(bx, by, bw, bh), tint(COLORS["periwinkle"], .62), z=2.5)
    c.text(bx + SHEAR * bh / 2 + bw / 2, by + bh / 2, "+ Matryoshka", size=7.8,
           weight=600, ha="center", zorder=3)

    # Shared lifecycle: the grey of the REPRESEN prefix.
    h2 = 3.4
    y2 = y1 - 4.0 - h2
    header(y2 + h2 + 1.1, "Shared lifecycle", "run_job")
    w2 = sw - SHEAR * h2
    c.tile(face(sx, y2, w2, h2), INK, dot=1.1)
    stages = ("stream", "compiled update", "checkpoint", "evaluate", "export")
    step = (w2 - 2.2) / len(stages)
    for i, s in enumerate(stages):
        xx = sx + SHEAR * h2 / 2 + 1.1 + step * (i + .5)
        c.text(xx, y2 + h2 / 2, s, size=8.3, ha="center")
        if i:
            c.text(xx - step / 2, y2 + h2 / 2, "›", size=9, ha="center", color=grey)

    # Execution: dots alone.
    h3 = 3.2
    y3 = y2 - 3.9 - h3
    header(y3 + h3 + 1.1, "Execution", "per device · not fingerprinted")
    w3 = (sw - SHEAR * h3 - 3 * g) / 4
    for i, s in enumerate(("GradCache chunks", "precision", "sharding", "prefetch")):
        x = sx + i * (w3 + g)
        c.tile(face(x, y3, w3, h3), grey, filled=False, dot=1.3)
        c.text(x + SHEAR * h3 / 2 + w3 / 2, y3 + h3 / 2, s, size=8.1, ha="center",
               color=grey)

    # Hardware, reached only through execution.
    y4 = y3 - 2.3
    c.text(sx, y4, "JAX · XLA", size=8.1, color=grey)
    xx = sx + 9.0
    for label, cols, rows in (("GPU", 1, 1), ("GPU node", 8, 1), ("TPU slice", 8, 2)):
        c.text(xx, y4, label, size=8.1, color=grey)
        xx += len(label) * .62 + .8
        for r in range(rows):
            for k in range(cols):
                yy = y4 - .26 + (r - (rows - 1) / 2) * .72
                c.poly(face(xx + k * .72, yy, .52, .52), tint(grey, .55))
        xx += cols * .72 + 2.4
    c.save("overview")


if __name__ == "__main__":
    overview()
