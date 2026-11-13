"""How the heat evacuation estimate reads space outside the design domain, for
`tex/main.tex`: a schematic of the domain on its build plate, the solid half-plane beyond
the plate, the void everywhere else, and an element's neighbourhood reaching into both.
Run from the repo root, via `python -m tex.make_figures`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Bbox

from tex.figures.designs import INK, MARK
from tex.figures.style import TEXT_WIDTH_IN

# Schematic units: the domain is W x H on the plate at y = 0, inside a view of the plane
W, H = 3.0, 1.6
VIEW = (-1.0, W + 1.0, -0.75, H + 0.45)
R_K = 0.55
ELEMENT = (0.3, 0.25)  # centre of the element whose neighbourhood is drawn
SOLID = "0.45"


def build(out_dir: Path) -> None:
    x0, x1, y0, y1 = VIEW
    width = 0.6 * TEXT_WIDTH_IN
    fig = plt.figure(figsize=(width, width * (y1 - y0) / (x1 - x0)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Edge to edge of the page, so it reads as the whole half-plane
    ax.add_patch(Rectangle((x0, y0), x1 - x0, -y0, facecolor=SOLID, lw=0))
    ax.add_patch(
        Rectangle((0, 0), W, H, facecolor="0.93", edgecolor=INK, lw=0.8, zorder=2)
    )
    text = dict(ha="center", va="center", fontsize=7)
    ax.text(
        W / 2,
        y0 / 2,
        "build plate: solid half-plane, printed at $t = -\\infty$",
        color="white",
        **text,
    )
    ax.text(
        W / 2, H + (y1 - H) / 2, "void, printed at $t = -\\infty$", color=INK, **text
    )
    ax.text(x0 / 2, H / 2, "void", color=INK, **text)
    ax.text(W + (x1 - W) / 2, H / 2, "void", color=INK, **text)
    ax.text(W / 2, H / 2, "design domain $\\Omega$", color=INK, **text)

    ax.add_patch(
        Circle(ELEMENT, R_K, facecolor="none", edgecolor=MARK, lw=1.0, zorder=3)
    )
    ax.plot(*ELEMENT, "s", color=MARK, ms=3, zorder=3)
    ax.annotate(
        "",
        xy=(ELEMENT[0] + R_K, ELEMENT[1]),
        xytext=ELEMENT,
        arrowprops=dict(arrowstyle="-|>", color=MARK, lw=0.8, mutation_scale=6),
        zorder=3,
    )
    ax.text(ELEMENT[0] + R_K / 2, ELEMENT[1] + 0.07, "$r_K$", color=MARK, **text)
    ax.text(ELEMENT[0] - 0.07, ELEMENT[1] + 0.09, "$e$", color=MARK, **text)
    # The whole page, not a tight box, which would pad the plate away from the edges
    fig.savefig(out_dir / "outside_domain.pdf", bbox_inches=Bbox.from_bounds(0, 0, *fig.get_size_inches()))
    plt.close(fig)
