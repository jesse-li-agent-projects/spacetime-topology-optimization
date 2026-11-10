"""The time-field formulations on S03bT06, as three subfigure panels: the problem setup,
then the time field of the direct and of the Laplace formulation, each optimized on the
Laplace run's grayscale density. Local minima of the time field over the part, the
elements printed without support, are circled.

Reads `output/tf_S03bT06_*`, made as in `configs/time_formulations/README.md`. Every
panel has one page size and plot box, so they line up side by side in `main.tex`; the
margins hold the setup's supports, loads and plate, and the colorbar. Run from the repo
root, via `python -m tex.make_figures`.
"""

import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from scipy import ndimage

import sttopt.checks as checks
import sttopt.torch_util as torch_util
from sttopt.run_config import LaplaceRunConfig
from tex.figures import designs

CONFIG = Path("configs/time_formulations/laplace.json")
README = "configs/time_formulations/README.md"
PANELS = (
    ("tf_direct", "tf_S03bT06_direct_continuity_fixed"),
    ("tf_laplace", "tf_S03bT06_laplace_fixed"),
)

# Margins around the square plot box, in inches: left, bottom, right, top
MARGIN_IN = (0.1, 0.14, 0.62, 0.24)
BOX_IN = 1.3
FIGSIZE_IN = (
    MARGIN_IN[0] + BOX_IN + MARGIN_IN[2],
    MARGIN_IN[1] + BOX_IN + MARGIN_IN[3],
)
INK = "#1a1a19"
MARK = "#b2182b"


def _panel() -> tuple[plt.Figure, Axes]:
    fig = plt.figure(figsize=FIGSIZE_IN)
    w, h = FIGSIZE_IN
    box = (MARGIN_IN[0] / w, MARGIN_IN[1] / h, BOX_IN / w, BOX_IN / h)
    return fig, fig.add_axes(box)


def _save(fig: plt.Figure, path: Path) -> None:
    # The whole page, not `viz`'s tight bbox, so every panel keeps the same box
    fig.savefig(path, bbox_inches=Bbox.from_bounds(0, 0, *FIGSIZE_IN))
    plt.close(fig)


def _draw_setup(ax: Axes, problem) -> None:
    """The domain, its clamped nodes, the loads, the build plate and, where the objective
    has gravity stages, gravity, read from `problem` rather than restated, so the panel
    follows the load case."""
    nelx, nely = problem.config.nelx, problem.config.nely
    grid = (nely + 1, nelx + 1)
    # Node (row, col) sits at (col, -row), as `designs` draws elements
    rows, cols = np.indices(grid)
    out = dict(clip_on=False)
    ax.add_patch(
        Rectangle(
            (0, -nely), nelx, nely, facecolor="0.93", edgecolor=INK, lw=0.6, **out
        )
    )

    plate = torch_util.to_numpy(problem.mesh.plate).reshape(grid)
    depth = 0.07 * nely
    x0, x1, y = cols[plate].min(), cols[plate].max(), -rows[plate].max()
    ax.add_patch(
        Rectangle((x0, y - depth), x1 - x0, depth, facecolor="0.45", lw=0, **out)
    )
    ax.text(
        (x0 + x1) / 2,
        y - depth / 2,
        "build plate",
        ha="center",
        va="center",
        fontsize=5.5,
        color="white",
    )

    clamped = ~torch_util.to_numpy(problem.free_mask).reshape(-1, 2).any(axis=1)
    clamped = clamped.reshape(grid)
    hatch = 0.05 * nely
    for r, c in zip(*np.nonzero(clamped)):
        if r % 8 == 4:
            ax.plot([c, c - hatch], [-r, -r - hatch], color=INK, lw=0.5, **out)
    ax.plot(
        cols[clamped], -rows[clamped], color=INK, lw=1.4, solid_capstyle="butt", **out
    )

    F = torch_util.to_numpy(problem.F).reshape(*grid, 2)
    loaded = (F != 0).any(axis=-1)
    labels, n = ndimage.label(loaded, structure=np.ones((3, 3)))
    length = 0.22 * nely
    for k in range(1, n + 1):
        at = labels == k
        u = F[at].sum(axis=0)
        u = u / np.hypot(*u)
        head = np.array([cols[at].mean(), -rows[at].mean()])
        # Head on the loaded nodes, tail outside the domain; else tail on the nodes
        tail = head - length * u
        if 0 < tail[0] < nelx and -nely < tail[1] < 0:
            head, tail = head + length * u, head
        ax.annotate(
            "",
            xy=head,
            xytext=tail,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color=MARK, lw=1.0, mutation_scale=7),
        )
    if problem.config.nStage > 0:
        gx = nelx * 1.2
        ax.annotate(
            "$g$",
            xy=(gx, -0.75 * nely),
            xytext=(gx, -0.45 * nely),
            ha="center",
            va="bottom",
            fontsize=7,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=6),
        )
    ax.set_xlim(0, nelx)
    ax.set_ylim(-nely, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _mark_minima(ax: Axes, d: designs.Design, problem) -> None:
    """Circles the local minima of the time field over the part, as `designs.evaluate`
    counts them."""
    solid = d.xPhys > problem.config.eta
    base = torch_util.to_numpy(problem.Nei)
    r, c = np.nonzero(checks.unsupported(solid, d.tPhys, base))
    ax.scatter(
        c + 0.5,
        -r - 0.5,
        marker="o",
        s=30,
        facecolors="none",
        edgecolors=MARK,
        linewidths=1.0,
        zorder=10,
        clip_on=False,
    )


def build(out_dir: Path, *, device: str | None = None) -> None:
    config = LaplaceRunConfig.from_dict(json.loads(CONFIG.read_text()))
    problem = designs.evaluator(config, device)

    fig, ax = _panel()
    _draw_setup(ax, problem)
    _save(fig, out_dir / "tf_setup.pdf")

    for name, tag in PANELS:
        d = designs.evaluate(problem, designs.run_dir(tag, README))
        fig, ax = _panel()
        # `t` is 1 at its maximum over the part; the void can run later, shown as 1
        designs.draw_time(ax, dataclasses.replace(d, tPhys=np.minimum(d.tPhys, 1)))
        _mark_minima(ax, d, problem)
        if name == PANELS[-1][0]:
            w, h = FIGSIZE_IN
            x = (MARGIN_IN[0] + BOX_IN + 0.06) / w
            cax = fig.add_axes((x, MARGIN_IN[1] / h, 0.07 / w, BOX_IN / h))
            fig.colorbar(
                ScalarMappable(Normalize(0, 1), "viridis"), cax=cax, extend="max"
            )
            cax.tick_params(labelsize=6)
            cax.set_ylabel("print time $t$", fontsize=7)
        _save(fig, out_dir / f"{name}.pdf")
