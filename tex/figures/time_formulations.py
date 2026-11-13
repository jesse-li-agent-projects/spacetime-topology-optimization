"""The time-field formulations on S03bT06, as six subfigure panels: the problem setup, the
plain TO design, and the initial time field, then the time field of the direct, the heat
and the Laplace formulation, each optimized on the plain TO's grayscale density. Local
minima of the time field over the part, the elements printed without support, are
circled.

Reads `output/tf_S03bT06_*`, made as in `configs/time_formulations/README.md`. Every
panel has one page size and plot box, centred with equal side margins, so they line up
in a grid in `main.tex` and each sits centred over its caption; the margins hold the
setup's supports, loads and plate, and the colorbars. Run from the repo root, via
`python -m tex.make_figures`.
"""

import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox

import sttopt.checks as checks
import sttopt.torch_util as torch_util
from sttopt.run_config import LaplaceRunConfig
from tex.figures import designs

CONFIG = Path("configs/time_formulations/laplace.json")
README = "configs/time_formulations/README.md"
PLAIN_TO = "tf_S03bT06_plain_to"
# The direct run starts from the linear ramp exactly; the PDE ones from their solution at
# the initial design variables, which only approximates it
INITIAL = "tf_S03bT06_direct_continuity_seq"
PANELS = (
    ("tf_direct", "tf_S03bT06_direct_continuity_seq"),
    ("tf_heat", "tf_S03bT06_heat_seq"),
    ("tf_laplace", "tf_S03bT06_laplace_seq"),
)
# The panels closing each row of the figure carry the time field's colorbar
COLORBAR = ("tf_time_init", "tf_laplace")

# Margins around the square plot box, in inches: side (both), bottom, top
SIDE_IN, BOTTOM_IN, TOP_IN = 0.5, 0.14, 0.24
BOX_IN = 1.3
FIGSIZE_IN = (2 * SIDE_IN + BOX_IN, BOTTOM_IN + BOX_IN + TOP_IN)


def _panel() -> tuple[plt.Figure, Axes]:
    fig = plt.figure(figsize=FIGSIZE_IN)
    w, h = FIGSIZE_IN
    box = (SIDE_IN / w, BOTTOM_IN / h, BOX_IN / w, BOX_IN / h)
    return fig, fig.add_axes(box)


def _save(fig: plt.Figure, path: Path) -> None:
    # The whole page, not `viz`'s tight bbox, so every panel keeps the same box
    fig.savefig(path, bbox_inches=Bbox.from_bounds(0, 0, *FIGSIZE_IN))
    plt.close(fig)


def _colorbar(fig: plt.Figure) -> None:
    w, h = FIGSIZE_IN
    x = (SIDE_IN + BOX_IN + 0.06) / w
    cax = fig.add_axes((x, BOTTOM_IN / h, 0.07 / w, BOX_IN / h))
    fig.colorbar(ScalarMappable(Normalize(0, 1), "viridis"), cax=cax, extend="max")
    cax.tick_params(labelsize=6)
    cax.set_ylabel("print time $t$", fontsize=7)


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
        s=120,
        facecolors="none",
        edgecolors=designs.MARK,
        linewidths=1.0,
        zorder=10,
        clip_on=False,
    )


def _draw_time(ax: Axes, d: designs.Design) -> None:
    # `t` is 1 at its maximum over the part; the void can run later, shown as 1
    designs.draw_time(ax, dataclasses.replace(d, tPhys=np.minimum(d.tPhys, 1)))


def build(out_dir: Path, *, device: str | None = None) -> None:
    config = LaplaceRunConfig.from_dict(json.loads(CONFIG.read_text()))
    problem = designs.evaluator(config, device)

    fig, ax = _panel()
    designs.draw_setup(ax, problem)
    _save(fig, out_dir / "tf_setup.pdf")

    plain = designs.evaluate(problem, designs.run_dir(PLAIN_TO, README))
    fig, ax = _panel()
    designs.draw_density(ax, plain)
    ax.set_aspect("equal")
    _save(fig, out_dir / "tf_pure_to.pdf")

    with np.load(designs.run_dir(INITIAL, README) / "design_it0000.npz") as data:
        t0 = data["tPhys"]
    fig, ax = _panel()
    _draw_time(ax, dataclasses.replace(plain, tPhys=t0))
    _colorbar(fig)
    _save(fig, out_dir / "tf_time_init.pdf")

    for name, tag in PANELS:
        d = designs.evaluate(problem, designs.run_dir(tag, README))
        fig, ax = _panel()
        _draw_time(ax, d)
        _mark_minima(ax, d, problem)
        if name in COLORBAR:
            _colorbar(fig)
        _save(fig, out_dir / f"{name}.pdf")
