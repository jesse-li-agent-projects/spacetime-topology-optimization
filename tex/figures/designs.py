"""Finished runs, as the figures draw them: each design's fields, compliance and hotspot
severity, all read the same way (`stto_laplace`'s hotspot and the grayscale density as
optimized) whichever time-field formulation produced the design, and the panels that draw
them. The runs themselves live under `output/`, made by the CLIs; nothing here optimizes.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy import ndimage

import sttopt.checks as checks
import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.stto_laplace as stto_laplace
import sttopt.torch_util as torch_util
import sttopt.viz as viz
from sttopt import run_config
from tex.figures import style  # noqa: F401  (applies the style on import)

OUTPUT = Path("output")
VOID_GRAY = "0.85"
SEVERITY_CMAP = matplotlib.colormaps["RdBu_r"].with_extremes(bad=VOID_GRAY)
INK = "#1a1a19"
MARK = "#b2182b"


@dataclass(frozen=True)
class Design:
    xPhys: Float[np.ndarray, "nely nelx"]
    tPhys: Float[np.ndarray, "nely nelx"]
    severity: Float[np.ndarray, "nely nelx"]  # per element, `-inf` where shielded
    compliance: float
    max_severity: float
    local_minima: int  # of `tPhys` over the part, off the print base
    saddles: int


def run_dir(tag: str, hint: str) -> Path:
    """`output/<tag>`, which must exist.

    :param hint: where the command that makes it is written down
    """
    path = OUTPUT / tag
    if not (path / "final_design.npz").exists():
        raise SystemExit(
            f"[error] {path}/final_design.npz is missing; run it as in {hint}"
        )
    return path


def evaluator(config: run_config.LaplaceRunConfig, device: str | None):
    """The `stto_laplace` problem whose hotspot and compliance every design is read with:
    a cell's `laplace` config."""
    return stto_laplace.build_problem(config, device=device)


def evaluate(problem: stto_laplace.Problem, run: Path) -> Design:
    """The run's final design, read with `problem`'s settings at their final values."""
    config = problem.config
    with np.load(run / "final_design.npz") as data:
        xPhys, tPhys = data["xPhys"], data["tPhys"]
    x_t, t_t = (
        torch_util.to_tensor(a, problem.device, problem.dtype) for a in (xPhys, tPhys)
    )
    with torch.no_grad():
        c, _, _ = compliance.batched_whole_and_gravity_compliance(
            x_t,
            torch.clamp(t_t, max=1),
            problem.KE,
            problem.edofMat,
            config.Emin,
            config.Emax,
            run_config.final_value(config.penal),
            problem.freedofs,
            problem.F,
            problem.ndof,
            problem.C,
            run_config.final_value(config.beta_t_schedule),
            [float(ti) for ti in np.linspace(0, 1, config.nStage + 1)[1:]],
        )
        K = problem.terms.estimated_conductivity(x_t, t_t)
        S = conductivity.severity(
            K, x_t, run_config.final_value(config.r), problem.terms.hotspot.void_switch
        ).reshape(xPhys.shape)
    solid = xPhys > config.eta
    base = torch_util.to_numpy(problem.Nei)
    return Design(
        xPhys=xPhys,
        tPhys=tPhys,
        severity=torch_util.to_numpy(S),
        compliance=float(c),
        max_severity=float(S.max()),
        local_minima=int(checks.unsupported(solid, tPhys, base).sum()),
        saddles=int(checks.saddles(tPhys, solid, solid).sum()),
    )


def severity_norm(designs: list[Design], Tcr: float) -> TwoSlopeNorm:
    """White at `Tcr`, red above and blue below, symmetric about it out to the worst
    design's excess; the low side clips."""
    top = max(d.max_severity for d in designs) - Tcr
    return TwoSlopeNorm(vcenter=0, vmin=-top, vmax=top)


def draw_density(ax: Axes, d: Design) -> None:
    nely, nelx = d.xPhys.shape
    ax.imshow(d.xPhys, cmap="gray_r", vmin=0, vmax=1, extent=(0, nelx, -nely, 0))
    _frame(ax, d)


def draw_time(ax: Axes, d: Design, contours: int = 20) -> None:
    """The time field with the void shaded translucent gray."""
    viz.timefield_filled_contour_plot(
        d.xPhys, d.tPhys, contours, show_void=True, colorbar=False, ax=ax
    )
    ax.set_title("")
    _frame(ax, d)


def draw_severity(ax: Axes, d: Design, Tcr: float, norm: Normalize) -> None:
    """Excess severity `S_e - T_cr` over the solid part (`xPhys >= 0.5`), void gray."""
    nely, nelx = d.xPhys.shape
    shown = (d.xPhys >= 0.5) & np.isfinite(d.severity)
    ax.imshow(
        np.where(shown, d.severity - Tcr, np.nan),
        cmap=SEVERITY_CMAP,
        norm=norm,
        extent=(0, nelx, -nely, 0),
    )
    _frame(ax, d)


def _frame(ax: Axes, d: Design) -> None:
    nely, nelx = d.xPhys.shape
    ax.set_xlim(0, nelx)
    ax.set_ylim(-nely, 0)
    ax.set_axis_off()


def draw_setup(ax: Axes, problem) -> None:
    """The domain, its clamped nodes, the loads, the build plate and, where the objective
    has gravity stages, gravity, read from `problem` rather than restated, so the panel
    follows the load case. Everything outside the domain is drawn unclipped, into the
    axes' margins."""
    nelx, nely = problem.config.nelx, problem.config.nely
    grid = (nely + 1, nelx + 1)
    # Node (row, col) sits at (col, -row), as the fields are drawn
    rows, cols = np.indices(grid)
    out = dict(clip_on=False)
    ax.add_patch(
        Rectangle(
            (0, -nely), nelx, nely, facecolor="0.93", edgecolor=INK, lw=0.6, **out
        )
    )
    unit = max(nelx, nely)

    clamped = ~torch_util.to_numpy(problem.free_mask).reshape(-1, 2).any(axis=1)
    clamped = clamped.reshape(grid)
    hatch = 0.04 * unit
    for r, c in zip(*np.nonzero(clamped)):
        if (r + c) % 8 == 4:
            ax.plot([c, c - hatch], [-r, -r - hatch], color=INK, lw=0.5, **out)
    ax.plot(
        cols[clamped], -rows[clamped], color=INK, lw=1.4, solid_capstyle="butt", **out
    )

    plate = torch_util.to_numpy(problem.mesh.plate).reshape(grid)
    depth = 0.07 * unit
    r0, r1 = rows[plate].min(), rows[plate].max()
    c0, c1 = cols[plate].min(), cols[plate].max()
    label = dict(ha="center", va="center", fontsize=5.5, color="white")
    if r0 == r1:  # along the bottom or top edge
        y = -r0 - depth if r0 == nely else -r0
        ax.add_patch(Rectangle((c0, y), c1 - c0, depth, facecolor="0.45", lw=0, **out))
        ax.text((c0 + c1) / 2, y + depth / 2, "build plate", **label)
    else:  # along the left or right edge, clear of the clamp's hatching
        x = c0 - hatch - depth if c0 == 0 else c0 + hatch
        ax.add_patch(Rectangle((x, -r1), depth, r1 - r0, facecolor="0.45", lw=0, **out))
        ax.text(x + depth / 2, -(r0 + r1) / 2, "build plate", rotation=90, **label)

    F = torch_util.to_numpy(problem.F).reshape(*grid, 2)
    loaded = (F != 0).any(axis=-1)
    labels, n = ndimage.label(loaded, structure=np.ones((3, 3)))
    length = 0.22 * min(nelx, nely)
    for k in range(1, n + 1):
        at = labels == k
        u = F[at].sum(axis=0)
        u = u / np.hypot(*u)
        head = np.array([cols[at].mean(), -rows[at].mean()])
        # Head on the loaded nodes, tail outside the domain; else tail on the nodes
        tail = head - length * u
        if 0 <= tail[0] <= nelx and -nely <= tail[1] <= 0:
            head, tail = head + length * u, head
        ax.annotate(
            "",
            xy=head,
            xytext=tail,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color=MARK, lw=1.0, mutation_scale=7),
        )
    if problem.config.nStage > 0:
        gx = nelx + 0.2 * unit
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
