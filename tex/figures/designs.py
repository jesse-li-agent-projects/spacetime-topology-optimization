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
