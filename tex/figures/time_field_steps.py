"""How `eqn:time_field_laplace`'s time field is computed, for `tex/main.tex`, as one image
of three panels on a small synthetic setup: the diffusivity and the raw wall design
beside each wall, extended past the build plate by its sign-flipped reflection; the
filtered wall data and the raw time field it gives; and that time field rescaled to 1
over the part, outlined.

Every panel is computed with the code the optimizer runs (`virtual_heat`), on a
hand-made design rather than an optimized one, so nothing is read from `output/`. Run
from the repo root, via `python -m tex.make_figures`.
"""

import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

import sttopt.units as units
import sttopt.virtual_heat as vh
from sttopt.run_config import LaplaceRunConfig
from tex.figures.designs import INK, MARK
from tex.figures.style import TEXT_WIDTH_IN

CONFIG = Path("configs/laplace_default.json")
NELX, NELY = 60, 30
SEED = 3

# Layout, in elements: wall graph offset from the wall and height per unit of wall data,
# the plate's drawn depth, and the page around the domain
GAP, AMP = 3.0, 16.0
PLATE = 10.0
LEFT, RIGHT, BOTTOM, TOP = 15.0, 15.0, 26.0, 18.0
PANEL_IN = TEXT_WIDTH_IN / 3
PLATE_GRAY = "0.45"
GRAPH_LW = 0.8


def _setup() -> tuple[LaplaceRunConfig, vh.ScalarMesh]:
    base = LaplaceRunConfig.from_dict(json.loads(CONFIG.read_text()))
    h = base.element_size_m
    config = dataclasses.replace(
        base,
        nelx=NELX,
        width_m=NELX * h,
        height_m=NELY * h,
        print_base="bottom_edge",
        laplace_cg_rtol=1e-10,
    )
    mesh = vh.ScalarMesh.build(
        NELX, NELY, "bottom_edge", torch.device("cpu"), torch.float64
    )
    return config, mesh


def _blobs(rng: np.random.Generator) -> Float[np.ndarray, "nely nelx"]:
    """A diffusivity design `mu` of a few smooth blobs about the uniform 1/2."""
    rows, cols = np.mgrid[:NELY, :NELX] + 0.5
    mu = np.full((NELY, NELX), 0.5)
    for _ in range(6):
        r, c = rng.uniform(0, NELY), rng.uniform(0, NELX)
        width = rng.uniform(5, 10)
        mu += rng.uniform(-0.5, 0.5) * np.exp(
            -((rows - r) ** 2 + (cols - c) ** 2) / (2 * width**2)
        )
    return np.clip(mu, 0, 1)


def _part() -> Bool[np.ndarray, "nely nelx"]:
    """An arch: two legs on the plate, joined by a beam below the domain's top."""
    rows, cols = np.mgrid[:NELY, :NELX] + 0.5
    height = NELY - rows  # above the plate
    legs = (np.abs(cols - 12) < 5) | (np.abs(cols - 44) < 5)
    beam = (height > 15) & (height < 22) & (cols > 7) & (cols < 49)
    return (legs & (height < 22)) | beam


def _raw_wall(
    mesh: vh.ScalarMesh, rng: np.random.Generator
) -> Float[np.ndarray, " n_arc"]:
    """The distance initialization with a bump, made rough so the filter shows."""
    wall = vh.distance_wall(mesh).numpy()
    k = np.arange(len(wall)) / len(wall)
    wall = wall + 0.12 * np.exp(-(((k - 0.62) / 0.08) ** 2))
    noise = np.convolve(rng.normal(0, 0.07, len(wall) + 2), np.ones(3) / 3, "valid")
    return np.clip(wall + noise, 0, 1)


def _node_xy(mesh: vh.ScalarMesh) -> tuple[np.ndarray, np.ndarray]:
    """Node coordinates in elements, `y` up from the plate."""
    ids = np.arange(mesh.ndof)
    return ids % (NELX + 1), NELY - ids // (NELX + 1)


def _wall_graphs(
    ax: Axes,
    mesh: vh.ScalarMesh,
    values: Float[np.ndarray, " n_arc"],
    label: str,
    reflect_to: float | None = None,
) -> None:
    """`values` drawn as a graph outside each wall, rising away from it; `reflect_to`
    continues the side walls' graphs that far below the plate with the sign-flipped
    reflection, dotted."""
    x, y = _node_xy(mesh)
    x, y = x[mesh.arc.numpy()], y[mesh.arc.numpy()]
    base = dict(color="0.6", lw=0.4)
    line = dict(color=INK, lw=GRAPH_LW)
    top = y == NELY
    ax.plot([0, NELX], [NELY + GAP] * 2, **base)
    ax.plot(x[top], NELY + GAP + AMP * values[top], **line)
    ax.text(
        -GAP,
        NELY + GAP + AMP * values[top][0],
        label,
        ha="right",
        va="center",
        fontsize=7,
    )
    for side, out in ((x == 0, -1), (x == NELX, 1)):
        offset = (0 if out < 0 else NELX) + out * GAP
        low = -(reflect_to or 0)
        ax.plot([offset] * 2, [low, NELY], **base)
        ax.plot(offset + out * AMP * values[side], y[side], **line)
        if reflect_to is not None:
            # Reflected about the plate corner, where the data is 0
            order = np.argsort(y[side])
            ys, vs = y[side][order], values[side][order]
            keep = ys <= reflect_to
            ax.plot(
                offset - out * AMP * np.r_[0.0, vs[keep]],
                -np.r_[0.0, ys[keep]],
                ls=":",
                **line,
            )


def _panel(fig: plt.Figure, i: int) -> Axes:
    """Panel `i` of the three, side by side, labelled (a), (b), (c)."""
    ax = fig.add_axes((i / 3, 0, 1 / 3, 1))
    ax.set_xlim(-LEFT, NELX + RIGHT)
    ax.set_ylim(-BOTTOM, NELY + TOP)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, -PLATE), NELX, PLATE, facecolor=PLATE_GRAY, lw=0))
    ax.text(
        NELX / 2,
        -PLATE / 2,
        "build plate",
        color="white",
        ha="center",
        va="center",
        fontsize=6,
    )
    ax.text(-LEFT + 1, NELY + TOP - 1, f"({'abc'[i]})", ha="left", va="top", fontsize=8)
    return ax


def _colorbar(
    fig: plt.Figure, i: int, mappable: ScalarMappable, label: str, **kw
) -> None:
    """A horizontal colorbar under panel `i`'s plate."""
    width = NELX + LEFT + RIGHT
    height = NELY + BOTTOM + TOP
    x0, y0 = (i + LEFT / width) / 3, (BOTTOM - PLATE - 4.0) / height
    cax = fig.add_axes((x0, y0, NELX / width / 3, 1.6 / height))
    fig.colorbar(mappable, cax=cax, orientation="horizontal", **kw)
    cax.tick_params(labelsize=5, length=2, pad=1)
    cax.set_xlabel(label, fontsize=6, labelpad=1)


def _nodal_field(ax: Axes, t: Float[np.ndarray, " ndof"], norm: Normalize) -> None:
    grid = t.reshape(NELY + 1, NELX + 1)[::-1]  # rows up from the plate
    levels = np.linspace(norm.vmin, norm.vmax, 17)
    style = dict(levels=levels, extend="max")
    ax.contourf(grid, cmap="viridis", norm=norm, **style)
    ax.contour(grid, colors="black", linewidths=0.25, **style)


def _outline(ax: Axes, part: Bool[np.ndarray, "nely nelx"]) -> None:
    """The part's element boundary: a contour of the mask, upsampled so it follows the
    element edges."""
    n = 8
    fine = np.kron(part[::-1].astype(float), np.ones((n, n)))
    coords = (np.arange(fine.shape[1]) + 0.5) / n, (np.arange(fine.shape[0]) + 0.5) / n
    ax.contour(*coords, fine, levels=[0.5], colors=MARK, linewidths=1.0)


def build(out_dir: Path) -> None:
    config, mesh = _setup()
    rng = np.random.default_rng(SEED)
    mu = _blobs(rng)
    part = _part()
    wall = _raw_wall(mesh, rng)
    r_b = units.in_elements(config.wall_filter_rmin_m, config.element_size_m)
    b = vh.wall_filter(len(wall), r_b) @ wall
    solution = vh.laplace_time_field(
        config,
        mesh,
        torch.as_tensor(part, dtype=torch.float64),
        torch.as_tensor(mu),
        torch.as_tensor(b),
    )
    t_r = solution.t.numpy()

    width = NELX + LEFT + RIGHT
    height = NELY + BOTTOM + TOP
    fig = plt.figure(figsize=(3 * PANEL_IN, PANEL_IN * height / width))

    chi = vh.diffusivity(torch.as_tensor(mu), config.chi_contrast).numpy()
    ax = _panel(fig, 0)
    chi_norm = LogNorm(1 / np.sqrt(config.chi_contrast), np.sqrt(config.chi_contrast))
    ax.imshow(
        chi[::-1],
        cmap="Greys",
        norm=chi_norm,
        origin="lower",
        extent=(0, NELX, 0, NELY),
        interpolation="nearest",
    )
    _wall_graphs(ax, mesh, wall, "$b_r$", reflect_to=PLATE)
    _colorbar(fig, 0, ScalarMappable(chi_norm, "Greys"), "diffusivity $\\chi$")

    ax = _panel(fig, 1)
    raw_norm = Normalize(0, float(t_r.max()))
    _nodal_field(ax, t_r, raw_norm)
    _wall_graphs(ax, mesh, b, "$b$")
    _colorbar(fig, 1, ScalarMappable(raw_norm, "viridis"), "raw time field $t_r$")

    ax = _panel(fig, 2)
    t = vh.element_means(solution.t, mesh.edof).numpy()
    scale = float(t[part.ravel()].max())  # as `virtual_heat.normalize`
    unit_norm = Normalize(0, 1)
    _nodal_field(ax, t_r / scale, unit_norm)
    _outline(ax, part)
    _colorbar(
        fig, 2, ScalarMappable(unit_norm, "viridis"), "time field $t$", extend="max"
    )
    # The whole page, not a tight box, which would shift the panels' shared layout
    fig.savefig(
        out_dir / "tf_steps.pdf",
        bbox_inches=Bbox.from_bounds(0, 0, *fig.get_size_inches()),
    )
    plt.close(fig)
