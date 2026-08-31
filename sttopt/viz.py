"""Plots for the printed structure: elements colored by print time, and the boundaries
between print stages.

When no `ax` is passed, both functions build their own `Figure` directly rather than
going through `pyplot`, so nothing is registered globally and there's nothing for the
caller to close; `savefig` still works. Pass your own `ax` (e.g. from `plt.subplots()`)
to draw into a pyplot-managed, interactive figure instead.

Coordinate conventions differ between the two functions, on purpose: `combination_plot`
places element `(row, col)` at `x in [col, col+1]`, `y in [-(row+1), -row]` (y flipped);
`stage_boundary_plot` places it at `x in [col+0.5, col+1.5]`, `y in [row+0.5, row+1.5]`
(no flip, half-cell offset). The two frames are related by `x' = x - 0.5, y' = 0.5 - y`;
`stage_boundary_plot(..., combination_coords=True)` applies it to compose both plots on
one `Axes` (as `cli.py` does). See `conventions.md`.

Run as a script (`python -m sttopt.viz <tag>`) to regenerate plots for a saved run from
its `output/<tag>/` artefacts, without rerunning the optimization. This reads
`final_design.npz`'s `xPhys`/`tPhys` -- the state *after* the last MMA update -- whereas
`cli.py`'s own end-of-run plot uses `prev_state` (the state entering that last update).
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure

matplotlib.rcParams["savefig.bbox"] = "tight"

_BOUNDARY_LINEWIDTH = 1.5


def _new_axes() -> Axes:
    """A standalone `Axes` on a `Figure` that pyplot does not own.

    Not registered with pyplot, so it needs no `plt.close()` -- it's reclaimed by
    ordinary garbage collection once the returned `Axes` goes out of scope. Creating
    `Figure` directly is matplotlib's documented route for library/application code.
    """
    return Figure().add_subplot()


def combination_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    values: Float[np.ndarray, "nely nelx"],
    eps: float,
    *,
    cmap: str = "viridis",
    colorbar_label: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draws only the elements with density `>= eps`, each colored by `values` (flat
    per-face color, not interpolated). `values` is any per-element scalar (e.g. `tPhys`,
    or a hotspot-severity score), not necessarily a time field.

    :param cmap: colormap name; callers below pick one that suits `values`.
    :param colorbar_label: if given, adds a horizontal colorbar below `ax` labelled with
        this string; omitted (the default) draws no colorbar.
    """
    if ax is None:
        ax = _new_axes()

    rows, cols = np.nonzero(xPhys >= eps)
    # Element (row, col) -> unit square x in [col, col+1], y in [-(row+1), -row].
    verts = np.stack(
        [
            np.stack([cols, -rows], axis=1),
            np.stack([cols + 1, -rows], axis=1),
            np.stack([cols + 1, -(rows + 1)], axis=1),
            np.stack([cols, -(rows + 1)], axis=1),
        ],
        axis=1,
    ).astype(float)
    values = values[rows, cols]

    coll = PolyCollection(verts, array=values, cmap=cmap, edgecolors="none")
    ax.add_collection(coll)
    ax.set_aspect("equal")
    ax.autoscale_view()
    if colorbar_label is not None:
        ax.figure.colorbar(coll, ax=ax, orientation="horizontal", label=colorbar_label)
    return ax


def stage_boundary_plot(
    tPhys: Float[np.ndarray, "nely nelx"],
    nStage: int,
    *,
    ax: Axes | None = None,
    combination_coords: bool = False,
) -> Axes:
    """Assigns each element to one of `nStage` print stages by its `tPhys` value
    (half-open `(tt[j], tt[j+1]]` bins except the first, which is closed on both ends),
    then draws a black line along every internal mesh edge whose two adjacent elements
    fall in different stages.

    :param combination_coords: remap edges into `combination_plot`'s coordinate frame
        (`x' = x - 0.5, y' = 0.5 - y`) before drawing, e.g. to overlay onto an `Axes` a
        prior `combination_plot` call already populated; leave `False` for a standalone
        plot. See the module docstring.
    """
    if ax is None:
        ax = _new_axes()

    tt = np.linspace(0.0, 1.0, nStage + 1)
    stage = np.zeros(tPhys.shape, dtype=int)
    for j in range(nStage):
        lo, hi = tt[j], tt[j + 1]
        mask = (tPhys >= lo) & (tPhys <= hi) if j == 0 else (tPhys > lo) & (tPhys <= hi)
        stage[mask] = j + 1

    def _pt(x: float, y: float) -> tuple[float, float]:
        return (x - 0.5, 0.5 - y) if combination_coords else (x, y)

    segments = []
    # Vertical edges: element (row, col) | (row, col+1), shared edge at x = col+1.5.
    rows, cols = np.nonzero(stage[:, :-1] != stage[:, 1:])
    for row, col in zip(rows, cols):
        x = col + 1.5
        segments.append([_pt(x, row + 0.5), _pt(x, row + 1.5)])
    # Horizontal edges: element (row, col) | (row+1, col), shared edge at y = row+1.5.
    rows, cols = np.nonzero(stage[:-1, :] != stage[1:, :])
    for row, col in zip(rows, cols):
        y = row + 1.5
        segments.append([_pt(col + 0.5, y), _pt(col + 1.5, y)])

    ax.add_collection(
        LineCollection(segments, colors="black", linewidths=_BOUNDARY_LINEWIDTH)
    )
    ax.set_aspect("equal")
    ax.autoscale_view()
    return ax


def hotspot_severity_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    hotspot_severity: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    nStage: int,
    *,
    ax: Axes | None = None,
) -> Axes:
    """`combination_plot` (binarized density, colored by `hotspot_severity`, `plasma`
    colormap, labelled horizontal colorbar) with `stage_boundary_plot` overlaid in its
    `combination_coords` frame -- the plot recipe `cli.py` saves as
    `hotspot_severity.png`.

    :param xPhys: physical density field (not yet binarized).
    :param hotspot_severity: per-element overheating severity, e.g. `(1 - K_est) * (xPhys > 0.5)`.
    :param tPhys: physical print-time field.
    :param nStage: number of print stages, for `stage_boundary_plot`'s binning.
    :return: the `Axes` drawn into.
    """
    XPhys = (xPhys > 0.5).astype(xPhys.dtype)
    ax = combination_plot(
        XPhys,
        hotspot_severity,
        eps=1.0e-1,
        cmap="plasma",
        colorbar_label="Hotspot severity",
        ax=ax,
    )
    stage_boundary_plot(tPhys, nStage, ax=ax, combination_coords=True)
    ax.set_title("Hotspot severity")
    return ax


def timefield_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    *,
    ax: Axes | None = None,
) -> Axes:
    """`combination_plot` (binarized density, colored by `tPhys`, `viridis` colormap,
    labelled horizontal colorbar) -- masked to the same `xPhys > 0.5` elements as
    `hotspot_severity_plot`, so the two plots cover the same printed region.

    :param xPhys: physical density field (not yet binarized).
    :param tPhys: physical print-time field.
    :return: the `Axes` drawn into.
    """
    XPhys = (xPhys > 0.5).astype(xPhys.dtype)
    ax = combination_plot(
        XPhys, tPhys, eps=1.0e-1, cmap="viridis", colorbar_label="Time field", ax=ax
    )
    ax.set_title("Time field")
    return ax


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate plots for a saved run from its output/<tag>/ "
        "artefacts, saved under plot/<tag>/."
    )
    parser.add_argument("tag", help="run identifier, matching output/<tag>/")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="parent directory of run artefacts (default: output)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("plot"),
        help="parent directory to save plots under (default: plot)",
    )
    return parser.parse_args(argv)


def _main(args: argparse.Namespace) -> None:
    import json

    import sttopt.conductivity as conductivity
    import sttopt.optimize as optimize
    import sttopt.torch_util as torch_util
    from sttopt.run_config import RunConfig

    run_dir = args.output_dir / args.tag
    config = RunConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    design = np.load(run_dir / "final_design.npz")
    xPhys, tPhys = design["xPhys"], design["tPhys"]

    # Only e1/e2/w/q/rouf (the conductivity-estimation neighborhood) are needed below,
    # but build_problem doesn't expose that setup on its own -- see cli.py's post-loop
    # section, which computes hotspot_severity the same way.
    problem = optimize.build_problem(config)
    K_est = conductivity.estimated_conductivity(
        torch_util.to_tensor(xPhys, device=problem.device, dtype=problem.dtype),
        torch_util.to_tensor(tPhys, device=problem.device, dtype=problem.dtype),
        problem.e1,
        problem.e2,
        problem.w,
        problem.config.q,
        problem.config.rouf,
    ).reshape(config.nely, config.nelx)
    K_est = torch_util.to_numpy(K_est)
    hotspot_severity = (1 - K_est) * (xPhys > 0.5)

    plot_dir = args.plot_dir / args.tag
    plot_dir.mkdir(parents=True, exist_ok=True)
    ax = hotspot_severity_plot(xPhys, hotspot_severity, tPhys, config.nStage)
    out_path = plot_dir / "hotspot_severity.png"
    ax.figure.savefig(out_path)
    print(f"Saved hotspot severity plot to {out_path}")

    ax = timefield_plot(xPhys, tPhys)
    out_path = plot_dir / "timefield.png"
    ax.figure.savefig(out_path)
    print(f"Saved time field plot to {out_path}")


if __name__ == "__main__":
    _args = _parse_args()  # early-exits on --help before the heavy imports below
    _main(_args)
