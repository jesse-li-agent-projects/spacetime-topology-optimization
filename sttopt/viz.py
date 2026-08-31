"""Plots for the printed structure: elements colored by print time, and the boundaries
between print stages -- ports `draw_combination1.m`/`draw_boundary.m` (the only two
plotting calls the main script actually exercises; see
`plans/archive/conductivity_estimation_2d_python_port.md`'s Scope section for why
`draw_combination2`/`3` and the main script's call to the undefined `draw_combination`
(line 577; no such file exists, only `draw_combination1`/`2`/`3`) are out of scope /
resolved to `draw_combination1`).

When no `ax` is passed, both functions build their own `Figure` directly rather than
going through `pyplot`, so importing or calling this module registers nothing in a
global figure registry and leaves nothing for the caller to close -- `savefig` still
works, picking its canvas from the output format. Pass your own `ax` (e.g. from
`plt.subplots()`) to draw into a pyplot-managed, interactive figure instead.

**Missing `draw_line` helper**: `draw_boundary.m` calls a `draw_line(V(e,:), 3, [0 0 0])`
helper that does not exist anywhere in the source repo (confirmed by search) -- an
unresolvable external dependency, not something with a real definition to recover. Its
call site is unambiguous though: two `(x, y)` node coordinates, linewidth 3, RGB black.
`stage_boundary_plot` below ports that directly as a black `LineCollection` (the vectorized
equivalent of calling `draw_line` once per boundary edge) rather than trying to
reverse-engineer a fancier helper that isn't there; the linewidth (`_BOUNDARY_LINEWIDTH`)
is a cosmetic default, not a MATLAB-parity number.

**Coordinate conventions differ between the two functions, on purpose**: `draw_combination1`
places element `(row, col)` at `x in [col, col+1]`, `y in [-(row+1), -row]` (y flipped, so
larger row -> more negative y); `draw_boundary` places the same element at
`x in [col+0.5, col+1.5]`, `y in [row+0.5, row+1.5]` (no flip, offset by half a cell). This
mirrors a real difference in the MATLAB source (`yElement = s(:)-0.5` vs.
`y = -(s(:)-1)`), not a typo -- `stage_boundary_plot` ports its own convention faithfully by
default (`combination_coords=False`); it is *not* pixel-aligned with `combination_plot` in
that mode. The two frames are related by a verified affine map, though (element `(row,
col)`'s footprint in one frame lands exactly on its footprint in the other under
`x' = x - 0.5, y' = 0.5 - y`) -- `stage_boundary_plot(..., combination_coords=True)` applies
it, for composing both plots on one `Axes` (as `cli.py` does). See `conventions.md`.

Unlike the MATLAB source (arbitrary-mesh node/face patch construction, needed for a
generic FEM mesh), every element here is an axis-aligned unit square on a regular grid,
so both functions below build polygons/edges directly from `(row, col)` -- no
`order='F'` flatten is needed (or used) anywhere in this module.

Run as a script (`python -m sttopt.viz <tag>`) to regenerate plots for a saved run from
its `output/<tag>/` artefacts, without rerunning the optimization. This reads
`final_design.npz`'s `xPhys`/`tPhys` -- the state *after* the last MMA update -- whereas
`cli.py`'s own end-of-run plot uses `prev_state` (the state entering that last update);
the two differ by one iteration. See `cli.py`'s module docstring for why it uses
`prev_state`.
"""

import argparse
from pathlib import Path

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure

_BOUNDARY_LINEWIDTH = 1.5


def _new_axes() -> Axes:
    """A standalone `Axes` on a `Figure` that pyplot does not own.

    Nothing needs to close this figure, because `plt.close()` is a deregistration
    rather than a destructor: pyplot holds a strong reference to every figure it
    creates, and *that* reference, not the caller's, is what keeps it alive. A
    directly-instantiated `Figure` never enters that registry, so it is reclaimed by
    ordinary garbage collection once the returned `Axes` goes out of scope. Creating
    `Figure` directly is matplotlib's documented route for library/application code
    (see `matplotlib.figure`'s module docstring); pyplot stays the right choice for
    interactive use, which callers reach by passing their own `ax` instead.
    """
    return Figure().add_subplot()


def combination_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    values: Float[np.ndarray, "nely nelx"],
    eps: float,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Ports `draw_combination1`: draws only the elements with density `>= eps`, each
    colored by `values` (jet colormap, per-face flat color -- not interpolated from
    vertices, matching the MATLAB source's `FaceColor='flat'`). Typically `tPhys`, but
    the MATLAB source also reuses this same call with a per-element hotspot-severity
    score in place of print time (see `cli.py`) -- `values` is any per-element scalar,
    not necessarily a time field.
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

    coll = PolyCollection(verts, array=values, cmap="jet", edgecolors="none")
    ax.add_collection(coll)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
    return ax


def stage_boundary_plot(
    tPhys: Float[np.ndarray, "nely nelx"],
    nStage: int,
    *,
    ax: Axes | None = None,
    combination_coords: bool = False,
) -> Axes:
    """Ports `draw_boundary`: assigns each element to one of `nStage` print stages by its
    `tPhys` value (the `tt`/`mt` binning loop -- half-open `(tt[j], tt[j+1]]` bins except
    the first, which is closed on both ends), then draws a black line along every internal
    mesh edge whose two adjacent elements fall in different stages.

    The MATLAB source finds these edges via a generic sparse node/edge/face table (built
    for an arbitrary mesh); since this mesh is always a regular `(nely, nelx)` grid, every
    internal edge is exactly the shared edge between two orthogonally-adjacent elements, so
    this ports the same result directly from array comparisons instead.

    `combination_coords`: `draw_boundary`'s native frame (the default) is *not* the same
    frame as `combination_plot`'s -- see the module docstring. Set this to remap edges into
    `combination_plot`'s frame (`x' = x - 0.5, y' = 0.5 - y`) before drawing, e.g. to overlay
    onto an `Axes` a prior `combination_plot` call already populated; leave it `False` for a
    standalone, MATLAB-faithful boundary plot.
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
    """`combination_plot` (binarized density, colored by `hotspot_severity`) with
    `stage_boundary_plot` overlaid in its `combination_coords` frame -- the plot recipe
    `cli.py` saves as `final_structure.png`, factored out here so both `cli.py` and this
    module's CLI (which reloads a saved run's fields instead of using an in-process
    state) draw the same thing.

    :param xPhys: physical density field (not yet binarized).
    :param hotspot_severity: per-element overheating severity, e.g. `(1 - K_est) * (xPhys > 0.5)`.
    :param tPhys: physical print-time field.
    :param nStage: number of print stages, for `stage_boundary_plot`'s binning.
    :return: the `Axes` drawn into.
    """
    XPhys = (xPhys > 0.5).astype(xPhys.dtype)
    ax = combination_plot(XPhys, hotspot_severity, eps=1.0e-1, ax=ax)
    stage_boundary_plot(tPhys, nStage, ax=ax, combination_coords=True)
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

    ax = hotspot_severity_plot(xPhys, hotspot_severity, tPhys, config.nStage)

    plot_dir = args.plot_dir / args.tag
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "final_structure.png"
    ax.figure.savefig(out_path, dpi=150)
    print(f"Saved final structure plot to {out_path}")


if __name__ == "__main__":
    _args = _parse_args()  # early-exits on --help before the heavy imports below
    _main(_args)
