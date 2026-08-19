"""Plots for the printed structure: elements colored by print time, and the boundaries
between print stages -- ports `draw_combination1.m`/`draw_boundary.m` (the only two
plotting calls the main script actually exercises; see the plan's Scope section and
Finding 2 for why `draw_combination2`/`3` and the broken `draw_combination` call at the
old line 577 are out of scope / resolved to `draw_combination1`).

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
"""

import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure

_BOUNDARY_LINEWIDTH = 1.5


def _new_axes() -> Axes:
    """A standalone `Axes` on a `Figure` that pyplot does not own, so it needs no
    corresponding `plt.close()` (see this module's docstring).
    """
    return Figure().add_subplot()


def combination_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    eps: float,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Ports `draw_combination1`: draws only the elements with density `>= eps`, each
    colored by its own `tPhys` value (jet colormap, per-face flat color -- not
    interpolated from vertices, matching the MATLAB source's `FaceColor='flat'`).
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
    values = tPhys[rows, cols]

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

    ax.add_collection(LineCollection(segments, colors="black", linewidths=_BOUNDARY_LINEWIDTH))
    ax.set_aspect("equal")
    ax.autoscale_view()
    return ax
