"""Initial "time" field variants for space-time topology optimization.

`tPhys` is initialized to one of these fields before the optimization loop starts,
encoding a spatial ordering of when each element is expected to be "active" (e.g. a
solidification or deposition front sweeping the domain). All three are normalized
Euclidean-distance or linear ramps over the `(nely, nelx)` element grid; see
`conventions.md` for the grid/array-order convention they follow.
"""

from enum import IntEnum

import numpy as np
from jaxtyping import Float


class TimeField(IntEnum):
    """Named time-field initialization variants, matching `tfield` in the MATLAB source."""

    CORNER = 1  # top-left corner distance
    EDGE = 2  # left-edge ramp
    OPPOSITE_CORNER = 3  # bottom-left corner distance


def _check_grid_size(nelx: int, nely: int) -> None:
    """Reject `nelx < 2` or `nely < 2`: `linspace(0, nel, nel)` degenerates at a single
    sample -- `nel == 1` divides by a zero max distance (corner variants) or fails to
    span `[0, 1]` (edge variant) -- so a one-element-wide/tall mesh is unsupported here.
    """
    if nelx < 2 or nely < 2:
        raise ValueError(f"nelx and nely must be >= 2, got nelx={nelx}, nely={nely}")


def _corner_distance_grid(
    nelx: int, nely: int, corner: tuple[float, float]
) -> Float[np.ndarray, "nely nelx"]:
    """Euclidean distance from `corner` (x, y) to each element grid position, normalized by its max.

    Grid coordinates are `linspace(0, nel, nel)` per the source -- `nel` points spanning
    `[0, nel]`, spacing `nel/(nel-1)`, not a unit-spaced grid. This is a faithful port of
    the MATLAB source, not an off-by-one to "fix": since the x- and y-axis spacings
    differ whenever `nelx != nely`, changing this shifts the field's shape (not just an
    overall scale), so it must stay exactly as the source has it to match the fixture.
    Requires `nelx, nely >= 2` (see `_check_grid_size`); undefined at `nel == 1`.
    """
    xpos = np.linspace(0, nelx, nelx)
    ypos = np.linspace(0, nely, nely)
    # default indexing='xy' matches MATLAB meshgrid
    xmesh, ymesh = np.meshgrid(xpos, ypos)
    dist = np.sqrt((xmesh - corner[0]) ** 2 + (ymesh - corner[1]) ** 2)
    return dist / dist.max()


def timefield_corner(nelx: int, nely: int) -> Float[np.ndarray, "nely nelx"]:
    """Normalized distance from the top-left grid corner (x=0, y=0)."""
    _check_grid_size(nelx, nely)
    return _corner_distance_grid(nelx, nely, (0, 0))


def timefield_edge(nelx: int, nely: int) -> Float[np.ndarray, "nely nelx"]:
    """Left-to-right linear ramp in x, constant down each column."""
    _check_grid_size(nelx, nely)
    return np.tile(np.linspace(0, 1, nelx), (nely, 1))


def timefield_opposite_corner(nelx: int, nely: int) -> Float[np.ndarray, "nely nelx"]:
    """Normalized distance from the bottom-left grid corner (x=0, y=nely)."""
    _check_grid_size(nelx, nely)
    return _corner_distance_grid(nelx, nely, (0, nely))


def init_timefield(
    nelx: int, nely: int, variant: TimeField
) -> Float[np.ndarray, "nely nelx"]:
    """Dispatch to one of the three named time-field initializations."""
    if variant == TimeField.CORNER:
        return timefield_corner(nelx, nely)
    elif variant == TimeField.EDGE:
        return timefield_edge(nelx, nely)
    elif variant == TimeField.OPPOSITE_CORNER:
        return timefield_opposite_corner(nelx, nely)
    else:
        raise ValueError(f"variant must be a TimeField member, got {variant!r}")
