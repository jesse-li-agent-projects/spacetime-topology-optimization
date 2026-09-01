"""The "time" field of space-time topology optimization: its initial variants, and
scalar measures of the optimized field.

`tPhys` is initialized to one of these fields before the optimization loop starts,
encoding a spatial ordering of when each element is expected to be "active" (e.g. a
solidification or deposition front sweeping the domain). All three are normalized
Euclidean-distance or linear ramps over the `(nely, nelx)` element grid; see
`conventions.md` for the grid/array-order convention they follow.

A lone-1 mesh (`nelx == 1` xor `nely == 1`) is well-defined but doesn't necessarily
span `[0, 1]`: EDGE is constant at 0 when `nelx == 1`, and OPPOSITE_CORNER never
reaches 0 when `nely == 1`. At `nelx == nely == 1` the two distance variants are
undefined (a zero max distance to normalize by, giving `nan`); rejecting that mesh is
`optimize.build_problem`'s job, since the same mesh also degenerates the continuity
filter and `build_problem` is where both are first constructed.
"""

from enum import IntEnum

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


class TimeField(IntEnum):
    """Named time-field initialization variants, matching `tfield` in the MATLAB source."""

    CORNER = 1  # top-left corner distance
    EDGE = 2  # left-edge ramp
    OPPOSITE_CORNER = 3  # bottom-left corner distance


def _corner_distance_grid(
    nelx: int, nely: int, corner: tuple[float, float]
) -> Float[np.ndarray, "nely nelx"]:
    """
    Euclidean distance from `corner` to each element grid position, normalized by its max.

    Grid coordinates are `linspace(0, nel, nel)` per the source -- `nel` points spanning
    `[0, nel]`, spacing `nel/(nel-1)`, not a unit-spaced grid. This is a faithful port of
    the MATLAB source, not an off-by-one to "fix": since the x- and y-axis spacings
    differ whenever `nelx != nely`, changing this shifts the field's shape (not just an
    overall scale), so it must stay exactly as the source has it to match the fixture.

    :param nelx: element count in x
    :param nely: element count in y
    :param corner: `(x, y)` position to measure distance from
    :return: normalized distances, shape `(nely, nelx)`
    """
    xpos = np.linspace(0, nelx, nelx)
    ypos = np.linspace(0, nely, nely)
    # default indexing='xy' matches MATLAB meshgrid
    xmesh, ymesh = np.meshgrid(xpos, ypos)
    dist = np.sqrt((xmesh - corner[0]) ** 2 + (ymesh - corner[1]) ** 2)
    return dist / dist.max()


def init_timefield(
    nelx: int, nely: int, variant: TimeField
) -> Float[np.ndarray, "nely nelx"]:
    """
    Build one of the three named time-field initializations.

    CORNER and OPPOSITE_CORNER are normalized distances from the top-left (x=0, y=0)
    and bottom-left (x=0, y=nely) grid corners; EDGE is a left-to-right linear ramp in
    x, constant down each column.

    :param nelx: element count in x
    :param nely: element count in y
    :param variant: which field to build
    :return: the time field, shape `(nely, nelx)`
    :raises ValueError: if `variant` is not a `TimeField` member
    """
    if variant == TimeField.CORNER:
        return _corner_distance_grid(nelx, nely, (0, 0))
    elif variant == TimeField.EDGE:
        return np.tile(np.linspace(0, 1, nelx), (nely, 1))
    elif variant == TimeField.OPPOSITE_CORNER:
        return _corner_distance_grid(nelx, nely, (0, nely))
    else:
        raise ValueError(f"variant must be a TimeField member, got {variant!r}")


# Keeps d|grad t|/d(grad t) finite where the gradient vanishes; small enough (relative
# to the squared gradients themselves, order (1/nelx)^2) to leave the magnitude
# unchanged to many digits wherever it is nonzero.
_GRAD_EPS = 1e-12


def gradient_magnitude(
    tPhys: Float[Tensor, "nely nelx"],
) -> Float[Tensor, "nely-2 nelx-2"]:
    """Magnitude of the time field's spatial gradient on the mesh interior.

    Gradients are 2nd-order central differences in element units, so they are defined
    only on interior elements; the border is excluded rather than one-sided. The
    magnitude is the reciprocal of the local deposited-layer thickness, so it is a
    per-element view of what `gradient_magnitude_std` reduces to one number.

    :param tPhys: filtered time field
    :return: `|grad tPhys|` on the interior elements, i.e. `tPhys[1:-1, 1:-1]`
    """
    dt_dx = (tPhys[1:-1, 2:] - tPhys[1:-1, :-2]) / 2
    dt_dy = (tPhys[2:, 1:-1] - tPhys[:-2, 1:-1]) / 2
    return torch.sqrt(dt_dx**2 + dt_dy**2 + _GRAD_EPS)


def gradient_magnitude_std(tPhys: Float[Tensor, "nely nelx"]) -> Float[Tensor, ""]:
    """Spread of the time field's spatial gradient magnitude over the mesh interior.

    The print-time gradient sets the local deposited-layer thickness (thickness goes as
    the reciprocal of the gradient magnitude), so a field whose gradient magnitude
    varies across the domain prints layers of uneven thickness. Penalizing the standard
    deviation of that magnitude -- rather than the magnitude itself -- pushes toward
    uniform layer thickness without prescribing what that thickness should be.

    :param tPhys: filtered time field
    :return: standard deviation of `gradient_magnitude(tPhys)`; zero when the interior
        holds fewer than two elements, since a standard deviation over fewer than two
        samples has no spread to measure
    """
    magnitude = gradient_magnitude(tPhys)
    if magnitude.numel() < 2:
        return tPhys.new_zeros(())
    return torch.std(magnitude)
