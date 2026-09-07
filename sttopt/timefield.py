"""The "time" field of space-time topology optimization: its initial variants, and
scalar measures of the optimized field.

`tPhys` is initialized to one of these fields before the optimization loop starts,
encoding a spatial ordering of when each element is expected to be "active" (e.g. a
solidification or deposition front sweeping the domain). CORNER/EDGE/OPPOSITE_CORNER/
BOTTOM_EDGE are normalized Euclidean-distance or linear ramps over the `(nely, nelx)`
element grid; `init_geodesic_timefield` (below) is the odd one out, since it depends
on the geometry rather than only on the mesh.
See `conventions.md` for the grid/array-order convention they follow.

A lone-1 mesh (`nelx == 1` xor `nely == 1`) is well-defined but doesn't necessarily
span `[0, 1]`: EDGE is constant at 0 when `nelx == 1`, BOTTOM_EDGE is constant at 0
when `nely == 1`, and OPPOSITE_CORNER never reaches 0 when `nely == 1`. At
`nelx == nely == 1` the two distance variants are undefined (a zero max distance to
normalize by, giving `nan`); rejecting that mesh is `stto.build_problem`'s job, since
the same mesh also degenerates the continuity filter and `build_problem` is where both
are first constructed.
"""

from enum import IntEnum, StrEnum

import numpy as np
import scipy.sparse as sp
import torch
from jaxtyping import Bool, Float, Int
from scipy.sparse.csgraph import dijkstra
from torch import Tensor

import sttopt.geometry as geometry


class TimeField(IntEnum):
    """Named time-field initialization variants, matching `tfield` in the MATLAB source
    for the first three; `BOTTOM_EDGE` is new (build plate at the bottom, matching
    every Wu2025 example) and `GEODESIC` is not a member here at all -- see its own
    `init_geodesic_timefield`, which does not fit this enum's plain
    `(nelx, nely, variant)` shape.
    """

    CORNER = 1  # top-left corner distance
    EDGE = 2  # left-edge ramp
    OPPOSITE_CORNER = 3  # bottom-left corner distance
    BOTTOM_EDGE = 4  # bottom-to-top ramp


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
    Build one of the named time-field initializations.

    CORNER and OPPOSITE_CORNER are normalized distances from the top-left (x=0, y=0)
    and bottom-left (x=0, y=nely) grid corners; EDGE is a left-to-right linear ramp in
    x, constant down each column; BOTTOM_EDGE is a bottom-to-top ramp (`t=0` on the
    bottom row, `t=1` on the top), for a build plate at the bottom.

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
    elif variant == TimeField.BOTTOM_EDGE:
        return np.tile(np.linspace(1, 0, nely)[:, None], (1, nelx))
    else:
        raise ValueError(f"variant must be a TimeField member, got {variant!r}")


def base_elements(nelx: int, nely: int, variant: TimeField) -> Int[np.ndarray, " k"]:
    """
    Candidate print-start element(s) for a time-field variant: the element(s) nearest
    its origin, before any geometry-dependent filtering (`geometry.base_elements`).

    `stto.build_problem` used to inline this exact logic for the three original
    variants; calling this instead keeps the two from drifting apart as new variants
    are added.

    :param nelx: element count in x
    :param nely: element count in y
    :param variant: which time field the candidates are for
    :return: 0-indexed element numbers
    :raises ValueError: if `variant` is not a `TimeField` member
    """
    if variant == TimeField.CORNER:
        return np.array([0])
    elif variant in (TimeField.EDGE, TimeField.OPPOSITE_CORNER):
        return np.arange(nely) * nelx  # column 0
    elif variant == TimeField.BOTTOM_EDGE:
        return (nely - 1) * nelx + np.arange(nelx)  # bottom row
    else:
        raise ValueError(f"variant must be a TimeField member, got {variant!r}")


def _solid_geodesic(
    solid: Bool[np.ndarray, " nel"], base: Int[np.ndarray, " k"], nelx: int, nely: int
) -> Float[np.ndarray, " nel"]:
    """Geodesic distance from `base` through material alone, in element units.

    :raises ValueError: if any solid element is unreachable -- an island that cannot be
        deposited from the build plate, which `geometry.drop_disconnected` removes
    """
    src, dst, step = geometry.neighbor_pairs(nelx, nely)
    within = solid[src] & solid[dst]
    graph = sp.csr_matrix(
        (step[within], (src[within], dst[within])), shape=(solid.size, solid.size)
    )
    dist = dijkstra(graph, directed=False, indices=base, min_only=True)

    unreachable = solid & ~np.isfinite(dist)
    if unreachable.any():
        raise ValueError(
            f"{int(unreachable.sum())} solid element(s) have no path of material back to the build plate, so their print time is undefined; call geometry.drop_disconnected first"
        )
    return dist


def _void_geodesic(
    solid: Bool[np.ndarray, " nel"],
    d_solid: Float[np.ndarray, " nel"],
    nelx: int,
    nely: int,
) -> Float[np.ndarray, " nel"]:
    """Distance to each void element, continuing `_solid_geodesic`'s metric outward
    from the material boundary -- same element units, so the two concatenate without a
    conversion factor.

    Every solid element seeds the traversal at its own print time, via one virtual
    super-source node joined to each by an edge of that length. Only `solid -> void`
    and `void -> void` edges exist, so a path leaves the material once and cannot
    re-enter it to take a shortcut along the part at step cost instead of at the
    part's own print times.
    """
    nel = solid.size
    src, dst, step = geometry.neighbor_pairs(nelx, nely)
    leaving = ~solid[dst]

    solid_idx = np.flatnonzero(solid)
    rows = np.concatenate([src[leaving], np.full(solid_idx.size, nel)])
    cols = np.concatenate([dst[leaving], solid_idx])
    weights = np.concatenate([step[leaving], d_solid[solid_idx]])

    graph = sp.csr_matrix((weights, (rows, cols)), shape=(nel + 1, nel + 1))
    return dijkstra(graph, directed=True, indices=nel)[:nel]


def init_geodesic_timefield(
    xPhys: Float[np.ndarray, "nely nelx"], base: Int[np.ndarray, " k"]
) -> Float[np.ndarray, "nely nelx"]:
    """
    Normalized geodesic distance from `base`, measured through the material, in two
    passes: the solid field first, then the void field continuing outward from it.

    Unlike the bounding-box ramps above, this depends on the geometry -- on a C-shape
    the two arms then start from times that reflect the real path back to the build
    plate, which a planar ramp gets wrong. Both passes matter for that: measuring
    solid and void in one traversal lets a path cross a narrow gap instead of following
    the material, so on that same C-shape the far arm's tip inherits its near
    neighbour's time across the slot rather than its own long way round. Solid
    distances here never leave the material, so no such shortcut exists.

    `seqopt`'s time field is a free design variable over every element, including void
    (see `plans/archive/fixed_geometry_sequence_optimization.md`), so void gets a real
    initialization rather than a fill value: the second pass seeds every solid element
    at its own time and spreads outward, so a void element prints after the solid it
    grows from and there is no discontinuity at the material boundary for the
    continuity constraint to fight. Note "grows from", not "touches" -- void between an
    early arm and a late one follows the early arm, and so can print before the late
    arm it also sits against.

    Normalization is by the largest *solid* distance, so the part's own times do not
    depend on how much empty space surrounds it -- padding a geometry file, or
    rasterizing a component into a roomier box, leaves them unchanged. Void beyond that
    distance clips to 1.

    Grid-graph metrication error (a few percent, from the 8-connected approximation to
    true Euclidean distance) is irrelevant for an initialization; this deliberately
    does not reach for fast marching.

    :param xPhys: density field, shape `(nely, nelx)`; every solid element must connect
        to `base` through material (`geometry.drop_disconnected`)
    :param base: 0-indexed element numbers to measure distance from
    :return: normalized distance, shape `(nely, nelx)`, in `[0, 1]`
    :raises ValueError: if any solid element has no path of material back to `base`, or
        if the material has no geodesic extent to normalize by
    """
    nely, nelx = xPhys.shape
    solid = geometry.solid_mask(xPhys)

    d_solid = _solid_geodesic(solid, base, nelx, nely)
    d_void = _void_geodesic(solid, d_solid, nelx, nely)
    dist = np.where(solid, d_solid, d_void)

    scale = float(d_solid[solid].max())
    if scale == 0:
        raise ValueError(
            "the geometry has no geodesic extent: every solid element is a print-start element, so the whole part deposits at t=0 and there is no print sequence to optimize"
        )
    return np.minimum(dist / scale, 1.0).reshape(nely, nelx)


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


class UniformityMetric(StrEnum):
    """Names for `uniformity_penalty`'s selectable layer-uniformity measure. Adding a
    measure is a matter of writing a function and a member here -- see
    `uniformity_penalty`'s docstring for what shape a new one must fit.
    """

    GRADIENT_CV = "gradient_cv"


def _gradient_cv(
    tPhys: Float[Tensor, "nely nelx"], weights: Float[Tensor, "nely nelx"] | None
) -> Float[Tensor, ""]:
    """Density-weighted coefficient of variation of `|grad tPhys|` over the mesh
    interior: `m = sum(w*g)/sum(w)`, `s = sqrt(sum(w*(g-m)^2)/sum(w))`, returning `s/m`.

    Weighted, so the statistic is taken over the part rather than the bounding box --
    `t` over void is pinned by nothing physical, so unweighted gradients there would be
    noise dominating the spread on any part that doesn't fill its box. Divided by the
    mean, so the measure is resolution-invariant: `gradient_magnitude` is a central
    difference in element units, so `|grad t|` (and its raw spread) scale as
    `1/nelx`/`1/nely` for a field spanning `[0, 1]`, but the ratio does not. That keeps
    a `uniformity_weight` meaningful across resolutions of the same component, and
    makes the value directly interpretable: 0.1 means layer thickness varies by about
    10% of its own mean.

    :param tPhys: physical time field
    :param weights: per-element weight, shape `(nely, nelx)`, cropped to the interior
        to line up with `gradient_magnitude`'s output; `None` weights every interior
        element equally
    :return: the coefficient of variation, or zero if the total weight or the mean
        gradient is zero (nothing to divide by)
    """
    g = gradient_magnitude(tPhys)
    w = tPhys.new_ones(g.shape) if weights is None else weights[1:-1, 1:-1]

    total_w = torch.sum(w)
    if total_w == 0:
        return tPhys.new_zeros(())
    mean = torch.sum(w * g) / total_w
    if mean == 0:
        return tPhys.new_zeros(())
    variance = torch.sum(w * (g - mean) ** 2) / total_w
    return torch.sqrt(variance) / mean


_UNIFORMITY_METRICS = {UniformityMetric.GRADIENT_CV: _gradient_cv}


def uniformity_penalty(
    tPhys: Float[Tensor, "nely nelx"],
    metric: UniformityMetric,
    weights: Float[Tensor, "nely nelx"] | None = None,
) -> Float[Tensor, ""]:
    """The selected layer-uniformity penalty, dispatched by `metric`.

    :param tPhys: physical time field
    :param metric: which measure to compute, by `UniformityMetric` name
    :param weights: per-element weight passed through to the selected measure (e.g.
        the density field, so void doesn't dominate the statistic -- see the
        individual metric's own docstring for how it uses this)
    :return: the penalty value
    :raises ValueError: if `metric` doesn't name a `UniformityMetric` member
    """
    try:
        fn = _UNIFORMITY_METRICS[UniformityMetric(metric)]
    except ValueError:
        raise ValueError(f"metric must be a UniformityMetric member, got {metric!r}")
    return fn(tPhys, weights)
