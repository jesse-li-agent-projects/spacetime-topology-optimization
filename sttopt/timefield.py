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
from scipy.sparse.linalg import spsolve
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


def _harmonic_void(
    solid: Bool[np.ndarray, " nel"],
    t_solid: Float[np.ndarray, " nel"],
    nelx: int,
    nely: int,
) -> Float[np.ndarray, " nel"]:
    """Harmonic extension of `t_solid` into the void: `grad^2 t = 0` there, Dirichlet
    `t = t_solid` on the material interface, natural (Neumann) conditions on the domain
    boundary.

    The 5-point Laplacian, restricted to the void unknowns; solid neighbours move to the
    right-hand side as known values, and out-of-domain neighbours are simply dropped,
    which is the reflecting condition. Every void component borders material on a
    domain with any solid at all, so the restricted operator is nonsingular.

    :param solid: flat material mask
    :param t_solid: flat field holding the interface values at the solid entries
    :param nelx: element count in x
    :param nely: element count in y
    :return: flat field, void entries filled; solid entries are `t_solid` unchanged
    """
    nel = solid.size
    void_idx = np.flatnonzero(~solid)
    out = t_solid.copy()
    if void_idx.size == 0:
        return out

    grid = np.arange(nel).reshape(nely, nelx)
    unknown = np.full(nel, -1)
    unknown[void_idx] = np.arange(void_idx.size)

    rows, cols, vals = [], [], []
    degree = np.zeros(void_idx.size)
    rhs = np.zeros(void_idx.size)
    for axis, step in ((0, nelx), (0, -nelx), (1, 1), (1, -1)):
        # `here`/`there` pair each element with one orthogonal neighbour, dropping the
        # border row or column that has none in this direction -- which is the Neumann
        # condition, since a missing neighbour then contributes to neither side.
        keep = slice(1, None) if step > 0 else slice(None, -1)
        here = (grid[keep, :] if axis == 0 else grid[:, keep]).flatten()
        there = here - step

        here, there = here[~solid[here]], there[~solid[here]]
        row = unknown[here]
        np.add.at(degree, row, 1.0)

        into_void = ~solid[there]
        rows.append(row[into_void])
        cols.append(unknown[there[into_void]])
        vals.append(np.full(int(into_void.sum()), -1.0))
        np.add.at(rhs, row[~into_void], t_solid[there[~into_void]])

    rows.append(np.arange(void_idx.size))
    cols.append(np.arange(void_idx.size))
    vals.append(degree)
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(void_idx.size, void_idx.size),
    ).tocsr()

    out[void_idx] = spsolve(A, rhs)
    return out


def init_harmonic_timefield(
    xPhys: Float[np.ndarray, "nely nelx"], base: Int[np.ndarray, " k"]
) -> Float[np.ndarray, "nely nelx"]:
    """`init_geodesic_timefield`'s solid pass, with the void filled by a harmonic
    extension (`_harmonic_void`) instead of by growing outward from the material.

    Same solid times, so the part's own initialization is untouched; the difference is
    entirely in the free variables. Growing outward makes `t` jump at the interface
    wherever the void inherits from a distant part of the material, and that jump is a
    large spurious contribution to any gradient statistic -- 99.7% of the c-shape's
    initial gradient variance sits on boundary-straddling cells under the geodesic fill
    against 86% under this one, and the initial CV falls from 2.22 to 0.097. An
    optimizer starting from a field with less to undo starts in a better basin.

    This deliberately drops the geodesic fill's "void prints after the solid it grows
    from" ordering: a harmonic extension interpolates, so void can land earlier than
    adjacent solid. `t` over void is pinned by nothing physical and is a free design
    variable (see `seqopt`'s module docstring), so for an initialization smoothness is
    the more useful property -- but this is a change of intent, not a bug fix.

    Continuity across the interface is all this buys, not smoothness: a harmonic
    extension matches value but not normal derivative, so `|grad t|` still kinks there.
    A biharmonic extension is the C1 fix should that residual prove to matter.

    :param xPhys: density field, shape `(nely, nelx)`; every solid element must connect
        to `base` through material (`geometry.drop_disconnected`)
    :param base: 0-indexed element numbers to measure distance from
    :return: normalized time field, shape `(nely, nelx)`, in `[0, 1]`
    :raises ValueError: as `init_geodesic_timefield`
    """
    nely, nelx = xPhys.shape
    solid = geometry.solid_mask(xPhys)

    d_solid = _solid_geodesic(solid, base, nelx, nely)
    scale = _geodesic_scale(d_solid, solid)

    t_solid = np.where(solid, d_solid / scale, 0.0)
    # The maximum principle keeps the extension inside the interface values' own range,
    # so the result lands in [0, 1] without a clip.
    return _harmonic_void(solid, t_solid, nelx, nely).reshape(nely, nelx)


def _geodesic_scale(
    d_solid: Float[np.ndarray, " nel"], solid: Bool[np.ndarray, " nel"]
) -> float:
    """The largest solid geodesic distance, which every initialization normalizes by.

    Normalizing by a *solid* extent is what keeps the part's own times independent of
    how much empty space surrounds it -- padding a geometry file, or rasterizing a
    component into a roomier box, leaves them unchanged.

    :raises ValueError: if the material has no geodesic extent to normalize by
    """
    scale = float(d_solid[solid].max())
    if scale == 0:
        raise ValueError(
            "the geometry has no geodesic extent: every solid element is a print-start element, so the whole part deposits at t=0 and there is no print sequence to optimize"
        )
    return scale


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

    scale = _geodesic_scale(d_solid, solid)
    return np.minimum(dist / scale, 1.0).reshape(nely, nelx)


class VoidExtension(StrEnum):
    """Names for how a geometry-dependent initialization fills the void, selectable per
    run. The solid times are the same either way; these differ only in the free
    variables. Adding one is a matter of writing the function and a member here.
    """

    GEODESIC = "geodesic"
    HARMONIC = "harmonic"


_VOID_EXTENSIONS = {
    VoidExtension.GEODESIC: init_geodesic_timefield,
    VoidExtension.HARMONIC: init_harmonic_timefield,
}


def init_geometry_timefield(
    xPhys: Float[np.ndarray, "nely nelx"],
    base: Int[np.ndarray, " k"],
    extension: VoidExtension,
) -> Float[np.ndarray, "nely nelx"]:
    """The geometry-dependent time-field initialization, dispatched by how it fills the
    void. See `init_geodesic_timefield` and `init_harmonic_timefield`.

    :param xPhys: density field, shape `(nely, nelx)`
    :param base: 0-indexed element numbers to measure distance from
    :param extension: which void fill to use, by `VoidExtension` name
    :return: normalized time field, shape `(nely, nelx)`, in `[0, 1]`
    :raises ValueError: if `extension` doesn't name a `VoidExtension` member
    """
    try:
        fn = _VOID_EXTENSIONS[VoidExtension(extension)]
    except ValueError:
        raise ValueError(f"extension must be a VoidExtension member, got {extension!r}")
    return fn(xPhys, base)


# Keeps d|grad t|/d(grad t) finite where the gradient vanishes; small enough (relative
# to the squared gradients themselves, order (1/nelx)^2) to leave the magnitude
# unchanged to many digits wherever it is nonzero.
_GRAD_EPS = 1e-12

# Natural-coordinate offset of a 2x2 Gauss point, and the four points as
# (eta, xi) sign pairs ordered to match `_GAUSS_CORNERS`.
_GAUSS_OFFSET = 3.0**-0.5
_GAUSS_SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
# For each Gauss point, the slice of the element grid holding the cell corner it sits
# nearest. Used to scatter a per-Gauss-point quantity back onto elements.
_GAUSS_CORNERS = (
    (slice(None, -1), slice(None, -1)),
    (slice(None, -1), slice(1, None)),
    (slice(1, None), slice(None, -1)),
    (slice(1, None), slice(1, None)),
)


def _q4_blends(
    eta_sign: int, xi_sign: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The bilinear shape functions at one 2x2 Gauss point, factored into their row and
    column parts: `((row_lo, row_hi), (col_lo, col_hi))`, each pair summing to 1.

    `N` for the corner at `(row_hi, col_lo)` is `row_hi * col_lo`, and so on, so the
    same two pairs weight both the gradient's one-sided differences and any nodal field
    interpolated to the point.

    :param eta_sign: sign of the point's row-direction natural coordinate
    :param xi_sign: sign of its column-direction natural coordinate
    :return: `((row_lo, row_hi), (col_lo, col_hi))`
    """
    eta, xi = eta_sign * _GAUSS_OFFSET, xi_sign * _GAUSS_OFFSET
    return ((1 - eta) / 2, (1 + eta) / 2), ((1 - xi) / 2, (1 + xi) / 2)


def _q4_gauss_gradient(
    tPhys: Float[Tensor, "nely nelx"],
) -> tuple[Float[Tensor, "4 nely-1 nelx-1"], Float[Tensor, "4 nely-1 nelx-1"]]:
    """`(dt/dx, dt/dy)` at the 2x2 Gauss points of every cell, in element units.

    Cells are the dual grid: cell `(i, j)` has the four elements `tPhys[i:i+2, j:j+2]`
    as its corner nodes, so `tPhys` is read as a bilinear (Q4) nodal field and the
    derivatives are that field's, evaluated by the standard shape-function
    derivatives. Each component is a blend of the cell's two one-sided differences in
    that direction.

    :param tPhys: filtered time field
    :return: `(dt/dx, dt/dy)`, each at every cell's four Gauss points
    """
    lo_lo, lo_hi = tPhys[:-1, :-1], tPhys[:-1, 1:]
    hi_lo, hi_hi = tPhys[1:, :-1], tPhys[1:, 1:]

    dt_dx, dt_dy = [], []
    for eta_sign, xi_sign in _GAUSS_SIGNS:
        (row_lo, row_hi), (col_lo, col_hi) = _q4_blends(eta_sign, xi_sign)
        dt_dx.append(row_lo * (lo_hi - lo_lo) + row_hi * (hi_hi - hi_lo))
        dt_dy.append(col_lo * (hi_lo - lo_lo) + col_hi * (hi_hi - lo_hi))
    return torch.stack(dt_dx), torch.stack(dt_dy)


def gradient_magnitude(
    tPhys: Float[Tensor, "nely nelx"],
) -> Float[Tensor, "4 nely-1 nelx-1"]:
    """Magnitude of the time field's spatial gradient, at the 2x2 Gauss points of every
    cell of the dual grid.

    The magnitude is the reciprocal of the local deposited-layer thickness, so this is
    the pointwise view of what `gradient_magnitude_std` and `_gradient_cv` reduce to one
    number. Both consume these samples directly; `gradient_magnitude_elements` scatters
    them back onto elements for plotting.

    Full 2x2 quadrature, not the cheaper single evaluation at the cell centre: the
    centre alone is blind to the `(-1)^(i+j)` hourglass mode, and a checkerboard the
    penalty cannot see is a free sawtooth for the optimizer (PR #91). As a linear map
    on `tPhys` this stencil's only null space is the constant field -- but the magnitude
    is not injective, since squaring erases the alternating sign a `(-1)^j` sawtooth
    puts on `dt/dx`. See `_gradient_cv` for what that costs.

    :param tPhys: filtered time field
    :return: `|grad tPhys|` at each cell's four Gauss points
    """
    dt_dx, dt_dy = _q4_gauss_gradient(tPhys)
    return torch.sqrt(dt_dx**2 + dt_dy**2 + _GRAD_EPS)


def interpolate_to_gauss(
    field: Float[Tensor, "nely nelx"],
) -> Float[Tensor, "4 nely-1 nelx-1"]:
    """Bilinear interpolation of a per-element field onto the same Gauss points
    `gradient_magnitude` evaluates at, so a weight lines up with the sample it weights.

    :param field: per-element values, read as Q4 nodal values like `tPhys` is
    :return: `field` at each cell's four Gauss points
    """
    lo_lo, lo_hi = field[:-1, :-1], field[:-1, 1:]
    hi_lo, hi_hi = field[1:, :-1], field[1:, 1:]

    out = []
    for eta_sign, xi_sign in _GAUSS_SIGNS:
        (row_lo, row_hi), (col_lo, col_hi) = _q4_blends(eta_sign, xi_sign)
        out.append(
            row_lo * (col_lo * lo_lo + col_hi * lo_hi)
            + row_hi * (col_lo * hi_lo + col_hi * hi_hi)
        )
    return torch.stack(out)


def gradient_magnitude_elements(
    tPhys: Float[Tensor, "nely nelx"],
) -> Float[Tensor, "nely nelx"]:
    """`gradient_magnitude` averaged back onto elements, for plotting on the element
    grid the rest of the visualization uses.

    Each Gauss point contributes to the one cell corner it sits nearest, so an interior
    element averages the four samples closest to it, one from each cell it belongs to,
    and a border element averages the fewer it has, so every element carries a value.

    :param tPhys: filtered time field
    :return: per-element mean of the neighbouring Gauss-point magnitudes
    """
    magnitude = gradient_magnitude(tPhys)
    total = torch.zeros_like(tPhys)
    count = torch.zeros_like(tPhys)
    for sample, (rows, cols) in zip(magnitude, _GAUSS_CORNERS):
        total[rows, cols] += sample
        count[rows, cols] += 1
    return total / count.clamp(min=1)


def gradient_magnitude_std(tPhys: Float[Tensor, "nely nelx"]) -> Float[Tensor, ""]:
    """Spread of the time field's spatial gradient magnitude over the mesh.

    The print-time gradient sets the local deposited-layer thickness (thickness goes as
    the reciprocal of the gradient magnitude), so a field whose gradient magnitude
    varies across the domain prints layers of uneven thickness. Penalizing the standard
    deviation of that magnitude -- rather than the magnitude itself -- pushes toward
    uniform layer thickness without prescribing what that thickness should be.

    :param tPhys: filtered time field
    :return: standard deviation of `gradient_magnitude(tPhys)`; zero when there are
        fewer than two samples, since a standard deviation over fewer than two samples
        has no spread to measure
    """
    magnitude = gradient_magnitude(tPhys)
    if magnitude.numel() < 2:
        return tPhys.new_zeros(())
    return torch.std(magnitude)


def roughness(
    tPhys: Float[Tensor, "nely nelx"],
    weights: Float[Tensor, "nely nelx"] | None = None,
) -> Float[Tensor, ""]:
    """Typical element-to-element wiggle amplitude of the time field, in units of `t`:
    the weighted RMS of the 5-point Laplacian residual `mean(4-neighbours) - tPhys`.

    The stencil annihilates any linear field and responds most strongly to the
    one-element modes, so a legitimate constant-thickness sweep reads ~0 at any
    orientation. That is what lets it cover `_gradient_cv`'s null space (PR #94).

    Being an RMS, this is in units of `t` and so shrinks under mesh refinement. Weight
    `relative_roughness` in an objective rather than this; read this one when an
    amplitude in the time field's own units is what you want.

    :param tPhys: physical time field
    :param weights: per-element weight over the interior, e.g. the density field so
        void does not dominate; `None` weights every interior element equally
    :return: the weighted RMS residual, or zero when the interior is empty
    """
    interior = tPhys[1:-1, 1:-1]
    neighbour_mean = (
        tPhys[:-2, 1:-1] + tPhys[2:, 1:-1] + tPhys[1:-1, :-2] + tPhys[1:-1, 2:]
    ) / 4
    residual = neighbour_mean - interior

    w = tPhys.new_ones(residual.shape) if weights is None else weights[1:-1, 1:-1]
    total_w = torch.sum(w)
    if total_w == 0:
        return tPhys.new_zeros(())
    return torch.sqrt(torch.sum(w * residual**2) / total_w)


def _weighted_gradient_mean(
    tPhys: Float[Tensor, "nely nelx"], weights: Float[Tensor, "nely nelx"] | None
) -> tuple[
    Float[Tensor, ""],
    Float[Tensor, "4 nely-1 nelx-1"],
    Float[Tensor, "4 nely-1 nelx-1"],
]:
    """Weighted mean of `|grad tPhys|` over the Gauss points, with the samples and the
    weights it was taken over.

    This mean is the reciprocal of the mean deposited-layer thickness, so dividing by it
    turns a quantity in units of `t` into one in units of a layer -- which is what makes
    a weight on that quantity portable across mesh resolutions. Both penalties here
    divide by it, so a weight *between* them is free of it entirely.

    :param tPhys: physical time field
    :param weights: per-element weight, interpolated to the Gauss points; `None` weights
        every sample equally
    :return: `(mean, samples, gauss_weights)`; `mean` is zero when the total weight is
    """
    g = gradient_magnitude(tPhys)
    w = tPhys.new_ones(g.shape) if weights is None else interpolate_to_gauss(weights)
    total_w = torch.sum(w)
    if total_w == 0:
        return tPhys.new_zeros(()), g, w
    return torch.sum(w * g) / total_w, g, w


def relative_roughness(
    tPhys: Float[Tensor, "nely nelx"],
    weights: Float[Tensor, "nely nelx"] | None = None,
) -> Float[Tensor, ""]:
    """`roughness` in units of the mean layer thickness: the objective's smoothness term.

    Raw `roughness` is in units of `t`, so it falls under mesh refinement and a weight
    tuned at one resolution silently means something else at the next. This ratio is
    what stays put -- refining the same physical field leaves it unchanged -- and it
    reads directly as "the wiggle is this fraction of a layer".

    :param tPhys: physical time field
    :param weights: per-element weight, e.g. the density field so void does not dominate
    :return: the ratio, or zero when the mean gradient is zero (nothing to divide by)
    """
    mean, _, _ = _weighted_gradient_mean(tPhys, weights)
    if mean == 0:
        return tPhys.new_zeros(())
    return roughness(tPhys, weights) / mean


def sawtooth_amplitude(
    tPhys: Float[Tensor, "nely nelx"],
    weights: Float[Tensor, "nely nelx"] | None = None,
) -> Float[Tensor, ""]:
    """Typical depth of a column-alternating corrugation in the time field, in units of
    `t`: the weighted mean of `|tPhys - smooth|`, where `smooth` is `tPhys` filtered
    1-2-1 along x.

    Unlike `roughness`, this looks along one axis only, because the mode it measures has
    one axis: the transverse sawtooth `_gradient_cv` rewards. That makes it the
    amplitude of the specific defect rather than a general wiggle size, which is what a
    "would a print show this" judgement needs. Read `relative_sawtooth_amplitude` for
    that judgement; this one is in `t` and so falls under mesh refinement.

    :param tPhys: physical time field
    :param weights: per-element weight over the columns with both neighbours, e.g. the
        density field so void does not dominate; `None` weights every one equally
    :return: the weighted mean deviation, or zero when the total weight is zero
    """
    interior = tPhys[:, 1:-1]
    smooth = (tPhys[:, :-2] + 2 * interior + tPhys[:, 2:]) / 4

    w = tPhys.new_ones(interior.shape) if weights is None else weights[:, 1:-1]
    total_w = torch.sum(w)
    if total_w == 0:
        return tPhys.new_zeros(())
    return torch.sum(w * torch.abs(interior - smooth)) / total_w


def _central_difference_gradient(
    tPhys: Float[Tensor, "nely nelx"], xPhys: Float[Tensor, "nely nelx"]
) -> Float[Tensor, " k"]:
    """`|grad tPhys|` from central differences, over the elements whose four orthogonal
    neighbours are all solid.

    The change of operator is the whole point of the diagnostics built on this. A
    central difference is blind to a one-element sawtooth -- both neighbours carry the
    same offset and it cancels, exactly so where the amplitude is locally constant --
    so these samples describe the field with the padding mode removed, which
    `gradient_magnitude` cannot do. That same blindness is why this must never be
    optimized: it would leave the mode entirely unconstrained.

    Stencils touching void are dropped rather than down-weighted: `t` over void is
    pinned by nothing physical, so a straddling difference reports on a free variable
    instead of on the part.

    :param tPhys: physical time field
    :param xPhys: density field, to locate the fully-solid stencils
    :return: the qualifying samples, flattened; empty if no stencil qualifies
    """
    dx = (tPhys[1:-1, 2:] - tPhys[1:-1, :-2]) / 2
    dy = (tPhys[2:, 1:-1] - tPhys[:-2, 1:-1]) / 2
    g = torch.sqrt(dx**2 + dy**2 + _GRAD_EPS)

    solid = xPhys > geometry.SOLID_THRESHOLD
    keep = (
        solid[1:-1, 1:-1]
        & solid[1:-1, 2:]
        & solid[1:-1, :-2]
        & solid[2:, 1:-1]
        & solid[:-2, 1:-1]
    )
    return g[keep]


def relative_sawtooth_amplitude(
    tPhys: Float[Tensor, "nely nelx"], xPhys: Float[Tensor, "nely nelx"]
) -> Float[Tensor, ""]:
    """`sawtooth_amplitude` in units of a layer thickness: "the corrugation is this
    fraction of a layer deep", the number that says whether a print would show it.

    The layer thickness here is the *central-difference* mean gradient, not
    `relative_roughness`'s `_weighted_gradient_mean`. The corrugation enters the latter
    in quadrature and inflates it, so normalizing by it would make a deepening
    corrugation report a smaller fraction -- the wrong direction for a measure of that
    corrugation. A sawtooth-blind denominator is the thickness the field would have
    without the defect, which is what the fraction is meant to be against.

    :param tPhys: physical time field
    :param xPhys: density field, weighting `sawtooth_amplitude` and masking the stencils
    :return: the ratio, or zero when no fully-solid stencil qualifies
    """
    g = _central_difference_gradient(tPhys, xPhys)
    if g.numel() == 0 or g.mean() == 0:
        return tPhys.new_zeros(())
    return sawtooth_amplitude(tPhys, xPhys) / g.mean()


def central_difference_cv(
    tPhys: Float[Tensor, "nely nelx"], xPhys: Float[Tensor, "nely nelx"]
) -> Float[Tensor, ""]:
    """Layer-uniformity CV over `_central_difference_gradient`'s samples: the
    sawtooth-blind counterpart of `_gradient_cv`, and a diagnostic, never an objective.

    `_gradient_cv` is the quantity a run optimizes and this is the quantity a run should
    be judged on; the gap between them is what the transverse sawtooth is padding, and
    has been measured at 4x on the c-shape.

    :param tPhys: physical time field
    :param xPhys: density field, to locate the fully-solid stencils
    :return: the coefficient of variation, or zero when fewer than two stencils qualify
        or the mean gradient is zero
    """
    sample = _central_difference_gradient(tPhys, xPhys)
    if sample.numel() < 2:
        return tPhys.new_zeros(())
    mean = sample.mean()
    if mean == 0:
        return tPhys.new_zeros(())
    # Population std, not torch's default sample std, so this matches the plain
    # `g.std()/g.mean()` of the numpy analyses this number is compared against.
    return torch.std(sample, unbiased=False) / mean


class UniformityMetric(StrEnum):
    """Names for `uniformity_penalty`'s selectable layer-uniformity measure. Adding a
    measure is a matter of writing a function and a member here -- see
    `uniformity_penalty`'s docstring for what shape a new one must fit.
    """

    GRADIENT_CV = "gradient_cv"


def _gradient_cv(
    tPhys: Float[Tensor, "nely nelx"], weights: Float[Tensor, "nely nelx"] | None
) -> Float[Tensor, ""]:
    """Density-weighted coefficient of variation of `|grad tPhys|` over the mesh:
    `m = sum(w*g)/sum(w)`, `s = sqrt(sum(w*(g-m)^2)/sum(w))`, returning `s/m`.

    Samples are `gradient_magnitude`'s Gauss points, so the sums are a 2x2 quadrature
    of the weighted statistic over the domain rather than an element average.

    Weighted, so the statistic is taken over the part rather than the bounding box --
    `t` over void is pinned by nothing physical, so unweighted gradients there would be
    noise dominating the spread on any part that doesn't fill its box. Divided by the
    mean, so the measure is resolution-invariant: `gradient_magnitude` is a difference
    in element units, so `|grad t|` (and its raw spread) scale as `1/nelx`/`1/nely` for
    a field spanning `[0, 1]`, but the ratio does not. That keeps a `uniformity_weight`
    meaningful across resolutions of the same component, and makes the value directly
    interpretable: 0.1 means layer thickness varies by about 10% of its own mean.

    **Not a smoothness measure, and not usable alone.** A sawtooth across the print
    direction is invisible here (see `gradient_magnitude`) and worse than free: it
    enters `|grad t|` in quadrature, so modulating its amplitude pads locally thin
    layers up to the thickest and drives this measure *down*. Optimizing it alone
    therefore reports a uniformity the field does not have -- measured at 2.7% against a
    true 11% on the c-shape. Pair it with `relative_roughness` (PR #94).

    :param tPhys: physical time field
    :param weights: per-element weight, shape `(nely, nelx)`, interpolated to the same
        Gauss points as the gradient; `None` weights every sample equally
    :return: the coefficient of variation, or zero if the total weight or the mean
        gradient is zero (nothing to divide by)
    """
    mean, g, w = _weighted_gradient_mean(tPhys, weights)
    if mean == 0:
        return tPhys.new_zeros(())
    variance = torch.sum(w * (g - mean) ** 2) / torch.sum(w)
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
