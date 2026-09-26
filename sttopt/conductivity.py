"""Print-time overheating proxy: local estimated conductivity, and the hotspot
constraint bounding its worst-case value.

`estimated_conductivity` scores each element by how strongly its already-printed
(cooler, earlier-`tPhys`) neighborhood shields it from residual heat -- a proxy for
overheating risk during additive deposition. A `PMean` or `LogSumExp` aggregation
collapses the severity `1 - K_est` (weighted toward already-dense, hot regions) into
the calibrated smooth maximum that the hotspot term reports, and the caller gets the
sensitivity from autograd through both (Phase 3.4, `plans/torch_port_part2.md`).

Two orthogonal choices shape that number, both selectable per run: `Normalization`
sets what `K_est` measures shielding against, and `Aggregation` sets how the field
collapses to one scalar. They pair -- HALF_STENCIL admits `K_est > 1`, which only
LOGSUMEXP accepts unconditionally -- but neither implies the other.

`tests/reference/conductivity.py` keeps `hotspot_constraint`, the hand-derived
predecessor that folds the P_MEAN `factor`/`Tcr` scaling and the density-filter chain
rule (`H`/`Hs`/`dx` from `filters.py`) directly into its sensitivities, as a cross-check
(`tests/test_reference_sweep.py`) and timing baseline
(`benchmarks/bench_sensitivities.py`). See `conventions.md` for array-order/tolerance
conventions.

The paper's neighbor weight is radial *times angular*, the angular part favoring the
build direction. `neighbor_weights` below supplies only the radial factor; `_lobe` adds
the angular one, read off the local `grad t` rather than off a fixed build direction,
since deposition order is itself a design variable here. It is off at `kappa = 0`, which
is exactly the radial-only stencil -- see `plans/angular_weight.md`.

One deliberate deviation from Das2025 Eq. (6) remains, load-bearing for this port and
not a bug: the paper's Eq. (6) numerator weights a neighbor by its raw density `rho_j`,
where `_conductivity_core` uses `x_j**q` with `q = 3` -- a SIMP-style penalization of
intermediate density the paper does not have.
"""

from enum import StrEnum
from functools import cache
from itertools import product
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from sttopt import run_config, smooth_max, timefield


class Normalization(StrEnum):
    """Names for what `K_est` measures shielding *against*, selectable per run.

    NEIGHBORHOOD divides by the weight of the caller's own already-printed neighbors,
    so `K_est` is a weighted average of neighbor density and is bounded by 1. Its
    denominator shrinks wherever the stencil leaves the mesh, which renormalizes the
    missing region away: an element on the domain edge whose in-mesh neighbors are all
    solid scores `K_est == 1` exactly, while an identical element beside an internal
    void does not. That makes free surfaces on the mesh boundary invisible, and leaves
    the print time of void -- a quantity nothing physical pins -- driving the result
    through the denominator (PR #98).

    HALF_STENCIL divides by `half_stencil_weight` instead: the shielding an infinite
    uniform layer schedule would supply at the same point in its own schedule. The
    denominator is then the same constant everywhere, so leaving the mesh costs
    exactly what an equal-sized internal void costs, and void print time drops out
    entirely (it only ever multiplies a zero density). `K_est` is no longer an average
    and may exceed 1, meaning "better shielded than an infinite uniform layer" -- see
    `PMean`/`LogSumExp` for what tolerates that.

    HALF_STENCIL is also the only normalization an angular weight can be paired with:
    the constant divisor stops being constant once the stencil stops being symmetric
    under `d -> -d`, and `directed_denominator` is what it generalizes to.
    """

    NEIGHBORHOOD = "neighborhood"
    HALF_STENCIL = "half_stencil"


class Aggregation(StrEnum):
    """Names for how the per-element severity field collapses to the one scalar the
    hotspot term reports, selectable per run: `PMean` or `LogSumExp`, as built by
    `make_aggregation`.

    Each aggregate is biased in the shape its own algebra gives it, which decides the
    shape of its calibration. With `N` elements sharing the maximum and the rest
    negligible, P_MEAN reads `true_max * (N/nel)**(1/p)` and LOGSUMEXP reads
    `true_max + log(N)/beta`: a ratio to undo in one case, a difference in the other.
    Correcting either one in the other's shape holds only at the instant it is measured
    and drifts everywhere else.
    """

    P_MEAN = "p_mean"
    LOGSUMEXP = "logsumexp"


def _stencil_offsets(rmin_cond: float) -> list[tuple[int, int, float]]:
    """Every `(di, dj, w)` grid offset inside the conductivity stencil's radial cutoff.

    The single definition of which offsets are in the stencil and what each weighs, so
    the neighbor pair list, its untruncated total, and the print-base exemption cannot
    drift apart about it.
    """
    r = int(np.ceil(rmin_cond)) - 1
    offsets = []
    for di, dj in product(range(-r, r + 1), repeat=2):
        dist = np.hypot(di, dj)
        if rmin_cond - dist < 0:
            continue
        offsets.append((di, dj, (rmin_cond - dist) / rmin_cond))
    return offsets


@cache
def half_stencil_weight(rmin_cond: float) -> float:
    """`Normalization.HALF_STENCIL`'s constant denominator: the stencil weight already
    deposited around any element under an infinite uniform layer schedule.

    Exactly half the untruncated stencil total, for any `rouf` and any layer spacing.
    Pairing offset `(di, +dj)` against `(di, -dj)` gives equal radial weight and
    `sigmoid(z) + sigmoid(-z) == 1`, so a time field linear in height puts precisely
    half the surrounding weight before each element -- including the `dj == 0` row,
    whose self-pair contributes `sigmoid(0) == 1/2`.
    """
    return sum(w for _, _, w in _stencil_offsets(rmin_cond)) / 2


def constant_denominator(
    normalization: Normalization, rmin_cond: float
) -> float | None:
    """The fixed `K_est` denominator a normalization variant calls for, or `None` for
    NEIGHBORHOOD, which derives a per-element one instead.
    """
    if normalization == Normalization.NEIGHBORHOOD:
        return None
    elif normalization == Normalization.HALF_STENCIL:
        return half_stencil_weight(rmin_cond)
    else:
        raise ValueError(
            f"normalization must be a Normalization member, got {normalization!r}"
        )


def infinite_base(
    normalization: Normalization, base: Int[np.ndarray, " k"]
) -> Int[np.ndarray, " k"] | None:
    """The print base, for the normalizations that model it as an infinitely dense heat
    sink, else `None`.

    The base is not part of the design: the material it stands in for is there whatever
    the deposition order. HALF_STENCIL's denominator assumes a full half-neighborhood of
    prior material everywhere, so the first deposited layer, which by definition has
    nothing before it, otherwise scores as the worst hotspot in the domain. An
    infinitely dense base sends `K_est` to infinity for every element whose stencil
    reaches it, which is what takes those elements out of the hotspot measure -- the
    reach is a consequence of the stencil, never a depth chosen here.

    NEIGHBORHOOD needs none: its denominator shrinks along with its numerator at the
    first deposited layer, so an element there already scores near zero severity
    without being told the base exists. Only a constant denominator has to be told.

    :param base: 0-indexed print-start elements, e.g. `geometry.base_elements`
    """
    return None if normalization == Normalization.NEIGHBORHOOD else base


def neighbor_weights(
    nelx: int, nely: int, rmin_cond: float
) -> tuple[
    Int[np.ndarray, " npairs"], Int[np.ndarray, " npairs"], Float[np.ndarray, " npairs"]
]:
    """Symmetric distance-weighted neighbor structure for the hotspot constraint, as COO
    triplets `(e1, e2, w)` (0-indexed element numbers; `e1 == e2` self-pairs included).

    `w = max(0, rmin_cond - dist(e1, e2)) / rmin_cond` for elements within `rmin_cond` of
    each other -- the same square-window/circular-cutoff pattern as
    `filters.density_filter`, but normalized by `rmin_cond` (density_filter's isn't).
    """
    e = np.arange(nelx * nely)
    i1, j1 = e % nelx, e // nelx
    e1s, e2s, ws = [], [], []
    for di, dj, weight in _stencil_offsets(rmin_cond):
        i2, j2 = i1 + di, j1 + dj
        valid = (i2 >= 0) & (i2 < nelx) & (j2 >= 0) & (j2 < nely)
        e1s.append(e[valid])
        e2s.append((j2 * nelx + i2)[valid])
        ws.append(np.full(valid.sum(), weight))
    return np.concatenate(e1s), np.concatenate(e2s), np.concatenate(ws)


@cache
def _offset_table(
    rmin_cond: float,
) -> tuple[Float[np.ndarray, "noffsets 2"], Float[np.ndarray, " noffsets"]]:
    """`_stencil_offsets` as arrays: the `(dx, dy)` of every offset, in elements, and its
    radial weight.

    The untruncated stencil, i.e. before any of it is cut away by a mesh edge, which is
    what `directed_denominator` has to sum over.
    """
    offsets = _stencil_offsets(rmin_cond)
    return (
        np.array([(di, dj) for di, dj, _ in offsets], dtype=float),
        np.array([w for _, _, w in offsets], dtype=float),
    )


class AngularStencil(NamedTuple):
    """The fixed stencil geometry the angular weight reads, built once per run by
    `angular_stencil`.

    `pair_dir` lines up with `neighbor_weights`' COO pair arrays. The `offset_*` fields
    are the untruncated stencil instead, which is deliberately a different list -- see
    `directed_denominator`.
    """

    pair_dir: Float[Tensor, "npairs 2"]
    offset: Float[Tensor, "noffsets 2"]
    offset_dir: Float[Tensor, "noffsets 2"]
    offset_w: Float[Tensor, " noffsets"]


def _unit_rows(v: Float[Tensor, "n 2"]) -> Float[Tensor, "n 2"]:
    """`v` with every row scaled to unit length, leaving a zero row at zero.

    A zero row means "no direction", which `_lobe` reads off the row itself rather than
    from a parallel mask that could drift away from it.
    """
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return v / torch.where(norm == 0, torch.ones_like(norm), norm)


def angular_stencil(
    e1: Int[Tensor, " npairs"],
    e2: Int[Tensor, " npairs"],
    nelx: int,
    rmin_cond: float,
    dtype: torch.dtype,
    kappa: "run_config.Scheduled",
) -> AngularStencil | None:
    """Build the angular weight's fixed geometry for one mesh, or `None` for a run whose
    lobe never opens.

    The pair directions are derived from the pair list itself rather than rebuilt from
    `_stencil_offsets`, so they cannot fall out of step with the pairs they weight. They
    are also one more `npairs`-sized pair of arrays, which a purely radial run has no
    use for -- hence the `None`.

    :param e1: first index of each COO pair, as `neighbor_weights` returns it
    :param e2: second index of each COO pair
    :param nelx: element count in x, to read an element number back as a grid position
    :param rmin_cond: conductivity-neighborhood radius, in elements
    :param dtype: floating dtype of the run's real-valued fields
    :param kappa: the run's lobe concentration setting; a constant `0` builds nothing,
        while a schedule is taken at its word that it will open the lobe at some point
    """
    if isinstance(kappa, float | int) and kappa == 0:
        return None

    def row(e: Int[Tensor, " npairs"]) -> Int[Tensor, " npairs"]:
        return torch.div(e, nelx, rounding_mode="floor")

    pair_d = torch.stack(
        ((e2 % nelx - e1 % nelx).to(dtype), (row(e2) - row(e1)).to(dtype)), dim=-1
    )
    offset_np, offset_w_np = _offset_table(rmin_cond)
    offset = torch.as_tensor(offset_np, dtype=dtype, device=e1.device)
    return AngularStencil(
        pair_dir=_unit_rows(pair_d),
        offset=offset,
        offset_dir=_unit_rows(offset),
        offset_w=torch.as_tensor(offset_w_np, dtype=dtype, device=e1.device),
    )


def _lobe(
    g_dot_dir: Float[Tensor, "..."],
    gmag: Float[Tensor, "..."],
    dirs: Float[Tensor, "n 2"],
    kappa: float,
    g0: float,
) -> Float[Tensor, "..."]:
    """The von Mises angular weight `exp(-kappa * (g . dhat + |g|) / (|g| + g0))`.

    `1` for a neighbor lying straight behind the print front, decaying as it rotates
    toward and then past the print direction. `kappa` sets the lobe width; 2.37 matches
    the linear ramp of Das2023 Eq. (3.5) at half maximum. `g0` sets how much layering
    must exist before the direction is believed, and is in the same per-unit-length
    units as `g`.

    Written so that normalizing the direction leaves no division by `|g|`: as the
    gradient vanishes the exponent goes to zero and the weight to exactly `1`, i.e. the
    isotropic radial-only stencil, which is the right answer where no layering defines a
    direction. `kappa = 0` is that same isotropic case at any gradient.

    :param g_dot_dir: `g . dhat`, the gradient projected on each direction
    :param gmag: `|g|`, broadcasting against `g_dot_dir`
    :param dirs: the unit directions, whose zero rows mark "no direction" -- the stencil
        origin, which takes weight `1` since there is no angle to penalize. That is the
        one offset no lobe suppresses, so it does not cancel out of a ratio the way it
        does on uniform material (`plans/angular_weight.md`).
    :param kappa: lobe concentration; `0` gives a uniform weight of `1`
    :param g0: the gradient scale at which the lobe reaches half its concentration
    """
    w = torch.exp(-kappa * (g_dot_dir + gmag) / (gmag + g0))
    return torch.where((dirs == 0).all(dim=-1), torch.ones_like(w), w)


def _gradient_magnitude(
    grad: tuple[Float[Tensor, "nely nelx"], Float[Tensor, "nely nelx"]],
) -> Float[Tensor, " nel"]:
    """`|grad t|` per element, flattened, kept differentiable where the gradient
    vanishes (`timefield.GRAD_EPS`)."""
    gx, gy = grad
    return torch.sqrt(gx.flatten() ** 2 + gy.flatten() ** 2 + timefield.GRAD_EPS)


def angular_pair_weights(
    grad: tuple[Float[Tensor, "nely nelx"], Float[Tensor, "nely nelx"]],
    e1: Int[Tensor, " npairs"],
    stencil: AngularStencil,
    kappa: float,
    g0: float,
) -> Float[Tensor, " npairs"]:
    """`_lobe` for every COO pair, against the print direction of the pair's own first
    element.

    :param grad: `(dt/dx, dt/dy)` per element, per unit length
    :param e1: first index of each COO pair, whose gradient each pair is weighted by
    :param stencil: the run's `AngularStencil`
    :param kappa: lobe concentration for this iteration
    :param g0: the lobe's gradient scale
    """
    gx, gy = grad[0].flatten(), grad[1].flatten()
    g_dot_dir = gx[e1] * stencil.pair_dir[:, 0] + gy[e1] * stencil.pair_dir[:, 1]
    return _lobe(g_dot_dir, _gradient_magnitude(grad)[e1], stencil.pair_dir, kappa, g0)


def directed_denominator(
    grad: tuple[Float[Tensor, "nely nelx"], Float[Tensor, "nely nelx"]],
    stencil: AngularStencil,
    kappa: float,
    g0: float,
    rouf: float,
    unit: float,
) -> Float[Tensor, " nel"]:
    """`Normalization.HALF_STENCIL`'s divisor once the stencil carries an angular
    weight: the stencil sum an ideal uniform layered fill would supply at each element's
    own print direction and layer thickness.

    The same sum the numerator computes, evaluated on a reference of full density whose
    time field is exactly linear with the element's own gradient. `K_est = 1` therefore
    still means "as well shielded as an ideal uniform layered fill", which is what the
    constant `half_stencil_weight` meant before the angular factor broke the `d -> -d`
    symmetry that made it a constant.

    Summed over the untruncated offset table, never over the in-mesh pair list: that is
    what keeps the normalization's defining property that leaving the mesh costs exactly
    what an equal-sized internal void costs.

    At `kappa = 0` this is `half_stencil_weight(rmin_cond)` for every direction and every
    gradient magnitude, by the pairing argument in that function's docstring.

    :param grad: `(dt/dx, dt/dy)` per element, per unit length
    :param stencil: the run's `AngularStencil`
    :param kappa: lobe concentration for this iteration
    :param g0: the lobe's gradient scale
    :param rouf: print-order sigmoid sharpness, as in `_pairwise_sigmoid`
    :param unit: `timefield.unit_length`, to read the per-unit-length gradient as a print
        time difference across an offset measured in elements
    """
    gx, gy = grad[0].flatten(), grad[1].flatten()
    g = torch.stack((gx, gy), dim=-1)
    # t_neighbor - t_self under the reference field, hence the sign: the sigmoid masks
    # in what was deposited *before* this element.
    mask = torch.sigmoid(-rouf * (g @ stencil.offset.T) / unit)
    w_ang = _lobe(
        g @ stencil.offset_dir.T,
        _gradient_magnitude(grad)[:, None],
        stencil.offset_dir,
        kappa,
        g0,
    )
    return (stencil.offset_w * w_ang * mask).sum(dim=-1)


def _pairwise_sigmoid(
    t: Float[Tensor, " nel"],
    a: Int[Tensor, " npairs"],
    b: Int[Tensor, " npairs"],
    rouf: float,
) -> Float[Tensor, " npairs"]:
    """
    `FT_el{a}[b]`: the neighbor-sigmoid weight of a COO pair array, a smooth mask on
    whether `b` was printed before `a`.

    `1/(1+exp(z))` is exactly `sigmoid(-z)`; `torch.sigmoid` evaluates it through its
    own overflow-safe form rather than the MATLAB source's literal expression, which
    for large `rouf*dt` overflows -- see `conventions.md`'s "Known deviations".

    :param t: per-element time field, `tPhys.flatten()`
    :param a: first index of each COO pair
    :param b: second index of each COO pair
    :param rouf: sigmoid sharpness
    :return: `FT`, one value per pair
    """
    return torch.sigmoid(rouf * (t[a] - t[b]))


class _ConductivityCore(NamedTuple):
    K_est: Float[Tensor, " nel"]
    denom: Float[Tensor, " nel"] | Float[Tensor, ""]
    FT_ab: Float[Tensor, " npairs"]
    xb_q: Float[Tensor, " npairs"]


def _conductivity_core(
    x: Float[Tensor, " nel"],
    t: Float[Tensor, " nel"],
    e1: Int[Tensor, " npairs"],
    e2: Int[Tensor, " npairs"],
    w: Float[Tensor, " npairs"],
    q: float,
    rouf: float,
    denom: float | Float[Tensor, " nel"] | None = None,
    base: Int[Tensor, " k"] | None = None,
) -> _ConductivityCore:
    """`K_est` and the divisor it was formed with, plus the `a->b` pair terms
    (`FT_ab`/`xb_q`) shared by `estimated_conductivity` and
    `tests/reference/conductivity.py`'s `_conductivity_terms` -- computed once here
    rather than redone by each.

    :param denom: a constant divisor for every element (`constant_denominator`), a
        per-element one (`directed_denominator`), or `None` for the per-element
        already-printed neighbor weight.
    :param base: print base elements, taken to be infinitely dense (`infinite_base`),
        or `None` to leave them ordinary design material.
    """
    nel = x.shape[0]
    FT_ab = _pairwise_sigmoid(t, e1, e2, rouf)
    xb_q = x[e2] ** q
    num = torch.zeros(nel, dtype=x.dtype, device=x.device)
    num.index_add_(0, e1, xb_q * w * FT_ab)
    if denom is None:
        divisor = torch.zeros(nel, dtype=x.dtype, device=x.device)
        divisor.index_add_(0, e1, w * FT_ab)
    else:
        divisor = torch.as_tensor(denom, dtype=x.dtype, device=x.device)
    K_est = num / divisor
    if base is not None:
        # An infinitely dense neighbor sends the sum above to infinity, so `K_est`
        # diverges wherever any stencil weight lands on the base at all. Selected
        # rather than evaluated as `inf * w`, whose backward is `nan` -- the same
        # safe-input-then-reselect pattern as `_safe_pmean`.
        on_base = torch.zeros(nel, dtype=x.dtype, device=x.device)
        on_base[base] = 1.0
        base_weight = torch.zeros(nel, dtype=x.dtype, device=x.device)
        base_weight.index_add_(0, e1, w * FT_ab.detach() * on_base[e2])
        K_est = torch.where(base_weight > 0, torch.inf, K_est)
    return _ConductivityCore(K_est, divisor, FT_ab, xb_q)


def _safe_pmean(u: Float[Tensor, ""], p: float) -> Float[Tensor, ""]:
    """
    `u**(1/p)`, with value and gradient both `0` at `u == 0`, rather than the finite
    forward value followed by a `nan` backward (`1/p - 1 < 0` for `p > 1`, so the naive
    gradient diverges there).

    Standard "safe input, then re-select" pattern: evaluate the singular branch at a
    substitute input that never actually triggers the singularity, so its local
    gradient is finite, then `where`-select the *value* -- `torch.where`'s backward
    routes zero incoming gradient into the discarded branch precisely where it was
    substituted, so the substitute's finite-but-irrelevant gradient never multiplies a
    `0 * inf`.

    :param u: a nonnegative mean of `p`-th powers. `u == 0` is a legitimate value, not
        a corner case -- it occurs whenever every element's contribution is zero.
    :param p: the power `u` is being inverted by; the singularity at `u == 0` requires
        `p > 1`.
    :return: `u**(1/p)`, gradient-safe at `u == 0`.
    """
    safe_u = torch.where(u == 0, torch.ones_like(u), u)
    val = safe_u ** (1 / p)
    return torch.where(u == 0, torch.zeros_like(val), val)


def severity(
    K_est: Float[Tensor, " nel"], xPhys: Float[Tensor, "nely nelx"], r: float
) -> Float[Tensor, " nel"]:
    """The per-element overheating severity `(1 - K_est) * xPhys**r`, `-inf` where an
    element is infinitely shielded and so has no severity to compare.

    The one no-grad statement of the quantity the aggregations smooth-max and the
    calibration targets, so a reading of the field cannot drift from the optimized one.
    The aggregations build their own gradient-safe variants of it.
    """
    with torch.no_grad():
        finite = torch.isfinite(K_est)
        return torch.where(finite, (1 - K_est) * xPhys.flatten() ** r, -torch.inf)


def _max_severity(
    K_est: Float[Tensor, " nel"], xPhys: Float[Tensor, "nely nelx"], r: float
) -> float:
    """The true maximum severity a smooth maximum stands in for."""
    return float(severity(K_est, xPhys, r).max())


class PMean:
    """`mean(T**p * x**(r*p)) ** (1/p)`, a scale-equivariant smooth maximum, carried
    onto the true maximum by a calibration ratio.

    It is built on powers, so it needs a non-negative field: `T` can go negative under
    a constant denominator, which `**p` only tolerates while `p` is integral. It
    approaches the true maximum from below by a factor `nel**(-1/p)` that worsens as the
    mesh grows -- 0.687 at `p=25` on a 120x100 mesh, a quarter low. Its density weight
    is `x**(r*p)`, so `p` cannot be changed without also changing how hard void is
    suppressed, and `r*p <= 1` silently makes the gradient diverge at `x == 0`.

    The calibration is run state, refreshed in place by a call with `recalibrate`.
    """

    def __init__(self, p: float, r: float):
        self.p = p
        self.r = r
        self.calibration = 1.0

    def resolve(self, loop: int) -> bool:
        """No scheduled parameters to adopt, so nothing that stales the calibration."""
        return False

    def __call__(
        self,
        K_est: Float[Tensor, " nel"],
        xPhys: Float[Tensor, "nely nelx"],
        recalibrate: bool = False,
    ) -> Float[Tensor, ""]:
        """The calibrated aggregate, differentiable in `K_est` and `xPhys`.

        Written in a NaN-safe form: `(T * x**r) ** p` differentiates to `inf` at
        `x == 0` (density does reach exact zero once the Heaviside projection
        saturates), so this computes the algebraically identical `T**p * x**(r*p)`,
        whose gradient is finite while `r*p > 1`.

        :param recalibrate: first refresh the calibration against this field's true
            maximum. Kept as it is where every element contributes zero, since the
            aggregate then says nothing about the maximum.
        """
        x = xPhys.flatten()
        # An infinitely shielded element contributes nothing, which `(-inf)**p` is not
        # a way to say.
        T = torch.where(torch.isinf(K_est), torch.zeros_like(K_est), 1 - K_est)
        numer = _safe_pmean(torch.mean(T**self.p * x ** (self.r * self.p)), self.p)
        if recalibrate and float(numer) != 0:
            self.calibration = _max_severity(K_est, xPhys, self.r) / float(numer)
        return self.calibration * numer


class LogSumExp(smooth_max.CalibratedLogSumExp):
    """The calibrated LogSumExp smooth maximum of the severity `T * x**r`.

    Suits `Normalization.HALF_STENCIL`, which pins the scale against a physical
    reference, so `PMean`'s scale equivariance buys nothing there. Its additive bias is
    +0.001 at `beta=200` where `PMean` at `p=25` is -26%.

    It aggregates exactly the quantity `PMean` and the calibration both target: density
    enters through the severity, so void is not suppressed beyond scoring zero severity
    -- a void element's share of the sum is `exp(-beta * max)`, which a sharp `beta`
    makes negligible.
    """

    def __init__(self, beta: "run_config.Scheduled", r: float):
        """
        :param beta: sharpness, possibly scheduled; `resolve` adopts one iteration's.
        :param r: the density exponent of the severity the calibration targets
        """
        super().__init__(beta)
        self.r = r

    def __call__(
        self,
        K_est: Float[Tensor, " nel"],
        xPhys: Float[Tensor, "nely nelx"],
        recalibrate: bool = False,
    ) -> Float[Tensor, ""]:
        """The calibrated aggregate, differentiable in `K_est` and `xPhys`.

        :param recalibrate: see `aggregate`.
        """
        shielded = torch.isinf(K_est)
        x_r = smooth_max.density_power(xPhys.flatten(), self.r)
        # `1 - inf` would poison the `-inf` reselect below through `inf * 0`.
        T = torch.where(shielded, torch.zeros_like(K_est), 1 - K_est)
        sev = torch.where(shielded, -torch.inf, T * x_r)
        return self.aggregate(sev, recalibrate)


def make_aggregation(
    aggregation: Aggregation, p: float, r: float, beta: "run_config.Scheduled"
) -> PMean | LogSumExp:
    """A fresh, uncalibrated aggregation for a run's settings.

    :param beta: `LogSumExp` sharpness, possibly scheduled; unused by `PMean`, which
        takes `p` instead.
    """
    if aggregation == Aggregation.P_MEAN:
        return PMean(p, r)
    elif aggregation == Aggregation.LOGSUMEXP:
        return LogSumExp(beta, r)
    else:
        raise ValueError(
            f"aggregation must be an Aggregation member, got {aggregation!r}"
        )


def estimated_conductivity(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    e1: Int[Tensor, " npairs"],
    e2: Int[Tensor, " npairs"],
    w: Float[Tensor, " npairs"],
    q: float,
    rouf: float,
    denom: float | None = None,
    base: Int[Tensor, " k"] | None = None,
    stencil: AngularStencil | None = None,
    kappa: float = 0.0,
    g0: float = 1.0,
) -> Float[Tensor, " nely*nelx"]:
    """Local estimated conductivity: how strongly each element's neighborhood has
    already solidified (cooler, earlier `tPhys`) around it, used as an overheating
    proxy by `PMean`/`LogSumExp`. `Normalization` sets what it is measured relative to,
    via `denom`, and `infinite_base`'s `base` is infinite wherever it is in reach.

    With `kappa > 0` the stencil additionally carries `_lobe`'s angular weight, favoring
    neighbors that lie behind the local print direction, and `denom` is replaced by
    `directed_denominator`'s per-element reference -- the two go together, since the
    angular factor is what stops a constant divisor from being the right one.

    :param stencil: the run's `AngularStencil`; required for `kappa > 0`
    :param kappa: angular lobe concentration for this iteration; `0` leaves the stencil
        purely radial, which is this function's behavior with no angular weight at all
    :param g0: the lobe's gradient scale
    :raises ValueError: if no element carrying density is left with a finite `K_est`,
        the whole part being within stencil reach of the print base; or if `kappa > 0`
        without the constant divisor the directional reference generalizes
    """
    x = xPhys.flatten()
    t = tPhys.flatten()
    if kappa != 0:
        if stencil is None or denom is None:
            raise ValueError(
                "an angular weight (kappa > 0) needs an AngularStencil and the HALF_STENCIL normalization, whose constant divisor it generalizes to a per-element directional reference; Normalization.NEIGHBORHOOD derives its own divisor and has no angular variant"
            )
        grad = timefield.central_difference_gradient_vector(tPhys)
        w = w * angular_pair_weights(grad, e1, stencil, kappa, g0)
        denom = directed_denominator(
            grad, stencil, kappa, g0, rouf, timefield.unit_length(tPhys)
        )
    K_est = _conductivity_core(x, t, e1, e2, w, q, rouf, denom, base).K_est
    if base is not None and not bool((torch.isfinite(K_est) & (x > 0)).any()):
        raise ValueError(
            "every element carrying density has infinite K_est, being within the conductivity stencil's reach of the print base, so the hotspot measure has nothing left to aggregate over; shrink rmin_cond or use a normalization that needs no print base (infinite_base)"
        )
    return K_est
