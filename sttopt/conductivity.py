"""Print-time overheating proxy: local estimated conductivity, and the hotspot
constraint bounding its worst-case value.

`estimated_conductivity` scores each element by how strongly its already-printed
(cooler, earlier-`tPhys`) neighborhood shields it from residual heat -- a proxy for
overheating risk during additive deposition. `hotspot_value` aggregates `1 - K_est`
(weighted toward already-dense, hot regions) into the smooth maximum that the hotspot
constraint bounds below a critical threshold `Tcr`; the caller (`stto.step`) applies
the `factor`/`Tcr` scaling and gets the sensitivity from autograd through this
(Phase 3.4, `plans/torch_port_part2.md`).

Two orthogonal choices shape that number, both selectable per run: `Normalization`
sets what `K_est` measures shielding against, and `Aggregation` sets how the field
collapses to one scalar. They pair -- HALF_STENCIL admits `K_est > 1`, which only
LOGSUMEXP accepts unconditionally -- but neither implies the other.

`tests/reference/conductivity.py` keeps `hotspot_constraint`, the hand-derived
predecessor that folds the `factor`/`Tcr` scaling and the density-filter chain rule
(`H`/`Hs`/`dx` from `filters.py`) directly into its sensitivities, as a cross-check
(`tests/test_reference_sweep.py`) and timing baseline
(`benchmarks/bench_sensitivities.py`). See `conventions.md` for array-order/tolerance
conventions.

Two deliberate deviations from Das2025 Eq. (6), both load-bearing for this port and
neither a bug: the paper's neighbor weight is radial *times angular*, the angular part
favoring the build direction, but `neighbor_weights` below keeps only the radial
factor -- the build direction is not fixed once deposition order is itself a design
variable (`plans/archive/fixed_geometry_sequence_optimization.md`), so this is a known
future direction rather than an oversight. And the paper's Eq. (6) numerator weights a
neighbor by its raw density `rho_j`, where `_conductivity_core` uses `x_j**q` with
`q = 3` -- a SIMP-style penalization of intermediate density the paper does not have.
"""

from enum import StrEnum
from functools import cache
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Normalization(StrEnum):
    """Names for what `K_est` measures shielding *against*, selectable per run.

    NEIGHBORHOOD divides by the weight of the caller's own already-printed neighbors,
    so `K_est` is a weighted average of neighbor density and is bounded by 1. Its
    denominator shrinks wherever the stencil leaves the mesh, which renormalizes the
    missing region away: an element on the domain edge whose in-mesh neighbors are all
    solid scores `K_est == 1` exactly, while an identical element beside an internal
    void does not. That makes free surfaces on the mesh boundary invisible, and leaves
    the print time of void -- a quantity nothing physical pins -- driving the result
    through the denominator (PR #96).

    HALF_STENCIL divides by `half_stencil_weight` instead: the shielding an infinite
    uniform layer schedule would supply at the same point in its own schedule. The
    denominator is then the same constant everywhere, so leaving the mesh costs
    exactly what an equal-sized internal void costs, and void print time drops out
    entirely (it only ever multiplies a zero density). `K_est` is no longer an average
    and may exceed 1, meaning "better shielded than an infinite uniform layer" -- see
    `Aggregation` for what tolerates that.
    """

    NEIGHBORHOOD = "neighborhood"
    HALF_STENCIL = "half_stencil"


class Aggregation(StrEnum):
    """Names for how the per-element severity field collapses to the one scalar the
    hotspot term reports, selectable per run.

    P_MEAN takes `mean(T**p * x**(r*p)) ** (1/p)`, a scale-equivariant smooth maximum.
    It is built on powers, so it needs a non-negative field, and it approaches the
    true maximum from below by a factor `nel**(-1/p)` that worsens as the mesh grows
    -- 0.687 at `p=25` on a 120x100 mesh, a quarter low. Its density weight is
    `x**(r*p)`, so `p` cannot be changed without also changing how hard void is
    suppressed, and `r*p <= 1` silently makes the gradient diverge at `x == 0`.

    LOGSUMEXP takes `log(sum(x**s * exp(beta*T))) / beta`. It is translation
    equivariant rather than scale equivariant, which is the property that fits once
    `Normalization.HALF_STENCIL` has pinned the scale against a physical reference:
    `exp` is defined on all of R, so `K_est > 1` needs no clamp. Its bias is additive
    and bounded by `log(nel_weighted)/beta` rather than multiplicative -- +0.001 at
    `beta=200` where P_MEAN at `p=25` is -26%. Sharpness (`beta`) and density
    suppression (`s`) are independent knobs.
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
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
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


def base_exempt_mask(
    nelx: int, nely: int, base: Int[np.ndarray, " k"], rmin_cond: float
) -> Bool[np.ndarray, " nel"]:
    """Elements whose conductivity stencil reaches the print base, and which are
    therefore excluded from the hotspot measure entirely.

    The print base is a heat sink, not part of the design: the material it stands in
    for is always already there, whatever the deposition order. `Normalization.
    HALF_STENCIL` has no way to say that -- its denominator assumes a full
    half-neighborhood of prior material everywhere, so the first-deposited layer,
    which by definition has nothing before it, scores as the worst hotspot in the
    domain. Modelling the base as infinitely dense makes `K_est` diverge for anything
    that can see it, which is what this mask is: the same statement, without an
    infinity to represent.

    Coarse on purpose. It exempts a band `rmin_cond` deep rather than shading the base
    into the field, so a part small enough to sit entirely within `rmin_cond` of its
    base has no hotspot measure left -- `seqopt.build_problem`/`stto.build_problem`
    reject that rather than silently aggregating nothing.

    :param base: 0-indexed print-start elements, e.g. `geometry.base_elements`
    """
    is_base = np.zeros((nely, nelx), dtype=bool)
    is_base.flat[base] = True
    reaches = np.zeros((nely, nelx), dtype=bool)
    # `reaches[j, i] |= is_base[j + dj, i + di]`, over the offsets that stay in-mesh.
    # A stencil wider than the mesh leaves offsets with no in-mesh destination at all.
    for di, dj, _ in _stencil_offsets(rmin_cond):
        j0, j1 = max(0, -dj), min(nely, nely - dj)
        i0, i1 = max(0, -di), min(nelx, nelx - di)
        if j0 >= j1 or i0 >= i1:
            continue
        reaches[j0:j1, i0:i1] |= is_base[j0 + dj : j1 + dj, i0 + di : i1 + di]
    return reaches.flatten()


def base_exemption(
    nelx: int,
    nely: int,
    base: Int[np.ndarray, " k"],
    rmin_cond: float,
    normalization: Normalization,
    solid: Bool[np.ndarray, " nel"] | None = None,
) -> Bool[np.ndarray, " nel"] | None:
    """`base_exempt_mask` for the normalizations that need one, else `None`.

    NEIGHBORHOOD needs none: its denominator shrinks along with its numerator at the
    first deposited layer, so an element there already scores near zero severity
    without being told the base exists. Only a constant denominator has to be told.

    :param solid: which elements hold material, where that is fixed up front. Pass it
        to have the check below see the part rather than the bounding box; omit it
        where density is itself a design variable and no such answer exists yet.
    :raises ValueError: if the exemption leaves the hotspot measure nothing to
        aggregate over, every candidate element being within one conductivity radius
        of the print base
    """
    if normalization == Normalization.NEIGHBORHOOD:
        return None
    exempt = base_exempt_mask(nelx, nely, base, rmin_cond)
    considered = ~exempt if solid is None else (~exempt & solid)
    if not considered.any():
        raise ValueError(
            f"the print-base exemption at rmin_cond={rmin_cond} covers every element the hotspot measure would aggregate over, leaving it nothing to report; shrink rmin_cond or use hotspot_normalization='neighborhood'"
        )
    return exempt


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
    denom: float | None = None,
) -> _ConductivityCore:
    """`K_est` and the divisor it was formed with, plus the `a->b` pair terms
    (`FT_ab`/`xb_q`) shared by `estimated_conductivity` and
    `tests/reference/conductivity.py`'s `_conductivity_terms` -- computed once here
    rather than redone by each.

    :param denom: a constant divisor for every element (`constant_denominator`), or
        `None` for the per-element already-printed neighbor weight. Only the `None`
        case makes `denom` indexable per element.
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
    return _ConductivityCore(num / divisor, divisor, FT_ab, xb_q)


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


def _weighted_logsumexp(
    T_val: Float[Tensor, " nel"], weight: Float[Tensor, " nel"], beta: float
) -> Float[Tensor, ""]:
    """`log(sum(weight * exp(beta * T_val))) / beta`, shifted by `max(T_val)` so the
    exponential cannot overflow at the large `beta` a sharp maximum needs.

    Shifting by a detached maximum is exact, not an approximation: the shift cancels
    between the sum and the added-back constant, and the gradient it produces,
    `weight * softmax(beta * T_val)`, is the same either way.

    :param weight: per-element weight, zero wherever an element takes no part
    :param beta: sharpness; the result exceeds `max(T_val)` over the weighted
        elements by at most `log(sum(weight))/beta`
    """
    shift = T_val.max().detach()
    total = torch.sum(weight * torch.exp(beta * (T_val - shift)))
    return shift + torch.log(total) / beta


def hotspot_value(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    e1: Int[Tensor, " npairs"],
    e2: Int[Tensor, " npairs"],
    w: Float[Tensor, " npairs"],
    p: float,
    q: float,
    r: float,
    rouf: float,
    denom: float | None = None,
    exempt: Bool[Tensor, " nel"] | None = None,
    aggregation: Aggregation = Aggregation.P_MEAN,
    beta: float = 200.0,
    density_exponent: float | None = None,
) -> tuple[Float[Tensor, ""], Float[Tensor, " nely*nelx"]]:
    """`hotspot_constraint`'s value alone (`numer`, `K_est`), differentiable end to end
    w.r.t. `xPhys`/`tPhys` (autograd sensitivity path, `plans/torch_port_part2.md`
    Phase 3.4 -- see `hotspot_constraint` for the hand-derived predecessor
    `bench_sensitivities.py` times this against).

    Written in the NaN-safe form the plan's Risks section requires: `cond_p =
    (T_val * x**r) ** p` differentiates to `inf` at `x == 0` (density does reach exact
    zero once the Heaviside projection saturates), so this computes the algebraically
    identical `T_val**p * x**(r*p)` instead, whose gradient is finite because
    `r*p > 1` at production settings (`r=0.05, p=25`). Also guards `_safe_pmean`'s
    input against the (rarer) fully-solid-part singularity. Callers needing the caller
    owned `factor`/`Tcr` scaling and the constraint value build them from `numer`
    directly, as `hotspot_constraint` does.

    :param denom: `constant_denominator` for the run's `Normalization`. Note that
        `T_val` can go negative under a constant denominator, which the `**p` here
        only tolerates while `p` is integral -- `Aggregation.LOGSUMEXP` is the pairing
        that does not care.
    :param exempt: elements to drop from the aggregate entirely, `base_exempt_mask`.
        Fixed by the mesh and the print base, so it is a constant weight here rather
        than anything autograd has to see through.
    :param aggregation: which smooth maximum collapses the severity field.
    :param beta: LOGSUMEXP sharpness; unused by P_MEAN, which takes `p` instead.
    :param density_exponent: LOGSUMEXP's void-suppression exponent, defaulting to
        P_MEAN's implicit `r * p`. Must exceed 1, or the weight's own gradient
        diverges at `x == 0`. Unused by P_MEAN, whose exponent is not separable.
    """
    nely, nelx = xPhys.shape
    nel = nely * nelx
    x = xPhys.flatten()
    t = tPhys.flatten()

    core = _conductivity_core(x, t, e1, e2, w, q, rouf, denom)
    K_est = core.K_est
    T_val = 1 - K_est

    if aggregation == Aggregation.P_MEAN:
        cond_p = T_val**p * x ** (r * p)
        if exempt is not None:
            cond_p = torch.where(exempt, torch.zeros_like(cond_p), cond_p)
        numer = _safe_pmean(torch.sum(cond_p) / nel, p)
    elif aggregation == Aggregation.LOGSUMEXP:
        s = r * p if density_exponent is None else density_exponent
        weight = x**s
        if exempt is not None:
            weight = torch.where(exempt, torch.zeros_like(weight), weight)
        numer = _weighted_logsumexp(T_val, weight, beta)
    else:
        raise ValueError(
            f"aggregation must be an Aggregation member, got {aggregation!r}"
        )
    return numer, K_est


def estimated_conductivity(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    e1: Int[Tensor, " npairs"],
    e2: Int[Tensor, " npairs"],
    w: Float[Tensor, " npairs"],
    q: float,
    rouf: float,
    denom: float | None = None,
) -> Float[Tensor, " nely*nelx"]:
    """Local estimated conductivity: how strongly each element's neighborhood has
    already solidified (cooler, earlier `tPhys`) around it, used as an overheating
    proxy by `hotspot_value`. `Normalization` sets what it is measured relative to,
    via `denom`.
    """
    x = xPhys.flatten()
    t = tPhys.flatten()
    return _conductivity_core(x, t, e1, e2, w, q, rouf, denom).K_est
