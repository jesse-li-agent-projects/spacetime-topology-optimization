"""Print-time overheating proxy: local estimated conductivity, and the hotspot
constraint bounding its worst-case value.

`estimated_conductivity` scores each element by how strongly its already-printed
(cooler, earlier-`tPhys`) neighborhood shields it from residual heat -- a proxy for
overheating risk during additive deposition. `hotspot_value` computes the p-norm of
`1 - K_est` (weighted toward already-dense, hot regions) that the hotspot constraint
bounds below a critical threshold `Tcr`, smoothly approximating a hard max via a
p-norm as `p -> inf`; the caller (`optimize.step`) applies the `factor`/`Tcr` scaling
and gets the sensitivity from autograd through this (Phase 3.4,
`plans/torch_port_part2.md`).

`tests/reference/conductivity.py` keeps `hotspot_constraint`, the hand-derived
predecessor that folds the `factor`/`Tcr` scaling and the density-filter chain rule
(`H`/`Hs`/`dx` from `filters.py`) directly into its sensitivities, as a cross-check
(`tests/test_reference_sweep.py`) and timing baseline
(`benchmarks/bench_sensitivities.py`). See `conventions.md` for array-order/tolerance
conventions.
"""

from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor


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
    r = int(np.ceil(rmin_cond)) - 1
    e = np.arange(nelx * nely)
    i1, j1 = e % nelx, e // nelx
    e1s, e2s, ws = [], [], []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            dist = np.hypot(di, dj)
            if rmin_cond - dist < 0:
                continue
            i2, j2 = i1 + di, j1 + dj
            valid = (i2 >= 0) & (i2 < nelx) & (j2 >= 0) & (j2 < nely)
            e1s.append(e[valid])
            e2s.append((j2 * nelx + i2)[valid])
            ws.append(np.full(valid.sum(), (rmin_cond - dist) / rmin_cond))
    return np.concatenate(e1s), np.concatenate(e2s), np.concatenate(ws)


def _pairwise_sigmoid_terms(
    t: Float[Tensor, " nel"],
    a: Int[Tensor, " npairs"],
    b: Int[Tensor, " npairs"],
    rouf: float,
) -> tuple[Float[Tensor, " npairs"], Float[Tensor, " npairs"]]:
    """
    `FT_el{a}[b]`/`DFT_el{a}[b]`: a neighbor-sigmoid weight and its t-derivative, for a
    COO pair array.

    Deviates from the MATLAB source in two ways -- see `conventions.md`'s "Known
    deviations" for why: `DFT` is zeroed only at `a == b` self-pairs rather than on any
    value tie, and both terms are evaluated through the overflow-safe `exp(-|z|)`
    instead of the source's literal (and, for large `rouf*dt`, NaN-producing) forms.

    :param t: per-element time field, `tPhys.flatten()`
    :param a: first index of each COO pair
    :param b: second index of each COO pair
    :param rouf: sigmoid sharpness
    :return: `(FT, DFT)`, one value per pair
    """
    ta, tb = t[a], t[b]
    z = rouf * (tb - ta)
    ez = torch.exp(-torch.abs(z))
    FT = torch.where(z >= 0, ez / (1.0 + ez), 1.0 / (1.0 + ez))
    # d/d(t[a]) of FT, which is even in z: exp(z)/(1+exp(z))^2 == exp(-|z|)/(1+exp(-|z|))^2.
    DFT = torch.where(a == b, torch.zeros_like(ez), rouf * ez / (1.0 + ez) ** 2)
    return FT, DFT


class _ConductivityCore(NamedTuple):
    K_est: Float[Tensor, " nel"]
    Nsum3: Float[Tensor, " nel"]
    FT_ab: Float[Tensor, " npairs"]
    DFT_ab: Float[Tensor, " npairs"]
    xb_q: Float[Tensor, " npairs"]


def _conductivity_core(
    x: Float[Tensor, " nel"],
    t: Float[Tensor, " nel"],
    e1: Int[Tensor, " npairs"],
    e2: Int[Tensor, " npairs"],
    w: Float[Tensor, " npairs"],
    q: float,
    rouf: float,
) -> _ConductivityCore:
    """`K_est`/`Nsum3` (its row-sum denominator), plus the `a->b` pair terms
    (`FT_ab`/`DFT_ab`/`xb_q`) shared by `estimated_conductivity` and
    `tests/reference/conductivity.py`'s `_conductivity_terms` -- computed once here
    rather than redone by each.
    """
    nel = x.shape[0]
    FT_ab, DFT_ab = _pairwise_sigmoid_terms(t, e1, e2, rouf)
    xb_q = x[e2] ** q
    Nsum3 = torch.zeros(nel, dtype=x.dtype, device=x.device)
    Nsum3.index_add_(0, e1, w * FT_ab)
    num = torch.zeros(nel, dtype=x.dtype, device=x.device)
    num.index_add_(0, e1, xb_q * w * FT_ab)
    K_est = num / Nsum3
    return _ConductivityCore(K_est, Nsum3, FT_ab, DFT_ab, xb_q)


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
    """
    nely, nelx = xPhys.shape
    nel = nely * nelx
    x = xPhys.flatten()
    t = tPhys.flatten()

    core = _conductivity_core(x, t, e1, e2, w, q, rouf)
    K_est = core.K_est
    T_val = 1 - K_est

    cond_p = T_val**p * x ** (r * p)
    sum_cond = torch.sum(cond_p)
    numer = _safe_pmean(sum_cond / nel, p)
    return numer, K_est


def estimated_conductivity(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    e1: Int[Tensor, " npairs"],
    e2: Int[Tensor, " npairs"],
    w: Float[Tensor, " npairs"],
    q: float,
    rouf: float,
) -> Float[Tensor, " nely*nelx"]:
    """Local estimated conductivity: a density/print-time-weighted average of how
    strongly each element's neighborhood has already solidified (cooler, earlier
    `tPhys`) around it, used as an overheating proxy by `hotspot_value`.
    """
    x = xPhys.flatten()
    t = tPhys.flatten()
    return _conductivity_core(x, t, e1, e2, w, q, rouf).K_est
