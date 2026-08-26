"""Print-time overheating proxy: local estimated conductivity, and the hotspot
constraint bounding its worst-case value.

`estimated_conductivity` scores each element by how strongly its already-printed
(cooler, earlier-`tPhys`) neighborhood shields it from residual heat -- a proxy for
overheating risk during additive deposition. `hotspot_constraint` bounds the p-norm
of `1 - K_est` (weighted toward already-dense, hot regions) below a critical
threshold `Tcr`, smoothly approximating a hard max via a p-norm as `p -> inf`.

Like `constraints.py`, this ports an *inline* main-loop block (not a standalone
MATLAB function), so `hotspot_constraint` bakes the density-filter chain rule
(`H`/`Hs`/`dx` from `filters.py`) into its returned sensitivities directly. See
`conventions.md` for array-order/tolerance conventions and this module's tests for
the (nontrivial, hand-derived) sensitivity algebra.
"""

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from jaxtyping import Float, Int


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
    t: Float[np.ndarray, " nel"],
    a: Int[np.ndarray, " npairs"],
    b: Int[np.ndarray, " npairs"],
    rouf: float,
) -> tuple[Float[np.ndarray, " npairs"], Float[np.ndarray, " npairs"]]:
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
    ez = np.exp(-np.abs(z))
    FT = np.where(z >= 0, ez / (1.0 + ez), 1.0 / (1.0 + ez))
    # d/d(t[a]) of FT, which is even in z: exp(z)/(1+exp(z))^2 == exp(-|z|)/(1+exp(-|z|))^2.
    DFT = np.where(a == b, 0.0, rouf * ez / (1.0 + ez) ** 2)
    return FT, DFT


class _ConductivityCore(NamedTuple):
    K_est: Float[np.ndarray, " nel"]
    Nsum3: Float[np.ndarray, " nel"]
    FT_ab: Float[np.ndarray, " npairs"]
    DFT_ab: Float[np.ndarray, " npairs"]
    xb_q: Float[np.ndarray, " npairs"]


class _ConductivityTerms(NamedTuple):
    K_est: Float[np.ndarray, " nel"]
    Nsum3: Float[np.ndarray, " nel"]
    FT_ba: Float[np.ndarray, " npairs"]
    DFT_ba: Float[np.ndarray, " npairs"]
    S1: Float[np.ndarray, " nel"]
    S2: Float[np.ndarray, " nel"]


def _conductivity_core(
    x: Float[np.ndarray, " nel"],
    t: Float[np.ndarray, " nel"],
    e1: Int[np.ndarray, " npairs"],
    e2: Int[np.ndarray, " npairs"],
    w: Float[np.ndarray, " npairs"],
    q: float,
    rouf: float,
) -> _ConductivityCore:
    """`K_est`/`Nsum3` (its row-sum denominator), plus the `a->b` pair terms
    (`FT_ab`/`DFT_ab`/`xb_q`) shared by both `estimated_conductivity` and
    `_conductivity_terms` -- computed once here rather than redone by each.
    """
    nel = x.shape[0]
    FT_ab, DFT_ab = _pairwise_sigmoid_terms(t, e1, e2, rouf)
    xb_q = x[e2] ** q
    Nsum3 = np.zeros(nel)
    np.add.at(Nsum3, e1, w * FT_ab)
    num = np.zeros(nel)
    np.add.at(num, e1, xb_q * w * FT_ab)
    K_est = num / Nsum3
    return _ConductivityCore(K_est, Nsum3, FT_ab, DFT_ab, xb_q)


def _conductivity_terms(
    x: Float[np.ndarray, " nel"],
    t: Float[np.ndarray, " nel"],
    e1: Int[np.ndarray, " npairs"],
    e2: Int[np.ndarray, " npairs"],
    w: Float[np.ndarray, " npairs"],
    q: float,
    rouf: float,
) -> _ConductivityTerms:
    """`_conductivity_core`'s `K_est`/`Nsum3`, plus the sensitivity-only terms
    `hotspot_constraint` needs: the neighbor-role-swapped sigmoid pair terms
    `FT_ba`/`DFT_ba` (its cross term) and `S1`/`S2` (its self/diagonal term).

    Reuses `e1`'s weight for both pair directions (`w[e1,e2] == w[e2,e1]` by
    construction of `neighbor_weights` -- confirmed against the MATLAB `WE` fixture in
    this module's tests) rather than a second lookup.
    """
    nel = x.shape[0]
    core = _conductivity_core(x, t, e1, e2, w, q, rouf)
    FT_ba, DFT_ba = _pairwise_sigmoid_terms(t, e2, e1, rouf)

    S1 = np.zeros(nel)
    np.add.at(S1, e1, w * core.DFT_ab)
    S2 = np.zeros(nel)
    np.add.at(S2, e1, core.xb_q * w * core.DFT_ab)

    return _ConductivityTerms(core.K_est, core.Nsum3, FT_ba, DFT_ba, S1, S2)


def estimated_conductivity(
    xPhys: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    e1: Int[np.ndarray, " npairs"],
    e2: Int[np.ndarray, " npairs"],
    w: Float[np.ndarray, " npairs"],
    q: float,
    rouf: float,
) -> Float[np.ndarray, " nely*nelx"]:
    """Local estimated conductivity: a density/print-time-weighted average of how
    strongly each element's neighborhood has already solidified (cooler, earlier
    `tPhys`) around it, used as an overheating proxy by `hotspot_constraint`.
    """
    x = xPhys.flatten()
    t = tPhys.flatten()
    return _conductivity_core(x, t, e1, e2, w, q, rouf).K_est


class HotspotConstraintResult(NamedTuple):
    fval: float
    df1: Float[np.ndarray, " nely*nelx"]
    dt1: Float[np.ndarray, " nely*nelx"]
    # factor-independent
    numer: float
    # factor-independent
    K_est: Float[np.ndarray, " nely*nelx"]


def hotspot_constraint(
    xPhys: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    e1: Int[np.ndarray, " npairs"],
    e2: Int[np.ndarray, " npairs"],
    w: Float[np.ndarray, " npairs"],
    dx: Float[np.ndarray, "nely nelx"],
    H: sp.spmatrix | sp.sparray,
    Hs: Float[np.ndarray, " nely*nelx"],
    factor: float,
    Tcr: float,
    p: float,
    q: float,
    r: float,
    rouf: float,
) -> HotspotConstraintResult:
    """Hotspot constraint: a p-norm of `1 - K_est` (weighted toward hot, dense regions)
    stays below `Tcr`, smoothly bounding the worst-case local overheating risk.

    `factor` is a periodically-refreshed rescaling constant the main optimization loop
    owns as persistent state (`optimize.State.factor`) -- pass it through, never
    recompute it here. `numer`/`K_est` are returned for the caller's periodic
    refresh (MATLAB's `rem(loop,25)==0` guard), which needs both but must not perturb
    this call's own `factor`-scaled result. Sensitivity algebra is hand-derived from the
    MATLAB source's per-element neighbor loop; see this module's tests for the derivation.
    """
    nely, nelx = xPhys.shape
    nel = nely * nelx
    x = xPhys.flatten()
    t = tPhys.flatten()

    terms = _conductivity_terms(x, t, e1, e2, w, q, rouf)
    K_est = terms.K_est
    T_val = 1 - K_est

    cond_p = (T_val * x**r) ** p
    sum_cond = np.sum(cond_p)
    numer = (sum_cond / nel) ** (1 / p)
    fval = float(factor * numer / Tcr - 1)

    xa, xb = x[e1], x[e2]
    Ka, Kb = K_est[e1], K_est[e2]
    Na, Nb = terms.Nsum3[e1], terms.Nsum3[e2]
    S1a, S2a = terms.S1[e1], terms.S2[e1]
    diag = e1 == e2

    # Cross term: shared by both branches (at a == b, xb/Nb reduce to xa/Na, and
    # FT_ba/w reduce to their diagonal identities 0.5/1 -- no separate diagonal
    # constant needed) and finite everywhere, since it has no negative powers of x.
    N_sub2 = -(xb**r) * q * xa ** (q - 1) * terms.FT_ba * w / Nb
    N_sub1 = np.where(
        diag,
        -(xa**r) * (S2a / Na - Ka * S1a / Na),
        -(w / Nb) * (Kb - xa**q) * xb**r * terms.DFT_ba,
    )

    Tsub_pow = (T_val[e2] * xb**r) ** (p - 1)
    cond_arr1 = np.zeros(nel)
    cond_arr2 = np.zeros(nel)
    np.add.at(cond_arr1, e1, Tsub_pow * N_sub1)
    np.add.at(cond_arr2, e1, Tsub_pow * N_sub2)

    # Diagonal self-heating correction to cond_arr2, kept out of the Tsub_pow * N_sub2
    # product above: on the diagonal that product is exactly r * T_val**p * x**(r*p-1),
    # one power of x rather than an x**(r-1) that diverges times a (x**r)**(p-1) that
    # vanishes -- whose inf * 0 is nan at x == 0, which the Heaviside projection reaches
    # routinely once beta_d saturates. Every element is its own neighbour exactly once,
    # so this is a per-element term needing no pair expansion.
    if r * p < 1 and np.any(x == 0):
        raise ValueError(
            f"hotspot_constraint: the self-heating term scales as x**(r*p - 1) with r*p - 1 = {r * p - 1} < 0, so it diverges at the exactly-zero element densities present here."
        )
    cond_arr2 += r * T_val**p * x ** (r * p - 1)

    # sum_cond == 0 (e.g. a fully solid part, T_val == 0 everywhere) makes the exponent
    # 1/p - 1 < 0 diverge; cond_arr1/cond_arr2 vanish exactly in that case too (every term
    # carries the same T_val**(p-1) factor), so the true limit of scale*cond_arr is 0.
    scale = (
        0.0 if sum_cond == 0 else factor * (sum_cond / nel) ** (1 / p - 1) / (nel * Tcr)
    )
    df1 = H @ ((scale * cond_arr2) * dx.flatten() / Hs)
    dt1 = H @ ((scale * cond_arr1) / Hs)
    return HotspotConstraintResult(fval, df1, dt1, float(numer), K_est)
