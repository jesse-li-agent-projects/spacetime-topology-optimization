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
    i1, j1 = e // nely, e % nely
    e1s, e2s, ws = [], [], []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            dist = np.hypot(di, dj)
            if rmin_cond - dist < 0:
                continue
            i2, j2 = i1 + di, j1 + dj
            valid = (i2 >= 0) & (i2 < nelx) & (j2 >= 0) & (j2 < nely)
            e1s.append(e[valid])
            e2s.append((i2 * nely + j2)[valid])
            ws.append(np.full(valid.sum(), (rmin_cond - dist) / rmin_cond))
    return np.concatenate(e1s), np.concatenate(e2s), np.concatenate(ws)


def _pairwise_sigmoid_terms(
    t: Float[np.ndarray, " nel"],
    a: Int[np.ndarray, " npairs"],
    b: Int[np.ndarray, " npairs"],
    rouf: float,
) -> tuple[Float[np.ndarray, " npairs"], Float[np.ndarray, " npairs"]]:
    """`FT_el{a}[b]`/`DFT_el{a}[b]` (a neighbor-sigmoid weight and its t-derivative) for a
    COO pair array. `DFT` is exactly 0 only at `a == b` self-pairs, where `FT(t[a], t[a])`
    is constant in `t[a]` so the true derivative really is 0; a genuine tie between two
    *distinct* elements (`a != b`, `t[a] == t[b]`) gets the ordinary `rouf/4` like any
    other point, since `FT` is smooth in `t[a]` there -- the source instead zeroed `DFT`
    on any value-tie regardless of `a == b`, a bug (not a deviation); see `conventions.md`.

    Both are evaluated through `exp(-|z|)`, which lies in `(0, 1]` for every `z` and so
    cannot overflow. The source's literal forms do: `(1+exp(z))^-1` saturates to 0 and
    `FT^2*rouf*exp(z)` becomes `0*inf = NaN` once `rouf*dt` exceeds ~709, silently
    poisoning `dt1`. The forms here are algebraically identical (they agree with the
    source to ~1e-16 wherever it is finite, tails included).
    """
    ta, tb = t[a], t[b]
    z = rouf * (tb - ta)
    ez = np.exp(-np.abs(z))
    FT = np.where(z >= 0, ez / (1.0 + ez), 1.0 / (1.0 + ez))
    # d/d(t[a]) of FT, which is even in z: exp(z)/(1+exp(z))^2 == exp(-|z|)/(1+exp(-|z|))^2.
    DFT = np.where(a == b, 0.0, rouf * ez / (1.0 + ez) ** 2)
    return FT, DFT


def _conductivity_terms(
    x: Float[np.ndarray, " nel"],
    t: Float[np.ndarray, " nel"],
    e1: Int[np.ndarray, " npairs"],
    e2: Int[np.ndarray, " npairs"],
    w: Float[np.ndarray, " npairs"],
    q: float,
    rouf: float,
) -> tuple[
    Float[np.ndarray, " nel"],
    Float[np.ndarray, " nel"],
    Float[np.ndarray, " npairs"],
    Float[np.ndarray, " npairs"],
    Float[np.ndarray, " nel"],
    Float[np.ndarray, " nel"],
]:
    """Shared per-element/per-pair terms for `estimated_conductivity` and
    `hotspot_constraint`: `K_est`, `Nsum3` (its row-sum denominator), the
    neighbor-role-swapped sigmoid pair terms `FT_ba`/`DFT_ba` (needed by the
    sensitivity's cross term), and `S1`/`S2` (needed by its self/diagonal term).

    Reuses `e1`'s weight for both pair directions (`w[e1,e2] == w[e2,e1]` by
    construction of `neighbor_weights` -- confirmed against the MATLAB `WE` fixture in
    this module's tests) rather than a second lookup.
    """
    nel = x.shape[0]
    FT_ab, DFT_ab = _pairwise_sigmoid_terms(t, e1, e2, rouf)
    FT_ba, DFT_ba = _pairwise_sigmoid_terms(t, e2, e1, rouf)

    xb_q = x[e2] ** q
    Nsum3 = np.zeros(nel)
    np.add.at(Nsum3, e1, w * FT_ab)
    num = np.zeros(nel)
    np.add.at(num, e1, xb_q * w * FT_ab)
    K_est = num / Nsum3

    S1 = np.zeros(nel)
    np.add.at(S1, e1, w * DFT_ab)
    S2 = np.zeros(nel)
    np.add.at(S2, e1, xb_q * w * DFT_ab)

    return K_est, Nsum3, FT_ba, DFT_ba, S1, S2


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
    x = xPhys.flatten(order="F")
    t = tPhys.flatten(order="F")
    K_est, *_ = _conductivity_terms(x, t, e1, e2, w, q, rouf)
    return K_est


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
) -> tuple[float, Float[np.ndarray, " nely*nelx"], Float[np.ndarray, " nely*nelx"]]:
    """Hotspot constraint: a p-norm of `1 - K_est` (weighted toward hot, dense regions)
    stays below `Tcr`, smoothly bounding the worst-case local overheating risk.

    `factor` is a periodically-refreshed rescaling constant the main optimization loop
    owns as persistent state (see `conventions.md`'s "Known traps") -- pass it through,
    never recompute it here. To refresh it (MATLAB's `rem(loop,25)==0` guard): this
    call's own `numer = (fval + 1) * Tcr / factor` is recoverable from `fval` alone
    (no extra return value needed, since `fval` is an affine function of `numer`); the
    other ingredient, `max((1 - K_est) * xPhys.flatten('F')**r)`, needs a separate
    `estimated_conductivity(...)` call (same `e1`/`e2`/`w`/`q`/`rouf`) to get `K_est` --
    cheap to duplicate since it only runs once per 25 iterations. Sensitivity algebra
    is hand-derived from the MATLAB source's per-element neighbor loop; see this
    module's tests for the derivation.
    """
    nely, nelx = xPhys.shape
    nel = nely * nelx
    x = xPhys.flatten(order="F")
    t = tPhys.flatten(order="F")

    K_est, Nsum3, FT_ba, DFT_ba, S1, S2 = _conductivity_terms(x, t, e1, e2, w, q, rouf)
    T_val = 1 - K_est

    cond_p = (T_val * x**r) ** p
    sum_cond = np.sum(cond_p)
    numer = (sum_cond / nel) ** (1 / p)
    fval = float(factor * numer / Tcr - 1)

    xa, xb = x[e1], x[e2]
    Ka, Kb = K_est[e1], K_est[e2]
    Na, Nb = Nsum3[e1], Nsum3[e2]
    S1a, S2a = S1[e1], S2[e1]
    diag = e1 == e2

    # N_sub2's cross term is shared by both branches (at a == b, xb/Nb reduce to
    # xa/Na, and FT_ba/w reduce to their diagonal identities 0.5/1 -- no separate
    # diagonal constant needed); only the self-heating correction is diagonal-only.
    N_sub2 = -(xb**r) * q * xa ** (q - 1) * FT_ba * w / Nb
    N_sub2 = N_sub2 + np.where(diag, (1 - Ka) * r * xa ** (r - 1), 0.0)
    N_sub1 = np.where(
        diag,
        -(xa**r) * (S2a / Na - Ka * S1a / Na),
        -(w / Nb) * (Kb - xa**q) * xb**r * DFT_ba,
    )

    Tsub_pow = (T_val[e2] * xb**r) ** (p - 1)
    cond_arr1 = np.zeros(nel)
    cond_arr2 = np.zeros(nel)
    np.add.at(cond_arr1, e1, Tsub_pow * N_sub1)
    np.add.at(cond_arr2, e1, Tsub_pow * N_sub2)

    scale = factor * (sum_cond / nel) ** (1 / p - 1) / (nel * Tcr)
    df1 = H @ ((scale * cond_arr2) * dx.flatten(order="F") / Hs)
    dt1 = H @ ((scale * cond_arr1) / Hs)
    return fval, df1, dt1
