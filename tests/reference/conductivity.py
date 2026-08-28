"""Hand-derived predecessor of `sttopt.conductivity.hotspot_value`'s sensitivity formula.

`sttopt.conductivity.hotspot_value` computes `numer`'s (and hence the hotspot
constraint's) sensitivity via autograd (Phase 3.4, `plans/torch_port_part2.md`).
`hotspot_constraint` here is the hand-derived predecessor Phase 3.2 ported from the
MATLAB source's inline per-element neighbor loop, kept only as a cross-check
(`tests/test_reference_sweep.py`) and a timing baseline
(`benchmarks/bench_sensitivities.py`). Nothing in `sttopt/` calls this module.

Unlike `hotspot_value`, this also folds in the caller-owned `factor`/`Tcr` scaling and
the density-filter chain rule (`H`/`Hs`/`dx`) directly into its returned sensitivities,
matching the MATLAB source's inline block.
"""

from typing import NamedTuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

import sttopt.conductivity as conductivity


class _ConductivityTerms(NamedTuple):
    K_est: Float[Tensor, " nel"]
    Nsum3: Float[Tensor, " nel"]
    FT_ba: Float[Tensor, " npairs"]
    DFT_ba: Float[Tensor, " npairs"]
    S1: Float[Tensor, " nel"]
    S2: Float[Tensor, " nel"]


def _conductivity_terms(
    x: Float[Tensor, " nel"],
    t: Float[Tensor, " nel"],
    e1: Int[Tensor, " npairs"],
    e2: Int[Tensor, " npairs"],
    w: Float[Tensor, " npairs"],
    q: float,
    rouf: float,
) -> _ConductivityTerms:
    """`conductivity._conductivity_core`'s `K_est`/`Nsum3`, plus the sensitivity-only
    terms `hotspot_constraint` needs: the neighbor-role-swapped sigmoid pair terms
    `FT_ba`/`DFT_ba` (its cross term) and `S1`/`S2` (its self/diagonal term).

    Reuses `e1`'s weight for both pair directions (`w[e1,e2] == w[e2,e1]` by
    construction of `neighbor_weights` -- confirmed against the MATLAB `WE` fixture in
    `tests/test_conductivity.py`) rather than a second lookup.
    """
    nel = x.shape[0]
    core = conductivity._conductivity_core(x, t, e1, e2, w, q, rouf)
    FT_ba, DFT_ba = conductivity._pairwise_sigmoid_terms(t, e2, e1, rouf)

    S1 = torch.zeros(nel, dtype=x.dtype, device=x.device)
    S1.index_add_(0, e1, w * core.DFT_ab)
    S2 = torch.zeros(nel, dtype=x.dtype, device=x.device)
    S2.index_add_(0, e1, core.xb_q * w * core.DFT_ab)

    return _ConductivityTerms(core.K_est, core.Nsum3, FT_ba, DFT_ba, S1, S2)


class HotspotConstraintResult(NamedTuple):
    fval: float
    df1: Float[Tensor, " nely*nelx"]
    dt1: Float[Tensor, " nely*nelx"]
    # factor-independent
    numer: float
    # factor-independent
    K_est: Float[Tensor, " nely*nelx"]


def hotspot_constraint(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    e1: Int[Tensor, " npairs"],
    e2: Int[Tensor, " npairs"],
    w: Float[Tensor, " npairs"],
    dx: Float[Tensor, "nely nelx"],
    H: Tensor,
    Hs: Float[Tensor, " nely*nelx"],
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
    MATLAB source's per-element neighbor loop; see `tests/test_conductivity.py` for the
    derivation.
    """
    nely, nelx = xPhys.shape
    nel = nely * nelx
    x = xPhys.flatten()
    t = tPhys.flatten()

    terms = _conductivity_terms(x, t, e1, e2, w, q, rouf)
    K_est = terms.K_est
    T_val = 1 - K_est

    cond_p = (T_val * x**r) ** p
    sum_cond = torch.sum(cond_p)
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
    N_sub1 = torch.where(
        diag,
        -(xa**r) * (S2a / Na - Ka * S1a / Na),
        -(w / Nb) * (Kb - xa**q) * xb**r * terms.DFT_ba,
    )

    Tsub_pow = (T_val[e2] * xb**r) ** (p - 1)
    cond_arr1 = torch.zeros(nel, dtype=x.dtype, device=x.device)
    cond_arr2 = torch.zeros(nel, dtype=x.dtype, device=x.device)
    cond_arr1.index_add_(0, e1, Tsub_pow * N_sub1)
    cond_arr2.index_add_(0, e1, Tsub_pow * N_sub2)

    # Diagonal self-heating correction to cond_arr2, kept out of the Tsub_pow * N_sub2
    # product above: on the diagonal that product is exactly r * T_val**p * x**(r*p-1),
    # one power of x rather than an x**(r-1) that diverges times a (x**r)**(p-1) that
    # vanishes -- whose inf * 0 is nan at x == 0, which the Heaviside projection reaches
    # routinely once beta_d saturates. Every element is its own neighbour exactly once,
    # so this is a per-element term needing no pair expansion.
    if r * p < 1 and bool(torch.any(x == 0)):
        raise ValueError(
            f"hotspot_constraint: the self-heating term scales as x**(r*p - 1) with r*p - 1 = {r * p - 1} < 0, so it diverges at the exactly-zero element densities present here."
        )
    cond_arr2 = cond_arr2 + r * T_val**p * x ** (r * p - 1)

    # sum_cond == 0 (e.g. a fully solid part, T_val == 0 everywhere) makes the exponent
    # 1/p - 1 < 0 diverge; cond_arr1/cond_arr2 vanish exactly in that case too (every term
    # carries the same T_val**(p-1) factor), so the true limit of scale*cond_arr is 0.
    scale = (
        0.0
        if float(sum_cond) == 0
        else factor * (sum_cond / nel) ** (1 / p - 1) / (nel * Tcr)
    )
    df1 = H @ ((scale * cond_arr2) * dx.flatten() / Hs)
    dt1 = H @ ((scale * cond_arr1) / Hs)
    return HotspotConstraintResult(fval, df1, dt1, float(numer), K_est)
