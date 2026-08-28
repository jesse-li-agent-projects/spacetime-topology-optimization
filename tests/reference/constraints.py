"""Hand-derived predecessors of `sttopt.constraints`'s constraint/sensitivity formulas.

`sttopt.constraints` computes sensitivities via autograd (Phase 3.4,
`plans/torch_port_part2.md`), through the caller's own filter/Heaviside chain rather
than baking `H`/`Hs`/`dx` into each returned row. The formulas here are the
hand-derived predecessors Phase 3.2 ported from the MATLAB source's inline main-loop
blocks, kept only as a cross-check (`tests/test_reference_sweep.py`). Nothing in
`sttopt/` calls this module.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor

import sttopt.compliance as compliance
import tests.reference.compliance as compliance_ref


def global_volume_fraction(
    xPhys: Float[Tensor, "nely nelx"],
    dx: Float[Tensor, "nely nelx"],
    H: Tensor,
    Hs: Float[Tensor, " nely*nelx"],
    volfrac: float,
) -> tuple[float, Float[Tensor, " nely*nelx"], Float[Tensor, " nely*nelx"]]:
    """Global printable-volume-fraction constraint: total deposited material vs. `volfrac`.

    `dft` is identically zero -- this constraint has no time-field dependence.
    """
    nely, nelx = xPhys.shape
    scale = nelx * nely * volfrac
    fval = float(torch.sum(xPhys) / scale - 1)
    dv = torch.ones((nely, nelx), dtype=xPhys.dtype, device=xPhys.device)
    dfx = H @ (dv.flatten() * dx.flatten() / Hs) / scale
    dft = torch.zeros(nelx * nely, dtype=xPhys.dtype, device=xPhys.device)
    return fval, dfx, dft


def time_field_continuity(
    tPhys: Float[Tensor, "nely nelx"],
    L: Tensor,
    H: Tensor,
    Hs: Float[Tensor, " nely*nelx"],
) -> tuple[float, Float[Tensor, " nely*nelx"], Float[Tensor, " nely*nelx"]]:
    """Time-field smoothness constraint: keeps each element's print time close to its local
    neighborhood average (`filters.continuity_filter`'s `L`), so the deposition sequence
    sweeps coherently across the mesh instead of jumping between distant elements.

    `dfx` is identically zero -- this constraint has no density dependence.
    """
    nely, nelx = tPhys.shape
    nel = nely * nelx
    # smoothness_weight = 2*nel is an overall tuning multiplier on the whole constraint
    # (MATLAB source's `kk`, commented "controlling the smoothness of the time field" --
    # not a paper symbol); it shows up in fval itself, so it isn't a derivative artifact.
    # The separate explicit `* 2` in dft below IS a derivative factor, from
    # d(deviation**2)/dt = 2*deviation * d(deviation)/dt.
    smoothness_weight = 2 * nel
    deviation = L @ tPhys.flatten()
    fval = float(smoothness_weight * (torch.sum(deviation**2 / nel) - 1.0e-6))
    dft = H @ ((smoothness_weight * 2 * (L.t() @ deviation)) / Hs) / nel
    dfx = torch.zeros(nel, dtype=tPhys.dtype, device=tPhys.device)
    return fval, dfx, dft


def start_point(
    tPhys: Float[Tensor, "nely nelx"],
    Nei: Int[Tensor, " k"],
    H: Tensor,
    Hs: Float[Tensor, " nely*nelx"],
) -> tuple[
    Float[Tensor, " k"],
    Float[Tensor, "k nely*nelx"],
    Float[Tensor, "k nely*nelx"],
]:
    """Print-start constraint(s): the deposition-origin element(s) `Nei` (0-indexed element
    numbers, per `conventions.md`) must start printing at t=0 (up to machine precision).

    Example: `Nei` is `[0]` for the single-origin time field (`tfield==1`) -- the elements nearest
    the print-start origin. `dfx` is identically zero (no density dependence).
    """
    nely, nelx = tPhys.shape
    nel = nely * nelx
    k = len(Nei)
    fval = tPhys.flatten()[Nei] - 1.0e-9
    ss = torch.zeros((nel, k), dtype=tPhys.dtype, device=tPhys.device)
    ss[Nei, torch.arange(k, device=tPhys.device)] = 1.0  # one-hot selector
    dft = (H @ (ss / Hs[:, None])).t()
    dfx = torch.zeros((k, nel), dtype=tPhys.dtype, device=tPhys.device)
    return fval, dfx, dft


def stage_volume_bounds(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    dx: Float[Tensor, "nely nelx"],
    H: Tensor,
    Hs: Float[Tensor, " nely*nelx"],
    t_stage: float,
    volfrac: float,
    beta_t: float,
) -> tuple[float, float, Float[Tensor, " nely*nelx"], Float[Tensor, " nely*nelx"]]:
    """Per-stage material deposition budget: at stage boundary `t_stage` (a fraction of the
    build, in (0, 1]), the volume fraction deposited so far must stay within a small slack
    of `t_stage` itself -- an even deposition schedule spends the build's material at the
    rate the build advances. Returns both the upper- and lower-bound constraint (MMA
    constraints are one-sided, so an equality-like budget needs both), via a smooth
    stage-membership mask (`compliance.time_mask`, sharpness `beta_t`) rather than a hard
    time cutoff.

    Returns `(fval_upper, fval_lower, dfx, dft)`: the lower bound's sensitivity rows are
    exactly `-dfx, -dft` (see the MATLAB source), so only one `(dfx, dft)` pair is returned.
    """
    nely, nelx = xPhys.shape
    scale = nelx * nely * volfrac
    t_mask = compliance.time_mask(tPhys, t_stage, beta_t)
    dfdt = compliance_ref.time_mask_derivative(tPhys, t_stage, beta_t)
    xtJoint = xPhys * t_mask

    deposited = torch.sum(xtJoint) / scale
    fval_upper = float(deposited - t_stage)
    fval_lower = float(-deposited + t_stage - 1.0e-5)

    dfx = H @ ((t_mask / scale).flatten() * dx.flatten() / Hs)
    dft = H @ ((xPhys * dfdt / scale).flatten() / Hs)
    return fval_upper, fval_lower, dfx, dft
