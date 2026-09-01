"""Design constraints for the printable-structure optimization: global and per-stage
material budgets, print-start timing, and time-field smoothness.

The MATLAB source computes these as inline blocks in the main optimization loop, not as
a separate function. Sensitivities come from autograd (Phase 3.4,
`plans/torch_port_part2.md`) through the caller's own filter/Heaviside chain -- no
`dx`/`H`/`Hs` arguments here, unlike the hand-derived predecessors kept as a cross-check
in `tests/reference/constraints.py` (`tests/test_reference_sweep.py`). See
`conventions.md` for array-order and tolerance conventions.

Each function returns the constraint's value alone; callers (the main optimization
loop) get sensitivities from autograd and assemble the final MMA-input matrix.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor

import sttopt.compliance as compliance


def global_volume_fraction(
    xPhys: Float[Tensor, "nely nelx"], volfrac: float
) -> Float[Tensor, ""]:
    """Global printable-volume-fraction constraint: total deposited material vs. `volfrac`,
    differentiable end to end w.r.t. `xPhys`.
    """
    nely, nelx = xPhys.shape
    scale = nelx * nely * volfrac
    return torch.sum(xPhys) / scale - 1


def time_field_continuity(
    tPhys: Float[Tensor, "nely nelx"], L: Tensor
) -> Float[Tensor, ""]:
    """Time-field smoothness constraint: keeps each element's print time close to its local
    neighborhood average (`filters.continuity_filter`'s `L`), so the deposition sequence
    sweeps coherently across the mesh instead of jumping between distant elements.
    """
    nely, nelx = tPhys.shape
    nel = nely * nelx
    smoothness_weight = 2 * nel
    deviation = L @ tPhys.flatten()
    return smoothness_weight * (torch.sum(deviation**2 / nel) - 1.0e-6)


def start_point(
    tPhys: Float[Tensor, "nely nelx"], Nei: Int[Tensor, " k"]
) -> Float[Tensor, " k"]:
    """Print-start constraint(s): the deposition-origin element(s) `Nei` (0-indexed
    element numbers, per `conventions.md`) must start printing at t=0 (up to machine
    precision).

    Example: `Nei` is `[0]` for the single-origin time field (`tfield==1`) -- the
    elements nearest the print-start origin.
    """
    return tPhys.flatten()[Nei] - 1.0e-9


def stage_volume_bounds(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    t_stage: float,
    volfrac: float,
    beta_t: float,
) -> Float[Tensor, ""]:
    """Per-stage material deposition budget's *upper* bound: at stage boundary `t_stage`
    (a fraction of the build, in (0, 1]), the volume fraction deposited so far must stay
    within a small slack of `t_stage` itself -- an even deposition schedule spends the
    build's material at the rate the build advances, via a smooth stage-membership mask
    (`compliance.time_mask`, sharpness `beta_t`) rather than a hard time cutoff.

    The lower bound is an explicit negation of the upper's *value*, so callers build
    `fval_lower = -fval_upper - 1.0e-5` themselves rather than evaluating a second
    expression. The sensitivity is not negated by convention -- a caller differentiating
    both rows gets `dfdx_lower == -dfdx_upper` to floating-point precision, not by a
    hand-applied negation.
    """
    nely, nelx = xPhys.shape
    scale = nelx * nely * volfrac
    t_mask = compliance.time_mask(tPhys, t_stage, beta_t)
    xtJoint = xPhys * t_mask
    return torch.sum(xtJoint) / scale - t_stage
