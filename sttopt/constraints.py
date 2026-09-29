"""Design constraints for the printable-structure optimization: global and per-stage
material budgets, print-start timing, time-field smoothness, and the printability
bounds on the time field (tool radius, gradient floor and smoothness). The last three
bound the worst element, through a calibrated smooth maximum the caller owns.

The MATLAB source computes its constraints as inline blocks in the main optimization
loop, not as separate functions. Sensitivities come from autograd (Phase 3.4,
`plans/torch_port_part2.md`) through the caller's own filter/Heaviside chain -- no
`dx`/`H`/`Hs` arguments here, unlike the hand-derived predecessors kept as a cross-check
in `tests/reference/constraints.py` (`tests/test_reference_sweep.py`). See
`conventions.md` for array-order and tolerance conventions.

Each function returns its constraint values alone, as a 1-D tensor with one entry per
constraint; callers (the main optimization loop) stack them and get sensitivities from
autograd.
"""

from typing import Sequence

import torch
from jaxtyping import Float, Int
from torch import Tensor

import sttopt.compliance as compliance
import sttopt.smooth_max as smooth_max
import sttopt.timefield as timefield


def global_volume_fraction(
    xPhys: Float[Tensor, "nely nelx"], volfrac: float
) -> Float[Tensor, " 1"]:
    """Global printable-volume-fraction constraint: total deposited material vs. `volfrac`,
    differentiable end to end w.r.t. `xPhys`.
    """
    nely, nelx = xPhys.shape
    scale = nelx * nely * volfrac
    return (torch.sum(xPhys) / scale - 1)[None]


def time_field_continuity(
    tPhys: Float[Tensor, "nely nelx"], L: Tensor, tolerance: float = 1.0e-6
) -> Float[Tensor, " 1"]:
    """Time-field smoothness constraint: keeps each element's print time close to its local
    neighborhood average (`filters.continuity_filter`'s `L`), so the deposition sequence
    sweeps coherently across the mesh instead of jumping between distant elements.

    This is the only bound on how jagged the field may get -- the layer-uniformity
    objective is scale-free and scores a uniformly-jagged field as well as a smooth one
    -- and it is a loose one. Tightening `tolerance` does not tighten it: the constraint
    aggregates over the whole field, so it cannot separate a sawtooth from a geometry's
    legitimate curvature, and below that curvature's own deviation it is infeasible from
    the first iteration rather than binding (PR #91).

    A mean over the domain relative to `tolerance`, so the constraint stays of order 1
    under mesh refinement; the MATLAB source's `2 * nel` weight grew with it.

    :param tPhys: physical time field
    :param L: continuity filter, from `filters.continuity_filter`
    :param tolerance: bound on the mean squared deviation from the neighborhood average,
        so the field's RMS deviation is held to `sqrt(tolerance)`. Must stay above the
        geometry's own curvature floor to be a constraint at all.
    :return: the constraint value, non-positive when satisfied
    """
    deviation = L @ tPhys.flatten()
    return (torch.mean(deviation**2) / tolerance - 1)[None]


def start_point(
    tPhys: Float[Tensor, "nely nelx"], Nei: Int[Tensor, " k"]
) -> Float[Tensor, " 1"]:
    """
    Print-start constraint: the deposition-origin element(s) must start printing at t=0
    (up to machine precision).

    One constraint on their mean print time rather than one per element, so the
    constraint count does not grow with the mesh. `tPhys` is non-negative, so a mean at
    0 pins every one.

    Example: `Nei` is `[0]` for the single-origin time field (`tfield==1`) -- the
    elements nearest the print-start origin.

    :param tPhys: physical time field
    :param Nei: deposition-origin element numbers, 0-indexed per `conventions.md`
    :return: the constraint value, non-positive when satisfied
    """
    return (tPhys.flatten()[Nei].mean() - 1.0e-9)[None]


def stage_volume_bounds(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    stage_times: Sequence[float],
    volfrac: float,
    beta_t: float,
) -> Float[Tensor, " 2*n_stage"]:
    """Per-stage material deposition budget: at each stage boundary `t_stage` (a
    fraction of the build, in (0, 1]), the volume fraction deposited so far must stay
    within a small slack of `t_stage` itself -- an even deposition schedule spends the
    build's material at the rate the build advances, via a smooth stage-membership mask
    (`compliance.time_mask`, sharpness `beta_t`) rather than a hard time cutoff.

    An upper and a lower bound per stage, interleaved. The lower bound is the negated
    upper one with a `1e-5` slack, so its sensitivity is exactly the negated upper's.

    Deprecated: it is incompatible with uniform layer heights, which deposit volume in
    proportion to the cross-section rather than to `t_stage`.
    """
    nely, nelx = xPhys.shape
    scale = nelx * nely * volfrac
    upper = torch.stack(
        [
            torch.sum(xPhys * compliance.time_mask(tPhys, t_stage, beta_t)) / scale
            - t_stage
            for t_stage in stage_times
        ]
    )
    return torch.stack([upper, -upper - 1.0e-5], dim=1).flatten()


def tool_radius(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    aggregate: smooth_max.CalibratedLogSumExp,
    tool_radius: float,
    r: float,
    loop: int,
) -> Float[Tensor, " 1"]:
    """Tool-radius constraint: the concave curvature of `t`'s iso-lines stays below
    `1 / tool_radius`, so the print tool cannot collide with printed material.

    The severity is `tool_radius * concave curvature`, density-weighted like the other
    aggregated constraints, so the constraint is its smooth maximum minus the bound of 1.

    :param aggregate: the smooth maximum, holding its own calibration
    :param tool_radius: this iteration's tool radius, in elements
    :param r: exponent of the density weight
    :param loop: iteration, for the smooth maximum's schedule
    """
    kappa_t = timefield.iso_curvature(tPhys, xPhys)
    unit = timefield.unit_length(tPhys)
    density_r = smooth_max.density_power(xPhys[1:-1, 1:-1].flatten(), r)
    concave_t = -kappa_t.flatten() / unit * density_r
    return (aggregate.aggregate(tool_radius * concave_t, loop) - 1)[None]


def min_gradient(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    aggregate: smooth_max.CalibratedLogSumExp,
    fraction: float,
    r: float,
    loop: int,
) -> Float[Tensor, " 1"]:
    """Gradient floor: each element's central-difference `|grad t|` stays at least
    `fraction` of the median, which rules out the interior saddles, extrema and flat
    valleys of a smooth field; none of these can be printed. A central difference never
    reads the element itself, so alone it passes an element-scale pit (PR #166);
    `gradient_smoothness` closes that.

    The severity, `x**r * (2 f - |grad t| / median)`, is `f` on the floor in a solid
    element and 0 in void, so it grows continuously as an element turns solid rather
    than jumping at a threshold (PR #166). The median, over the whole mesh, is held out
    of the gradient, as a calibration is, so the constraint cannot be met by lowering
    it.

    :param aggregate: the smooth maximum, holding its own calibration
    :param fraction: this iteration's floor, as a fraction of the median
    :param r: exponent of the density weight
    :param loop: iteration, for the smooth maximum's schedule
    """
    grad, _ = timefield.central_derivatives(tPhys)
    if grad.numel() == 0:
        # Nothing to bound; `grad.sum()` is a zero that keeps the constraint in the
        # graph.
        return (grad.sum() - 1)[None]
    slope = torch.linalg.vector_norm(grad, dim=-1)
    median = slope.detach().median()
    density_r = smooth_max.density_power(xPhys[1:-1, 1:-1].flatten(), r)
    severity = density_r * (2 * fraction - slope / median)
    return (aggregate.aggregate(severity, loop) - fraction)[None]


def gradient_smoothness(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    aggregate: smooth_max.CalibratedLogSumExp,
    length: float,
    r: float,
    loop: int,
) -> Float[Tensor, " 1"]:
    """Hessian bound, `||H||_F <= median |grad t| / length`, which closes the pits that
    `min_gradient` passes: the local quadratic's gradient stays within
    `h ||H||_F / sqrt(2)` of `grad t` across the element, so no critical point fits in
    one while `length` exceeds `h / (sqrt(2) * fraction)`.

    Density-weighted, with the median held out of the gradient, like `min_gradient`.

    :param aggregate: the smooth maximum, holding its own calibration
    :param length: this iteration's shortest length over which the gradient may change
        by the median gradient, in elements
    :param r: exponent of the density weight
    :param loop: iteration, for the smooth maximum's schedule
    """
    grad, hess = timefield.central_derivatives(tPhys)
    if grad.numel() == 0:
        # Nothing to bound; `grad.sum()` is a zero that keeps the constraint in the
        # graph.
        return (grad.sum() - 1)[None]
    median = torch.linalg.vector_norm(grad, dim=-1).detach().median()
    density_r = smooth_max.density_power(xPhys[1:-1, 1:-1].flatten(), r)
    # `vector_norm`, unlike `sqrt`, has a finite gradient where `H` vanishes.
    frobenius = torch.linalg.vector_norm(
        hess * hess.new_tensor([1.0, 1.0, 2**0.5]), dim=-1
    )
    severity = density_r * frobenius * length / median
    return (aggregate.aggregate(severity, loop) - 1)[None]
