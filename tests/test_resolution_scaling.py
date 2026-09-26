"""Mesh refinement of one physical part: what the physics reads must converge as the
element size shrinks, rather than drift with the element count.

Each test builds the same part at several resolutions with smooth fields defined in
physical coordinates, and checks the order of convergence the successive differences
show. Pointwise maxima at a boundary are left out: a finer mesh samples closer to the
boundary, so those converge only as fast as the sampling does.

The `xfail` tests are the quantities not yet invariant; each names the step of
`plans/physical_units.md` that should make it pass.
"""

import math

import numpy as np
import pytest
import torch
from jaxtyping import Float

import sttopt.compliance as compliance
import sttopt.constraints as constraints
import sttopt.filters as filters
import sttopt.seqopt as seqopt
import sttopt.stto as stto
import sttopt.torch_util as torch_util
from conftest import default_run_config, default_seq_run_config

WIDTH_M, HEIGHT_M = 0.18, 0.06
NELX = (30, 60, 120, 240)
# Below this, a difference sequence is not converging but drifting, as a point load's
# log singularity does.
MIN_ORDER = 0.75


def _problem(nelx: int, **overrides) -> stto.Problem:
    config = dict(
        width_m=WIDTH_M,
        height_m=HEIGHT_M,
        nelx=nelx,
        rmin_m=0.012,
        time_filter_rmin_m=0.012,
        lrmin_m=0.009,
        rmin_cond_m=0.006,
        tool_radius_m=0.01,
        print_base="bottom_edge",
    )
    return stto.build_problem(default_run_config(**(config | overrides)))


def _fields(
    problem: stto.Problem,
) -> tuple[Float[torch.Tensor, "nely nelx"], Float[torch.Tensor, "nely nelx"]]:
    """A smooth density, and a time field whose iso-lines are circles about a point
    outside the bottom-left corner, printed from the outside in, so they are concave."""
    config = problem.config
    h = config.element_size_m
    X, Y = np.meshgrid(
        (np.arange(config.nelx) + 0.5) * h,
        HEIGHT_M - (np.arange(config.nely) + 0.5) * h,  # row 0 is the top
    )
    x = 0.6 + 0.3 * np.sin(2 * np.pi * X / WIDTH_M) * np.cos(np.pi * Y / HEIGHT_M)
    r = np.hypot(X + 0.05, Y + 0.05)
    t = (r.max() - r) / (r.max() - r.min())
    to = lambda a: torch_util.to_tensor(a, problem.device, problem.dtype)  # noqa: E731
    return to(x), to(t)


def _order(values: list[float]) -> float:
    """The order of convergence the last three of `values` show, each at half the
    previous element size; `inf` where they already agree exactly."""
    d1, d2 = abs(values[-2] - values[-3]), abs(values[-1] - values[-2])
    return math.inf if d2 == 0 else math.log2(d1 / d2)


def _at_each_resolution(quantity, nelx=NELX, **overrides) -> list[float]:
    values = []
    for n in nelx:
        problem = _problem(n, **overrides)
        values.append(float(quantity(problem, *_fields(problem))))
    return values


def _gravity_compliance(problem, x, t):
    c = problem.config
    cg, _ = compliance.gravity_compliance(
        x, t, problem.KE, problem.edofMat, c.Emin, c.Emax, 3.0, 1.0, problem.C,
        50.0, problem.freedofs, problem.ndof,
    )  # fmt: skip
    return cg


def _whole_compliance(problem, x, t):
    c = problem.config
    cw, _ = compliance.whole_compliance(
        x, problem.KE, problem.edofMat, c.Emin, c.Emax, 3.0, problem.freedofs,
        problem.F, problem.ndof,
    )  # fmt: skip
    return cw


def test_filtered_density_converges():
    values = _at_each_resolution(
        lambda p, x, t: (filters.apply_density_filter(x, p.H, p.Hs) ** 2).mean()
    )
    assert _order(values) >= MIN_ORDER, values


def test_self_weight_compliance_converges():
    values = _at_each_resolution(_gravity_compliance)
    assert _order(values) >= MIN_ORDER, values


def test_mean_hotspot_severity_converges():
    """Over the elements the print base does not shield, so every one is finite."""

    def mean_severity(p, x, t):
        K = stto.estimated_conductivity(p, x, t, 0)
        finite = torch.isfinite(K)
        return ((1 - K[finite]) * x.flatten()[finite] ** p.config.r).mean()

    values = _at_each_resolution(mean_severity, nelx=NELX[:3], rmin_cond_m=0.018)
    assert _order(values) >= MIN_ORDER, values


def test_tool_radius_row_converges():
    """At iteration 0 the calibration puts the row on the tool radius times the true
    concave curvature, so it is a physical quantity."""
    values = _at_each_resolution(lambda p, x, t: stto._tool_radius_row(p, x, t, 0)[0])
    assert values[-1] > -1  # non-vacuous: the iso-lines are concave
    assert _order(values) >= MIN_ORDER, values


@pytest.mark.xfail(
    strict=True, reason="step 4: a point load's compliance grows as log(1/h)"
)
def test_tip_load_compliance_converges():
    values = _at_each_resolution(_whole_compliance)
    assert _order(values) >= MIN_ORDER, values


@pytest.mark.xfail(
    strict=True,
    reason="step 3 (a mean-form row) and the lrmin redesign (a physical window)",
)
def test_continuity_row_converges():
    values = _at_each_resolution(
        lambda p, x, t: constraints.time_field_continuity(t, p.L, 1e-6),
        nelx=NELX[:3],
    )
    assert _order(values) >= MIN_ORDER, values


@pytest.mark.xfail(
    strict=True, reason="step 3: one start-point row, not one per base element"
)
def test_constraint_row_count_is_independent_of_resolution():
    rows = []
    for n in NELX[:2]:
        problem = _problem(n, nStage=2)
        _, record = stto.step(problem, stto.init_state(problem))
        rows.append(len(record.g))
    assert rows[0] == rows[1]


@pytest.mark.parametrize("kappa", [0.0, 2.3666])
def test_void_padding_leaves_the_parts_conductivity_unchanged(kappa):
    """The domain box is not physics: widening it with void the conductivity stencil
    cannot reach from the part must not change what the part measures."""
    ny, nx, pad = 20, 30, 20

    def K_est(padding: int) -> np.ndarray:
        xPhys = np.hstack([np.ones((ny, nx)), np.zeros((ny, padding))])
        # Linear across the padding too, so the part's edge column reads the same
        # gradient from a one-sided difference as from a central one.
        rows, cols = np.indices(xPhys.shape)
        t = ((ny - 1 - rows) + 0.3 * cols) / ny
        config = default_seq_run_config(hotspot_kappa=kappa, rmin_cond_m=0.012)
        problem = seqopt.build_problem(config, xPhys, 2e-3, device="cpu")
        K = seqopt.estimated_conductivity(problem, torch.from_numpy(t))
        return K.reshape(ny, nx + padding)[:, :nx].numpy()

    np.testing.assert_allclose(K_est(pad), K_est(0), rtol=1e-12)
