"""End-to-end test for sttopt.stto against the full small-grid trajectory
(`tests/fixtures/generate_fixtures.py`'s main loop) -- a golden-regression fixture,
not a MATLAB cross-check (see conftest.py, conventions.md).

Split into three layers, ordered from most to least diagnostic on failure:
  1. `test_iteration1_assembly_matches_fixture` -- iteration 1's assembled `.f`/`.df`
     and `.g`/`.dg` against `mma.npz`'s single-shot snapshot (the only ground truth for
     the *assembled* objective, since no other fixture covers `.df`). A pass here rules
     out objective/constraint-stacking bugs before trajectory drift can hide them.
  2. `test_mma_state_threading_matches_fixture` -- per-iteration xmma/low/upp/lam against
     `mma.npz`'s xmma_all/low_all/upp_all/lam_all, validating mmasub's stateful low/upp
     threading across multiple calls (no other test exercises this: test_mma.py only
     covers iteration 1, where low/upp start at 0 and are simply reinitialized).
  3. `test_e2e_trajectory_matches_fixture` -- the primary regression check: xPhys/tPhys
     trajectory, objf, vol, tru_max_all, via stto.run().
"""

import numpy as np
import torch

import sttopt.conductivity as conductivity
import sttopt.filters as filters
import sttopt.stto as stto
import tests.reference.conductivity as conductivity_ref
from conftest import assert_close, default_run_config, load_fixture_npz

NELX, NELY = 7, 5
NSTAGE = 3
VOLFRAC = 0.5
THETA = 0.1
TCR = 0.8
PRINT_BASE = "opposite_corner"
NLOOP = 3
RMIN, LRMIN, RMIN_COND = 2, 2, 3
BETA_INIT = 1.0

CONFIG = default_run_config(
    nelx=NELX,
    nely=NELY,
    nStage=NSTAGE,
    volfrac=VOLFRAC,
    Theta=THETA,
    Tcr=TCR,
    print_base=PRINT_BASE,
    rmin=RMIN,
    lrmin=LRMIN,
    rmin_cond=RMIN_COND,
    nloop=NLOOP,
)


def _run():
    problem = stto.build_problem(CONFIG)
    return stto.run_from_state(problem, stto.init_state(problem, BETA_INIT), NLOOP)


def test_iteration1_assembly_matches_fixture():
    """Checks .f/.df and .g/.dg at iteration 1 against mma.npz's single-shot
    snapshot. .f in particular has no other coverage anywhere in this test suite --
    mmasub itself never reads it (see mma.py's docstring), so a wrong Theta-weighting
    or wrong per-stage `ti` in the objective sum would pass every other test here.
    """
    fx = load_fixture_npz("mma")
    problem = stto.build_problem(CONFIG)
    state = stto.init_state(problem, BETA_INIT)

    _, record = stto.step(problem, state)

    # .f/.df are downstream of a sparse linear solve (compliance), so "solved" tier.
    assert_close(record.f, fx["f0val_1"], tier="solved")
    assert_close(record.df, fx["df0dx_1"], tier="solved")
    assert_close(record.g, fx["fval_1"], tier="algebraic")
    assert_close(record.dg, fx["dfdx_1"], tier="algebraic")


def test_constraints_stacking_matches_fixture():
    """Cheap, order-sensitive check on top of test_constraints.py's per-constraint
    fixture tests: this validates that stto.step stacks .g/.dg rows in the
    same order the reference loop does, which per-constraint tests can't catch (a
    swapped-but-correctly-shaped row wouldn't fail them).

    Only the first iteration is compared. `.g`/`.dg` are built before `mmasub` runs, so
    at iteration 1 -- where both implementations sit on the same initial design -- they
    still agree exactly. From iteration 2 the designs diverge, because `sttopt` no
    longer uses the MATLAB source's trust-region parameterization (see
    `mma.trust_region_params`), and the comparison would say nothing about row order.
    """
    fx = load_fixture_npz("constraints")
    problem = stto.build_problem(CONFIG)
    state = stto.init_state(problem, BETA_INIT)

    _, record = stto.step(problem, state)
    assert_close(record.g, fx["fval_all"][:, 0], tier="e2e", iteration=1)
    assert_close(record.dg, fx["dfdx_all"][:, :, 0], tier="e2e", iteration=1)


def test_hotspot_factor_refresh_at_loop_25():
    """Targeted coverage for step()'s `loop % 25 == 0` factor-refresh branch: no fixture
    exercises it (NLOOP=3), so this drives 24 *real* iterations from init_state (rather
    than fabricating a `loop=24` state directly -- stale `low`/`upp`/`xold1`/`xold2` at an
    unrealistic loop count makes `mmasub`'s inner Newton loop fail to converge, an
    incidental warning unrelated to what this test targets) and checks the 25th call's
    returned `factor`, `.g`, and `.dg` against an independent recomputation, rather
    than relying on `step`'s internals to be self-consistently correct.

    The refresh takes effect starting the *next* iteration, not the one that computes
    it: loop 25's own `.g`/`.dg` are evaluated at the old `factor`, and the new
    `factor` only lands in `new_state.factor` for loop 26 onward.
    """
    problem = stto.build_problem(CONFIG)
    state = stto.init_state(problem, BETA_INIT)
    for _ in range(24):
        state, _ = stto.step(problem, state)
    assert state.loop == 24
    assert state.factor == 1.0  # never refreshed before loop 25

    new_state, record = stto.step(problem, state)
    assert new_state.loop == 25

    xPhys = state.xPhys
    tPhys = state.tPhys

    # Independent recomputation of the refresh formula (factor = max_g / numer), using
    # the pre-update xPhys/tPhys/dx the refresh actually saw.
    dx = filters.heaviside_projection_derivative(
        state.xTilde, state.beta_d, problem.config.eta
    )
    old = conductivity_ref.hotspot_constraint(
        xPhys,
        tPhys,
        problem.e1,
        problem.e2,
        problem.w,
        dx,
        problem.H,
        problem.Hs,
        state.factor,
        problem.config.Tcr,
        problem.config.p,
        problem.config.q,
        problem.config.r,
        problem.config.rouf,
    )
    numer = (old.fval + 1) * problem.config.Tcr / state.factor
    K_est = conductivity.estimated_conductivity(
        xPhys,
        tPhys,
        problem.e1,
        problem.e2,
        problem.w,
        problem.config.q,
        problem.config.rouf,
    )
    max_g = float(torch.max((1 - K_est) * xPhys.flatten() ** problem.config.r))
    expected_factor = max_g / numer

    assert not np.isclose(expected_factor, state.factor), "refresh must be non-vacuous"
    assert_close(new_state.factor, expected_factor, tier="algebraic")

    # Loop 25's own fv/df1/dt1 in `record` are evaluated at the *old* (pre-refresh)
    # factor -- they must match `old` (already computed above at `state.factor`), not
    # a recompute at `expected_factor`. `tru_max`, a pure diagnostic, uses the refreshed
    # factor immediately.
    nel = problem.config.nelx * problem.config.nely
    assert_close(record.g[-1], old.fval, tier="algebraic")
    assert_close(record.dg[-1, :nel], old.df1, tier="algebraic")
    assert_close(record.dg[-1, nel:], old.dt1, tier="algebraic")
    assert_close(record.tru_max, expected_factor * numer, tier="algebraic")
