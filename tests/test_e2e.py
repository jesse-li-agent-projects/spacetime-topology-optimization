"""End-to-end test for sttopt.optimize against the full small-grid MATLAB run
(`tests/fixtures/generate_fixtures.m`'s main loop).

Split into three layers, ordered from most to least diagnostic on failure:
  1. `test_iteration1_assembly_matches_fixture` -- iteration 1's assembled f0val/df0dx
     and fval/dfdx against `mma.mat`'s single-shot snapshot (the only ground truth for
     the *assembled* objective, since no other fixture covers df0dx). A pass here rules
     out objective/constraint-stacking bugs before trajectory drift can hide them.
  2. `test_mma_state_threading_matches_fixture` -- per-iteration xmma/low/upp/lam against
     `mma.mat`'s xmma_all/low_all/upp_all/lam_all, validating mmasub's stateful low/upp
     threading across multiple calls (no other test exercises this: test_mma.py only
     covers iteration 1, where low/upp start at 0 and are simply reinitialized).
  3. `test_e2e_trajectory_matches_fixture` -- the primary correctness check: xPhys/tPhys
     trajectory, objf, vol, tru_max_all, via optimize.run().

See conventions.md's tolerance policy: iteration 3 is where mmasub's `zzz == 0` asymptote
ties (test_mma.py's `test_mmasub_asymptote_update_branch` docstring) can amplify a ~1e-15
input difference into a 20-30% swing in a handful of low/upp entries -- a known, documented
hazard, not evidence of a wiring bug, if iterations 1-2 match tightly and only isolated
entries at iteration 3 exceed tolerance.
"""

import numpy as np

import sttopt.conductivity as conductivity
import sttopt.optimize as optimize
from conftest import assert_close, load_fixture

NELX, NELY = 7, 5
NSTAGE = 3
VOLFRAC = 0.5
THETA = 0.1
TCR = 0.8
TFIELD = 3
NLOOP = 3
RMIN, LRMIN, RMIN_COND = 2, 2, 3
BETA_INIT = 1.0


def _run():
    return optimize.run(
        NELX,
        NELY,
        NLOOP,
        NSTAGE,
        VOLFRAC,
        THETA,
        TCR,
        TFIELD,
        RMIN,
        LRMIN,
        RMIN_COND,
        beta=BETA_INIT,
    )


def test_iteration1_assembly_matches_fixture():
    """Checks f0val/df0dx and fval/dfdx at iteration 1 against mma.mat's single-shot
    snapshot. f0val in particular has no other coverage anywhere in this test suite --
    mmasub itself never reads it (see mma.py's docstring), so a wrong Theta-weighting
    or wrong per-stage `ti` in the objective sum would pass every other test here.
    """
    fx = load_fixture("mma")
    problem = optimize.build_problem(
        NELX, NELY, NSTAGE, VOLFRAC, THETA, TCR, TFIELD, RMIN, LRMIN, RMIN_COND
    )
    state = optimize.init_state(problem, BETA_INIT)

    _, record = optimize.step(problem, state)

    # f0val/df0dx are downstream of a sparse linear solve (compliance), so "solved" tier.
    assert_close(record.f0val, fx["f0val_1"], tier="solved")
    assert_close(record.df0dx, fx["df0dx_1"], tier="solved")
    assert_close(record.fval, fx["fval_1"], tier="algebraic")
    assert_close(record.dfdx, fx["dfdx_1"], tier="algebraic")


def test_mma_state_threading_matches_fixture():
    fx = load_fixture("mma")
    problem = optimize.build_problem(
        NELX, NELY, NSTAGE, VOLFRAC, THETA, TCR, TFIELD, RMIN, LRMIN, RMIN_COND
    )
    state = optimize.init_state(problem, BETA_INIT)

    for k in range(NLOOP):
        state, record = optimize.step(problem, state)
        assert_close(record.xmma, fx["xmma_all"][:, k], tier="e2e", iteration=k + 1)
        assert_close(record.low, fx["low_all"][:, k], tier="e2e", iteration=k + 1)
        assert_close(record.upp, fx["upp_all"][:, k], tier="e2e", iteration=k + 1)
        assert_close(record.lam, fx["lam_all"][:, k], tier="e2e", iteration=k + 1)


def test_constraints_stacking_matches_fixture():
    """Cheap, order-sensitive check on top of test_constraints.py's per-constraint
    fixture tests: this validates that optimize.step stacks fval/dfdx rows in the
    same order the reference loop does, which per-constraint tests can't catch (a
    swapped-but-correctly-shaped row wouldn't fail them).
    """
    fx = load_fixture("constraints")
    problem = optimize.build_problem(
        NELX, NELY, NSTAGE, VOLFRAC, THETA, TCR, TFIELD, RMIN, LRMIN, RMIN_COND
    )
    state = optimize.init_state(problem, BETA_INIT)

    for k in range(NLOOP):
        state, record = optimize.step(problem, state)
        assert_close(record.fval, fx["fval_all"][:, k], tier="e2e", iteration=k + 1)
        assert_close(record.dfdx, fx["dfdx_all"][:, :, k], tier="e2e", iteration=k + 1)


def test_e2e_trajectory_matches_fixture():
    fx = load_fixture("e2e")
    result = _run()

    # Slice 0: the initial field, before any iteration runs -- pins init_state's
    # unfiltered-at-init xTilde/tPhys directly, rather than only via iteration 1.
    assert_close(result.xPhys_traj[0], fx["xPhys_traj"][:, :, 0], tier="algebraic")
    assert_close(result.tPhys_traj[0], fx["tPhys_traj"][:, :, 0], tier="algebraic")

    for k in range(1, NLOOP + 1):
        assert_close(
            result.xPhys_traj[k], fx["xPhys_traj"][:, :, k], tier="e2e", iteration=k
        )
        assert_close(
            result.tPhys_traj[k], fx["tPhys_traj"][:, :, k], tier="e2e", iteration=k
        )
        assert_close(
            result.records[k - 1].obj, fx["objf"][k - 1], tier="e2e", iteration=k
        )
        assert_close(
            result.records[k - 1].vol, fx["vol"][k - 1], tier="e2e", iteration=k
        )
        assert_close(
            result.records[k - 1].tru_max,
            fx["tru_max_all"][k - 1],
            tier="e2e",
            iteration=k,
        )


def test_hotspot_factor_refresh_at_loop_25():
    """Targeted coverage for step()'s `loop % 25 == 0` factor-refresh branch: no fixture
    exercises it (NLOOP=3), so this drives 24 *real* iterations from init_state (rather
    than fabricating a `loop=24` state directly -- stale `low`/`upp`/`xold1`/`xold2` at an
    unrealistic loop count makes `mmasub`'s inner Newton loop fail to converge, an
    incidental warning unrelated to what this test targets) and checks the 25th call's
    returned `factor` against an independent recomputation of `hotspot_constraint`'s own
    documented refresh recipe, rather than relying on `step`'s internals to be
    self-consistently correct.
    """
    problem = optimize.build_problem(
        NELX, NELY, NSTAGE, VOLFRAC, THETA, TCR, TFIELD, RMIN, LRMIN, RMIN_COND
    )
    state = optimize.init_state(problem, BETA_INIT)
    for _ in range(24):
        state, _ = optimize.step(problem, state)
    assert state.loop == 24
    assert state.factor == 1.0  # never refreshed before loop 25

    new_state, _ = optimize.step(problem, state)
    assert new_state.loop == 25

    # Independent recomputation (conductivity.hotspot_constraint's own docstring recipe),
    # using the pre-update xPhys/tPhys/dx the refresh actually saw.
    dx = np.ones_like(
        state.xTilde
    )  # unused by K_est; only scales density sensitivities
    fv_old, _, _ = conductivity.hotspot_constraint(
        state.xPhys,
        state.tPhys,
        problem.e1,
        problem.e2,
        problem.w,
        dx,
        problem.H,
        problem.Hs,
        state.factor,
        problem.Tcr,
        problem.p,
        problem.q,
        problem.r,
        problem.rouf,
    )
    numer = (fv_old + 1) * problem.Tcr / state.factor
    K_est = conductivity.estimated_conductivity(
        state.xPhys,
        state.tPhys,
        problem.e1,
        problem.e2,
        problem.w,
        problem.q,
        problem.rouf,
    )
    max_g = np.max((1 - K_est) * state.xPhys.flatten(order="F") ** problem.r)
    expected_factor = max_g / numer

    assert not np.isclose(expected_factor, state.factor), "refresh must be non-vacuous"
    assert_close(new_state.factor, expected_factor, tier="algebraic")
