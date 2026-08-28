"""Tests sttopt.mma against small optimization problems independent of the MATLAB
port the rest of this codebase is validated against: two with closed-form optima
(ground truth), and one (the 8-bar truss) as a regression/golden-value pin -- see its
docstring for why no independent ground truth is available for it.
"""

import warnings

import numpy as np

import sttopt.mma as mma
from conftest import tt


def _run_mmasub(n, m, xmin, xmax, x0, a0, a, c, d, objective, constraint, n_iterations):
    """Drives mmasub's own default asymptote/move-limit heuristics for `n_iterations`
    starting from `x0`, given per-iteration `objective(x) -> (f0val, df0dx)` and
    `constraint(x) -> (fval, dfdx)` callbacks. Returns the final x.

    `objective`/`constraint` are plain-NumPy callbacks (independent test-problem
    formulas); `x` round-trips to NumPy between `mmasub` calls so they stay unchanged.

    Runs with warnings promoted to errors, since subsolv's inner-Newton-cap warning
    (CLAUDE.local.md: don't ignore test warnings) usually means the problem's bounds
    or starting point put it in a regime the port wasn't meant to handle.
    """
    x = np.asarray(x0, dtype=float)
    xold1, xold2 = x.copy(), x.copy()
    low, upp = tt(np.zeros(n)), tt(np.zeros(n))
    xmin_t, xmax_t, a_t, c_t, d_t = tt(xmin), tt(xmax), tt(a), tt(c), tt(d)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for iteration in range(1, n_iterations + 1):
            f0val, df0dx = objective(x)
            fval, dfdx = constraint(x)
            xmma, _, _, _, _, _, _, _, _, low, upp = mma.mmasub(
                m,
                n,
                iteration,
                tt(x),
                xmin_t,
                xmax_t,
                tt(xold1),
                tt(xold2),
                float(f0val),
                tt(df0dx),
                tt(fval),
                tt(dfdx),
                low,
                upp,
                a0,
                a_t,
                c_t,
                d_t,
            )
            xold2, xold1 = xold1, x
            x = xmma.numpy()
    return x


# Test problem 1 (cantilever beam), eq. (22): minimize C1*sum(x), x_j > 0, subject to
# sum(coef_j / x_j^3) <= C2. From Svanberg (1987) section 6; see
# resources/Svanberg1987_MethodOfMovingAsymptotes.pdf.
_BEAM_COEF = np.array([61.0, 37.0, 19.0, 7.0, 1.0])
_BEAM_C1 = 0.0624
_BEAM_C2 = 1.0


def _beam_analytic_optimum():
    """Closed-form optimum of eq. (22): stationarity of the Lagrangian makes x_j
    proportional to coef_j^(1/4), and the constraint being active at the optimum
    fixes the overall scale.
    """
    root4 = _BEAM_COEF**0.25
    x_star = np.sum(root4) ** (1 / 3) * root4
    f0_star = _BEAM_C1 * np.sum(x_star)
    return x_star, f0_star


def test_mmasub_cantilever_beam_converges_to_analytic_optimum():
    """Drives mmasub (its own default heuristics -- not the paper's eq. (23)/(24)
    asymptote/move-limit rules or its Fortran dual solver, which differ too much from
    this port's interior-point subsolv for a per-iteration trajectory match to be
    meaningful) on Svanberg's test problem 1, and checks it converges to the
    closed-form optimum. df0dx > 0 and dfdx < 0 throughout, exercising the `p0`
    (objective) / `Q` (constraint) branches of mmasub's subproblem construction.
    """
    x_star, f0_star = _beam_analytic_optimum()
    n, m = 5, 1

    def objective(x):
        return _BEAM_C1 * np.sum(x), np.full(n, _BEAM_C1)

    def constraint(x):
        g = np.sum(_BEAM_COEF / x**3) - _BEAM_C2
        dgdx = -3 * _BEAM_COEF / x**4
        return np.array([g]), dgdx[None, :]

    x = _run_mmasub(
        n,
        m,
        xmin=np.full(n, 0.1),
        xmax=np.full(n, 10.0),
        x0=np.full(n, 5.0),
        a0=1.0,
        a=np.zeros(m),
        c=np.array([1000.0]),
        d=np.zeros(m),
        objective=objective,
        constraint=constraint,
        n_iterations=20,
    )

    f0val, _ = objective(x)
    gval, _ = constraint(x)
    infeas = max(0.0, gval[0] / _BEAM_C2)  # eq. (25)

    np.testing.assert_allclose(x, x_star, rtol=1e-4)
    np.testing.assert_allclose(f0val, f0_star, rtol=1e-4)
    assert infeas < 1e-6


def test_mmasub_quadratic_objective_linear_constraint_converges_to_analytic_optimum():
    """Minimize (x1-4)^2 + (x2-4)^2 subject to x1+x2 <= 2, starting from the feasible
    interior point (0,0). The unconstrained minimum (4,4) violates the constraint, so
    by symmetry and KKT stationarity (2*(x_j-4) + lambda = 0 for both j, constraint
    active) the optimum is the projection of (4,4) onto the line x1+x2=2: x* = (1,1),
    f0* = 18, with lambda = 6 >= 0 confirming validity.

    Complements the cantilever-beam test's sign pattern: df0dx < 0 here (near the
    optimum), exercising mmasub's `q0` branch on the objective, and dfdx is constant
    and positive, exercising the `P` branch on the constraint -- the two branches the
    beam problem doesn't reach.
    """
    x_star = np.array([1.0, 1.0])
    f0_star = 18.0
    n, m = 2, 1

    def objective(x):
        return np.sum((x - 4.0) ** 2), 2 * (x - 4.0)

    def constraint(x):
        return np.array([np.sum(x) - 2.0]), np.array([[1.0, 1.0]])

    x = _run_mmasub(
        n,
        m,
        xmin=np.full(n, -10.0),
        xmax=np.full(n, 10.0),
        x0=np.zeros(n),
        a0=1.0,
        a=np.zeros(m),
        c=np.array([1000.0]),
        d=np.zeros(m),
        objective=objective,
        constraint=constraint,
        n_iterations=10,
    )

    f0val, _ = objective(x)
    gval, _ = constraint(x)
    infeas = max(0.0, gval[0])

    np.testing.assert_allclose(x, x_star, atol=1e-6)
    np.testing.assert_allclose(f0val, f0_star, rtol=1e-6)
    assert infeas < 1e-6


# Test problem 2 (8-bar truss), section 6: 8 elements from the fixed nodes to a single
# free apex node (5), a linear-elastic space truss under one load case, weight
# objective, +-100 N/mm^2 stress constraint per bar. Geometry/topology/load from
# Figure 2 and its tables; the material is unstated in the paper's text but a density
# of 7800 kg/m^3 (steel) reproduces the paper's stated starting weight (13.05 kg)
# exactly, corroborating the geometry transcription below.
_TRUSS_NODES = {
    1: (-250.0, -250.0, 0.0),
    2: (-250.0, 250.0, 0.0),
    3: (250.0, 250.0, 0.0),
    4: (250.0, -250.0, 0.0),
    5: (0.0, 0.0, 375.0),  # free node; all 8 bars connect here
    6: (-375.0, 0.0, 0.0),
    7: (0.0, 375.0, 0.0),
    8: (375.0, 0.0, 0.0),
    9: (0.0, -375.0, 0.0),
}
_TRUSS_BASE_NODES = [1, 2, 3, 4, 6, 7, 8, 9]  # bar j's fixed end
_TRUSS_FREE = np.array(_TRUSS_NODES[5])
_TRUSS_L = np.array(
    [np.linalg.norm(_TRUSS_FREE - np.array(_TRUSS_NODES[b])) for b in _TRUSS_BASE_NODES]
)
_TRUSS_E = np.array(
    [
        (_TRUSS_FREE - np.array(_TRUSS_NODES[b])) / L
        for b, L in zip(_TRUSS_BASE_NODES, _TRUSS_L)
    ]
)
_TRUSS_FORCE = np.array([40e3, 20e3, 200e3])  # N, at node 5
_TRUSS_RHO = 7800e-9  # kg/mm^3
_TRUSS_SIGMA_LIM = 100.0  # N/mm^2


def _truss_stress(A):
    """Member stresses (tension positive) and their Jacobian d(sigma)/d(A), for the
    8-bar space truss under _TRUSS_FORCE.

    All 8 bars share a common free node and Young's modulus, so E cancels out of the
    force distribution: assembling K = sum_j (A_j/L_j) e_j e_j^T (E-free) and solving
    K u = F for a "pseudo-displacement" u gives sigma_j = (e_j . u) / L_j directly (the
    true displacement is u/E, and sigma_j = (E/L_j)*(e_j . (u/E)) -- E drops out).
    Differentiating that solve gives d(sigma_j)/d(A_i) = -(sigma_i/L_j) * (e_j . Kinv .
    e_i), verified against finite differences during development (see PR discussion).
    """
    Kp = (_TRUSS_E * (A / _TRUSS_L)[:, None]).T @ _TRUSS_E
    Kpinv = np.linalg.inv(Kp)
    sigma = (_TRUSS_E @ (Kpinv @ _TRUSS_FORCE)) / _TRUSS_L
    M = _TRUSS_E @ Kpinv @ _TRUSS_E.T
    dsigma = -(sigma[None, :] / _TRUSS_L[:, None]) * M
    return sigma, dsigma


def _truss_objective(A):
    return _TRUSS_RHO * np.sum(A * _TRUSS_L), _TRUSS_RHO * _TRUSS_L


def _truss_constraint(A):
    """Two one-sided linearizable constraints per bar (sigma <= lim, -sigma <= lim)
    rather than |sigma| <= lim, so every constraint stays smooth in A.
    """
    sigma, dsigma = _truss_stress(A)
    fval = np.concatenate([sigma - _TRUSS_SIGMA_LIM, -sigma - _TRUSS_SIGMA_LIM])
    dfdx = np.concatenate([dsigma, -dsigma], axis=0)
    return fval, dfdx


def test_truss_starting_weight_matches_paper():
    """Independent check (not a golden value) that the truss geometry/topology/density
    above was transcribed correctly: the paper reports 13.05 kg at x_j=400 for all j.
    """
    weight0, _ = _truss_objective(np.full(8, 400.0))
    np.testing.assert_allclose(weight0, 13.05, atol=0.01)


def test_mmasub_eight_bar_truss_matches_golden_regression_value():
    """Unlike the other two tests in this file, this is a regression/golden-value
    pin, not a check against independently derived ground truth: Table II's
    per-iteration values used a different (Fletcher-Reeves dual) subproblem solver
    (same caveat as test problem 1's Table I), and the paper only reports its final
    optimum "approximately" (x1..x8 rounded to the nearest 10 mm^2, no digits given
    for the objective) -- not precise enough to assert against directly. So instead
    this pins mmasub's actual current output after 30 iterations at rtol=1e-6, to
    catch future accidental behavior changes.

    The regression target is still grounded in the paper: it lands close to the
    paper's reported approximate optimum (x1..x4 near 880/720/260/520 mm^2, x5..x8 at
    the 100 mm^2 lower bound, weight near 11.23 kg, all constrained bars near the
    +-100 N/mm^2 stress limit) -- see test_truss_starting_weight_matches_paper and PR
    discussion for how the geometry was cross-checked against the paper.
    """
    n, m = 8, 16

    x = _run_mmasub(
        n,
        m,
        xmin=np.full(n, 100.0),
        xmax=np.full(n, 5000.0),
        x0=np.full(n, 400.0),
        a0=1.0,
        a=np.zeros(m),
        c=np.full(m, 1000.0),
        d=np.zeros(m),
        objective=_truss_objective,
        constraint=_truss_constraint,
        n_iterations=30,
    )

    golden_x = np.array(
        [
            849.94135375,
            753.17038548,
            231.47550478,
            547.01510523,
            100.0002238,
            100.00022557,
            100.00022551,
            100.00022469,
        ]
    )
    golden_f0val = 11.22874168217898

    np.testing.assert_allclose(x, golden_x, rtol=1e-6)
    np.testing.assert_allclose(_truss_objective(x)[0], golden_f0val, rtol=1e-6)
