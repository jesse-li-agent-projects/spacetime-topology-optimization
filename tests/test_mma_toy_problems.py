"""Tests sttopt.mma against small optimization problems with closed-form optima --
ground truth independent of the MATLAB port the rest of this codebase is validated
against.
"""

import warnings

import numpy as np

import sttopt.mma as mma


def _run_mmasub(n, m, xmin, xmax, x0, a0, a, c, d, objective, constraint, n_iterations):
    """Drives mmasub's own default asymptote/move-limit heuristics for `n_iterations`
    starting from `x0`, given per-iteration `objective(x) -> (f0val, df0dx)` and
    `constraint(x) -> (fval, dfdx)` callbacks. Returns the final x.

    Runs with warnings promoted to errors, since subsolv's inner-Newton-cap warning
    (CLAUDE.local.md: don't ignore test warnings) usually means the problem's bounds
    or starting point put it in a regime the port wasn't meant to handle.
    """
    x = np.asarray(x0, dtype=float)
    xold1, xold2 = x.copy(), x.copy()
    low, upp = np.zeros(n), np.zeros(n)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for iteration in range(1, n_iterations + 1):
            f0val, df0dx = objective(x)
            fval, dfdx = constraint(x)
            xmma, _, _, _, _, _, _, _, _, low, upp = mma.mmasub(
                m,
                n,
                iteration,
                x,
                xmin,
                xmax,
                xold1,
                xold2,
                f0val,
                df0dx,
                fval,
                dfdx,
                low,
                upp,
                a0,
                a,
                c,
                d,
            )
            xold2, xold1 = xold1, x
            x = xmma
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
