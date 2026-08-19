"""Tests sttopt.mma against Svanberg (1987)'s cantilever-beam toy problem (test
problem 1, section 6), which has a closed-form optimum -- ground truth independent of
the MATLAB port the rest of this codebase is validated against. See
resources/Svanberg1987_MethodOfMovingAsymptotes.pdf.
"""

import warnings

import numpy as np

import sttopt.mma as mma

# Test problem 1 (cantilever beam), eq. (22): minimize C1*sum(x), x_j > 0, subject to
# sum(coef_j / x_j^3) <= C2.
COEF = np.array([61.0, 37.0, 19.0, 7.0, 1.0])
C1 = 0.0624
C2 = 1.0


def _analytic_optimum():
    """Closed-form optimum of eq. (22): stationarity of the Lagrangian makes x_j
    proportional to coef_j^(1/4), and the constraint being active at the optimum
    fixes the overall scale.
    """
    root4 = COEF**0.25
    x_star = np.sum(root4) ** (1 / 3) * root4
    f0_star = C1 * np.sum(x_star)
    return x_star, f0_star


def _objective(x):
    return C1 * np.sum(x), np.full_like(x, C1)


def _constraint(x):
    g = np.sum(COEF / x**3) - C2
    dgdx = -3 * COEF / x**4
    return g, dgdx


def test_mmasub_cantilever_beam_converges_to_analytic_optimum():
    """Drives the production mmasub loop (its own default asymptote/move-limit
    heuristics -- not the paper's eq. (23)/(24) rules or its Fortran dual solver,
    which differ too much from this port's interior-point subsolv for a per-iteration
    trajectory match to be meaningful) on Svanberg's test problem 1, and checks it
    converges to the closed-form optimum.
    """
    x_star, f0_star = _analytic_optimum()

    n, m = 5, 1
    xmin = np.full(n, 0.1)
    xmax = np.full(n, 10.0)
    a0, a, c, d = 1.0, np.zeros(m), np.array([1000.0]), np.zeros(m)

    x = np.full(n, 5.0)
    xold1, xold2 = x.copy(), x.copy()
    low, upp = np.zeros(n), np.zeros(n)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for iteration in range(1, 21):
            f0val, df0dx = _objective(x)
            gval, dgdx = _constraint(x)
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
                np.array([gval]),
                dgdx[None, :],
                low,
                upp,
                a0,
                a,
                c,
                d,
            )
            xold2, xold1 = xold1, x
            x = xmma

    f0val, _ = _objective(x)
    gval, _ = _constraint(x)
    infeas = max(0.0, gval / C2)  # eq. (25)

    np.testing.assert_allclose(x, x_star, rtol=1e-4)
    np.testing.assert_allclose(f0val, f0_star, rtol=1e-4)
    assert infeas < 1e-6
