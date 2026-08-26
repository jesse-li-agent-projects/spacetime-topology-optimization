"""Tests for sttopt.mma against a golden-regression fixture (see conftest.py,
conventions.md)."""

import numpy as np

import sttopt.mma as mma
from conftest import assert_close, load_fixture_npz

# a0 = 1; a = zeros(m,1); c_ = ones(m,1)*2500; d = zeros(m,1) -- from
# generate_fixtures.py, not saved to the fixture.
A0 = 1.0
C_VAL = 2500.0


def test_mmasub_iteration1():
    fx = load_fixture_npz("mma")
    m, n = int(fx["m"]), int(fx["n"])
    a = np.zeros(m)
    c = np.ones(m) * C_VAL
    d = np.zeros(m)

    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mma.mmasub(
        m,
        n,
        1,
        fx["xval_1"],
        fx["xmin_1"],
        fx["xmax_1"],
        fx["xold1_1"],
        fx["xold2_1"],
        float(fx["f0val_1"]),
        fx["df0dx_1"],
        fx["fval_1"],
        fx["dfdx_1"],
        # low_1/upp_1 are unused at iteration 1 (see mma.mmasub: `iteration < 2.5`
        # recomputes them from scratch) -- the fixture stores them as zeros.
        np.zeros(n),
        np.zeros(n),
        A0,
        a,
        c,
        d,
    )

    assert xmma.shape == fx["xmma_all"][:, 0].shape
    assert_close(xmma, fx["xmma_all"][:, 0], tier="algebraic")
    assert_close(low, fx["low_all"][:, 0], tier="algebraic")
    assert_close(upp, fx["upp_all"][:, 0], tier="algebraic")
    assert_close(lam, fx["lam_all"][:, 0], tier="algebraic")


def test_mmasub_asymptote_update_branch():
    """Exercises the `iteration >= 2.5` asymptote branch (factor/asyincr/asydecr and
    the four min/max clamps in mma.mmasub), which iteration 1 never reaches.

    A fixture-chained version of this test (feeding iteration 2/3 state reconstructed
    from xmma_all/low_all/upp_all) was tried and dropped: xval at iteration 2 depends
    on this port's own iteration-1 xmma, which agrees with MATLAB's only to ~1e-15 (not
    bit-exactly -- see conventions.md's tolerance policy) -- and a handful of elements
    have `zzz = (xval-xold1)*(xold1-xold2)` sitting right at 0, where that tiny
    difference flips `factor` between asyincr/1.0/asydecr (a 20-30% swing), not a
    smoothly growing error. That's a real trajectory-divergence hazard for any future
    end-to-end MMA-loop comparison (Phase 8) to be aware of, but it makes a fixture
    comparison at iteration 2+ fail for reasons unrelated to port correctness.

    Instead, this constructs a small synthetic problem hand-picked to exercise every
    branch (zzz > 0, zzz < 0, zzz == 0 exactly, and each of the lowmin/lowmax/uppmin/
    uppmax clamps) unambiguously, with expected low/upp values computed independently
    of mma.py from the same formulas in mmasub.m. No MATLAB ground truth is involved.
    """
    n, m = 6, 1
    xval = np.full(n, 5.0)
    xmin = np.zeros(n)
    xmax = np.full(n, 10.0)
    # Per-index scenario: 0 zzz>0, 1 zzz<0, 2 zzz==0, 3 lowmin clamp, 4 uppmax clamp,
    # 5 lowmax and uppmin clamp.
    xold1 = np.array([4.0, 4.0, 5.0, 4.0, 4.0, 4.0])
    xold2 = np.array([3.0, 5.0, 2.0, 3.0, 3.0, 4.0])
    low = np.array([2.0, 2.0, 2.0, -1000.0, 2.0, 4.8])
    upp = np.array([8.0, 8.0, 8.0, 8.0, 1000.0, 4.0])
    expected_low = np.array([2.6, 3.6, 2.0, -95.0, 2.6, 4.9])
    expected_upp = np.array([9.8, 7.8, 8.0, 9.8, 105.0, 5.1])

    _, _, _, _, _, _, _, _, _, low_out, upp_out = mma.mmasub(
        m,
        n,
        3,
        xval,
        xmin,
        xmax,
        xold1,
        xold2,
        0.0,
        np.zeros(n),
        np.zeros(m),
        np.zeros((m, n)),
        low,
        upp,
        A0,
        np.zeros(m),
        np.ones(m) * C_VAL,
        np.zeros(m),
    )
    assert_close(low_out, expected_low, tier="algebraic")
    assert_close(upp_out, expected_upp, tier="algebraic")


def _assert_subsolv_kkt(
    solution, *, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, epsimin
):
    """Assert a `subsolv` return satisfies the relaxed KKT system it solves.

    `subsolv` returns the solution of the *log-barrier subproblem* at the last barrier
    parameter it ran, not of the unrelaxed problem: its outer loop stops once
    `epsi <= epsimin`, so the final pass ran at `epsi = 10*epsimin`. Residuals are
    therefore bounded by that value and complementarity products equal it rather than
    zero -- exactly the conditions `subsolv`'s own inner convergence test uses. This
    checks the returned point against those conditions from the MMA subproblem's own
    definition (Svanberg's dual/primal-dual formulation), independent of any fixture.
    """
    xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma = solution

    for arr in (xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma):
        assert np.all(np.isfinite(arr))
    assert np.all(xmma > alfa) and np.all(xmma < beta)
    assert np.all(ymma >= 0)
    assert zmma >= 0
    assert np.all(lamma >= 0)
    assert (
        np.all(xsimma >= 0)
        and np.all(etamma >= 0)
        and np.all(mumma >= 0)
        and zetmma >= 0
    )

    epsi_final = 10 * epsimin
    ux1, xl1 = upp - xmma, xmma - low
    plam, qlam = p0 + P.T @ lamma, q0 + Q.T @ lamma
    rex = plam / ux1**2 - qlam / xl1**2 - xsimma + etamma
    rey = c + d * ymma - mumma - lamma
    rez = a0 - zetmma - a @ lamma
    relam = P @ (1 / ux1) + Q @ (1 / xl1) - a * zmma - ymma + smma - b
    tol = 10 * epsi_final
    for r in (rex, rey, np.array([rez]), relam):
        assert np.max(np.abs(r)) < tol
    complementarity = (
        xsimma * (xmma - alfa),
        etamma * (beta - xmma),
        mumma * ymma,
        lamma * smma,
        np.array([zetmma * zmma]),
    )
    for prod in complementarity:
        assert np.all(np.abs(prod - epsi_final) < tol)


def test_subsolv_m_lt_n_kkt():
    """KKT check for subsolv's `m < n` branch (primal-space elimination) -- the branch
    the real space-time problem always takes (m ~ 17 vs. n = 2*nelx*nely), and the one
    every fixture drives, but only ever *through* a MATLAB trajectory comparison. Nothing
    outside those fixtures says the point it returns actually solves the subproblem.

    The two branches share the outer Newton loop but assemble and factor entirely
    different systems, so passing on `m >= n` says nothing here. Nonzero `a`/`d` keep the
    z- and y-coupled terms live: the production loop passes `a = d = 0`, which zeroes
    them and would hide a sign or coefficient error in any of them.
    """
    n, m = 8, 3
    rng = np.random.default_rng(7)
    low = np.full(n, -1.0)
    upp = np.full(n, 1.0)
    alfa = np.full(n, -0.9)
    beta = np.full(n, 0.9)
    p0 = rng.uniform(0.5, 2.0, n)
    q0 = rng.uniform(0.5, 2.0, n)
    P = rng.uniform(0.1, 1.0, (m, n))
    Q = rng.uniform(0.1, 1.0, (m, n))
    a0 = 1.0
    a = np.array([0.3, 0.1, 0.2])
    # Below each row's value at x = 0 (9.66, 7.97, 8.16), so all three constraints
    # actually bind -- with slack bounds every lam collapses to ~0 and the residual
    # checks below pass without the dual terms ever being exercised.
    b = np.array([8.0, 7.0, 7.0])
    c = np.full(m, 10.0)
    d = np.array([0.5, 1.0, 0.25])
    epsimin = 1e-7

    solution = mma.subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
    )
    _assert_subsolv_kkt(
        solution,
        low=low,
        upp=upp,
        alfa=alfa,
        beta=beta,
        p0=p0,
        q0=q0,
        P=P,
        Q=Q,
        a0=a0,
        a=a,
        b=b,
        c=c,
        d=d,
        epsimin=epsimin,
    )
    # Non-vacuity: a solve that parked every dual at zero, or left every primal at the
    # midpoint, would satisfy the residuals above without exercising the branch.
    xmma, ymma = solution[0], solution[1]
    lamma = solution[3]
    assert np.max(lamma) > 1e-3
    assert np.max(np.abs(xmma - 0.5 * (alfa + beta))) > 1e-3
    assert np.all(np.isfinite(ymma))


def test_subsolv_m_gt_n_smoke():
    """Smoke test for subsolv's `m >= n` branch (dual elimination), which no fixture
    exercises -- the real problem always has m << n (see mma.py's comment on this
    branch). No ground truth is available, so this checks that the returned point
    satisfies the KKT stationarity/feasibility/complementarity conditions to within
    the barrier tolerance the solve stopped at, not that it matches MATLAB.
    """
    n, m = 2, 3
    low = np.array([-1.0, -1.0])
    upp = np.array([1.0, 1.0])
    alfa = np.array([-0.9, -0.9])
    beta = np.array([0.9, 0.9])
    p0 = np.array([1.0, 2.0])
    q0 = np.array([2.0, 1.0])
    P = np.array([[1.0, 0.5], [0.2, 1.0], [0.3, 0.3]])
    Q = np.array([[0.5, 1.0], [1.0, 0.2], [0.3, 0.3]])
    a0 = 1.0
    # Nonzero, so this actually exercises the a-coupled terms (AA's border, axz, azz,
    # bz, and dlam's -dz*(a/diaglamyi) term) -- a=0 (as used elsewhere in this file)
    # zeroes all of them and would let a sign/coefficient bug in any of those terms
    # through undetected. See review discussion, PR history.
    a = np.array([0.3, 0.1, 0.2])
    b = np.array([1.0, 1.0, 1.0])
    c = np.ones(m) * 10.0
    d = np.zeros(m)
    epsimin = 1e-7

    solution = mma.subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
    )
    _assert_subsolv_kkt(
        solution,
        low=low,
        upp=upp,
        alfa=alfa,
        beta=beta,
        p0=p0,
        q0=q0,
        P=P,
        Q=Q,
        a0=a0,
        a=a,
        b=b,
        c=c,
        d=d,
        epsimin=epsimin,
    )
