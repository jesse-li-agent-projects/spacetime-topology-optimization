"""Tests for sttopt.mma against a MATLAB fixture (see conftest.py, conventions.md)."""

import numpy as np

import sttopt.mma as mma
from conftest import assert_close, load_fixture

# a0 = 1; a = zeros(m,1); c_ = ones(m,1)*2500; d = zeros(m,1) -- from
# generate_fixtures.m lines ~455-457, not saved to the fixture.
A0 = 1.0
C_VAL = 2500.0


def test_mmasub_iteration1():
    fx = load_fixture("mma")
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
        # recomputes them from scratch) -- the fixture stores them as scalar 0.
        np.zeros(n),
        np.zeros(n),
        A0,
        a,
        c,
        d,
    )

    assert xmma.shape == fx["xmma_all"][:, 0].shape
    # "algebraic" tier is a deliberate carve-out from conventions.md's letter (this *is*
    # downstream of subsolv's Newton iteration, which would nominally route it to
    # "solved"): mmasub/subsolv are a deterministic algorithm run on identical inputs in
    # both languages, not two different solvers of the same problem, so near-machine-
    # precision agreement is the right expectation and what's actually observed
    # (measured ~1e-15 for xmma/lam here, not just "passes at 1e-10").
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

    xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma = mma.subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
    )

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

    # KKT residuals, evaluated the same way subsolv's own convergence check does. The
    # outer loop's last pass runs at epsi = 10*epsimin (the barrier value from just
    # before the epsi > epsimin test failed), so residuals/complementarity products
    # are bounded by that, not by zero.
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
