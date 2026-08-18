"""Tests for sttopt.compliance against MATLAB fixtures and finite-difference checks.

See conftest.py/conventions.md for fixture format and tolerance policy.
"""

import numpy as np

import sttopt.compliance as compliance
import sttopt.fem as fem
import sttopt.gravity as gravity
from conftest import assert_close, load_fixture


def _build_point_load_problem(nelx, nely):
    """F/freedofs for the fixed point-load problem, matching test_fem.py/generate_fixtures.m."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    F[2 * (nelx + 1) * (nely + 1) - 1] = -1.0
    fixeddofs = np.arange(2 * (nely + 1))
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    return F, freedofs, ndof


def test_whole_compliance_matches_fixture():
    fx = load_fixture("compliance")
    e2e = load_fixture("e2e")
    nelx, nely, nloop = int(fx["nelx"]), int(fx["nely"]), e2e["xPhys_traj"].shape[2] - 1
    Emin, Emax, penal = 1e-9, 1.0, 3

    KE = fem.plane_stress_KE(nu=0.3)
    edofMat = fem.element_dof_map(nelx, nely)
    F, freedofs, ndof = _build_point_load_problem(nelx, nely)

    for k in range(nloop):
        xPhys = e2e["xPhys_traj"][:, :, k]
        c, dcx = compliance.whole_compliance(
            xPhys, KE, edofMat, Emin, Emax, penal, freedofs, F, ndof
        )
        assert_close(c, fx["c_whole_all"][k], tier="solved")
        assert_close(dcx, fx["dcx_whole_all"][:, :, k], tier="solved")


def test_gravity_compliance_matches_fixture():
    fx = load_fixture("compliance")
    e2e = load_fixture("e2e")
    grav = load_fixture("gravity")
    nelx, nely, nStage = int(fx["nelx"]), int(fx["nely"]), int(fx["nStage"])
    nloop = e2e["xPhys_traj"].shape[2] - 1
    Emin, Emax, penal = 1e-9, 1.0, 3
    lam = 10.0  # rou=10 fixed for loop=1..3: mod(loop,30)==0 never triggers (see generate_fixtures.m)

    KE = fem.plane_stress_KE(nu=0.3)
    edofMat = fem.element_dof_map(nelx, nely)
    _, freedofs, ndof = _build_point_load_problem(nelx, nely)
    C = grav["C"]

    tP = np.linspace(0, 1, nStage + 1)

    for k in range(nloop):
        xPhys = e2e["xPhys_traj"][:, :, k]
        tPhys = e2e["tPhys_traj"][:, :, k]
        for i in range(nStage):
            ti = tP[i + 1]
            c, dcx, dct = compliance.gravity_compliance(
                xPhys, tPhys, KE, edofMat, Emin, Emax, penal, ti, C, lam, freedofs, ndof
            )
            assert_close(c, fx["c_grav_all"][k, i], tier="solved")
            assert_close(dcx, fx["dcx_grav_all"][:, i, k], tier="solved")
            assert_close(dct, fx["dct_grav_all"][:, i, k], tier="solved")


# --- Finite-difference internal-consistency checks (pure Python, no MATLAB fixture) ---

def _random_field(rng, nely, nelx):
    """Asymmetric density/time fields, kept away from the [0,1] boundary (near-void/near-solid
    densities make the stiffness matrix ill-conditioned; see `well_conditioned` below for the
    stronger safeguard the gravity FD test needs)."""
    return rng.uniform(0.2, 0.8, size=(nely, nelx))


def test_whole_compliance_fd_dcx():
    nelx, nely = 6, 4
    Emin, Emax, penal = 1e-9, 1.0, 3
    KE = fem.plane_stress_KE(nu=0.3)
    edofMat = fem.element_dof_map(nelx, nely)
    F, freedofs, ndof = _build_point_load_problem(nelx, nely)
    h = 1e-6

    rng = np.random.default_rng(0)
    for seed in range(5):
        xPhys = _random_field(rng, nely, nelx)
        _, dcx = compliance.whole_compliance(
            xPhys, KE, edofMat, Emin, Emax, penal, freedofs, F, ndof
        )
        fd = np.zeros_like(xPhys)
        for j in range(nely):
            for i in range(nelx):
                xp = xPhys.copy(); xp[j, i] += h
                xm = xPhys.copy(); xm[j, i] -= h
                cp, _ = compliance.whole_compliance(xp, KE, edofMat, Emin, Emax, penal, freedofs, F, ndof)
                cm, _ = compliance.whole_compliance(xm, KE, edofMat, Emin, Emax, penal, freedofs, F, ndof)
                fd[j, i] = (cp - cm) / (2 * h)
        np.testing.assert_allclose(dcx, fd, rtol=1e-4, atol=1e-6)


def test_gravity_compliance_fd_dcx_and_dct():
    nelx, nely = 6, 4
    Emin, Emax, penal = 1e-9, 1.0, 3
    lam = 10.0
    ti = 0.5
    KE = fem.plane_stress_KE(nu=0.3)
    edofMat = fem.element_dof_map(nelx, nely)
    _, freedofs, ndof = _build_point_load_problem(nelx, nely)

    C = gravity.gravity_load_matrix(nelx, nely)
    h = 1e-4

    # Self-weight-only loading (no external point load) on a small random mesh can
    # land near a mechanism -- K_free's condition number spans 1e4 to 1e8 across
    # random draws below, and FD noise blows up (verified by an h-convergence sweep)
    # long before the analytic gradient does. Reject ill-conditioned draws rather than
    # loosen tolerances to paper over them: this is a numerical-conditioning artifact
    # of the FD probe, not evidence about the analytic formula.
    def well_conditioned(xPhys, tPhys):
        ft = compliance.time_mask(tPhys, ti, lam)
        K = fem.assemble_stiffness(KE, xPhys * ft, Emin, Emax, penal, edofMat, ndof)
        Kfree = K[np.ix_(freedofs, freedofs)].toarray()
        return np.linalg.cond(Kfree) < 1e5

    rng = np.random.default_rng(1)
    accepted = 0
    tries = 0
    max_tries = 200
    while accepted < 5:
        assert tries < max_tries, f"couldn't find 5 well-conditioned draws in {max_tries} tries"
        xPhys = _random_field(rng, nely, nelx)
        tPhys = _random_field(rng, nely, nelx)
        tries += 1
        if not well_conditioned(xPhys, tPhys):
            continue
        accepted += 1

        _, dcx, dct = compliance.gravity_compliance(
            xPhys, tPhys, KE, edofMat, Emin, Emax, penal, ti, C, lam, freedofs, ndof
        )

        fd_x = np.zeros(nely * nelx)
        fd_t = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            j, i = e % nely, e // nely  # Fortran-order element -> (row, col), per conventions.md

            xp = xPhys.copy(); xp[j, i] += h
            xm = xPhys.copy(); xm[j, i] -= h
            cp, _, _ = compliance.gravity_compliance(xp, tPhys, KE, edofMat, Emin, Emax, penal, ti, C, lam, freedofs, ndof)
            cm, _, _ = compliance.gravity_compliance(xm, tPhys, KE, edofMat, Emin, Emax, penal, ti, C, lam, freedofs, ndof)
            fd_x[e] = (cp - cm) / (2 * h)

            tp = tPhys.copy(); tp[j, i] += h
            tm = tPhys.copy(); tm[j, i] -= h
            cp, _, _ = compliance.gravity_compliance(xPhys, tp, KE, edofMat, Emin, Emax, penal, ti, C, lam, freedofs, ndof)
            cm, _, _ = compliance.gravity_compliance(xPhys, tm, KE, edofMat, Emin, Emax, penal, ti, C, lam, freedofs, ndof)
            fd_t[e] = (cp - cm) / (2 * h)

        np.testing.assert_allclose(dcx, fd_x, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(dct, fd_t, rtol=1e-3, atol=1e-6)


def test_time_mask_derivative_matches_fd():
    rng = np.random.default_rng(2)
    tPhys = rng.uniform(0.05, 0.95, size=(4, 6))
    ti, lam = 0.4, 10.0
    h = 1e-6
    analytic = compliance.time_mask_derivative(tPhys, ti, lam)
    fd = (compliance.time_mask(tPhys + h, ti, lam) - compliance.time_mask(tPhys - h, ti, lam)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=1e-8)
