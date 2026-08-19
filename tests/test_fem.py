"""Tests for sttopt.fem against MATLAB fixtures (see conftest.py, conventions.md)."""

import numpy as np

import sttopt.fem as fem
from conftest import assert_close, load_fixture


def test_plane_stress_KE():
    fx = load_fixture("fem_setup")
    KE = fem.plane_stress_KE(nu=0.3)
    assert KE.shape == fx["KE"].shape
    assert_close(KE, fx["KE"], tier="algebraic")


def test_element_dof_map():
    fx = load_fixture("fem_setup")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])
    edofMat = fem.element_dof_map(nelx, nely)
    expected = fx["edofMat"].astype(np.int64) - 1  # MATLAB 1-indexed -> 0-indexed
    assert edofMat.shape == expected.shape
    assert_close(edofMat, expected, tier="algebraic")


def test_assemble_and_solve():
    fx = load_fixture("fem_solve")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])
    xPhys0 = fx["xPhys0"]
    U0 = fx["U0"]
    assert xPhys0.shape == (nely, nelx)

    # Fixed problem constants used throughout generate_fixtures.m (not saved to any
    # fixture); mirrors generate_fixtures.m lines ~66 (Emin/Emax/penal) and ~184-188
    # (F/freedofs), converted from MATLAB's 1-indexing to Python's 0-indexing.
    Emin, Emax, penal = 1e-9, 1.0, 3
    ndof = 2 * (nelx + 1) * (nely + 1)

    F = np.zeros(ndof)
    F[2 * (nelx + 1) * (nely + 1) - 1] = -1.0  # MATLAB dof 2*(nelx+1)*(nely+1), 1-indexed
    fixeddofs = np.arange(2 * (nely + 1))  # MATLAB 1:2*(nely+1), 1-indexed -> 0-indexed
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    KE = fem.plane_stress_KE(nu=0.3)
    edofMat = fem.element_dof_map(nelx, nely)
    K = fem.assemble_stiffness(KE, xPhys0, Emin, Emax, penal, edofMat, ndof)
    U = fem.solve_fe(K, F, freedofs)

    assert U.shape == U0.shape
    assert_close(U, U0, tier="solved")


def test_assemble_respects_column_major_element_order():
    # fem_solve.mat's xPhys0 is uniform (repmat with tanh(0)=0 offset cancels), so the
    # U comparison above can't distinguish order='F' from order='C' -- conventions.md
    # calls this out explicitly. Pin a single asymmetric density element instead.
    nelx, nely = 7, 5
    edofMat = fem.element_dof_map(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)
    xPhys = np.zeros((nely, nelx))
    i, j = 4, 1  # x-index, y-index; asymmetric so F-order and C-order disagree
    xPhys[j, i] = 1.0
    K = fem.assemble_stiffness(fem.plane_stress_KE(0.3), xPhys, 0.0, 1.0, 3, edofMat, ndof)
    element = i * nely + j  # Fortran-order (column-major) element index per conventions.md
    touched_dofs = np.unique(K.nonzero()[0])
    assert np.array_equal(touched_dofs, np.sort(edofMat[element]))
