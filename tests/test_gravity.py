"""Tests for sttopt.gravity against MATLAB fixtures (see conftest.py, conventions.md)."""

import numpy as np

import sttopt.fem as fem
import sttopt.gravity as gravity
from conftest import assert_close, load_fixture


def test_gravity_load_matrix():
    fx = load_fixture("gravity")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])

    C = gravity.gravity_load_matrix(nelx, nely)
    # sparse fixture, small problem size -- densify for comparison only
    expected = fx["C"].toarray()

    assert C.shape == expected.shape == ((nelx + 1) * (nely + 1), nelx * nely)
    assert_close(C.toarray(), expected, tier="algebraic")


# --- Closed-form patch test ---------------------------------------------------------
#
# For a bilinear Q4 element, integrating each corner's shape function over the unit
# square gives exactly 1/4 -- so `fe/4` (see gravity.py's docstring) is the *consistent*
# (work-equivalent) nodal load for a uniform body force, not a lumped approximation.
# That makes a column hanging under its own weight (fixed at the top, free at the
# bottom) checkable against elasticity theory: at nu=0 axial and lateral strain
# decouple, so the exact continuum solution is a 1D bar under self-weight (ux=0
# everywhere, uy quadratic in depth alone) -- and linear elements with consistent loads
# are nodally exact for that constant-body-force bar problem, so the FEM solution
# should match the closed form at every node, not just converge to it (verified
# empirically before writing this: max error ~1e-14 across several mesh shapes at
# nu=0, and it does NOT hold at nu=0.3, confirming the decoupling argument is load-
# bearing here).

EMIN, EMAX, PENAL = 1e-9, 1.0, 3


def test_gravity_self_weight_column_patch():
    # asymmetric mesh so a row/col transposition wouldn't slip through
    nelx, nely = 3, 6
    ndof = 2 * (nelx + 1) * (nely + 1)
    xPhys = np.ones((nely, nelx))
    KE = fem.plane_stress_KE(nu=0.0)
    edofMat = fem.element_dof_map(nelx, nely)
    K = fem.assemble_stiffness(KE, xPhys, EMIN, EMAX, PENAL, edofMat, ndof)

    C = gravity.gravity_load_matrix(nelx, nely)
    F = np.zeros(ndof)
    # sign/scatter convention matches compliance.gravity_compliance: gravity acts in -y
    F[1::2] = -C @ xPhys.flatten(order="F")

    num_nodes = (nelx + 1) * (nely + 1)
    node_row = np.arange(num_nodes) % (nely + 1)
    # row=0 is the top, per test_fem.py's convention
    top_nodes = np.where(node_row == 0)[0]
    fixeddofs = np.concatenate([2 * top_nodes, 2 * top_nodes + 1])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    U = fem.solve_fe(K, F, freedofs)

    # Bar-under-self-weight closed form: axial force at depth `row` is the weight of
    # everything below it, so strain is linear in row and displacement is quadratic.
    # fe = 1/(nelx*nely) is the total domain weight spread evenly over its elements.
    fe = 1 / (nelx * nely)
    expected = np.zeros(ndof)
    expected[1::2] = -(fe / EMAX) * (nely * node_row - node_row**2 / 2)
    np.testing.assert_allclose(U, expected, atol=1e-10)

    # Cheap magnitude cross-check, independent of the quadratic-profile derivation
    # above: reactions at the fixed top row must balance the domain's total weight.
    reactions = K @ U - F
    assert np.isclose(reactions[2 * top_nodes + 1].sum(), 1.0)
