"""Tests for sttopt.fem: MATLAB-fixture regression tests, plus closed-form elasticity
patch tests that check the FEM assembly against elasticity theory directly rather than
against the MATLAB port (see conftest.py, conventions.md).
"""

import numpy as np
import pytest

import sttopt.fem as fem
from conftest import assert_close, load_fixture

# SIMP constants for the patch tests below; matches test_assemble_and_solve's convention.
EMIN, EMAX, PENAL = 1e-9, 1.0, 3


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
    F[2 * (nelx + 1) * (nely + 1) - 1] = (
        -1.0
    )  # MATLAB dof 2*(nelx+1)*(nely+1), 1-indexed
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
    K = fem.assemble_stiffness(
        fem.plane_stress_KE(0.3), xPhys, 0.0, 1.0, 3, edofMat, ndof
    )
    element = (
        i * nely + j
    )  # Fortran-order (column-major) element index per conventions.md
    touched_dofs = np.unique(K.nonzero()[0])
    assert np.array_equal(touched_dofs, np.sort(edofMat[element]))


# --- Closed-form patch tests -------------------------------------------------------
#
# A bilinear (Q4) element is exact for any constant-strain field, so a load case that
# produces uniform strain everywhere should match its closed-form elasticity solution to
# near machine precision. Unlike the MATLAB-fixture tests above, this doesn't depend on
# the MATLAB port being correct -- it catches a bug the port might share with the source.
#
# Node (row, col) sits at physical (x, y) = (col, -row): `element_dof_map`'s local node
# order is CCW read against y = -row, not row itself (checked directly against edofMat's
# node ordering, not assumed) -- so `-row`, not `row`, stands for the physical y
# coordinate throughout.


def _node_id(row: int, col: int, nely: int) -> int:
    """0-indexed global node number for grid position (row, col); matches `element_dof_map`."""
    return col * (nely + 1) + row


def _dofs(row: int, col: int, nely: int) -> tuple[int, int]:
    """0-indexed (x, y) global dof pair for grid position (row, col)."""
    n = _node_id(row, col, nely)
    return 2 * n, 2 * n + 1


def _add_edge_traction(F, nodes, traction):
    """Add consistent nodal forces for a uniform traction along ordered, unit-spaced edge nodes."""
    tx, ty = traction
    n = len(nodes)
    for i, node in enumerate(nodes):
        weight = 0.5 if i in (0, n - 1) else 1.0
        F[2 * node] += weight * tx
        F[2 * node + 1] += weight * ty


@pytest.mark.parametrize("nu", [0.0, 0.3])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_uniaxial_tension_patch(axis, nu):
    """Roller BCs plus a uniform edge traction should reproduce a bar-in-tension's affine
    displacement field: axial strain t/E, lateral (Poisson) strain -nu times that.
    """
    nelx, nely = 7, 5
    t = 1.0
    ndof = 2 * (nelx + 1) * (nely + 1)
    xPhys = np.ones((nely, nelx))
    KE = fem.plane_stress_KE(nu)
    edofMat = fem.element_dof_map(nelx, nely)
    K = fem.assemble_stiffness(KE, xPhys, EMIN, EMAX, PENAL, edofMat, ndof)

    # Rollers pin the x=0 and y=0 lines, matching the field's zeros there.
    fixeddofs = [_dofs(row, 0, nely)[0] for row in range(nely + 1)]
    fixeddofs += [_dofs(0, col, nely)[1] for col in range(nelx + 1)]
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    eps_axial = t / EMAX  # xPhys == 1 everywhere -> E == Emax regardless of penal
    eps_lateral = -nu * eps_axial
    F = np.zeros(ndof)
    if axis == "x":
        _add_edge_traction(
            F, [_node_id(row, nelx, nely) for row in range(nely + 1)], (t, 0.0)
        )
        eps_x, eps_y = eps_axial, eps_lateral
    else:
        _add_edge_traction(
            F, [_node_id(nely, col, nely) for col in range(nelx + 1)], (0.0, -t)
        )
        eps_x, eps_y = eps_lateral, eps_axial

    U = fem.solve_fe(K, F, freedofs)

    num_nodes = (nelx + 1) * (nely + 1)
    node_row = np.arange(num_nodes) % (nely + 1)
    node_col = np.arange(num_nodes) // (nely + 1)
    expected = np.zeros(ndof)
    expected[0::2] = eps_x * node_col
    expected[1::2] = eps_y * -node_row
    np.testing.assert_allclose(U, expected, atol=1e-10)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_uniaxial_tension_patch_graded_density(axis):
    """Same patch test, but with density graded along the load axis and uniform across it.

    nu=0 only: at nu!=0, a density gradient perpendicular to it needs nonzero shear strain
    to stay compatible at the interfaces (a real bimaterial effect), so there's no simple
    closed form there. At nu=0 the axes decouple and this reduces to a 1D bar-in-series:
    stress is uniform along the bar regardless of the modulus profile, so each element's own
    SIMP-interpolated modulus sets its local strain, and displacement is the running sum.
    """
    nelx, nely = 6, 4
    t = 1.0
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE = fem.plane_stress_KE(nu=0.0)
    edofMat = fem.element_dof_map(nelx, nely)

    fixeddofs = [_dofs(row, 0, nely)[0] for row in range(nely + 1)]
    fixeddofs += [_dofs(0, col, nely)[1] for col in range(nelx + 1)]
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    F = np.zeros(ndof)
    if axis == "x":
        density = np.array(
            [0.3, 0.55, 1.0, 0.45, 0.8, 0.2]
        )  # len == nelx, asymmetric, all > 0
        xPhys = np.tile(density[None, :], (nely, 1))
        _add_edge_traction(
            F, [_node_id(row, nelx, nely) for row in range(nely + 1)], (t, 0.0)
        )
    else:
        density = np.array([0.3, 0.55, 1.0, 0.45])  # len == nely, asymmetric, all > 0
        xPhys = np.tile(density[:, None], (1, nelx))
        _add_edge_traction(
            F, [_node_id(nely, col, nely) for col in range(nelx + 1)], (0.0, -t)
        )

    K = fem.assemble_stiffness(KE, xPhys, EMIN, EMAX, PENAL, edofMat, ndof)
    U = fem.solve_fe(K, F, freedofs)

    E_elem = EMIN + density**PENAL * (EMAX - EMIN)
    disp_at_boundary = np.concatenate([[0.0], np.cumsum(t / E_elem)])

    num_nodes = (nelx + 1) * (nely + 1)
    node_row = np.arange(num_nodes) % (nely + 1)
    node_col = np.arange(num_nodes) // (nely + 1)
    expected = np.zeros(ndof)
    if axis == "x":
        expected[0::2] = disp_at_boundary[node_col]
    else:
        expected[1::2] = -disp_at_boundary[node_row]
    np.testing.assert_allclose(U, expected, atol=1e-10)
