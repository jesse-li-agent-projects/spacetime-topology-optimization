"""Tests for sttopt.fem: MATLAB-fixture regression tests, plus closed-form elasticity
patch tests that check the FEM assembly against elasticity theory directly rather than
against the MATLAB port (see conftest.py, conventions.md).
"""

import numpy as np
import pytest

import sttopt.fem as fem
from conftest import (
    assert_close,
    fixture_dof_perm,
    load_fixture,
    node_positions,
    point_load_problem,
    reindex_fixture,
)

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
    # The fixture is F-order on both axes: its rows are F-order elements, and the dof
    # numbers it stores are F-order node numbers. Reindex the rows, relabel the values.
    expected = reindex_fixture(expected, nelx, nely, axis=0)
    expected = fixture_dof_perm(nelx, nely)[expected]
    assert edofMat.shape == expected.shape
    assert_close(edofMat, expected, tier="algebraic")


def test_node_grid():
    """Pins the node numbering itself, which the patch tests below take as given."""
    nelx, nely = 7, 5
    nodes = fem.node_grid(nelx, nely)
    assert nodes.shape == (nely + 1, nelx + 1)
    for row in range(nely + 1):
        for col in range(nelx + 1):
            assert nodes[row, col] == row * (nelx + 1) + col


@pytest.mark.parametrize("nelx, nely", [(1, 1), (3, 2), (7, 5), (4, 9)])
def test_element_dof_map_corner_geometry(nelx, nely):
    """The fixture above pins one mesh shape only; restate the map's geometry -- each
    element's dofs are the (x, y) pairs of its 4 corner nodes -- across several shapes.
    """
    nodes = fem.node_grid(nelx, nely)
    edofMat = fem.element_dof_map(nelx, nely)
    assert edofMat.shape == (nelx * nely, 8)
    for e in range(nelx * nely):
        row, col = e // nelx, e % nelx
        corners = [
            (row + 1, col),  # bottom-left, per the local node order
            (row + 1, col + 1),
            (row, col + 1),
            (row, col),
        ]
        expected = _dofs([nodes[c] for c in corners])
        assert list(edofMat[e]) == list(expected)


def test_assemble_and_solve():
    fx = load_fixture("fem_solve")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])
    xPhys0 = fx["xPhys0"]
    U0 = fx["U0"]
    assert xPhys0.shape == (nely, nelx)

    # Fixed problem constants used throughout generate_fixtures.m (not saved to any
    # fixture); mirrors generate_fixtures.m line ~66.
    Emin, Emax, penal = 1e-9, 1.0, 3
    F, freedofs, ndof = point_load_problem(nelx, nely)

    KE = fem.plane_stress_KE(nu=0.3)
    edofMat = fem.element_dof_map(nelx, nely)
    K = fem.assemble_stiffness(KE, xPhys0, Emin, Emax, penal, edofMat, ndof)
    U = fem.solve_fe(K, F, freedofs)

    # U0 is indexed by the fixture's F-order node numbering; relabel to sttopt's.
    expected = np.empty_like(U0)
    expected[fixture_dof_perm(nelx, nely)] = U0
    assert U.shape == expected.shape
    assert_close(U, expected, tier="solved")


def test_assemble_respects_row_major_element_order():
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
    # C-order (row-major) element index per conventions.md
    element = j * nelx + i
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


def _x_dofs(nodes) -> np.ndarray:
    """0-indexed x dofs of the given nodes."""
    return 2 * np.asarray(nodes)


def _y_dofs(nodes) -> np.ndarray:
    """0-indexed y dofs of the given nodes."""
    return 2 * np.asarray(nodes) + 1


def _dofs(nodes) -> np.ndarray:
    """Both (x, y) dofs of the given nodes, interleaved per node."""
    return np.stack([_x_dofs(nodes), _y_dofs(nodes)], axis=-1).ravel()


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
    nodes = fem.node_grid(nelx, nely)
    fixeddofs = np.concatenate([_x_dofs(nodes[:, 0]), _y_dofs(nodes[0, :])])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    eps_axial = t / EMAX  # xPhys == 1 everywhere -> E == Emax regardless of penal
    eps_lateral = -nu * eps_axial
    F = np.zeros(ndof)
    if axis == "x":
        _add_edge_traction(F, nodes[:, -1], (t, 0.0))
        eps_x, eps_y = eps_axial, eps_lateral
    else:
        _add_edge_traction(F, nodes[-1, :], (0.0, -t))
        eps_x, eps_y = eps_lateral, eps_axial

    U = fem.solve_fe(K, F, freedofs)

    node_row, node_col = node_positions(nelx, nely)
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

    nodes = fem.node_grid(nelx, nely)
    fixeddofs = np.concatenate([_x_dofs(nodes[:, 0]), _y_dofs(nodes[0, :])])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    F = np.zeros(ndof)
    if axis == "x":
        # len == nelx, asymmetric, all > 0
        density = np.array([0.3, 0.55, 1.0, 0.45, 0.8, 0.2])
        xPhys = np.tile(density[None, :], (nely, 1))
        _add_edge_traction(F, nodes[:, -1], (t, 0.0))
    else:
        density = np.array([0.3, 0.55, 1.0, 0.45])  # len == nely, asymmetric, all > 0
        xPhys = np.tile(density[:, None], (1, nelx))
        _add_edge_traction(F, nodes[-1, :], (0.0, -t))

    K = fem.assemble_stiffness(KE, xPhys, EMIN, EMAX, PENAL, edofMat, ndof)
    U = fem.solve_fe(K, F, freedofs)

    E_elem = EMIN + density**PENAL * (EMAX - EMIN)
    disp_at_boundary = np.concatenate([[0.0], np.cumsum(t / E_elem)])

    node_row, node_col = node_positions(nelx, nely)
    expected = np.zeros(ndof)
    if axis == "x":
        expected[0::2] = disp_at_boundary[node_col]
    else:
        expected[1::2] = -disp_at_boundary[node_row]
    np.testing.assert_allclose(U, expected, atol=1e-10)


@pytest.mark.parametrize("nu", [0.0, 0.3, 0.45])
def test_pure_shear_patch(nu):
    """Rollers plus edge shear tractions on all four sides should reproduce simple shear:
    u = -gamma*row (i.e. gamma*y), v = 0, with gamma = tau/G and G = E/(2*(1+nu)) the
    plane-stress shear modulus. Unlike uniaxial tension this exercises the shear
    ((1-nu)/2, off-diagonal) terms of `plane_stress_KE` -- the constant strain here is
    entirely eps_xy, whereas tension's is entirely eps_x/eps_y.

    A uniform shear stress state has nonzero traction on all four edges, not just one (its
    stress tensor is off-diagonal, so every face normal picks up a component). u = v = 0
    along the whole y=0 (row=0) line, though, so that edge's tractions are absorbed by
    rollers there instead of applied explicitly; the other three edges get explicit
    consistent-nodal shear tractions.
    """
    nelx, nely = 7, 5
    tau = 1.0
    ndof = 2 * (nelx + 1) * (nely + 1)
    xPhys = np.ones((nely, nelx))
    KE = fem.plane_stress_KE(nu)
    edofMat = fem.element_dof_map(nelx, nely)
    K = fem.assemble_stiffness(KE, xPhys, EMIN, EMAX, PENAL, edofMat, ndof)

    # Rollers: both dofs = 0 along row=0 (matches the field's zeros there), plus v = 0
    # along the other three edges (matches v == 0 everywhere).
    nodes = fem.node_grid(nelx, nely)
    fixeddofs = np.concatenate(
        [
            _dofs(nodes[0, :]),
            _y_dofs(nodes[-1, :]),
            _y_dofs(nodes[:, 0]),
            _y_dofs(nodes[:, -1]),
        ]
    )
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    F = np.zeros(ndof)
    _add_edge_traction(F, nodes[-1, :], (-tau, 0.0))
    U = fem.solve_fe(K, F, freedofs)

    G = EMAX / (2 * (1 + nu))  # xPhys == 1 everywhere -> E == Emax regardless of penal
    gamma = tau / G

    node_row, _ = node_positions(nelx, nely)
    expected = np.zeros(ndof)
    expected[0::2] = -gamma * node_row
    np.testing.assert_allclose(U, expected, atol=1e-10)
