"""Tests for sttopt.fem: golden-regression fixture tests for the mesh/element setup
`plane_stress_KE`/`node_grid`/`element_dof_map` (see conftest.py, conventions.md).
`assemble_stiffness`/`solve_fe`'s tests moved to `tests/reference/test_fem.py`
alongside the functions themselves (Phase 3.7, `plans/torch_port_review_followup.md`).
"""

import numpy as np
import pytest

import sttopt.fem as fem
from conftest import assert_close, load_fixture_npz


def test_plane_stress_KE():
    fx = load_fixture_npz("fem_setup")
    KE = fem.plane_stress_KE(nu=0.3)
    assert KE.shape == fx["KE"].shape
    assert_close(KE, fx["KE"], tier="algebraic")


def test_element_dof_map():
    fx = load_fixture_npz("fem_setup")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])
    edofMat = fem.element_dof_map(nelx, nely)
    assert edofMat.shape == fx["edofMat"].shape
    assert_close(edofMat, fx["edofMat"], tier="algebraic")


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


def _x_dofs(nodes):
    """0-indexed x dofs of the given nodes."""
    return 2 * np.asarray(nodes)


def _y_dofs(nodes):
    """0-indexed y dofs of the given nodes."""
    return 2 * np.asarray(nodes) + 1


def _dofs(nodes):
    """Both (x, y) dofs of the given nodes, interleaved per node."""
    return np.stack([_x_dofs(nodes), _y_dofs(nodes)], axis=-1).ravel()


@pytest.mark.parametrize("length", [0.0, 0.25, 1.0, 1.5, 3.0, 5.0])
def test_edge_load_shares_carry_the_whole_force(length):
    assert fem.edge_load_shares(6, length).sum() == pytest.approx(1.0, rel=1e-14)


def test_edge_load_shares_integrate_the_hat_functions():
    """Over 1.5 elements from node 0: node 0's half hat whole, node 1's rising half and
    `(2 - y)` up to 1.5, node 2's `(y - 1)` up to 1.5, each over the span."""
    np.testing.assert_allclose(
        fem.edge_load_shares(4, 1.5), [0.5 / 1.5, 0.875 / 1.5, 0.125 / 1.5, 0.0]
    )


def test_edge_load_shares_tend_to_a_point_load():
    np.testing.assert_array_equal(fem.edge_load_shares(4, 0.0), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        fem.edge_load_shares(4, 1e-9), [1.0, 0.0, 0.0, 0.0], atol=1e-9
    )


def test_edge_load_shares_reject_a_span_longer_than_the_edge():
    with pytest.raises(ValueError, match="does not fit"):
        fem.edge_load_shares(4, 3.5)
