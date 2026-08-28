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
