"""Plane-stress mesh/element setup for the 2D density-based topology optimization mesh.

Builds the element stiffness matrix and mesh connectivity for a regular grid of
bilinear quad elements under plane-stress -- setup the torch path (`torch_fem.py`)
uses directly. See `conventions.md` for array-order and tolerance conventions;
`edofMat`/`plane_stress_KE` here are the Python (0-indexed) counterparts of the
MATLAB source's `edofMat`/`KE`. `tests/reference/fem.py` keeps this module's
predecessor `assemble_stiffness`/`solve_fe` (plain `scipy.sparse` assembly and
`spsolve`), as the independent oracle the torch multigrid-CG solve is checked
against.
"""

import numpy as np
from jaxtyping import Float, Int


def plane_stress_KE(nu: float) -> Float[np.ndarray, "8 8"]:
    """Element stiffness matrix for a unit-square bilinear plane-stress quad.

    Dof order per node is (x, y); nodes are ordered as in the source mesh
    convention (see `element_dof_map`). Independent of element size since the
    reference mesh uses unit squares.
    """
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    A = np.block([[A11, A12], [A12.T, A11]])
    B = np.block([[B11, B12], [B12.T, B11]])
    return (A + nu * B) / (1 - nu**2) / 24


def node_grid(nelx: int, nely: int) -> Int[np.ndarray, "nely+1 nelx+1"]:
    """Global node number (0-indexed) at each mesh corner position `(row, col)`.

    The single definition of the mesh's node numbering, in C order per conventions.md.
    Anything that needs to name specific nodes -- element connectivity, self-weight
    loads, boundary conditions -- should index this grid geometrically rather than
    re-deriving the linear-index formula, so node numbering stays a choice made in one
    place.
    """
    return np.arange((nely + 1) * (nelx + 1)).reshape(nely + 1, nelx + 1)


def element_dof_map(nelx: int, nely: int) -> Int[np.ndarray, "n 8"]:
    """Per-element global dof indices (0-indexed), in element order matching `xPhys.flatten()`.

    Each row lists the 8 dofs (x, y for each of the 4 corner nodes) of one element,
    the corners taken in the local node order `plane_stress_KE` expects. Element `e`
    (0-indexed, C order per conventions.md) corresponds to grid position
    `(e // nelx, e % nelx)`, and node numbering follows `node_grid`.
    """
    nodes = node_grid(nelx, nely)
    # Local node order, counterclockwise against the physical y = -row axis: bottom
    # left, bottom right, top right, top left, relative to each element's top-left node.
    corners = np.stack(
        [nodes[1:, :-1], nodes[1:, 1:], nodes[:-1, 1:], nodes[:-1, :-1]], axis=-1
    ).reshape(-1, 4)
    return np.stack([2 * corners, 2 * corners + 1], axis=-1).reshape(-1, 8)
