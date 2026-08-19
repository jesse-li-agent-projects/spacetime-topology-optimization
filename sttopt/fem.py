"""Plane-stress FEM assembly for the 2D density-based topology optimization mesh.

Builds the element stiffness matrix and mesh connectivity for a regular grid of
bilinear quad elements under plane-stress, assembles the global (SIMP-penalized)
stiffness matrix from an element density field, and solves the free-dof system for
nodal displacements. See `conventions.md` for array-order and tolerance conventions;
`edofMat`/`plane_stress_KE` here are the Python (0-indexed) counterparts of the
MATLAB source's `edofMat`/`KE`.
"""

import numpy as np
import scipy.sparse as sp
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


def element_dof_map(nelx: int, nely: int) -> Int[np.ndarray, "n 8"]:
    """Per-element global dof indices (0-indexed), in element order matching `xPhys.flatten('F')`.

    Each row lists the 8 dofs (x, y for each of the 4 corner nodes) of one element.
    Node numbering follows the mesh's column-major node grid, so element `e` (0-indexed,
    Fortran order) corresponds to grid position `(e % nely, e // nely)` per conventions.md.
    """
    # Mirrors the MATLAB source's 1-indexed dof numbering exactly (nodenrs 1..N, dof
    # 2*node+1), then shifts to 0-indexed at the end -- doing the arithmetic directly
    # in 0-indexed terms is off by an additive constant per node, not just a shift.
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(
        1 + nely, 1 + nelx, order="F"
    )
    edof_vec = 2 * nodenrs[:-1, :-1].flatten(order="F") + 1
    offsets = np.array(
        [0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1]
    )
    return edof_vec[:, None] + offsets[None, :] - 1


def assemble_stiffness(
    KE: Float[np.ndarray, "8 8"],
    xPhys: Float[np.ndarray, "nely nelx"],
    Emin: float,
    Emax: float,
    penal: float,
    edofMat: Int[np.ndarray, "nelx*nely 8"],
    ndof: int,
) -> sp.csr_matrix:
    """Assemble the global SIMP-penalized stiffness matrix from per-element densities.

    Element density `xPhys.flatten('F')[e]` scales `KE` via the SIMP interpolation
    `Emin + xPhys**penal * (Emax - Emin)`; overlapping dof contributions from adjacent
    elements are summed (COO duplicate-index accumulation), then the result is
    symmetrized to cancel floating-point asymmetry from the summation order.
    """
    nel = edofMat.shape[0]
    # Matches MATLAB's iK = kron(edofMat,ones(8,1))', jK = kron(edofMat,ones(1,8))'
    # pairing exactly (iK[k] = row[k%8], jK[k] = row[k//8]) -- inert here since KE is
    # symmetric and the result is symmetrized below, but kept literal to the source.
    # NB: these three flattens are row-major ('C', NumPy's default) over the per-element
    # (nel, 64) block layout built by tile/repeat -- that's correct here and not a
    # violation of conventions.md's order='F' rule, which is about mirroring a MATLAB
    # (:) or reshape over the (nely, nelx) grid; this isn't that.
    iK = np.tile(edofMat, (1, 8)).flatten()
    jK = np.repeat(edofMat, 8, axis=1).flatten()
    density = Emin + xPhys.flatten(order="F") ** penal * (Emax - Emin)
    sK = (KE.flatten(order="F")[None, :] * density[:, None]).flatten()
    assert iK.shape == jK.shape == sK.shape == (64 * nel,)
    K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsr()
    return (K + K.T) / 2


def solve_fe(
    K: sp.spmatrix,
    F: Float[np.ndarray, " ndof"],
    freedofs: Int[np.ndarray, " n_free"],
) -> Float[np.ndarray, " ndof"]:
    """Solve the free-dof partition of `K @ U = F`, zero-filling fixed dofs."""
    U = np.zeros(K.shape[0])
    K_free = K[np.ix_(freedofs, freedofs)]
    U[freedofs] = sp.linalg.spsolve(K_free.tocsc(), F[freedofs])
    return U
