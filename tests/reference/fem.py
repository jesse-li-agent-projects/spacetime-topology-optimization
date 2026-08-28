"""NumPy FEM assembly and solve, kept as the independent oracle the torch multigrid-CG
path (`sttopt.torch_fem`/`sttopt.torch_solve`) is checked against.

`sttopt.fem` still owns mesh/element setup (`plane_stress_KE`, `node_grid`,
`element_dof_map`) -- the torch path uses those directly. `assemble_stiffness`/
`solve_fe` here are the plain `scipy.sparse.linalg.spsolve` path Phase 3.1
(`plans/torch_port_part2.md`) ported before the MGCG solve replaced it; nothing in
`sttopt/` calls them.
"""

import numpy as np
import scipy.sparse as sp
from jaxtyping import Float, Int


def assemble_from_density(
    KE: Float[np.ndarray, "8 8"],
    density: Float[np.ndarray, " nel"],
    edofMat: Int[np.ndarray, "nelx*nely 8"],
    ndof: int,
) -> sp.csr_matrix:
    """Assemble the global stiffness matrix from an explicit per-element stiffness scale.

    `density` is the quantity `K` is actually linear in -- the same variable
    `sttopt.torch_solve.FemSolve` takes -- so a caller differentiating the solve needs
    no SIMP algebra of its own. Overlapping dof contributions from adjacent elements
    are summed (COO duplicate-index accumulation), then the result is symmetrized to
    cancel floating-point asymmetry from the summation order.
    """
    nel = edofMat.shape[0]
    # Matches MATLAB's iK = kron(edofMat,ones(8,1))', jK = kron(edofMat,ones(1,8))'
    # pairing exactly (iK[k] = row[k%8], jK[k] = row[k//8]). KE.flatten() (C order) pairs
    # with this as KE[k//8, k%8], the transpose of what (iK[k], jK[k]) addresses -- inert
    # since KE is symmetric, and the result is symmetrized below regardless.
    iK = np.tile(edofMat, (1, 8)).flatten()
    jK = np.repeat(edofMat, 8, axis=1).flatten()
    sK = (KE.flatten()[None, :] * density[:, None]).flatten()
    assert iK.shape == jK.shape == sK.shape == (64 * nel,)
    K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsr()
    return (K + K.T) / 2


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

    Element density `xPhys.flatten()[e]` scales `KE` via the SIMP interpolation
    `Emin + xPhys**penal * (Emax - Emin)`.
    """
    density = Emin + xPhys.flatten() ** penal * (Emax - Emin)
    return assemble_from_density(KE, density, edofMat, ndof)


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
