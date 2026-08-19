"""Self-weight gravity load matrix for the 2D density-based topology optimization mesh.

Maps a per-element uniform density field to y-direction nodal loads: each element's
weight is distributed equally (`fe/4`) to its 4 corner nodes, where `fe = 1/(nelx*nely)`
is the per-element weight for unit total density. `C @ x.flatten('F')` (see
`conventions.md`) gives the nodal self-weight load vector for density field `x`.
"""

import numpy as np
import scipy.sparse as sp


def gravity_load_matrix(nelx: int, nely: int) -> sp.csr_matrix:
    """Sparse matrix distributing per-element self-weight to corner-node loads.

    Shape `((nelx+1)*(nely+1), nelx*nely)`: column `e` (element index, 0-indexed,
    Fortran order per conventions.md) has `fe/4` in the rows of its element's 4 corner
    nodes (node numbering column-major over the `(nely+1, nelx+1)` node grid, matching
    `fem.element_dof_map`'s `nodenrs`).
    """
    fe = 1 / (nelx * nely)
    nodenrs = np.arange((nelx + 1) * (nely + 1)).reshape(nely + 1, nelx + 1, order="F")
    top_left = nodenrs[:-1, :-1].flatten(order="F")
    bottom_left = nodenrs[1:, :-1].flatten(order="F")
    top_right = nodenrs[:-1, 1:].flatten(order="F")
    bottom_right = nodenrs[1:, 1:].flatten(order="F")

    I = np.concatenate([top_left, bottom_left, top_right, bottom_right])
    element = np.arange(nelx * nely)
    J = np.tile(element, 4)
    S = np.full(4 * nelx * nely, fe / 4)

    return sp.coo_matrix((S, (I, J)), shape=((nelx + 1) * (nely + 1), nelx * nely)).tocsr()
