"""Self-weight gravity load matrix for the 2D density-based topology optimization mesh.

Maps a per-element uniform density field to y-direction nodal loads: each element's
weight is distributed equally (`fe/4`) to its 4 corner nodes, where `fe = 1/(nelx*nely)`
is the per-element weight for unit total density. `C @ x.flatten()` (see
`conventions.md`) gives the nodal self-weight load vector for density field `x`.
"""

import numpy as np
import scipy.sparse as sp

from sttopt import fem


def gravity_load_matrix(nelx: int, nely: int) -> sp.csr_matrix:
    """Sparse matrix distributing per-element self-weight to corner-node loads.

    Shape `((nelx+1)*(nely+1), nelx*nely)`: column `e` (element index, 0-indexed,
    C order per conventions.md) has `fe/4` in the rows of its element's 4 corner
    nodes, numbered by `fem.node_grid`.
    """
    fe = 1 / (nelx * nely)
    nodenrs = fem.node_grid(nelx, nely)
    top_left = nodenrs[:-1, :-1].flatten()
    bottom_left = nodenrs[1:, :-1].flatten()
    top_right = nodenrs[:-1, 1:].flatten()
    bottom_right = nodenrs[1:, 1:].flatten()

    I = np.concatenate([top_left, bottom_left, top_right, bottom_right])
    element = np.arange(nelx * nely)
    J = np.tile(element, 4)
    S = np.full(4 * nelx * nely, fe / 4)

    return sp.coo_matrix(
        (S, (I, J)), shape=((nelx + 1) * (nely + 1), nelx * nely)
    ).tocsr()
