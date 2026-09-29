"""The structural load cases: the nodal force vector and the free dofs of each.

Positions are fractions of the domain, measured from the left edge in x and from the
bottom edge in y; spans are in metres (`load_length_m`, `support_length_m`), so a case
means the same structure at any resolution. Nodes are named geometrically through
`fem.node_grid`, never by a linear-index formula.
"""

from enum import StrEnum

import numpy as np
from jaxtyping import Float, Int

import sttopt.fem as fem
import sttopt.units as units


class LoadCase(StrEnum):
    """Named load cases.

    CANTILEVER: the left edge clamped, a unit downward load at the bottom-right corner.
    CORNER_LOADS: the left edge clamped, a unit load at each right corner, down-right at
    the top and down-left at the bottom.
    EDGE_TRACTION: three clamped patches (two on the left edge, one on the bottom), a
    unit +x traction over the whole right edge.
    """

    CANTILEVER = "cantilever"
    CORNER_LOADS = "corner_loads"
    EDGE_TRACTION = "edge_traction"


def _patch_nodes(
    edge: Int[np.ndarray, " n"], centre: float, length: float
) -> Int[np.ndarray, " k"]:
    """The nodes of `edge` inside the closed span `length` centred at `centre`, both in
    elements from the edge's first node.

    :raises ValueError: if the span leaves the edge or holds no node
    """
    lo, hi = centre - length / 2, centre + length / 2
    if lo < 0 or hi > len(edge) - 1:
        raise ValueError(
            f"a support patch over [{lo}, {hi}] elements does not fit on an edge of {len(edge) - 1} elements"
        )
    pos = np.arange(len(edge))
    # Tolerance so a span that ends on a node keeps it despite round-off
    inside = (pos >= lo - 1e-9) & (pos <= hi + 1e-9)
    if not inside.any():
        raise ValueError(
            f"a support patch over [{lo}, {hi}] elements holds no node; lengthen support_length_m"
        )
    return edge[inside]


def _clamp(nodes: Int[np.ndarray, " k"]) -> Int[np.ndarray, " 2k"]:
    """Both dofs of each node."""
    return np.stack([2 * nodes, 2 * nodes + 1], axis=-1).ravel()


def load_case(
    case: LoadCase,
    nelx: int,
    nely: int,
    load_length_m: float,
    support_length_m: float,
    element_size_m: float,
) -> tuple[Float[np.ndarray, " ndof"], Int[np.ndarray, " n_free"]]:
    """The force vector and free dofs of `case` on a `nelx` x `nely` mesh.

    :param load_length_m: span of each corner load, a uniform traction from its corner
        along the right edge
    :param support_length_m: span of each clamped patch, where the case has patches
    :return: `(F, freedofs)`
    """
    nodes = fem.node_grid(nelx, nely)
    ndof = 2 * nodes.size
    F = np.zeros(ndof)
    load_length = units.in_elements(load_length_m, element_size_m)
    right_down = nodes[:, -1]  # row 0 is the top
    right_up = right_down[::-1]
    left_up = nodes[::-1, 0]
    if case == LoadCase.CANTILEVER:
        F[2 * right_up + 1] = -fem.edge_load_shares(nely + 1, load_length)
        fixed = _clamp(left_up)
    elif case == LoadCase.CORNER_LOADS:
        # Each corner's span starts at its own corner, so the shares are the same
        share = fem.edge_load_shares(nely + 1, load_length) / np.sqrt(2)
        F[2 * right_down] += share
        F[2 * right_down + 1] -= share
        F[2 * right_up] -= share
        F[2 * right_up + 1] -= share
        fixed = _clamp(left_up)
    elif case == LoadCase.EDGE_TRACTION:
        F[2 * right_up] = fem.edge_load_shares(nely + 1, nely)
        support_length = units.in_elements(support_length_m, element_size_m)
        patches = [
            _patch_nodes(left_up, 0.25 * nely, support_length),
            _patch_nodes(left_up, 0.80 * nely, support_length),
            _patch_nodes(nodes[-1, :], 0.30 * nelx, support_length),
        ]
        fixed = _clamp(np.concatenate(patches))
    else:
        raise ValueError(f"case must be a LoadCase member, got {case!r}")
    return F, np.setdiff1d(np.arange(ndof), fixed)
