"""Density and continuity filters, and the Heaviside projection, for the density field.

The density filter (`H`, `Hs`) is a local-neighborhood blur that regularizes the SIMP
density field against checkerboarding; the continuity filter (`L`) instead penalizes a
field for *disagreeing* with its local neighborhood average, used on the time field to
encourage smooth print sequencing. The Heaviside projection sharpens a filtered density
field toward 0/1 while keeping it differentiable. See `conventions.md` for array-order
and tolerance conventions.
"""

import math

import numpy as np
import scipy.sparse as sp
import torch
from jaxtyping import Float
from torch import Tensor


def _neighbor_offsets(radius: float) -> list[tuple[int, int]]:
    """Integer (di, dj) offsets covering the square neighborhood the MATLAB source loops over."""
    r = int(np.ceil(radius)) - 1
    return [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)]


def _element_grid(nelx: int, nely: int) -> tuple[np.ndarray, np.ndarray]:
    """0-indexed (i, j) grid coordinates of every element, in linear-index order.

    `e = j*nelx + i` is exactly the row-major element index from `conventions.md`
    (0-indexed), so `np.arange(nelx*nely)` already lists elements in that order.
    """
    e = np.arange(nelx * nely)
    return e % nelx, e // nelx


def density_filter(
    nelx: int, nely: int, rmin: float
) -> tuple[sp.csr_matrix, Float[np.ndarray, " n"]]:
    """Distance-weighted density-blur filter, and its per-element row-sum normalizer.

    `H[e1, e2] = max(0, rmin - dist(e1, e2))` for elements within `rmin` of each other
    (including `e1 == e2`); `Hs` normalizes a filtered field back to the original
    density's scale (`xTilde = H @ x.flatten() / Hs`).
    """
    i1, j1 = _element_grid(nelx, nely)
    e1_all = np.arange(nelx * nely)
    rows, cols, vals = [], [], []
    for di, dj in _neighbor_offsets(rmin):
        i2, j2 = i1 + di, j1 + dj
        valid = (i2 >= 0) & (i2 < nelx) & (j2 >= 0) & (j2 < nely)
        weight = max(0.0, rmin - np.hypot(di, dj))
        if weight == 0.0:
            continue
        rows.append(e1_all[valid])
        cols.append((j2 * nelx + i2)[valid])
        vals.append(np.full(valid.sum(), weight))
    H = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nelx * nely, nelx * nely),
    ).tocsr()
    Hs = np.asarray(H.sum(axis=1)).flatten()
    return H, Hs


def continuity_filter(nelx: int, nely: int, lrmin: float) -> sp.csr_matrix:
    """Identity-minus-row-normalized-neighbor-average matrix, for the print-time smoothness penalty.

    `L @ t` measures each element's time-field value minus the (unweighted) average of
    its neighbors within `lrmin`; built entirely from sparse ops (row-scale a sparse
    adjacency matrix, subtract from a sparse identity) so it never materializes a dense
    `n x n` array, unlike the MATLAB source's `eye(n) - L./M` (see `conventions.md`).
    """
    i1, j1 = _element_grid(nelx, nely)
    e1_all = np.arange(nelx * nely)
    rows, cols = [], []
    for di, dj in _neighbor_offsets(lrmin):
        if di == 0 and dj == 0:
            continue
        i2, j2 = i1 + di, j1 + dj
        valid = (i2 >= 0) & (i2 < nelx) & (j2 >= 0) & (j2 < nely)
        rows.append(e1_all[valid])
        cols.append((j2 * nelx + i2)[valid])
    n = nelx * nely
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adjacency = sp.coo_matrix((np.ones(rows.shape), (rows, cols)), shape=(n, n)).tocsr()
    row_sum = np.asarray(adjacency.sum(axis=1)).flatten()
    normalized = sp.diags(1.0 / row_sum) @ adjacency
    return (sp.eye(n, format="csr") - normalized).tocsr()


def heaviside_projection(
    xTilde: Float[Tensor, "*dims"], beta: float, eta: float
) -> Float[Tensor, "*dims"]:
    """Project a filtered density field toward 0/1 via a smoothed threshold at `eta`.

    `beta` controls sharpness (linear as `beta -> 0`, a step function as `beta -> inf`).
    """
    denom = math.tanh(beta * eta) + math.tanh(beta * (1 - eta))
    return (math.tanh(beta * eta) + torch.tanh(beta * (xTilde - eta))) / denom


def heaviside_projection_derivative(
    xTilde: Float[Tensor, "*dims"], beta: float, eta: float
) -> Float[Tensor, "*dims"]:
    """Derivative of `heaviside_projection` w.r.t. `xTilde`, for the chain rule through sensitivities."""
    denom = math.tanh(beta * eta) + math.tanh(beta * (1 - eta))
    return beta * (1 - torch.tanh(beta * (xTilde - eta)) ** 2) / denom
