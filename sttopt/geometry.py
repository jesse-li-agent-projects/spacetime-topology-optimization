"""Loading and interrogating a fixed density field: the geometry a `seqopt` run holds
constant, read from an `.npz` file so any component -- a hand-built thesis shape
(`geometry_builders.py`) or a topology-optimized layout -- can be studied.

Deliberately NumPy-only, no torch: this is CPU-side setup at the same boundary
`timefield.init_timefield` sits at, not part of the autograd graph.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from jaxtyping import Bool, Float, Int
from scipy.sparse.csgraph import connected_components

# Elements within this of 0 or 1 count as solid/void; anything strictly between is
# "intermediate" and trips load_geometry's binary check.
_BINARY_TOL = 1e-3

# Density above which an element counts as solid. The one cutoff every solid/void
# decision outside the physics uses -- the physics itself never thresholds, taking
# density continuously (`conductivity`'s `x**q`, the uniformity weights). On the
# intended path `load_geometry` has already required a binary field, so any cutoff in
# `(0, 1)` would agree with this one; it matters only under `--non-binarized`.
SOLID_THRESHOLD = 0.5

_NEIGHBOR_OFFSETS_8 = [
    (di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)
]


def load_geometry(path: Path, *, binary: bool = True) -> Float[np.ndarray, "nely nelx"]:
    """
    Load a fixed density field from an `.npz` file's `xPhys` key.

    :param path: path to the `.npz` file
    :param binary: if True (default), require every element to lie within
        `_BINARY_TOL` of 0 or 1. A topology-optimized `xPhys` rarely does -- see
        `plans/fixed_geometry_sequence_optimization.md`'s measured figures -- so
        turning one into a printable geometry is its own step
        (`python -m sttopt.geometry_builders binarize`), not a hidden transform here.
        Pass `False` (the CLI's `--non-binarized`) to study a grey field deliberately.
    :return: the density field, shape `(nely, nelx)`
    :raises ValueError: if `xPhys` is missing, not 2-D, has non-finite or
        out-of-`[0, 1]` values, or (when `binary`) is not binary
    """
    with np.load(path) as data:
        if "xPhys" not in data:
            raise ValueError(
                f"{path}: no 'xPhys' key found; keys present: {sorted(data.keys())}"
            )
        xPhys = np.asarray(data["xPhys"])

    if xPhys.ndim != 2:
        raise ValueError(f"{path}: xPhys must be 2-D, got shape {xPhys.shape}")
    if not np.all(np.isfinite(xPhys)):
        raise ValueError(f"{path}: xPhys must be finite")
    if xPhys.min() < 0.0 or xPhys.max() > 1.0:
        raise ValueError(
            f"{path}: xPhys is not a density field: values must lie in [0, 1], got "
            f"min={xPhys.min():.6g}, max={xPhys.max():.6g}"
        )

    if binary:
        intermediate = (xPhys > _BINARY_TOL) & (xPhys < 1.0 - _BINARY_TOL)
        if np.any(intermediate):
            vals = xPhys[intermediate]
            raise ValueError(
                f"{path}: xPhys is not binary: {int(intermediate.sum())} of "
                f"{xPhys.size} elements lie strictly between 0 and 1 (tolerance "
                f"{_BINARY_TOL:g}), ranging {vals.min():.6g} to {vals.max():.6g}. "
                f"Pass --non-binarized to study the grey field as given, or binarize "
                f"it first (python -m sttopt.geometry_builders binarize)."
            )

    return xPhys


def base_elements(
    xPhys: Float[np.ndarray, "nely nelx"],
    candidates: Int[np.ndarray, " k"],
) -> Int[np.ndarray, " j"]:
    """
    Filter a set of candidate print-start elements (e.g. `timefield.base_elements`)
    down to those that actually hold material.

    This is the load-bearing piece for a geometry whose build-plate-facing edge only
    partly touches the plate -- e.g. the Wu2025 overhang bracket, solid over only the
    left half of its bottom row -- where "the whole bottom row" is the wrong
    print-start set.

    :param xPhys: density field, shape `(nely, nelx)`
    :param candidates: 0-indexed element numbers of the candidate print-start set
    :return: the subset of `candidates` that hold material, at `SOLID_THRESHOLD`
    :raises ValueError: if no candidate touches the build plate -- a geometry that
        does not touch its build plate cannot be printed from it
    """
    solid = candidates[solid_mask(xPhys)[candidates]]
    if solid.size == 0:
        raise ValueError(
            "no candidate print-start element holds material: the geometry is lifted clear of its build plate"
        )
    return solid


def neighbor_pairs(
    nelx: int, nely: int
) -> tuple[
    Int[np.ndarray, " npairs"], Int[np.ndarray, " npairs"], Float[np.ndarray, " npairs"]
]:
    """
    Every ordered 8-connected element adjacency on the `(nely, nelx)` grid.

    The shared grid-graph backbone: callers select the subset of pairs they want (both
    ends solid, destination void, ...) and attach their own edge costs, so connectivity
    and distance stay one adjacency rather than two that can disagree.

    :param nelx: element count in x
    :param nely: element count in y
    :return: `(src, dst, step)` -- 0-indexed element numbers and the Euclidean step
        length between them (`1` or `sqrt(2)`)
    """
    e = np.arange(nelx * nely)
    i, j = e % nelx, e // nelx

    src, dst, step = [], [], []
    for di, dj in _NEIGHBOR_OFFSETS_8:
        i2, j2 = i + di, j + dj
        valid = (i2 >= 0) & (i2 < nelx) & (j2 >= 0) & (j2 < nely)
        src.append(e[valid])
        dst.append((j2 * nelx + i2)[valid])
        step.append(np.full(int(valid.sum()), math.hypot(di, dj)))
    return np.concatenate(src), np.concatenate(dst), np.concatenate(step)


def solid_mask(xPhys: Float[np.ndarray, "nely nelx"]) -> Bool[np.ndarray, " nel"]:
    """Flat boolean "this element holds material" mask, at `SOLID_THRESHOLD`."""
    return xPhys.flatten() > SOLID_THRESHOLD


def drop_disconnected(
    xPhys: Float[np.ndarray, "nely nelx"], base: Int[np.ndarray, " k"]
) -> Float[np.ndarray, "nely nelx"]:
    """
    Void out every solid element with no 8-connected path of material back to `base`.

    Material that does not connect to the build plate cannot be deposited from it, so
    it is not part of the component being printed; carrying it would let it contribute
    to the hotspot objective and to the volume the stage budgets normalize by. Removing
    it here means the geometry a run optimizes is one that can actually be built --
    which also lets `timefield.init_geodesic_timefield` treat an infinite geodesic
    distance as a contradiction rather than a case to smooth over.

    :param xPhys: density field, shape `(nely, nelx)`
    :param base: 0-indexed print-start elements (`base_elements`); their own components
        are the ones kept
    :return: `xPhys` with disconnected solid set to 0, or `xPhys` itself when nothing
        is disconnected
    :warns UserWarning: naming how many elements were dropped, since this silently
        changes the component being optimized
    """
    nely, nelx = xPhys.shape
    solid = solid_mask(xPhys)
    src, dst, _ = neighbor_pairs(nelx, nely)

    within = solid[src] & solid[dst]
    graph = sp.csr_matrix(
        (np.ones(int(within.sum())), (src[within], dst[within])),
        shape=(solid.size, solid.size),
    )
    _, labels = connected_components(graph, directed=False)

    dropped = solid & ~np.isin(labels, labels[base])
    if not dropped.any():
        return xPhys

    warnings.warn(
        f"dropping {int(dropped.sum())} solid element(s) with no path of material back to the build plate: they cannot be deposited, so they are not part of the component being optimized"
    )
    cleaned = xPhys.copy()
    cleaned.reshape(-1)[dropped] = 0.0
    return cleaned
