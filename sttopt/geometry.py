"""Loading and interrogating a fixed density field: the geometry a `seqopt` run holds
constant, read from an `.npz` file so any component -- a hand-built thesis shape
(`geometry_builders.py`) or a topology-optimized layout -- can be studied.

Deliberately NumPy-only, no torch: this is CPU-side setup at the same boundary
`timefield.init_timefield` sits at, not part of the autograd graph.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

# Elements within this of 0 or 1 count as solid/void; anything strictly between is
# "intermediate" and trips load_geometry's binary check.
_BINARY_TOL = 1e-3


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
    solid_threshold: float = 0.5,
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
    :param solid_threshold: density above which an element counts as touching the
        build plate; a grey topology-optimized field and a hand-drawn binary mask
        don't want the same cutoff
    :return: the subset of `candidates` whose density exceeds `solid_threshold`
    :raises ValueError: if no candidate touches the build plate -- a geometry that
        does not touch its build plate cannot be printed from it
    """
    flat = xPhys.flatten()
    solid = candidates[flat[candidates] > solid_threshold]
    if solid.size == 0:
        raise ValueError(
            f"no candidate print-start element touches the build plate "
            f"(solid_threshold={solid_threshold}): the geometry is lifted clear of it"
        )
    return solid
