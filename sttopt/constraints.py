"""Design constraints for the printable-structure optimization: global and per-stage
material budgets, print-start timing, and time-field smoothness.

The MATLAB source computes these as inline blocks in the main optimization loop, not as
a separate function, and each block bakes the density-filter chain rule (`H`, `Hs` from
`filters.density_filter`, and the Heaviside-projection derivative `dx` from
`filters.heaviside_projection_derivative`) directly into its sensitivity -- so each
function here does the same: callers pass `H`/`Hs`/`dx` and get back sensitivities that
are already the finished `dfdx` rows MMA consumes, not raw `d(.)/d(xPhys)` gradients like
`compliance.py` returns. See `conventions.md` for array-order and tolerance conventions.

Each function returns `(fval, dfx, dft)` (density- and time-sensitivity rows separately,
one of which may be identically zero) rather than a concatenated `dfdx` row -- callers
(the main optimization loop) assemble the final MMA-input matrix.
"""

import numpy as np
import scipy.sparse as sp
from jaxtyping import Float, Int

import sttopt.compliance as compliance


def global_volume_fraction(
    xPhys: Float[np.ndarray, "nely nelx"],
    dx: Float[np.ndarray, "nely nelx"],
    H: sp.spmatrix | sp.sparray,
    Hs: Float[np.ndarray, " nely*nelx"],
    volfrac: float,
) -> tuple[float, Float[np.ndarray, " nely*nelx"], Float[np.ndarray, " nely*nelx"]]:
    """Global printable-volume-fraction constraint: total deposited material vs. `volfrac`.

    `dft` is identically zero -- this constraint has no time-field dependence.
    """
    nely, nelx = xPhys.shape
    scale = nelx * nely * volfrac
    fval = float(np.sum(xPhys) / scale - 1)
    dv = np.ones((nely, nelx))
    dfx = H @ (dv.flatten() * dx.flatten() / Hs) / scale
    dft = np.zeros(nelx * nely)
    return fval, dfx, dft


def time_field_continuity(
    tPhys: Float[np.ndarray, "nely nelx"],
    L: sp.spmatrix | sp.sparray,
    H: sp.spmatrix | sp.sparray,
    Hs: Float[np.ndarray, " nely*nelx"],
) -> tuple[float, Float[np.ndarray, " nely*nelx"], Float[np.ndarray, " nely*nelx"]]:
    """Time-field smoothness constraint: keeps each element's print time close to its local
    neighborhood average (`filters.continuity_filter`'s `L`), so the deposition sequence
    sweeps coherently across the mesh instead of jumping between distant elements.

    `dfx` is identically zero -- this constraint has no density dependence.
    """
    nely, nelx = tPhys.shape
    nel = nely * nelx
    kk = 2 * nel
    A = L @ tPhys.flatten()
    fval = float(kk * (np.sum(A**2 / nel) - 1.0e-6))
    dft = H @ ((kk * 2 * (L.T @ A)) / Hs) / nel
    dfx = np.zeros(nel)
    return fval, dfx, dft


def start_point(
    tPhys: Float[np.ndarray, "nely nelx"],
    Nei: Int[np.ndarray, " k"],
    H: sp.spmatrix | sp.sparray,
    Hs: Float[np.ndarray, " nely*nelx"],
) -> tuple[
    Float[np.ndarray, " k"],
    Float[np.ndarray, "k nely*nelx"],
    Float[np.ndarray, "k nely*nelx"],
]:
    """Print-start constraint(s): the deposition-origin element(s) `Nei` (0-indexed element
    numbers, per `conventions.md`) must start printing at t=0 (up to machine precision).

    Example: `Nei` is `[0]` for the single-origin time field (`tfield==1`) -- the elements nearest
    the print-start origin. `dfx` is identically zero (no density dependence).
    """
    nely, nelx = tPhys.shape
    nel = nely * nelx
    k = len(Nei)
    fval = tPhys.flatten()[Nei] - 1.0e-9
    ss = np.zeros((nel, k))
    ss[Nei, np.arange(k)] = 1.0
    dft = (H @ (ss / Hs[:, None])).T
    dfx = np.zeros((k, nel))
    return fval, dfx, dft


def stage_volume_bounds(
    xPhys: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    dx: Float[np.ndarray, "nely nelx"],
    H: sp.spmatrix | sp.sparray,
    Hs: Float[np.ndarray, " nely*nelx"],
    stage: int,
    nStage: int,
    volfrac: float,
    rou: float,
) -> tuple[
    float, float, Float[np.ndarray, " nely*nelx"], Float[np.ndarray, " nely*nelx"]
]:
    """Per-stage material deposition budget: the volume fraction deposited by the end of
    stage `stage` (1-indexed, out of `nStage`) must stay within a small slack of the even
    schedule `stage/nStage`. Returns both the upper- and lower-bound constraint (MMA
    constraints are one-sided, so an equality-like budget needs both), via a smooth
    stage-membership mask (`compliance.time_mask`, sharpness `rou`) rather than a hard
    time cutoff.

    Returns `(fval_upper, fval_lower, dfx, dft)`: the lower bound's sensitivity rows are
    exactly `-dfx, -dft` (see the MATLAB source), so only one `(dfx, dft)` pair is returned.
    """
    nely, nelx = xPhys.shape
    scale = nelx * nely * volfrac
    # matches MATLAB's `tP(i+1)`, not just `stage/nStage`
    ti = np.linspace(0, 1, nStage + 1)[stage]
    ft = compliance.time_mask(tPhys, ti, rou)
    dfdt = compliance.time_mask_derivative(tPhys, ti, rou)
    xtJoint = xPhys * ft

    budget = stage / nStage
    deposited = np.sum(xtJoint) / scale
    fval_upper = float(deposited - budget)
    fval_lower = float(-deposited + budget - 1.0e-5)

    dfx = H @ ((ft / scale).flatten() * dx.flatten() / Hs)
    dft = H @ ((xPhys * dfdt / scale).flatten() / Hs)
    return fval_upper, fval_lower, dfx, dft
