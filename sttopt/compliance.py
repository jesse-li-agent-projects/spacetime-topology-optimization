"""SIMP compliance and its density/time sensitivities, under two load models.

`whole_compliance` is the "whole structure" objective under a fixed external point
load. `gravity_compliance` is the self-weight objective for one deposition-order
stage: only elements already "active" by stage time `ti` (per `time_mask`) carry
load, so the gravity load itself depends on both density and time, giving both a
`dcx` and `dct` sensitivity plus an extra adjoint term (see `gravity_compliance`).

Both assemble/solve via `fem.assemble_stiffness`/`fem.solve_fe`; array-order and mesh
conventions (element order, dof layout) follow `conventions.md` and `fem.py`.
"""

import numpy as np
import scipy.sparse as sp
from jaxtyping import Float, Int

import sttopt.fem as fem


def time_mask(
    tPhys: Float[np.ndarray, "nely nelx"], ti: float, lam: float
) -> Float[np.ndarray, "nely nelx"]:
    """Smooth indicator of whether each element is "active" by stage time `ti`.

    Sigmoid transition (MATLAB source's `ft`, called with `lamda`/`rou` depending on
    caller) from ~1 (element active, `tPhys < ti`) to ~0 (not yet active), with
    sharpness `lam`.
    """
    num = np.tanh(lam * ti) + np.tanh(lam * (tPhys - ti))
    den = np.tanh(lam * ti) + np.tanh(lam * (1 - ti))
    return 1 - num / den


def time_mask_derivative(
    tPhys: Float[np.ndarray, "nely nelx"], ti: float, lam: float
) -> Float[np.ndarray, "nely nelx"]:
    """d(`time_mask`)/d(tPhys), element-wise."""
    num = lam * (np.tanh(lam * (tPhys - ti)) ** 2 - 1)
    den = np.tanh(lam * (ti - 1)) - np.tanh(lam * ti)
    return -num / den


def _element_strain_energy(
    U: Float[np.ndarray, " ndof"],
    edofMat: Int[np.ndarray, "nel 8"],
    KE: Float[np.ndarray, "8 8"],
    nely: int,
    nelx: int,
) -> Float[np.ndarray, "nely nelx"]:
    """Per-element strain energy `U_e^T KE U_e`, reshaped to the `(nely, nelx)` grid."""
    Ue = U[edofMat]  # (nel, 8) -- per-element nodal displacements
    ce = np.sum((Ue @ KE) * Ue, axis=1)
    return ce.reshape(nely, nelx)


def whole_compliance(
    xPhys: Float[np.ndarray, "nely nelx"],
    KE: Float[np.ndarray, "8 8"],
    edofMat: Int[np.ndarray, "nelx*nely 8"],
    Emin: float,
    Emax: float,
    penal: float,
    freedofs: Int[np.ndarray, " n_free"],
    F: Float[np.ndarray, " ndof"],
    ndof: int,
) -> tuple[float, Float[np.ndarray, "nely nelx"]]:
    """SIMP compliance and its density sensitivity under a fixed external load `F`."""
    nely, nelx = xPhys.shape
    K = fem.assemble_stiffness(KE, xPhys, Emin, Emax, penal, edofMat, ndof)
    U = fem.solve_fe(K, F, freedofs)
    ce = _element_strain_energy(U, edofMat, KE, nely, nelx)

    simp = Emin + xPhys**penal * (Emax - Emin)
    c = float(np.sum(simp * ce))
    dcx = -penal * (Emax - Emin) * xPhys ** (penal - 1) * ce
    return c, dcx


def gravity_compliance(
    xPhys: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    KE: Float[np.ndarray, "8 8"],
    edofMat: Int[np.ndarray, "nelx*nely 8"],
    Emin: float,
    Emax: float,
    penal: float,
    ti: float,
    # gravity.gravity_load_matrix's output, or a loaded fixture's
    C: sp.spmatrix | sp.sparray,
    lam: float,
    freedofs: Int[np.ndarray, " n_free"],
    ndof: int,
) -> tuple[float, Float[np.ndarray, " nely*nelx"], Float[np.ndarray, " nely*nelx"]]:
    """SIMP compliance and its density/time sensitivities under self-weight gravity.

    Only elements active by stage time `ti` (per `time_mask`, sharpness `lam` --
    MATLAB source's `lamda`/`rou`) carry density-weighted self-weight load, built via
    `gravity.gravity_load_matrix`'s `C`. Because the load itself depends on `xPhys`
    (through the joint density-time field `xtJoint = xPhys * ft`), the sensitivities
    pick up an extra adjoint term (`dcx2`/`dct2` below, from differentiating the load
    through the displacement it produces) beyond the direct SIMP-stiffness term
    (`dcx1`/`dct1`); the factor of 2 combining them is the standard self-adjoint
    compliance-sensitivity result for a density-dependent load, not a simplification.
    """
    nely, nelx = xPhys.shape
    ft = time_mask(tPhys, ti, lam)
    dfdt = time_mask_derivative(tPhys, ti, lam)
    xtJoint = xPhys * ft

    K = fem.assemble_stiffness(KE, xtJoint, Emin, Emax, penal, edofMat, ndof)

    f = -C @ xtJoint.flatten()
    F = np.zeros(ndof)
    F[1::2] = f  # y-dof of each node; x-dof stays 0 (gravity acts in -y)

    U = fem.solve_fe(K, F, freedofs)
    ce = _element_strain_energy(U, edofMat, KE, nely, nelx)

    simp = Emin + xtJoint**penal * (Emax - Emin)
    c = float(np.sum(simp * ce))

    dcx1 = -penal * (Emax - Emin) * xtJoint ** (penal - 1) * ce * ft
    dct1 = -penal * (Emax - Emin) * xtJoint ** (penal - 1) * ce * xPhys * dfdt

    Uy = U[1::2]  # y-displacement dof of every node, in the node order C's rows use
    adjoint = -(C.T @ Uy)  # (nel,): d(load)/d(density) term, adjoint-contracted with U

    dcx2 = adjoint * ft.flatten()
    dct2 = adjoint * xPhys.flatten() * dfdt.flatten()

    dcx = 2 * dcx2 + dcx1.flatten()
    dct = 2 * dct2 + dct1.flatten()
    return c, dcx, dct
