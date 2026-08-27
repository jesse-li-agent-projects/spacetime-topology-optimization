"""SIMP compliance and its density/time sensitivities, under two load models.

`whole_compliance` is the "whole structure" objective under a fixed external point
load. `gravity_compliance` is the self-weight objective for one deposition-order
stage: only elements already "active" by stage time `ti` (per `time_mask`) carry
load, so the gravity load itself depends on both density and time, giving both a
`dcx` and `dct` sensitivity plus an extra adjoint term (see `gravity_compliance`).

Both solve via `torch_solve.FemSolve` (`plans/torch_port_part2.md` Phase 3.3), the
multigrid-CG solve as an autograd `Function` -- no NumPy round trip. The sensitivities
below stay the hand-derived formulas Phase 3.2 ported (Phase 3.4 replaces them with
autograd); `FemSolve`'s adjoint isn't exercised by this module, only its forward.
Array-order and mesh conventions (element order, dof layout) follow `conventions.md`
and `fem.py`.
"""

import math

import torch
from jaxtyping import Float, Int
from torch import Tensor

import sttopt.torch_fem as torch_fem
import sttopt.torch_solve as torch_solve


def time_mask(
    tPhys: Float[Tensor, "nely nelx"], ti: float, beta: float
) -> Float[Tensor, "nely nelx"]:
    """Smooth indicator of whether each element is "active" by stage time `ti`.

    Sigmoid transition (MATLAB source's `ft`, called with `lamda`/`rou` depending on
    caller; Wang et al. 2019's beta_t) from ~1 (element active, `tPhys < ti`) to ~0
    (not yet active), with sharpness `beta`.
    """
    num = math.tanh(beta * ti) + torch.tanh(beta * (tPhys - ti))
    den = math.tanh(beta * ti) + math.tanh(beta * (1 - ti))
    return 1 - num / den


def time_mask_derivative(
    tPhys: Float[Tensor, "nely nelx"], ti: float, beta: float
) -> Float[Tensor, "nely nelx"]:
    """d(`time_mask`)/d(tPhys), element-wise."""
    num = beta * (torch.tanh(beta * (tPhys - ti)) ** 2 - 1)
    den = math.tanh(beta * (ti - 1)) - math.tanh(beta * ti)
    return -num / den


def _element_strain_energy(
    U: Float[Tensor, " ndof"],
    edofMat: Int[Tensor, "nel 8"],
    KE: Float[Tensor, "8 8"],
    nely: int,
    nelx: int,
) -> Float[Tensor, "nely nelx"]:
    """Per-element strain energy `U_e^T KE U_e`, reshaped to the `(nely, nelx)` grid."""
    Ue = U[edofMat]  # (nel, 8) -- per-element nodal displacements
    ce = torch.sum((Ue @ KE) * Ue, dim=1)
    return ce.reshape(nely, nelx)


def _solve_fe(
    KE: Float[Tensor, "8 8"],
    xPhys: Float[Tensor, "nely nelx"],
    Emin: float,
    Emax: float,
    penal: float,
    edofMat: Int[Tensor, "nel 8"],
    freedofs: Int[Tensor, " n_free"],
    F: Float[Tensor, " ndof"],
    ndof: int,
) -> Float[Tensor, " ndof"]:
    """Solve `K @ U = F` via `torch_solve.FemSolve`'s multigrid-CG, `K` implicit from
    `xPhys`'s SIMP-scaled stiffness. `xPhys`'s own gradient reaches `FemSolve`'s
    `density` input through `torch_fem.simp_density`'s ordinary (non-`Function`) power
    law, so the caller's autograd graph sees one differentiable chain end to end.
    """
    nely, nelx = xPhys.shape
    density = torch_fem.simp_density(xPhys, Emin, Emax, penal)
    mask = torch_fem.free_mask(ndof, freedofs, device=xPhys.device)
    return torch_solve.femsolve(density.flatten(), F, edofMat, KE, mask, nelx, nely)


def whole_compliance(
    xPhys: Float[Tensor, "nely nelx"],
    KE: Float[Tensor, "8 8"],
    edofMat: Int[Tensor, "nelx*nely 8"],
    Emin: float,
    Emax: float,
    penal: float,
    freedofs: Int[Tensor, " n_free"],
    F: Float[Tensor, " ndof"],
    ndof: int,
) -> tuple[float, Float[Tensor, "nely nelx"]]:
    """SIMP compliance and its density sensitivity under a fixed external load `F`."""
    nely, nelx = xPhys.shape
    U = _solve_fe(KE, xPhys, Emin, Emax, penal, edofMat, freedofs, F, ndof)
    ce = _element_strain_energy(U, edofMat, KE, nely, nelx)

    simp = Emin + xPhys**penal * (Emax - Emin)
    c = float(torch.sum(simp * ce))
    dcx = -penal * (Emax - Emin) * xPhys ** (penal - 1) * ce
    return c, dcx


def gravity_compliance(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    KE: Float[Tensor, "8 8"],
    edofMat: Int[Tensor, "nelx*nely 8"],
    Emin: float,
    Emax: float,
    penal: float,
    ti: float,
    # gravity.gravity_load_matrix's output, converted to a sparse CSR tensor
    C: Tensor,
    beta_t: float,
    freedofs: Int[Tensor, " n_free"],
    ndof: int,
) -> tuple[float, Float[Tensor, " nely*nelx"], Float[Tensor, " nely*nelx"]]:
    """SIMP compliance and its density/time sensitivities under self-weight gravity.

    Only elements active by stage time `ti` (per `time_mask`, sharpness `beta_t` --
    MATLAB source's `lamda`/`rou`) carry density-weighted self-weight load, built via
    `gravity.gravity_load_matrix`'s `C`. Because the load itself depends on `xPhys`
    (through the joint density-time field `xtJoint = xPhys * t_mask`), the sensitivities
    pick up an extra adjoint term (`dcx2`/`dct2` below, from differentiating the load
    through the displacement it produces) beyond the direct SIMP-stiffness term
    (`dcx1`/`dct1`); the factor of 2 combining them is the standard self-adjoint
    compliance-sensitivity result for a density-dependent load, not a simplification.
    """
    nely, nelx = xPhys.shape
    nel = nely * nelx
    t_mask = time_mask(tPhys, ti, beta_t)
    dfdt = time_mask_derivative(tPhys, ti, beta_t)
    xtJoint = xPhys * t_mask

    f = -(C @ xtJoint.flatten())
    F = torch.zeros(ndof, dtype=xPhys.dtype, device=xPhys.device)
    F[1::2] = f  # y-dof of each node; x-dof stays 0 (gravity acts in -y)

    U = _solve_fe(KE, xtJoint, Emin, Emax, penal, edofMat, freedofs, F, ndof)
    ce = _element_strain_energy(U, edofMat, KE, nely, nelx)

    simp = Emin + xtJoint**penal * (Emax - Emin)
    c = float(torch.sum(simp * ce))

    dcx1 = -penal * (Emax - Emin) * xtJoint ** (penal - 1) * ce * t_mask
    dct1 = -penal * (Emax - Emin) * xtJoint ** (penal - 1) * ce * xPhys * dfdt

    Uy = U[1::2]  # y-displacement dof of every node, in the node order C's rows use
    adjoint = -(
        C.t() @ Uy
    )  # (nel,): d(load)/d(density) term, adjoint-contracted with U

    dcx2 = adjoint * t_mask.flatten()
    dct2 = adjoint * xPhys.flatten() * dfdt.flatten()

    dcx = 2 * dcx2 + dcx1.flatten()
    dct = 2 * dct2 + dct1.flatten()
    return c, dcx, dct
