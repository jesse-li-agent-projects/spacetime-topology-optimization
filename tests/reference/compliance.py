"""Hand-derived predecessors of `sttopt.compliance`'s compliance/sensitivity formulas.

`sttopt.compliance` computes sensitivities via autograd (Phase 3.4,
`plans/torch_port_part2.md`). The formulas here are the hand-derived versions Phase 3.2
ported from the MATLAB source, kept only as a cross-check: `tests/test_reference_sweep.py`
compares them against the autograd path, and `benchmarks/bench_sensitivities.py` times
one against the other. Nothing in `sttopt/` calls this module.

Shares its solve/assembly helpers (`_solve_fe`, `_solve_fe_batched`, `_element_strain_energy`,
`_gravity_load`, `time_mask`) with `sttopt.compliance`, which still owns them -- both the
production and reference paths solve the same way, only the sensitivity algebra differs.
"""

import math

import torch
from jaxtyping import Float, Int
from torch import Tensor

import sttopt.compliance as compliance
import sttopt.torch_fem as torch_fem


def time_mask_derivative(
    tPhys: Float[Tensor, "nely nelx"], ti: float, beta: float
) -> Float[Tensor, "nely nelx"]:
    """d(`compliance.time_mask`)/d(tPhys), element-wise."""
    num = beta * (torch.tanh(beta * (tPhys - ti)) ** 2 - 1)
    den = math.tanh(beta * (ti - 1)) - math.tanh(beta * ti)
    return -num / den


def _whole_compliance_from_U(
    xPhys: Float[Tensor, "nely nelx"],
    KE: Float[Tensor, "8 8"],
    edofMat: Int[Tensor, "nelx*nely 8"],
    Emin: float,
    Emax: float,
    penal: float,
    U: Float[Tensor, " ndof"],
) -> tuple[float, Float[Tensor, "nely nelx"]]:
    """`whole_compliance`'s value/sensitivity algebra, given an already-solved `U`."""
    nely, nelx = xPhys.shape
    ce = compliance._element_strain_energy(U, edofMat, KE, nely, nelx)
    simp = Emin + xPhys**penal * (Emax - Emin)
    c = float(torch.sum(simp * ce))
    dcx = -penal * (Emax - Emin) * xPhys ** (penal - 1) * ce
    return c, dcx


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
    *,
    x0: Float[Tensor, " ndof"] | None = None,
) -> tuple[float, Float[Tensor, "nely nelx"]]:
    """SIMP compliance and its density sensitivity under a fixed external load `F`.

    :param x0: optional warm start for the solve.
    """
    U = compliance._solve_fe(
        KE, xPhys, Emin, Emax, penal, edofMat, freedofs, F, ndof, x0=x0
    )
    return _whole_compliance_from_U(xPhys, KE, edofMat, Emin, Emax, penal, U)


def _gravity_compliance_from_U(
    xPhys: Float[Tensor, "nely nelx"],
    xtJoint: Float[Tensor, "nely nelx"],
    t_mask: Float[Tensor, "nely nelx"],
    dfdt: Float[Tensor, "nely nelx"],
    KE: Float[Tensor, "8 8"],
    edofMat: Int[Tensor, "nelx*nely 8"],
    Emin: float,
    Emax: float,
    penal: float,
    C: Tensor,
    U: Float[Tensor, " ndof"],
) -> tuple[float, Float[Tensor, " nely*nelx"], Float[Tensor, " nely*nelx"]]:
    """`gravity_compliance`'s value/sensitivity algebra, given an already-solved `U`.

    See `gravity_compliance`'s docstring for the extra adjoint term this computes.
    """
    nely, nelx = xPhys.shape
    ce = compliance._element_strain_energy(U, edofMat, KE, nely, nelx)

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
    *,
    x0: Float[Tensor, " ndof"] | None = None,
) -> tuple[float, Float[Tensor, " nely*nelx"], Float[Tensor, " nely*nelx"]]:
    """SIMP compliance and its density/time sensitivities under self-weight gravity.

    Only elements active by stage time `ti` (per `compliance.time_mask`, sharpness
    `beta_t` -- MATLAB source's `lamda`/`rou`) carry density-weighted self-weight load,
    built via `gravity.gravity_load_matrix`'s `C`. Because the load itself depends on
    `xPhys` (through the joint density-time field `xtJoint = xPhys * t_mask`), the
    sensitivities pick up an extra adjoint term (`dcx2`/`dct2` below, from
    differentiating the load through the displacement it produces) beyond the direct
    SIMP-stiffness term (`dcx1`/`dct1`); the factor of 2 combining them is the standard
    self-adjoint compliance-sensitivity result for a density-dependent load, not a
    simplification.

    :param x0: optional warm start for the solve.
    """
    t_mask, xtJoint, F = compliance._gravity_load(xPhys, tPhys, ti, C, beta_t, ndof)
    dfdt = time_mask_derivative(tPhys, ti, beta_t)
    U = compliance._solve_fe(
        KE, xtJoint, Emin, Emax, penal, edofMat, freedofs, F, ndof, x0=x0
    )
    return _gravity_compliance_from_U(
        xPhys, xtJoint, t_mask, dfdt, KE, edofMat, Emin, Emax, penal, C, U
    )


def batched_whole_and_gravity_compliance(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    KE: Float[Tensor, "8 8"],
    edofMat: Int[Tensor, "nelx*nely 8"],
    Emin: float,
    Emax: float,
    penal: float,
    freedofs: Int[Tensor, " n_free"],
    F: Float[Tensor, " ndof"],
    ndof: int,
    C: Tensor,
    beta_t: float,
    stage_times: list[float],
    *,
    x0: Float[Tensor, "n_stage_plus_1 ndof"] | None = None,
) -> tuple[
    float,
    Float[Tensor, "nely nelx"],
    list[tuple[float, Float[Tensor, " nely*nelx"], Float[Tensor, " nely*nelx"]]],
    Float[Tensor, "n_stage_plus_1 ndof"],
]:
    """`whole_compliance`'s solve plus every stage's `gravity_compliance` solve, as one
    batched `FemSolve` call (`plans/torch_port_part2.md` Phase 3.3's batching
    requirement) -- the hand-derived counterpart of
    `sttopt.compliance.batched_whole_and_gravity_compliance`.

    Only the solve is batched; the value/sensitivity algebra afterwards is exactly
    `_whole_compliance_from_U`/`_gravity_compliance_from_U`, applied per row -- the same
    hand-derived formulas `whole_compliance`/`gravity_compliance` use standalone.

    :param stage_times: this iteration's `ti` for each gravity stage, in order.
    :param x0: optional warm start, `(1 + len(stage_times), ndof)` -- typically the
        previous iteration's returned `U`.
    :return: `(c, dcx, stages, U)` where `stages[i]` is `(cg, dcx_g, dct_g)` for
        `stage_times[i]` and `U` is `(1 + len(stage_times), ndof)`, for the next
        iteration's `x0`.
    """
    nely, nelx = xPhys.shape
    mask = torch_fem.free_mask(ndof, freedofs, device=xPhys.device)

    density_rows = [torch_fem.simp_density(xPhys, Emin, Emax, penal).flatten()]
    F_rows = [F]
    stage_loads = []
    for ti in stage_times:
        t_mask, xtJoint, F_g = compliance._gravity_load(
            xPhys, tPhys, ti, C, beta_t, ndof
        )
        dfdt = time_mask_derivative(tPhys, ti, beta_t)
        density_rows.append(
            torch_fem.simp_density(xtJoint, Emin, Emax, penal).flatten()
        )
        F_rows.append(F_g)
        stage_loads.append((xtJoint, t_mask, dfdt))

    density = torch.stack(density_rows)
    F_stack = torch.stack(F_rows)
    U = compliance._solve_fe_batched(
        KE, density, edofMat, mask, nelx, nely, F_stack, x0=x0
    )

    c, dcx = _whole_compliance_from_U(xPhys, KE, edofMat, Emin, Emax, penal, U[0])
    stages = [
        _gravity_compliance_from_U(
            xPhys, xtJoint, t_mask, dfdt, KE, edofMat, Emin, Emax, penal, C, U[1 + i]
        )
        for i, (xtJoint, t_mask, dfdt) in enumerate(stage_loads)
    ]
    return c, dcx, stages, U
