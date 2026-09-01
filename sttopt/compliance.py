"""SIMP compliance and its density/time sensitivities, under two load models.

`whole_compliance` is the "whole structure" objective under a fixed external point
load. `gravity_compliance` is the self-weight objective for one deposition-order
stage: only elements already "active" by stage time `ti` (per `time_mask`) carry
load, so the gravity load itself depends on both density and time, giving both a
`dcx` and `dct` sensitivity plus an extra adjoint term (see `gravity_compliance`).

Both solve via `torch_solve.FemSolve` (`plans/torch_port_part2.md` Phase 3.3), the
multigrid-CG solve as an autograd `Function` -- no NumPy round trip -- and get their
sensitivities from autograd through it (Phase 3.4), rather than hand-derived algebra.
`tests/reference/compliance.py` keeps the hand-derived predecessor these replaced, as
a cross-check (`tests/test_reference_sweep.py`) and a timing baseline
(`benchmarks/bench_sensitivities.py`). Array-order and mesh conventions (element
order, dof layout) follow `conventions.md` and `fem.py`.

`batched_whole_and_gravity_compliance` is `optimize.step`'s entry point: one `FemSolve`
call for `whole_compliance`'s solve plus every stage's `gravity_compliance` solve,
instead of `1 + nStage` separate calls. `whole_compliance`/`gravity_compliance` stay as
the single-solve API, and as what the batched function is checked against.
"""

import math

import torch
from jaxtyping import Bool, Float, Int
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
    *,
    x0: Float[Tensor, " ndof"] | None = None,
) -> Float[Tensor, " ndof"]:
    """Solve `K @ U = F` via `torch_solve.FemSolve`'s multigrid-CG, `K` implicit from
    `xPhys`'s SIMP-scaled stiffness. `xPhys`'s own gradient reaches `FemSolve`'s
    `density` input through `torch_fem.simp_density`'s ordinary (non-`Function`) power
    law, so the caller's autograd graph sees one differentiable chain end to end.

    :param x0: optional warm start (e.g. the previous iteration's `U`).
    """
    nely, nelx = xPhys.shape
    density = torch_fem.simp_density(xPhys, Emin, Emax, penal)
    mask = torch_fem.free_mask(ndof, freedofs, device=xPhys.device)
    return torch_solve.femsolve(
        density.flatten(), F, edofMat, KE, mask, nelx, nely, x0=x0
    )


def _solve_fe_batched(
    KE: Float[Tensor, "8 8"],
    density: Float[Tensor, "n_batch nel"],
    edofMat: Int[Tensor, "nel 8"],
    mask: Bool[Tensor, " ndof"],
    nelx: int,
    nely: int,
    F: Float[Tensor, "n_batch ndof"],
    *,
    x0: Float[Tensor, "n_batch ndof"] | None = None,
) -> Float[Tensor, "n_batch ndof"]:
    """Batched counterpart of `_solve_fe`: `density`/`F` already carry the leading batch
    dim (`batched_whole_and_gravity_compliance` has already applied SIMP and built the
    mask, unlike `_solve_fe`, since those are per-row there but shared here). Kept as
    its own patch point -- symmetric with `_solve_fe` -- so a solver substitution (e.g.
    `benchmarks/calibrate_cg_rtol.py`'s `mgcg_backend`) can intercept the batched path
    too, rather than only ever seeing it as unbatched calls.
    """
    return torch_solve.femsolve(density, F, edofMat, KE, mask, nelx, nely, x0=x0)


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
) -> tuple[Float[Tensor, ""], Float[Tensor, " ndof"]]:
    """SIMP compliance under a fixed external load `F`, differentiable end to end
    w.r.t. `xPhys` (autograd sensitivity path, `plans/torch_port_part2.md` Phase 3.4).

    :param x0: optional warm start for the solve.
    :return: `(c, U)` -- `U` for the caller's next-iteration warm start.
    """
    U = _solve_fe(KE, xPhys, Emin, Emax, penal, edofMat, freedofs, F, ndof, x0=x0)
    nely, nelx = xPhys.shape
    ce = _element_strain_energy(U, edofMat, KE, nely, nelx)
    simp = Emin + xPhys**penal * (Emax - Emin)
    return torch.sum(simp * ce), U


def _gravity_load(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    ti: float,
    C: Tensor,
    beta_t: float,
    ndof: int,
) -> tuple[
    Float[Tensor, "nely nelx"],
    Float[Tensor, "nely nelx"],
    Float[Tensor, " ndof"],
]:
    """One gravity stage's `t_mask`/`xtJoint`/load vector `F` -- the part of
    `gravity_compliance` that has to run before the solve, shared with the batched
    path. Autograd differentiates through `t_mask`/`xtJoint` itself, so unlike
    `tests/reference/compliance.py`'s hand-derived counterpart, this has no separate
    `dfdt` to compute.
    """
    t_mask = time_mask(tPhys, ti, beta_t)
    xtJoint = xPhys * t_mask

    f = -(C @ xtJoint.flatten())
    F = torch.zeros(ndof, dtype=xPhys.dtype, device=xPhys.device)
    F[1::2] = f  # y-dof of each node; x-dof stays 0 (gravity acts in -y)
    return t_mask, xtJoint, F


def gravity_compliance(
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    KE: Float[Tensor, "8 8"],
    edofMat: Int[Tensor, "nelx*nely 8"],
    Emin: float,
    Emax: float,
    penal: float,
    ti: float,
    C: Tensor,
    beta_t: float,
    freedofs: Int[Tensor, " n_free"],
    ndof: int,
    *,
    x0: Float[Tensor, " ndof"] | None = None,
) -> tuple[Float[Tensor, ""], Float[Tensor, " ndof"]]:
    """SIMP compliance under self-weight gravity, differentiable end to end w.r.t.
    `xPhys`/`tPhys` -- autograd counterpart of `whole_compliance`.

    Only elements active by stage time `ti` (per `time_mask`, sharpness `beta_t` --
    MATLAB source's `lamda`/`rou`) carry density-weighted self-weight load, built via
    `gravity.gravity_load_matrix`'s `C`.

    :param x0: optional warm start for the solve.
    :return: `(cg, U)`.
    """
    _, xtJoint, F = _gravity_load(xPhys, tPhys, ti, C, beta_t, ndof)
    U = _solve_fe(KE, xtJoint, Emin, Emax, penal, edofMat, freedofs, F, ndof, x0=x0)
    nely, nelx = xPhys.shape
    ce = _element_strain_energy(U, edofMat, KE, nely, nelx)
    simp = Emin + xtJoint**penal * (Emax - Emin)
    return torch.sum(simp * ce), U


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
    Float[Tensor, ""],
    list[Float[Tensor, ""]],
    Float[Tensor, "n_stage_plus_1 ndof"],
]:
    """`whole_compliance`'s solve plus every stage's `gravity_compliance` solve, as one
    batched `FemSolve` call -- `optimize.step`'s entry point.

    :param stage_times: this iteration's `ti` for each gravity stage, in order.
    :param x0: optional warm start, `(1 + len(stage_times), ndof)`.
    :return: `(c, [cg, ...], U)`, `U` for the next iteration's `x0`.
    """
    nely, nelx = xPhys.shape
    mask = torch_fem.free_mask(ndof, freedofs, device=xPhys.device)

    density_rows = [torch_fem.simp_density(xPhys, Emin, Emax, penal).flatten()]
    F_rows = [F]
    xtJoints = []
    for ti in stage_times:
        _, xtJoint, F_g = _gravity_load(xPhys, tPhys, ti, C, beta_t, ndof)
        density_rows.append(
            torch_fem.simp_density(xtJoint, Emin, Emax, penal).flatten()
        )
        F_rows.append(F_g)
        xtJoints.append(xtJoint)

    density = torch.stack(density_rows)
    F_stack = torch.stack(F_rows)
    U = _solve_fe_batched(KE, density, edofMat, mask, nelx, nely, F_stack, x0=x0)

    ce_whole = _element_strain_energy(U[0], edofMat, KE, nely, nelx)
    simp_whole = Emin + xPhys**penal * (Emax - Emin)
    c = torch.sum(simp_whole * ce_whole)

    stages = []
    for i, xtJoint in enumerate(xtJoints):
        ce = _element_strain_energy(U[1 + i], edofMat, KE, nely, nelx)
        simp = Emin + xtJoint**penal * (Emax - Emin)
        stages.append(torch.sum(simp * ce))
    return c, stages, U
