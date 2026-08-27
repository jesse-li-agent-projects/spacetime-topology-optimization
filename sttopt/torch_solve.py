"""The multigrid-CG FEM solve as an autograd `Function`, `plans/torch_port_part2.md`
Phase 3.3.

For `K U = F` with `K = sum_e d_e KE` symmetric positive definite on the free dofs and
a downstream scalar `L(U)` with `g = dL/dU`, the adjoint is

    lambda = K^-1 g
    dL/dF   = lambda
    dL/dd_e = -(lambda_e @ KE) . U_e      (elementwise contraction over 8 element dofs)

-- the same operator as the forward solve, so `backward` runs one more MGCG solve
against `torch_mg`'s hierarchy rather than differentiating through the CG iteration
itself (which would be both data-dependent in iteration count and ruinous in memory).

`density` here is `torch_fem.simp_density`'s *output* -- the per-element stiffness
scale `K` is actually built from -- not `xPhys`. The SIMP power law is ordinary
differentiable torch code, so a caller composes `simp_density(xPhys, ...)` with
`FemSolve` and autograd chains the two; `FemSolve` itself stays free of the physics
constants (`Emin`/`Emax`/`penal`) that would otherwise have to round-trip through it.

**The self-adjoint warm start.** A generic adjoint costs a second CG solve per forward
solve. For the compliance objectives this is avoidable: `L = sum_e simp_e * ce_e` with
`ce_e = Ue^T KE Ue` has `dL/dU = 2 K U = 2 F`, so `lambda = 2 U` exactly and the adjoint
solve should converge at iteration zero. Rather than special-casing compliance --
a correctness trap the moment a non-compliance scalar is differentiated -- `backward`
always warm-starts from `alpha * U`, `alpha = (U . g) / (U . F)`, the least-squares fit
of a multiple of `U` to `K^-1 g`. When `g` is parallel to `F` this lands on the exact
answer and `pcg` returns at iteration zero (it checks convergence before the first
iteration for exactly this reason); otherwise it is still a good warm start and the
solve is still exact.

**Hierarchy reuse.** The forward and backward of one solve share the same `K`, so
`forward` saves the hierarchy it built and `backward` reuses it rather than rebuilding
it (~24% of one solve's cost, per part 1's profile). The hierarchy cannot be reused
*across* solves (a new `FemSolve.apply` call sees a new density field), which is why
the hierarchy lives on `ctx` rather than anywhere longer-lived.
"""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from sttopt import torch_fem, torch_mg


class FemSolve(torch.autograd.Function):
    """`K @ U = F` as a differentiable op, `K` implicit from `density` and `KE`.

    See the module docstring for the adjoint and the self-adjoint warm start.
    Supports the same batching as `torch_mg.solve`: `density` and `F` may each carry
    leading batch dims, broadcasting against each other as `torch_fem.matvec` does.

    A `CGConvergenceError` from either the forward or the adjoint solve propagates
    through unchanged -- there is no silent fallback and no returning an unconverged
    `U`/`lambda`.
    """

    @staticmethod
    def forward(
        ctx,
        density: Float[Tensor, "*dbatch nel"],
        F: Float[Tensor, "*fbatch ndof"],
        edofMat: Int[Tensor, "nel 8"],
        KE: Float[Tensor, "8 8"],
        mask: Bool[Tensor, " ndof"],
        nelx: int,
        nely: int,
        rtol: float,
        max_iter: int,
        x0: Float[Tensor, "*batch ndof"] | None,
        omega: float,
        n_smooth: int,
        gamma: int,
        max_coarse_elements: int,
        info: dict | None,
    ) -> Float[Tensor, "*batch ndof"]:
        levels = torch_mg.build_hierarchy(
            density,
            edofMat,
            KE,
            mask,
            nelx,
            nely,
            max_coarse_elements=max_coarse_elements,
        )
        precond = torch_mg.VCycle(levels, omega=omega, n_smooth=n_smooth, gamma=gamma)
        b = torch_fem.project(F, mask)
        U, n_iter = torch_fem.pcg(
            levels[0].apply_A, b, precond, rtol=rtol, max_iter=max_iter, x0=x0
        )
        U = torch_fem.project(U, mask)

        ctx.save_for_backward(U, density, F)
        # The hierarchy, KE and solver settings are constants of this solve, not
        # autograd inputs -- kept on ctx directly (not save_for_backward) so backward
        # can reuse them without re-deriving anything from density/F.
        ctx.levels = levels
        ctx.edofMat = edofMat
        ctx.KE = KE
        ctx.mask = mask
        ctx.rtol = rtol
        ctx.max_iter = max_iter
        ctx.omega, ctx.n_smooth, ctx.gamma = omega, n_smooth, gamma
        ctx.info = info
        if info is not None:
            info["forward_n_iter"] = n_iter
        return U

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "*batch ndof"]):
        U, density, F = ctx.saved_tensors

        # alpha = (U . g) / (U . F): the least-squares fit of alpha*U to K^-1 g, exact
        # when g is parallel to F (the compliance case) and a good warm start otherwise.
        UF = (U * F).sum(dim=-1)
        Ug = (U * grad_output).sum(dim=-1)
        alpha = torch_fem.safe_div(Ug, UF)
        x0 = alpha[..., None] * U

        precond = torch_mg.VCycle(
            ctx.levels, omega=ctx.omega, n_smooth=ctx.n_smooth, gamma=ctx.gamma
        )
        b = torch_fem.project(grad_output, ctx.mask)
        lam, n_iter = torch_fem.pcg(
            ctx.levels[0].apply_A,
            b,
            precond,
            rtol=ctx.rtol,
            max_iter=ctx.max_iter,
            x0=x0,
        )
        lam = torch_fem.project(lam, ctx.mask)
        if ctx.info is not None:
            ctx.info["backward_n_iter"] = n_iter

        grad_density = None
        if ctx.needs_input_grad[0]:
            Ue = U[..., ctx.edofMat]  # (*batch, nel, 8)
            lam_e = lam[..., ctx.edofMat]  # (*batch, nel, 8)
            grad_density = -torch.sum((lam_e @ ctx.KE) * Ue, dim=-1)

        grad_F = lam if ctx.needs_input_grad[1] else None

        return (
            grad_density,
            grad_F,
            None,  # edofMat
            None,  # KE
            None,  # mask
            None,  # nelx
            None,  # nely
            None,  # rtol
            None,  # max_iter
            None,  # x0
            None,  # omega
            None,  # n_smooth
            None,  # gamma
            None,  # max_coarse_elements
            None,  # info
        )


def femsolve(
    density: Float[Tensor, "*dbatch nel"],
    F: Float[Tensor, "*fbatch ndof"],
    edofMat: Int[Tensor, "nel 8"],
    KE: Float[Tensor, "8 8"],
    mask: Bool[Tensor, " ndof"],
    nelx: int,
    nely: int,
    *,
    rtol: float = 1e-8,
    max_iter: int = 500,
    x0: Float[Tensor, "*batch ndof"] | None = None,
    omega: float = 0.6,
    n_smooth: int = 2,
    gamma: int = 1,
    max_coarse_elements: int = torch_mg.MAX_COARSE_ELEMENTS,
    info: dict | None = None,
) -> Float[Tensor, "*batch ndof"]:
    """`FemSolve.apply` with `torch_mg.solve`'s keyword defaults bound in.

    :param density: SIMP-scaled per-element stiffness, `torch_fem.simp_density`'s output
        (the differentiable leaf `K` is built from -- not `xPhys`).
    :param F: right-hand side load vector(s).
    :param edofMat: element dof map.
    :param KE: reference element stiffness matrix.
    :param mask: free-dof mask.
    :param nelx: fine-mesh elements in x.
    :param nely: fine-mesh elements in y.
    :param rtol: CG relative-residual tolerance, forward and adjoint alike.
    :param max_iter: CG iteration cap.
    :param x0: optional warm start for the forward solve (the adjoint warm-starts on
        its own, from `alpha * U` -- see the module docstring).
    :param omega: damped-Jacobi relaxation factor for the V-cycle smoother.
    :param n_smooth: pre-/post-smoothing sweeps per level.
    :param gamma: coarse cycles per visit; 1 is a V-cycle, 2 a W-cycle.
    :param max_coarse_elements: coarsening stops at or below this element count.
    :param info: optional dict that receives `forward_n_iter`/`backward_n_iter`, for
        tests that assert on iteration counts (e.g. the self-adjoint zero-iteration
        adjoint solve).
    :return: `U`.
    """
    return FemSolve.apply(
        density,
        F,
        edofMat,
        KE,
        mask,
        nelx,
        nely,
        rtol,
        max_iter,
        x0,
        omega,
        n_smooth,
        gamma,
        max_coarse_elements,
        info,
    )
