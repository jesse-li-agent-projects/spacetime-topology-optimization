"""Matrix-free element-by-element FEM stiffness operator and Jacobi-PCG solver.

Alongside `sttopt.fem`'s assemble-then-`spsolve` path (unchanged, still the code the
optimizer runs today -- see `plans/archive/torch_port.md`). Nothing here is wired into any
production call site yet; this module is the correctness scaffold Phase 1 of that plan
asks for. Every element shares the same reference stiffness matrix `KE`
(`fem.plane_stress_KE`), so the global matvec is done without ever assembling a global
matrix: gather per-element dofs, multiply by `KE`, scale by density, scatter back.

Boundary conditions are applied by projection rather than by extracting a free-dof
submatrix: `A(v) = P(K @ P(v))` with `P` zeroing the fixed dofs. This is exactly the
free-dof system (a fixed dof of `v` cannot affect `K @ P(v)` at another fixed dof after
the second projection, and the fixed rows of the "matrix" this defines are the identity
on the zero vector), and it keeps the whole operator matrix-free.

Every function here takes torch tensors, is dtype/device-agnostic (float64 is the
caller's responsibility -- nothing here calls `torch.set_default_dtype`), and supports
an optional leading batch shape so a batch of right-hand sides and/or a batch of density
fields can share one call. See `matvec`'s docstring for the broadcasting rule.
"""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

#: Compact the CG batch only once the active rows have fallen to this fraction of their
#: count at the last compaction. Restricting the operator gathers the whole multigrid
#: hierarchy -- roughly one V-cycle of work -- so compacting on every single retired row
#: can cost more than the rows it drops save.
COMPACTION_RATIO = 0.75


class CGConvergenceError(RuntimeError):
    """Raised when preconditioned CG fails to reach `rtol` within `max_iter`.

    Carries the achieved relative residual(s) and iteration count so a caller can log
    or debug the failure; there is no "return the unconverged answer" path in this
    module; see `plans/archive/torch_port.md`'s Phase 1 design.
    """

    def __init__(
        self, residuals: Float[Tensor, " *batch"], n_iter: int, rtol: float
    ) -> None:
        self.residuals = residuals
        self.n_iter = n_iter
        self.rtol = rtol
        worst = residuals.max().item()
        failed = torch.nonzero(residuals > rtol).flatten().tolist()
        super().__init__(
            f"CG did not converge to rtol={rtol:g} within {n_iter} iterations "
            f"(worst relative residual {worst:g}); failed batch indices {failed}"
        )


def simp_density(
    xPhys: Float[Tensor, "*batch nel"], Emin: float, Emax: float, penal: float
) -> Float[Tensor, "*batch nel"]:
    """SIMP-interpolated per-element stiffness scale `Emin + xPhys**penal * (Emax - Emin)`."""
    return Emin + xPhys**penal * (Emax - Emin)


def matvec(
    U: Float[Tensor, "*ubatch ndof"],
    density: Float[Tensor, "*dbatch nel"],
    edofMat: Int[Tensor, "nel 8"],
    KE: Float[Tensor, "8 8"],
    ndof: int,
) -> Float[Tensor, "*batch ndof"]:
    """Matrix-free global stiffness matvec `K @ U`, `K` implicit from `density` and `KE`.

    `U`'s and `density`'s leading batch dims broadcast against each other (standard
    NumPy/torch right-aligned broadcasting) to a combined `*batch`; this is what lets a
    single call batch over displacement vectors, density fields, or both together (e.g.
    the `nStage` gravity solves, which share `U`'s shape but differ in density).
    """
    Ue = U[..., edofMat]  # (*ubatch, nel, 8)
    UeKE = Ue @ KE  # (*ubatch, nel, 8)
    scaled = UeKE * density[..., :, None]  # (*batch, nel, 8), batch dims broadcast
    batch_shape = scaled.shape[:-2]
    flat = scaled.reshape(*batch_shape, -1)  # (*batch, nel*8)
    out = flat.new_zeros(*batch_shape, ndof)
    out.index_add_(-1, edofMat.reshape(-1), flat)
    return out


def matvec_diagonal(
    density: Float[Tensor, "*batch nel"],
    edofMat: Int[Tensor, "nel 8"],
    KE: Float[Tensor, "8 8"],
    ndof: int,
) -> Float[Tensor, "*batch ndof"]:
    """Matrix-free `diag(K)`: scatter-add of `diag(KE)` scaled by `density`."""
    contrib = density[..., :, None] * torch.diagonal(KE)  # (*batch, nel, 8)
    batch_shape = contrib.shape[:-2]
    flat = contrib.reshape(*batch_shape, -1)
    out = flat.new_zeros(*batch_shape, ndof)
    out.index_add_(-1, edofMat.reshape(-1), flat)
    return out


def free_mask(
    ndof: int, freedofs: Int[Tensor, " n_free"], device=None
) -> Bool[Tensor, " ndof"]:
    """Boolean mask, `True` at free dofs and `False` at fixed dofs -- the `P` in `A(v) = P(K @ P(v))`."""
    mask = torch.zeros(ndof, dtype=torch.bool, device=device)
    mask[freedofs] = True
    return mask


def project(
    v: Float[Tensor, "*batch ndof"], mask: Bool[Tensor, " ndof"]
) -> Float[Tensor, "*batch ndof"]:
    """Zero the fixed dofs of `v` (the dofs where `mask` is `False`)."""
    return v * mask


def operator(
    v: Float[Tensor, "*batch ndof"],
    density: Float[Tensor, "*dbatch nel"],
    edofMat: Int[Tensor, "nel 8"],
    KE: Float[Tensor, "8 8"],
    ndof: int,
    mask: Bool[Tensor, " ndof"],
) -> Float[Tensor, "*batch ndof"]:
    """The projected, matrix-free operator `A(v) = P(K @ P(v))` that PCG runs against."""
    return project(matvec(project(v, mask), density, edofMat, KE, ndof), mask)


def jacobi_preconditioner_diag(
    density: Float[Tensor, "*batch nel"],
    edofMat: Int[Tensor, "nel 8"],
    KE: Float[Tensor, "8 8"],
    ndof: int,
    mask: Bool[Tensor, " ndof"],
) -> Float[Tensor, "*batch ndof"]:
    """Diagonal of the projected operator, with fixed dofs set to 1.0.

    The 1.0 makes the diagonal (and hence Jacobi's `1/diag` preconditioner) invertible
    on the projected system, where the true diagonal of `P(K @ P(.))` is 0 at fixed
    dofs.
    """
    diag = matvec_diagonal(density, edofMat, KE, ndof)
    return torch.where(mask, diag, torch.ones((), dtype=diag.dtype, device=diag.device))


def safe_div(
    num: Float[Tensor, " *batch"], den: Float[Tensor, " *batch"]
) -> Float[Tensor, " *batch"]:
    """`num / den`, `0` where `den` is exactly zero -- a converged batch member's
    `(p, Ap)` inner product (for `alpha`) or `rz_old` (for `beta`), which would
    otherwise divide `0/0` into a nan. See `pcg`'s docstring.
    """
    safe_den = torch.where(den != 0, den, torch.ones_like(den))
    return torch.where(den != 0, num / safe_den, torch.zeros_like(den))


def pcg(
    apply_A,
    b: Float[Tensor, "*batch ndof"],
    apply_M,
    *,
    rtol: float = 1e-8,
    max_iter: int = 10000,
    x0: Float[Tensor, "*batch ndof"] | None = None,
) -> tuple[Float[Tensor, "*batch ndof"], int]:
    """Preconditioned CG on the batched system `apply_A(x) = b`.

    Iterates until every batch row has reached `rtol` on `||r|| / ||b||` or `max_iter`
    is hit. Supports warm-starting via `x0`. Raises `CGConvergenceError` -- never
    returns an unconverged result -- if any batch row fails to converge within
    `max_iter`.

    **Rows retire as they converge.** Rows converge tens of iterations apart on the
    production batch, so running every row for the slowest row's iteration count wastes
    a third of the work at the larger meshes. A converged row's `x` is therefore moved
    into the output and the row dropped from the batch, and the loop continues on a
    restricted operator. This is exact, not an approximation: `alpha`/`beta`/`rz` are
    per-row scalars and the operator is block-diagonal across rows, so retiring a row
    changes no other row's arithmetic -- results match the all-rows schedule up to
    kernel reduction order, not bitwise. Retirement needs `apply_A` and `apply_M` to
    supply `select` (see `torch_mg._Level.select`) and a single batch dim; with a plain
    callable, or any other batch rank, the whole batch runs one shared schedule as
    before.

    `apply_M` must be a fixed symmetric positive-definite linear operator; anything
    state-dependent or nonsymmetric (an inner CG, an unequal-sweep multigrid cycle)
    silently invalidates the short recurrence this algorithm relies on.

    `safe_div` guards `alpha`/`beta` against an exact-zero denominator, which a row with
    an all-zero `b` reaches (it converges and retires at iteration 0, but the guard also
    covers a row whose `rz` underflows to zero mid-solve).

    :param apply_A: callable, `Tensor -> Tensor`, the (implicitly batched) operator;
        optionally with `select(rows)` returning the same operator on a row subset.
    :param b: right-hand side.
    :param apply_M: callable, `Tensor -> Tensor`, the preconditioner approximating `A^-1`
        (`1/jacobi_preconditioner_diag` for the diagonal one, `torch_mg.VCycle` for
        multigrid); same optional `select`.
    :param rtol: relative residual tolerance `||r|| / ||b||`.
    :param max_iter: maximum CG iterations before raising.
    :param x0: optional warm-start initial guess, defaults to zero.
    :return: `(x, n_iter)` -- the solution and the number of iterations actually run,
        i.e. the count the last row to converge needed.
    """
    b_norm = b.norm(dim=-1)
    b_norm_safe = torch.where(b_norm > 0, b_norm, torch.ones_like(b_norm))

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - apply_A(x)
    rel_resid = r.norm(dim=-1) / b_norm_safe

    # Checked before the first iteration, not only after: a warm start can already be
    # converged, and an exact one makes the first alpha a 0/0 nan.
    converged = rel_resid <= rtol
    n_remaining = int(torch.count_nonzero(~converged))
    if n_remaining == 0:
        return x, 0

    can_retire = (
        b.ndim == 2
        and r.ndim == 2
        and hasattr(apply_A, "select")
        and hasattr(apply_M, "select")
    )
    if can_retire:
        # Retired rows land in these full-size buffers; `active` maps a row of the
        # shrinking batch back to its original index, for the output and the error.
        x_out = torch.zeros_like(b)
        resid_out = torch.zeros_like(rel_resid)
        active = torch.arange(b.shape[0], device=b.device)
        n_at_last_compaction = b.shape[0]

    def worth_compacting() -> bool:
        return can_retire and n_remaining <= COMPACTION_RATIO * n_at_last_compaction

    def retire() -> Int[Tensor, " k"]:
        """Bank the converged rows, shrink the operators, and return the rows kept.

        The caller slices its own iteration vectors by the returned indices -- which
        ones exist depends on whether the loop has started yet.
        """
        nonlocal apply_A, apply_M, active, n_at_last_compaction, b_norm_safe, rel_resid
        done, keep = (torch.nonzero(m).flatten() for m in (converged, ~converged))
        x_out[active[done]] = x[done]
        resid_out[active[done]] = rel_resid[done]
        active = active[keep]
        n_at_last_compaction = keep.numel()
        apply_A, apply_M = apply_A.select(keep), apply_M.select(keep)
        b_norm_safe, rel_resid = b_norm_safe[keep], rel_resid[keep]
        return keep

    if worth_compacting():
        keep = retire()
        x, r = x[keep], r[keep]

    z = apply_M(r)
    p = z.clone()
    rz_old = (r * z).sum(dim=-1)

    for n_iter in range(1, max_iter + 1):
        Ap = apply_A(p)
        alpha = safe_div(rz_old, (p * Ap).sum(dim=-1))[..., None]
        x.addcmul_(alpha, p)
        r.addcmul_(alpha, Ap, value=-1)
        rel_resid = r.norm(dim=-1) / b_norm_safe
        converged = rel_resid <= rtol
        n_remaining = int(torch.count_nonzero(~converged))
        if n_remaining == 0:
            break
        z = apply_M(r)
        rz_new = (r * z).sum(dim=-1)
        p.mul_(safe_div(rz_new, rz_old)[..., None]).add_(z)
        rz_old = rz_new

        if worth_compacting():
            keep = retire()
            x, r, p, rz_old = x[keep], r[keep], p[keep], rz_old[keep]
    else:
        if can_retire:
            resid_out[active] = rel_resid
            rel_resid = resid_out
        raise CGConvergenceError(rel_resid, max_iter, rtol)

    if can_retire:
        x_out[active] = x
        return x_out, n_iter
    return x, n_iter


def solve(
    F: Float[Tensor, "*batch ndof"],
    xPhys: Float[Tensor, "*dbatch nel"],
    edofMat: Int[Tensor, "nel 8"],
    KE: Float[Tensor, "8 8"],
    Emin: float,
    Emax: float,
    penal: float,
    mask: Bool[Tensor, " ndof"],
    *,
    rtol: float = 1e-8,
    max_iter: int = 10000,
    x0: Float[Tensor, "*batch ndof"] | None = None,
) -> tuple[Float[Tensor, "*batch ndof"], int]:
    """Convenience wrapper: SIMP density from `xPhys`, then Jacobi-PCG for `K @ U = F`.

    Boundary conditions are handled by projection (see module docstring), so fixed dofs
    of `F` are ignored and come back exactly 0.0 in `U` regardless of their input value.

    :return: `(U, n_iter)`, matching `pcg`.
    """
    ndof = mask.shape[-1]
    density = simp_density(xPhys, Emin, Emax, penal)

    def apply_A(v):
        return operator(v, density, edofMat, KE, ndof, mask)

    diag = jacobi_preconditioner_diag(density, edofMat, KE, ndof, mask)
    b = project(F, mask)
    U, n_iter = pcg(
        apply_A,
        b,
        lambda r: r / diag,
        rtol=rtol,
        max_iter=max_iter,
        x0=x0,
    )
    return project(U, mask), n_iter
