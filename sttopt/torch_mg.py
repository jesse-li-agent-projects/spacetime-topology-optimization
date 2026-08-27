"""Geometric-multigrid V-cycle preconditioner for `torch_fem`'s matrix-free operator.

Jacobi-preconditioned CG on this problem costs O(sqrt(cond)) iterations, and `cond`
carries both the O(h^-2) mesh factor and the SIMP stiffness contrast (`Emax/Emin`,
~1e9 at a near-binary design). Multigrid removes the mesh factor and -- with a Galerkin
coarse operator -- most of the contrast factor too. See `plans/torch_port.md` Phase 1.

Three choices make this work and are worth stating up front:

**Exact Galerkin coarse operators, still matrix-free.** Piecewise-linear interpolation
inside a coarse element uses only that element's own four corner nodes, so the Galerkin
product `P^T K P` decomposes element by element: a coarse element's matrix is
`sum_children S_c^T K_child S_c` with `S_c` a fixed 8x8 interpolation matrix depending
only on the child's position in the block. So each level is again "one 8x8 matrix per
element, gather-multiply-scatter" -- no global sparse matrix anywhere, and no accuracy
lost to re-discretization with averaged densities (which is the usual matrix-free
shortcut and the one that degrades under high contrast).

**Coarsening by a per-dimension integer factor, not strictly by 2**, and **stopping
early on purpose.** 180x60 halves twice and then reaches 45x15, where neither dimension
is even, so `coarsening_factors` also admits a factor of 3. But coarsening as far as the
grid allows is actively harmful here: a coarse grid can only help if it still resolves
the design's bars, and on a hard 0/1 cantilever at 90x30, descending 90x30 -> 45x15 ->
15x5 -> 5x5 costs 119 CG iterations against 31 for stopping at 45x15. `MAX_COARSE_ELEMENTS`
therefore stops the hierarchy while the coarse grid is still meaningful and solves that
level by dense Cholesky. The default is sized so 90x30, 180x60 and 360x120 all bottom out
at 45x15 (1472 dofs, ~10 ms to factorize on GPU).

**Boundary conditions by projection at every level**, matching `torch_fem`'s
`A(v) = P(K P(v))`. Coarse masks are the fine mask sampled at the collocated coarse
nodes, and both transfers are projected, which keeps restriction the exact adjoint of
prolongation and hence keeps the whole V-cycle a symmetric positive-definite
preconditioner -- a requirement of CG's convergence theory, not a nicety.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import sttopt.fem as fem
from sttopt import torch_fem

#: Stop coarsening once a level has at most this many elements; it is solved densely.
#: Trades a bigger dense factorization against a coarse grid too coarse to resolve the
#: design -- see the module docstring for the measurements behind the default.
MAX_COARSE_ELEMENTS = 700


def coarsening_factors(nelx: int, nely: int) -> tuple[int, int]:
    """Per-dimension coarsening factors for one multigrid level, `(1, 1)` if none apply.

    Each dimension independently takes the smallest of 2 or 3 that divides it, so a
    dimension only ever coarsens onto a node subset of itself (node `i` of the coarse
    grid is node `k*i` of the fine one) and the transfers stay exact. `(1, 1)` means the
    mesh cannot be coarsened further and the caller should solve this level directly.

    Admitting 3 as well as 2 matters: the production meshes reach 45x15 after halving,
    where a factor of 2 no longer divides either dimension.

    :param nelx: elements in x at this level.
    :param nely: elements in y at this level.
    :return: `(kx, ky)`, each in `{1, 2, 3}`.
    """

    def factor(n: int) -> int:
        return next((k for k in (2, 3) if n % k == 0), 1)

    return factor(nelx), factor(nely)


def _interp_axis(c: Tensor, k: int) -> Tensor:
    """Piecewise-linear interpolation by factor `k` along the last axis (`nc+1 -> k*nc+1`)."""
    if k == 1:
        return c
    w = torch.arange(k, device=c.device, dtype=c.dtype) / k
    body = c[..., :-1, None] * (1 - w) + c[..., 1:, None] * w
    return torch.cat([body.reshape(*c.shape[:-1], -1), c[..., -1:]], dim=-1)


def _restrict_axis(f: Tensor, k: int) -> Tensor:
    """Exact transpose of `_interp_axis` -- full weighting along the last axis."""
    if k == 1:
        return f
    w = torch.arange(k, device=f.device, dtype=f.dtype) / k
    body = f[..., :-1].reshape(*f.shape[:-1], -1, k)
    out = torch.nn.functional.pad((body * (1 - w)).sum(-1), (0, 1))
    out = out + torch.nn.functional.pad((body * w).sum(-1), (1, 0))
    out[..., -1] += f[..., -1]
    return out


def _on_node_grid(
    v: Float[Tensor, "*batch ndof"], nnx: int, nny: int, kx: int, ky: int, axis_op
) -> Tensor:
    """Apply `axis_op(tensor, k)` along the node grid's x then y axes of a dof vector.

    Reshapes `(*batch, ndof)` to the `(*batch, nny, nnx, 2)` node grid (row-major nodes,
    interleaved x/y dofs -- `fem.node_grid`'s convention), applies the 1D operator along
    each spatial axis in turn, and flattens back. Both dof components transfer
    identically, which is what "vector-valued transfer, acting per component" means here.
    """
    g = v.reshape(*v.shape[:-1], nny, nnx, 2)
    g = axis_op(g.movedim(-2, -1), kx).movedim(-1, -2)
    g = axis_op(g.movedim(-3, -1), ky).movedim(-1, -3)
    return g.reshape(*v.shape[:-1], -1)


def _child_elements(
    nelx: int, nely: int, kx: int, ky: int, device
) -> Int[Tensor, "nel_coarse kx*ky"]:
    """Fine element indices under each coarse element, ordered as `fem.element_dof_map(kx, ky)`."""
    e = torch.arange(nelx * nely, device=device).reshape(nely, nelx)
    blocks = e.reshape(nely // ky, ky, nelx // kx, kx)
    return blocks.permute(0, 2, 1, 3).reshape(-1, ky * kx)


def child_interpolation_matrices(
    kx: int, ky: int, device=None, dtype=torch.float64
) -> Float[Tensor, "kx*ky 8 8"]:
    """Interpolation from one coarse element's 8 dofs to each child element's 8 dofs.

    `S[m]` maps the coarse element's nodal values to child element `m`'s nodal values
    under the same piecewise-linear interpolation the grid transfer uses. Derived by
    running that transfer on a one-coarse-element mesh rather than by writing the
    weights out, so the two can never drift apart.
    """
    edof_coarse = torch.tensor(fem.element_dof_map(1, 1)[0], dtype=torch.int64)
    edof_fine = torch.tensor(fem.element_dof_map(kx, ky), dtype=torch.int64)
    basis = torch.zeros(8, 8, dtype=dtype)
    basis[torch.arange(8), edof_coarse] = 1.0
    fine = _on_node_grid(
        basis, 2, 2, kx, ky, _interp_axis
    )  # (8 coarse dofs, ndof_fine)
    S = fine[:, edof_fine].permute(1, 2, 0)  # (m, fine local dof, coarse local dof)
    return S.to(device=device, dtype=dtype).contiguous()


def _scatter(
    per_element: Float[Tensor, "*batch nel d"], edofMat: Int[Tensor, "nel 8"], ndof: int
) -> Float[Tensor, "*batch ndof"]:
    """Scatter-add per-element dof contributions into a global dof vector."""
    batch_shape = per_element.shape[:-2]
    flat = per_element.reshape(*batch_shape, -1)
    out = flat.new_zeros(*batch_shape, ndof)
    out.index_add_(-1, edofMat.reshape(-1), flat)
    return out


def keff_matvec(
    U: Float[Tensor, "*ubatch ndof"],
    keff: Float[Tensor, "*dbatch nel 8 8"],
    edofMat: Int[Tensor, "nel 8"],
    ndof: int,
) -> Float[Tensor, "*batch ndof"]:
    """Matrix-free matvec for a level carrying an explicit 8x8 matrix per element.

    The coarse-level counterpart of `torch_fem.matvec`, whose single shared `KE` scaled
    by a density no longer suffices once Galerkin coarsening has made every element's
    matrix different. Batch dims of `U` and `keff` broadcast as in `torch_fem.matvec`.
    """
    return _scatter(
        torch.einsum("...ei,...eij->...ej", U[..., edofMat], keff), edofMat, ndof
    )


@dataclass
class _Level:
    """One grid in the hierarchy: its operator, its Jacobi diagonal, and how it coarsens."""

    nelx: int
    nely: int
    ndof: int
    mask: Bool[Tensor, " ndof"]
    diag: Float[Tensor, "*batch ndof"]
    apply_A: Callable[[Tensor], Tensor]
    kx: int = 1
    ky: int = 1
    chol: Float[Tensor, "*batch n n"] | None = None


def _dense_cholesky(
    keff: Float[Tensor, "*batch nel 8 8"],
    edofMat: Int[Tensor, "nel 8"],
    ndof: int,
    mask: Bool[Tensor, " ndof"],
) -> Float[Tensor, "*batch ndof ndof"]:
    """Cholesky factor of the coarsest level's projected operator.

    Fixed dofs get an identity row/column, which makes the projected operator invertible
    (it is singular there by construction) without perturbing the free-dof block, so
    solving against a projected right-hand side returns exactly zero at fixed dofs.
    """
    idx = (edofMat[:, :, None] * ndof + edofMat[:, None, :]).reshape(-1)
    batch_shape = keff.shape[:-3]
    dense = keff.new_zeros(*batch_shape, ndof * ndof)
    dense.index_add_(-1, idx, keff.reshape(*batch_shape, -1))
    dense = dense.reshape(*batch_shape, ndof, ndof)
    dense = (dense + dense.transpose(-1, -2)) / 2
    free = mask[:, None] & mask[None, :]
    dense = torch.where(free, dense, dense.new_zeros(()))
    dense = dense + torch.diag((~mask).to(dense.dtype))
    return torch.linalg.cholesky(dense)


def build_hierarchy(
    density: Float[Tensor, "*batch nel"],
    edofMat: Int[Tensor, "nel 8"],
    KE: Float[Tensor, "8 8"],
    mask: Bool[Tensor, " ndof"],
    nelx: int,
    nely: int,
    *,
    max_coarse_elements: int = MAX_COARSE_ELEMENTS,
) -> list[_Level]:
    """Build the multigrid levels for one density field (or a batch of them).

    Level 0 is the fine mesh and reuses `torch_fem`'s density-times-`KE` operator; each
    coarser level carries the exact Galerkin element matrices built from the level above.
    The last level is factorized densely.

    :param density: SIMP-interpolated element stiffness scale, `torch_fem.simp_density`.
    :param edofMat: fine-mesh element dof map.
    :param KE: reference element stiffness matrix.
    :param mask: fine-mesh free-dof mask.
    :param nelx: fine-mesh elements in x.
    :param nely: fine-mesh elements in y.
    :param max_coarse_elements: stop coarsening at or below this element count.
    :return: levels from finest to coarsest.
    """
    ndof = mask.shape[-1]
    device, dtype = density.device, density.dtype

    diag = torch_fem.jacobi_preconditioner_diag(density, edofMat, KE, ndof, mask)
    levels = [
        _Level(
            nelx=nelx,
            nely=nely,
            ndof=ndof,
            mask=mask,
            diag=diag,
            apply_A=lambda v: torch_fem.operator(v, density, edofMat, KE, ndof, mask),
        )
    ]

    keff = None
    while True:
        top = levels[-1]
        if top.nelx * top.nely <= max_coarse_elements:
            break
        kx, ky = coarsening_factors(top.nelx, top.nely)
        if (kx, ky) == (1, 1):
            break
        S = child_interpolation_matrices(kx, ky, device=device, dtype=dtype)
        children = _child_elements(top.nelx, top.nely, kx, ky, device)
        if keff is None:
            # Level 0 is stored implicitly as density * KE, so start the recursion from
            # the four (kx*ky) fixed products S^T KE S rather than materializing it.
            G = torch.einsum("mpi,pq,mqj->mij", S, KE, S)
            keff = torch.einsum("...em,mij->...eij", density[..., children], G)
        else:
            keff = torch.einsum(
                "mpi,...empq,mqj->...eij", S, keff[..., children, :, :], S
            )

        top.kx, top.ky = kx, ky
        cnelx, cnely = top.nelx // kx, top.nely // ky
        cedof = torch.tensor(
            fem.element_dof_map(cnelx, cnely), dtype=torch.int64, device=device
        )
        cndof = 2 * (cnelx + 1) * (cnely + 1)
        cmask = top.mask.reshape(top.nely + 1, top.nelx + 1, 2)[::ky, ::kx].reshape(-1)
        cdiag = _scatter(torch.diagonal(keff, dim1=-2, dim2=-1), cedof, cndof)
        cdiag = torch.where(cmask, cdiag, cdiag.new_ones(()))
        levels.append(
            _Level(
                nelx=cnelx,
                nely=cnely,
                ndof=cndof,
                mask=cmask,
                diag=cdiag,
                apply_A=(
                    lambda v, k=keff, e=cedof, n=cndof, m=cmask: torch_fem.project(
                        keff_matvec(torch_fem.project(v, m), k, e, n), m
                    )
                ),
            )
        )

    coarsest = levels[-1]
    if keff is None:
        # No coarsening was possible at all: factorize the fine level itself.
        keff = density[..., :, None, None] * KE
        cedof = edofMat
    else:
        cedof = torch.tensor(
            fem.element_dof_map(coarsest.nelx, coarsest.nely),
            dtype=torch.int64,
            device=device,
        )
    coarsest.chol = _dense_cholesky(keff, cedof, coarsest.ndof, coarsest.mask)
    return levels


class VCycle:
    """Multigrid V-cycle acting as an SPD preconditioner `z = M(r)` for CG.

    Damped Jacobi pre- and post-smoothing with equal sweep counts, an exact (dense
    Cholesky) coarse solve, and adjoint transfer operators together make `M` symmetric
    and positive definite, which CG requires. Applying it is a pure function of `r` --
    no internal state carries between calls.
    """

    def __init__(
        self,
        levels: list[_Level],
        *,
        omega: float = 0.6,
        n_smooth: int = 2,
        gamma: int = 1,
    ) -> None:
        """
        :param levels: hierarchy from `build_hierarchy`, finest first.
        :param omega: damped-Jacobi relaxation factor.
        :param n_smooth: pre- and post-smoothing sweeps per level (equal, for symmetry).
        :param gamma: coarse-level cycles per visit; 1 is a V-cycle, 2 a W-cycle.
        """
        self.levels = levels
        self.omega = omega
        self.n_smooth = n_smooth
        self.gamma = gamma

    def _smooth(self, level: _Level, x: Tensor, b: Tensor) -> Tensor:
        for _ in range(self.n_smooth):
            x = x + self.omega * (b - level.apply_A(x)) / level.diag
        return torch_fem.project(x, level.mask)

    def _cycle(self, i: int, b: Tensor) -> Tensor:
        level = self.levels[i]
        if i == len(self.levels) - 1:
            x = torch.linalg.solve_triangular(level.chol, b.unsqueeze(-1), upper=False)
            x = torch.linalg.solve_triangular(
                level.chol.transpose(-1, -2), x, upper=True
            ).squeeze(-1)
            return torch_fem.project(x, level.mask)

        x = self._smooth(level, torch.zeros_like(b), b)
        r = b - level.apply_A(x)
        coarse = self.levels[i + 1]
        rc = _on_node_grid(
            r, level.nelx + 1, level.nely + 1, level.kx, level.ky, _restrict_axis
        )
        rc = torch_fem.project(rc, coarse.mask)
        ec = self._cycle(i + 1, rc)
        for _ in range(self.gamma - 1):
            ec = ec + self._cycle(i + 1, rc - coarse.apply_A(ec))
        ef = _on_node_grid(
            torch_fem.project(ec, coarse.mask),
            coarse.nelx + 1,
            coarse.nely + 1,
            level.kx,
            level.ky,
            _interp_axis,
        )
        x = x + torch_fem.project(ef, level.mask)
        return self._smooth(level, x, b)

    def __call__(self, r: Float[Tensor, "*batch ndof"]) -> Float[Tensor, "*batch ndof"]:
        return self._cycle(0, torch_fem.project(r, self.levels[0].mask))


def solve(
    F: Float[Tensor, "*batch ndof"],
    xPhys: Float[Tensor, "*dbatch nel"],
    edofMat: Int[Tensor, "nel 8"],
    KE: Float[Tensor, "8 8"],
    Emin: float,
    Emax: float,
    penal: float,
    mask: Bool[Tensor, " ndof"],
    nelx: int,
    nely: int,
    *,
    rtol: float = 1e-8,
    max_iter: int = 500,
    x0: Float[Tensor, "*batch ndof"] | None = None,
    check_every: int = 1,
    omega: float = 0.6,
    n_smooth: int = 2,
    gamma: int = 1,
    max_coarse_elements: int = MAX_COARSE_ELEMENTS,
) -> tuple[Float[Tensor, "*batch ndof"], int]:
    """Multigrid-preconditioned CG for `K @ U = F`, the MGCG counterpart of `torch_fem.solve`.

    Same contract as `torch_fem.solve` -- boundary conditions by projection, batching over
    right-hand sides and/or density fields, warm start via `x0`, and a raised
    `CGConvergenceError` rather than a silently unconverged result.

    :param nelx: fine-mesh elements in x (multigrid needs the grid shape, unlike Jacobi).
    :param nely: fine-mesh elements in y.
    :param omega: damped-Jacobi relaxation factor for the smoother.
    :param n_smooth: pre- and post-smoothing sweeps per level.
    :param gamma: coarse cycles per visit; 1 is a V-cycle, 2 a W-cycle.
    :param check_every: iterations between convergence checks; see `torch_fem.pcg`.
    :param max_coarse_elements: coarsening stops at or below this element count.
    :return: `(U, n_iter)`.
    """
    density = torch_fem.simp_density(xPhys, Emin, Emax, penal)
    levels = build_hierarchy(
        density,
        edofMat,
        KE,
        mask,
        nelx,
        nely,
        max_coarse_elements=max_coarse_elements,
    )
    precond = VCycle(levels, omega=omega, n_smooth=n_smooth, gamma=gamma)
    U, n_iter = torch_fem.pcg(
        levels[0].apply_A,
        torch_fem.project(F, mask),
        precond,
        rtol=rtol,
        max_iter=max_iter,
        x0=x0,
        check_every=check_every,
    )
    return torch_fem.project(U, mask), n_iter
