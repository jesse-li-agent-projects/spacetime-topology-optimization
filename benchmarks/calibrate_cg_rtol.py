"""Calibrate the MGCG relative-residual tolerance against the *sensitivities*, not the
compliance, for `plans/archive/torch_port.md` Phase 1 ("Accuracy calibration") and Phase 2.

The asymmetry this exists to measure: `U` minimizes the potential energy, so the
compliance `c` is stationary at the solution and its error is second-order in the error
of `U`; the sensitivities are not stationary, because `dcx` is built from per-element
`ce = Ue^T KE Ue` and MMA reads every element of it. Calibrating a solver tolerance
against `c` would therefore pick a tolerance that looks excellent and quietly degrades
the optimizer's search direction.

Method: differentiate the *unmodified* `sttopt.compliance` objectives twice per design --
once with `spsolve` underneath and once with MGCG at each candidate `rtol` -- and compare
the two sensitivities element-wise. Swapping the solver underneath rather than
reimplementing the algebra is deliberate: any reimplementation could drift from the
production formulas and would then be calibrating the wrong thing. Both backends are
autograd `Function`s over the same downstream code (`SpsolveFE` and
`torch_solve.FemSolve`), so the linear solve is the only thing that differs between the
two columns.

The adjoint adds no solver error of its own to calibrate for. `FemSolve.backward`
warm-starts from `alpha * U`, which for any compliance scalar (`dL/dU = 2KU`) is already
the answer, so the adjoint returns at CG iteration zero and `lambda` is a closed-form
multiple of the forward `U` -- pinned by
`tests/test_torch_solve.py::test_self_adjoint_shortcut_gives_zero_adjoint_iterations`.
The candidate `rtol` therefore enters the sensitivities only through the forward solve,
which is what this table measures.

Two element-wise error measures are reported per sensitivity array, because a bare
per-element relative error is not meaningful on this field. At a near-binary design most
elements have `xPhys == 0`, where `dcx` is ~1e-30 and its relative error is pure noise
about a quantity MMA cannot respond to. So:

- `rel@active`: max over elements of `|d_cg - d_ref| / |d_ref|`, restricted to elements
  carrying at least `ACTIVE_FRACTION` of the largest `|d_ref|` -- the elements that
  actually steer the design.
- `max_abs/peak`: max over elements of `|d_cg - d_ref|`, divided by `max |d_ref|`. Still
  element-wise (a single bad element shows up in full), but well-defined everywhere.

Neither is an L2 norm; both are max-over-elements, per the plan's requirement.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=["90x30", "180x60"],
        help="fixture mesh keys to calibrate on",
    )
    parser.add_argument(
        "--iteration",
        default="0800",
        help="fixture snapshot iteration (late = near-binary = hardest)",
    )
    parser.add_argument(
        "--rtols",
        nargs="+",
        type=float,
        default=None,
        help="candidate CG relative-residual tolerances",
    )
    parser.add_argument(
        "--device", default="cpu", help="torch device for the CG solver (cpu / cuda)"
    )
    parser.add_argument(
        "--nstage", type=int, default=8, help="gravity stages (production value)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

import contextlib
from pathlib import Path

import numpy as np
import torch

import sttopt.compliance as compliance
import sttopt.fem as fem
import tests.reference.fem as fem_ref
import sttopt.gravity as gravity
import sttopt.torch_fem as torch_fem
import sttopt.torch_solve as torch_solve
import sttopt.torch_util as torch_util

FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "torch_port_designs.npz"
)

# Production physics constants (optimize.build_problem's defaults).
EMIN, EMAX, PENAL, NU = 1e-9, 1.0, 3.0, 0.3
BETA_T = 50.0  # beta_t continuation saturates well before loop 800

DEFAULT_RTOLS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)

#: Calibrated CG relative-residual tolerance: the loosest one whose element-wise
#: sensitivity error clears `conftest.assert_close`'s "solved" tier with an order of
#: margin. Tightening further buys nothing -- `spsolve` and MGCG stop agreeing
#: element-wise at ~1e-8 regardless, that being what float64 pins down at this
#: conditioning. See `plans/archive/torch_port.md`'s Phase 2 results for the full table.
RECOMMENDED_RTOL = 1e-8

#: The accuracy bar every solver configuration must clear: the repo's "solved" tier.
SENSITIVITY_TOL = 1e-6

#: An element is "active" for `rel@active` if `|d_ref|` is at least this fraction of the
#: largest `|d_ref|` on the mesh.
ACTIVE_FRACTION = 1e-6


def mesh_setup(nelx: int, nely: int, device="cpu", dtype=torch.float64) -> dict:
    """Everything both solver paths need for one mesh: the cantilever load case, the
    numpy FEM tables, and their torch counterparts.

    :param nelx: elements in x.
    :param nely: elements in y.
    :param device: torch device for the CG-side tensors.
    :return: dict of numpy and torch setup objects.
    """
    nodes = fem.node_grid(nelx, nely)
    ndof = 2 * nodes.size
    F = np.zeros(ndof)
    F[2 * nodes[-1, -1] + 1] = -1.0
    left_edge = nodes[:, 0]
    fixeddofs = np.stack([2 * left_edge, 2 * left_edge + 1], axis=-1).ravel()
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    KE = fem.plane_stress_KE(NU)
    edofMat = fem.element_dof_map(nelx, nely)
    C = gravity.gravity_load_matrix(nelx, nely)
    return {
        "nelx": nelx,
        "nely": nely,
        "ndof": ndof,
        "F": F,
        "freedofs": freedofs,
        "KE": KE,
        "edofMat": edofMat,
        "C": C,
        # torch counterparts -- sttopt.compliance is torch-native (Phase 3.2,
        # plans/torch_port_part2.md), so these (not the plain-NumPy fields above) are
        # what this module's own calls into it use.
        "KE_t": torch.tensor(KE, dtype=dtype, device=device),
        "edofMat_t": torch.tensor(edofMat, dtype=torch.int64, device=device),
        "freedofs_t": torch.tensor(freedofs, dtype=torch.int64, device=device),
        "F_t": torch.tensor(F, dtype=dtype, device=device),
        "C_t": torch_util.csr_to_tensor(C, device, dtype),
        "mask": torch_fem.free_mask(
            ndof, torch.tensor(freedofs, dtype=torch.int64, device=device), device
        ),
        "device": device,
        "dtype": dtype,
    }


@contextlib.contextmanager
def mgcg_backend(setup: dict, *, rtol: float, max_iter: int = 2000):
    """Run `sttopt.compliance` with MGCG at a chosen `rtol`, in place of its own default.

    Monkeypatches `compliance._solve_fe` (Phase 3.3, `plans/torch_port_part2.md`:
    `compliance.py` talks to `torch_solve.FemSolve` directly now, so there is no
    `fem.assemble_stiffness`/`fem.solve_fe` call left to intercept) rather than
    reimplementing its sensitivity algebra -- any reimplementation could drift from the
    production formulas and would then be calibrating the wrong thing.

    :param setup: `mesh_setup` output.
    :param rtol: CG relative-residual tolerance.
    :param max_iter: CG iteration cap; exceeding it raises `CGConvergenceError`.
    :yield: a list that receives one iteration count per solve performed (one entry per
        batch member for a batched `optimize.step` call -- see `_solve_fe_batched`
        below -- since `pcg` runs every batch member for the same iteration count).
    """
    orig_solve_fe = compliance._solve_fe
    orig_solve_fe_batched = compliance._solve_fe_batched
    iters: list[int] = []

    def solve_fe(KE, xPhys, Emin, Emax, penal, edofMat, freedofs, F, ndof, *, x0=None):
        nely, nelx = xPhys.shape
        density = torch_fem.simp_density(xPhys, Emin, Emax, penal)
        mask = torch_fem.free_mask(ndof, freedofs, device=xPhys.device)
        info: dict = {}
        U = torch_solve.femsolve(
            density.flatten(),
            F,
            edofMat,
            KE,
            mask,
            nelx,
            nely,
            rtol=rtol,
            max_iter=max_iter,
            x0=x0,
            info=info,
        )
        iters.append(info["forward_n_iter"])
        return U

    def solve_fe_batched(KE, density, edofMat, mask, nelx, nely, F, *, x0=None):
        info: dict = {}
        U = torch_solve.femsolve(
            density,
            F,
            edofMat,
            KE,
            mask,
            nelx,
            nely,
            rtol=rtol,
            max_iter=max_iter,
            x0=x0,
            info=info,
        )
        # One batched FemSolve call runs every row for the same iteration count
        # (pcg's batching contract), recorded once per row for parity with solve_fe.
        iters.extend([info["forward_n_iter"]] * density.shape[0])
        return U

    compliance._solve_fe = solve_fe
    compliance._solve_fe_batched = solve_fe_batched
    try:
        yield iters
    finally:
        compliance._solve_fe = orig_solve_fe
        compliance._solve_fe_batched = orig_solve_fe_batched


class SpsolveFE(torch.autograd.Function):
    """`K @ U = F` by assemble-plus-`spsolve`, differentiable in `density` and `F`.

    `torch_solve.FemSolve`'s direct-solve counterpart, and for the same reason it is a
    `Function` at all: the linear solve is where the two paths differ, so it is the only
    place the substitution belongs. Its adjoint is the same algebra `FemSolve.backward`
    runs (`lambda = K^-1 g`, `dL/dF = lambda`, `dL/dd_e = -(lambda_e @ KE) . U_e`) over a
    direct solve rather than MGCG, so nothing downstream of the solve -- SIMP, the
    gravity load, the strain-energy contraction -- is duplicated here; autograd chains
    it exactly as it does in production.

    Takes `density` (`torch_fem.simp_density`'s output) rather than `xPhys`, matching
    `FemSolve`: the SIMP power law stays ordinary torch code on both paths.
    """

    @staticmethod
    def forward(ctx, density, F, KE, edofMat, freedofs, ndof):
        KE_np = torch_util.to_numpy(KE)
        edofMat_np = torch_util.to_numpy(edofMat)
        freedofs_np = torch_util.to_numpy(freedofs)
        K = fem_ref.assemble_from_density(
            KE_np, torch_util.to_numpy(density), edofMat_np, ndof
        )
        U = fem_ref.solve_fe(K, torch_util.to_numpy(F), freedofs_np)

        ctx.K, ctx.freedofs_np, ctx.KE_np, ctx.edofMat_np = (
            K,
            freedofs_np,
            KE_np,
            edofMat_np,
        )
        U_t = torch_util.to_tensor(U, density.device, density.dtype)
        ctx.save_for_backward(U_t)
        return U_t

    @staticmethod
    def backward(ctx, grad_output):
        (U_t,) = ctx.saved_tensors
        # solve_fe reads only the free dofs of its right-hand side and zero-fills the
        # rest, which is the projection FemSolve.backward applies explicitly.
        lam = fem_ref.solve_fe(ctx.K, torch_util.to_numpy(grad_output), ctx.freedofs_np)
        lam_t = torch_util.to_tensor(lam, grad_output.device, grad_output.dtype)

        edofMat_t = torch.as_tensor(ctx.edofMat_np, device=U_t.device)
        Ue = U_t[edofMat_t]
        lam_e = lam_t[edofMat_t]
        KE_t = torch_util.to_tensor(ctx.KE_np, U_t.device, U_t.dtype)
        grad_density = -torch.sum((lam_e @ KE_t) * Ue, dim=-1)
        return grad_density, lam_t, None, None, None, None


@contextlib.contextmanager
def spsolve_backend():
    """Run `sttopt.compliance` over assemble-plus-`spsolve`, the reference this module
    calibrates MGCG's `rtol` against.

    Phase 3.3 (`plans/torch_port_part2.md`) moved `compliance._solve_fe`'s default
    backend from `assemble_stiffness`/`solve_fe` (now `tests/reference/fem.py`'s) to
    `torch_solve.FemSolve`'s MGCG, so this monkeypatches `_solve_fe` back to the
    NumPy/SciPy path rather than assuming it is still the default -- symmetric with
    `mgcg_backend` above, and for the same reason: swap the solver underneath, never
    reimplement the algebra.

    The replacement is `SpsolveFE`, an autograd `Function`, not a bare NumPy round trip:
    `sensitivities` reads autograd gradients, so a backend that detached the graph could
    not serve as their reference at all.
    """
    orig_solve_fe = compliance._solve_fe

    def solve_fe(KE, xPhys, Emin, Emax, penal, edofMat, freedofs, F, ndof, *, x0=None):
        # spsolve is a direct solve with no notion of a warm start; x0 is accepted
        # (for signature parity with compliance._solve_fe) and ignored.
        density = torch_fem.simp_density(xPhys, Emin, Emax, penal)
        return SpsolveFE.apply(density.flatten(), F, KE, edofMat, freedofs, ndof)

    compliance._solve_fe = solve_fe
    try:
        yield
    finally:
        compliance._solve_fe = orig_solve_fe


def _whole_compliance_and_grad(setup: dict, x: np.ndarray):
    """`(c, dc/dxPhys)` from `compliance.whole_compliance` via autograd."""
    device, dtype = setup["device"], setup["dtype"]
    x_t = torch.tensor(x, dtype=dtype, device=device, requires_grad=True)
    c, _ = compliance.whole_compliance(
        x_t,
        setup["KE_t"],
        setup["edofMat_t"],
        EMIN,
        EMAX,
        PENAL,
        setup["freedofs_t"],
        setup["F_t"],
        setup["ndof"],
    )
    (dcx,) = torch.autograd.grad(c, x_t)
    return float(c.detach()), dcx.detach().cpu().numpy()


def sensitivities(setup: dict, x: np.ndarray, t: np.ndarray, nstage: int) -> dict:
    """Compliances and sensitivities for one design: the whole-structure solve plus every
    gravity stage, exactly as `optimize.step` computes them.

    :param setup: `mesh_setup` output.
    :param x: `xPhys`, shape `(nely, nelx)`.
    :param t: `tPhys`, shape `(nely, nelx)`.
    :param nstage: number of deposition stages.
    :return: dict with `c` (whole), `dcx` (whole), and stacked per-stage `cg`, `dcx_g`,
        `dct_g`.
    """
    device, dtype = setup["device"], setup["dtype"]
    c, dcx = _whole_compliance_and_grad(setup, x)

    cg, dcx_g, dct_g = [], [], []
    for ti in np.linspace(0, 1, nstage + 1)[1:]:
        x_t = torch.tensor(x, dtype=dtype, device=device, requires_grad=True)
        t_t = torch.tensor(t, dtype=dtype, device=device, requires_grad=True)
        c_s, _ = compliance.gravity_compliance(
            x_t,
            t_t,
            setup["KE_t"],
            setup["edofMat_t"],
            EMIN,
            EMAX,
            PENAL,
            float(ti),
            setup["C_t"],
            BETA_T,
            setup["freedofs_t"],
            setup["ndof"],
        )
        dcx_s, dct_s = torch.autograd.grad(c_s, (x_t, t_t))
        cg.append(float(c_s.detach()))
        dcx_g.append(dcx_s.detach().cpu().numpy())
        dct_g.append(dct_s.detach().cpu().numpy())
    return {
        "c": c,
        "dcx": dcx.flatten(),
        "cg": np.array(cg),
        "dcx_g": np.array(dcx_g),
        "dct_g": np.array(dct_g),
    }


def elementwise_errors(actual: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """`(rel@active, max_abs/peak)` for one sensitivity array -- see the module docstring.

    :param actual: sensitivities from the CG solve.
    :param ref: sensitivities from `spsolve`.
    :return: the two max-over-elements error measures.
    """
    diff = np.abs(actual - ref)
    peak = np.abs(ref).max()
    active = np.abs(ref) >= ACTIVE_FRACTION * peak
    rel_active = float((diff[active] / np.abs(ref[active])).max())
    return rel_active, float(diff.max() / peak)


def finite_difference_check(
    setup: dict, x: np.ndarray, rtol: float, elements: np.ndarray, h: float = 1e-4
) -> float:
    """Max relative error of `dcx` against a central difference taken *through* the CG solve.

    An oracle-free check: it validates the whole chain (matrix-free operator, V-cycle,
    CG, sensitivity algebra) against the definition of a derivative, so a consistent
    error shared by CG and `spsolve` cannot hide in it.

    :param setup: `mesh_setup` output.
    :param x: density field, shape `(nely, nelx)`; should be well inside `(0, 1)` so the
        perturbed fields stay in range.
    :param rtol: CG tolerance.
    :param elements: flat indices of the elements to difference.
    :param h: central-difference step. The default balances the O(h^2) truncation error
        against the O(eps*|c|/h) cancellation error; both larger and smaller steps
        measurably degrade the check.
    :return: max over `elements` of the relative error.
    """
    with mgcg_backend(setup, rtol=rtol):
        _, dcx = _whole_compliance_and_grad(setup, x)
        worst = 0.0
        for e in elements:
            cs = []
            for sign in (+1, -1):
                xp = x.copy()
                xp.flat[e] += sign * h
                c_p, _ = _whole_compliance_and_grad(setup, xp)
                cs.append(c_p)
            fd = (cs[0] - cs[1]) / (2 * h)
            worst = max(worst, abs(fd - dcx.flat[e]) / abs(dcx.flat[e]))
    return worst


def calibrate(mesh: str, iteration: str, rtols, device: str, nstage: int) -> None:
    """Print the calibration table for one mesh: sensitivity error vs `rtol`, plus the
    compliance error at the same points to show the asymmetry.
    """
    nelx, nely = (int(v) for v in mesh.split("x"))
    with np.load(FIXTURES) as data:
        x = data[f"x_{mesh}_it{iteration}"]
        t = data[f"t_{mesh}_it{iteration}"]
    setup = mesh_setup(nelx, nely, device=device)
    with spsolve_backend():
        ref = sensitivities(setup, x, t, nstage)

    print(
        f"\n=== {mesh} (ndof={setup['ndof']}), snapshot it{iteration}, "
        f"nStage={nstage}, device={device} ==="
    )
    print(
        f"{'rtol':>8} {'iters(min-max)':>16} {'dcx rel@act':>13} {'dcx abs/peak':>13}"
        f" {'dcxg rel@act':>13} {'dcxg abs/pk':>13} {'dctg rel@act':>13}"
        f" {'dctg abs/pk':>13} {'|dc/c|':>10} {'|dcg/cg|':>10}"
    )
    for rtol in rtols:
        with mgcg_backend(setup, rtol=rtol) as iters:
            got = sensitivities(setup, x, t, nstage)
        e_dcx = elementwise_errors(got["dcx"], ref["dcx"])
        e_dcxg = elementwise_errors(got["dcx_g"].ravel(), ref["dcx_g"].ravel())
        e_dctg = elementwise_errors(got["dct_g"].ravel(), ref["dct_g"].ravel())
        c_err = abs(got["c"] - ref["c"]) / abs(ref["c"])
        cg_err = float(np.abs(got["cg"] - ref["cg"]).max() / np.abs(ref["cg"]).max())
        print(
            f"{rtol:>8.0e} {min(iters):>7d}-{max(iters):<8d} "
            f"{e_dcx[0]:>13.2e} {e_dcx[1]:>13.2e} {e_dcxg[0]:>13.2e} {e_dcxg[1]:>13.2e}"
            f" {e_dctg[0]:>13.2e} {e_dctg[1]:>13.2e} {c_err:>10.2e} {cg_err:>10.2e}"
        )


def main():
    rtols = args.rtols if args.rtols else DEFAULT_RTOLS
    for mesh in args.meshes:
        calibrate(mesh, args.iteration, rtols, args.device, args.nstage)

    # Finite-difference check on a small mesh with mid-range densities: the near-binary
    # fixture sits at the bounds of [0, 1], where a central difference is not defined.
    nelx, nely = 24, 16
    rng = np.random.default_rng(0)
    x_fd = rng.uniform(0.3, 0.7, (nely, nelx))
    setup = mesh_setup(nelx, nely, device=args.device)
    elements = rng.choice(nelx * nely, 8, replace=False)
    for rtol in (1e-6, 1e-8, 1e-10):
        err = finite_difference_check(setup, x_fd, rtol, elements)
        print(f"\nFD check {nelx}x{nely}, rtol={rtol:.0e}: max rel error {err:.3e}")


if __name__ == "__main__":
    main()
