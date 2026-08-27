"""Calibrate the MGCG relative-residual tolerance against the *sensitivities*, not the
compliance, for `plans/torch_port.md` Phase 1 ("Accuracy calibration") and Phase 2.

The asymmetry this exists to measure: `U` minimizes the potential energy, so the
compliance `c` is stationary at the solution and its error is second-order in the error
of `U`; the sensitivities are not stationary, because `dcx` is built from per-element
`ce = Ue^T KE Ue` and MMA reads every element of it. Calibrating a solver tolerance
against `c` would therefore pick a tolerance that looks excellent and quietly degrades
the optimizer's search direction.

Method: run the *unmodified* `sttopt.compliance` sensitivity algebra twice per design --
once over `spsolve`'s `U` and once over MGCG's `U` at each candidate `rtol` -- and
compare the two element-wise. Swapping the solver underneath rather than reimplementing
the algebra is deliberate: any reimplementation could drift from the production formulas
and would then be calibrating the wrong thing.

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
import sttopt.gravity as gravity
import sttopt.torch_fem as torch_fem
import sttopt.torch_mg as torch_mg

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
#: conditioning. See `plans/torch_port.md`'s Phase 2 results for the full table.
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
    return {
        "nelx": nelx,
        "nely": nely,
        "ndof": ndof,
        "F": F,
        "freedofs": freedofs,
        "KE": KE,
        "edofMat": edofMat,
        "C": gravity.gravity_load_matrix(nelx, nely),
        "KE_t": torch.tensor(KE, dtype=dtype, device=device),
        "edofMat_t": torch.tensor(edofMat, dtype=torch.int64, device=device),
        "mask": torch_fem.free_mask(
            ndof, torch.tensor(freedofs, dtype=torch.int64, device=device), device
        ),
        "device": device,
        "dtype": dtype,
    }


@contextlib.contextmanager
def mgcg_backend(setup: dict, *, rtol: float, max_iter: int = 2000):
    """Run `sttopt.compliance` with MGCG substituted for assemble-plus-`spsolve`.

    `fem.assemble_stiffness` is replaced by a recorder that captures the density field
    the caller assembled from and returns `None` (nothing but `fem.solve_fe` consumes
    its result, and the matrix-free operator does not want it); `fem.solve_fe` is
    replaced by `torch_mg.solve` over that recorded field. Recording is what makes this
    work for `gravity_compliance`, whose operative density is the internal `xtJoint`
    rather than any argument.

    :param setup: `mesh_setup` output.
    :param rtol: CG relative-residual tolerance.
    :param max_iter: CG iteration cap; exceeding it raises `CGConvergenceError`.
    :yield: a list that receives one iteration count per solve performed.
    """
    orig_assemble, orig_solve = fem.assemble_stiffness, fem.solve_fe
    recorded: dict = {}
    iters: list[int] = []

    def assemble(KE_, xPhys, Emin, Emax, penal, edofMat_, ndof_):
        recorded["xPhys"] = xPhys
        return None

    def solve_fe(K, F, freedofs):
        # The mask comes from the caller's own `freedofs`, not from `setup`, so this
        # backend honours whatever boundary conditions the substituted-into code uses
        # rather than assuming `mesh_setup`'s cantilever.
        mask = torch_fem.free_mask(
            F.shape[-1],
            torch.tensor(freedofs, dtype=torch.int64, device=setup["device"]),
            setup["device"],
        )
        U, n_iter = torch_mg.solve(
            torch.tensor(F, dtype=setup["dtype"], device=setup["device"]),
            torch.tensor(
                recorded["xPhys"].flatten(),
                dtype=setup["dtype"],
                device=setup["device"],
            ),
            setup["edofMat_t"],
            setup["KE_t"],
            EMIN,
            EMAX,
            PENAL,
            mask,
            setup["nelx"],
            setup["nely"],
            rtol=rtol,
            max_iter=max_iter,
        )
        iters.append(n_iter)
        return U.cpu().numpy()

    fem.assemble_stiffness, fem.solve_fe = assemble, solve_fe
    try:
        yield iters
    finally:
        fem.assemble_stiffness, fem.solve_fe = orig_assemble, orig_solve


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
    c, dcx = compliance.whole_compliance(
        x,
        setup["KE"],
        setup["edofMat"],
        EMIN,
        EMAX,
        PENAL,
        setup["freedofs"],
        setup["F"],
        setup["ndof"],
    )
    cg, dcx_g, dct_g = [], [], []
    for ti in np.linspace(0, 1, nstage + 1)[1:]:
        c_s, dcx_s, dct_s = compliance.gravity_compliance(
            x,
            t,
            setup["KE"],
            setup["edofMat"],
            EMIN,
            EMAX,
            PENAL,
            float(ti),
            setup["C"],
            BETA_T,
            setup["freedofs"],
            setup["ndof"],
        )
        cg.append(c_s)
        dcx_g.append(dcx_s)
        dct_g.append(dct_s)
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
        _, dcx = compliance.whole_compliance(
            x,
            setup["KE"],
            setup["edofMat"],
            EMIN,
            EMAX,
            PENAL,
            setup["freedofs"],
            setup["F"],
            setup["ndof"],
        )
        worst = 0.0
        for e in elements:
            cs = []
            for sign in (+1, -1):
                xp = x.copy()
                xp.flat[e] += sign * h
                c_p, _ = compliance.whole_compliance(
                    xp,
                    setup["KE"],
                    setup["edofMat"],
                    EMIN,
                    EMAX,
                    PENAL,
                    setup["freedofs"],
                    setup["F"],
                    setup["ndof"],
                )
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
