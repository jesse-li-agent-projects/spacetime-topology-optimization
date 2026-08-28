"""Calibrate the MGCG relative-residual tolerance against *autograd gradients*,
`plans/torch_port_part2.md` Phase 3.6.

`calibrate_cg_rtol.py`'s table was built before `torch_solve.FemSolve` existed: it
compares the hand-derived `dcx`/`dct` computed from a solved `U`, so it only ever
exercises one CG solve per sensitivity. Autograd's `dL/dxPhys` instead runs `FemSolve`
forward *and* its adjoint `backward` -- a second CG solve (warm-started, but still a
solve at the same `rtol`) whose error compounds with the forward's. This module re-runs
`calibrate_cg_rtol.py`'s method against that longer chain: `compliance.whole_compliance_
value`/`gravity_compliance_value` (Phase 3.4's autograd-differentiable value functions)
through `torch.autograd.grad`, with `compliance._solve_fe`/`_solve_fe_batched`
monkeypatched to run MGCG at each candidate `rtol` -- same monkeypatch mechanism as
`calibrate_cg_rtol.mgcg_backend`, reused here rather than reimplemented.

**Reference.** `calibrate_cg_rtol.py` compares against `spsolve`. That backend
(`spsolve_backend`) round-trips through NumPy and detaches the autograd graph, so it
cannot serve as an autograd reference here -- there is no gradient to read on the other
side. Instead the reference is MGCG at a much tighter `rtol` (`REFERENCE_RTOL = 1e-12`),
i.e. the same solver, believed converged to float64's own precision floor rather than
to a solver-independent oracle. `finite_difference_check` below is the
solver-independent cross-check: it differences `whole_compliance_value` itself (not a
returned sensitivity) at a fixed `rtol`, so an error shared by every `rtol` -- e.g. a
wrong adjoint sign -- cannot hide behind "MGCG agrees with a tighter MGCG".
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

import numpy as np
import torch

import sttopt.compliance as compliance
from benchmarks.calibrate_cg_rtol import (
    BETA_T,
    EMAX,
    EMIN,
    FIXTURES,
    PENAL,
    elementwise_errors,
    mesh_setup,
)

#: The candidate `rtols` are compared against MGCG at this much tighter tolerance,
#: standing in for a solver-independent oracle -- see the module docstring for why
#: `spsolve` (`calibrate_cg_rtol.py`'s reference) cannot be used directly here.
REFERENCE_RTOL = 1e-12

DEFAULT_RTOLS = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10)


@contextlib.contextmanager
def mgcg_backend(setup: dict, *, rtol: float, max_iter: int = 2000):
    """Same monkeypatch as `calibrate_cg_rtol.mgcg_backend`, plus the info dicts the
    backward pass fills in -- needed here to report adjoint iteration counts, which
    the forward-only original had no reason to capture.

    :yield: a list of `info` dicts, one per `_solve_fe`/`_solve_fe_batched` call, each
        gaining `backward_n_iter` once a later `torch.autograd.grad` call runs that
        solve's adjoint.
    """
    import sttopt.torch_fem as torch_fem
    import sttopt.torch_solve as torch_solve

    orig_solve_fe = compliance._solve_fe
    orig_solve_fe_batched = compliance._solve_fe_batched
    infos: list[dict] = []

    def solve_fe(KE, xPhys, Emin, Emax, penal, edofMat, freedofs, F, ndof, *, x0=None):
        nely, nelx = xPhys.shape
        density = torch_fem.simp_density(xPhys, Emin, Emax, penal)
        mask = torch_fem.free_mask(ndof, freedofs, device=xPhys.device)
        info: dict = {}
        infos.append(info)
        return torch_solve.femsolve(
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

    compliance._solve_fe = solve_fe
    compliance._solve_fe_batched = orig_solve_fe_batched  # unused by this module
    try:
        yield infos
    finally:
        compliance._solve_fe = orig_solve_fe
        compliance._solve_fe_batched = orig_solve_fe_batched


def autograd_sensitivities(setup: dict, x: np.ndarray, t: np.ndarray, nstage: int):
    """`dL/dxPhys` (and `dL/dtPhys`) from autograd, through `FemSolve`'s forward and
    adjoint alike, for the whole-structure and every gravity-stage objective.

    Mirrors `calibrate_cg_rtol.sensitivities`, but reads gradients off
    `whole_compliance_value`/`gravity_compliance_value` via `torch.autograd.grad`
    instead of the hand-derived `dcx`/`dct` `whole_compliance`/`gravity_compliance`
    return directly.

    :return: dict with `c`, `dcx`, stacked per-stage `cg`/`dcx_g`/`dct_g`, and
        `fwd_iters`/`bwd_iters` (one entry per `FemSolve` call: 1 whole + `nstage`
        gravity).
    """
    device, dtype = setup["device"], setup["dtype"]
    fwd_iters, bwd_iters = [], []

    with mgcg_backend(setup, rtol=setup["rtol"]) as infos:
        x_t = torch.tensor(x, dtype=dtype, device=device, requires_grad=True)
        c, _ = compliance.whole_compliance_value(
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
        fwd_iters.append(infos[-1]["forward_n_iter"])
        bwd_iters.append(infos[-1]["backward_n_iter"])
    c = float(c.detach())
    dcx = dcx.detach().cpu().numpy().flatten()

    cg_list, dcxg_list, dctg_list = [], [], []
    for ti in np.linspace(0, 1, nstage + 1)[1:]:
        with mgcg_backend(setup, rtol=setup["rtol"]) as infos:
            x_t = torch.tensor(x, dtype=dtype, device=device, requires_grad=True)
            t_t = torch.tensor(t, dtype=dtype, device=device, requires_grad=True)
            cg, _ = compliance.gravity_compliance_value(
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
            dcxg, dctg = torch.autograd.grad(cg, (x_t, t_t))
            fwd_iters.append(infos[-1]["forward_n_iter"])
            bwd_iters.append(infos[-1]["backward_n_iter"])
        cg_list.append(float(cg.detach()))
        dcxg_list.append(dcxg.detach().cpu().numpy().flatten())
        dctg_list.append(dctg.detach().cpu().numpy().flatten())

    return {
        "c": c,
        "dcx": dcx,
        "cg": np.array(cg_list),
        "dcx_g": np.array(dcxg_list),
        "dct_g": np.array(dctg_list),
        "fwd_iters": fwd_iters,
        "bwd_iters": bwd_iters,
    }


def finite_difference_check(
    setup: dict, x: np.ndarray, rtol: float, elements: np.ndarray, h: float = 1e-4
) -> float:
    """Max relative error of the autograd `dcx` against a central difference of
    `whole_compliance_value` itself, both at the same `rtol` -- the solver-independent
    cross-check the module docstring describes.

    :param setup: `mesh_setup` output.
    :param x: density field, shape `(nely, nelx)`; should be well inside `(0, 1)`.
    :param rtol: CG tolerance, forward and adjoint alike.
    :param elements: flat indices of the elements to difference.
    :param h: central-difference step -- see `calibrate_cg_rtol.finite_difference_check`.
    :return: max over `elements` of the relative error.
    """
    device, dtype = setup["device"], setup["dtype"]
    with mgcg_backend(setup, rtol=rtol):
        x_t = torch.tensor(x, dtype=dtype, device=device, requires_grad=True)
        c, _ = compliance.whole_compliance_value(
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
        dcx = dcx.detach().cpu().numpy()

    worst = 0.0
    with mgcg_backend(setup, rtol=rtol):
        for e in elements:
            cs = []
            for sign in (+1, -1):
                xp = x.copy()
                xp.flat[e] += sign * h
                c_p, _ = compliance.whole_compliance_value(
                    torch.tensor(xp, dtype=dtype, device=device),
                    setup["KE_t"],
                    setup["edofMat_t"],
                    EMIN,
                    EMAX,
                    PENAL,
                    setup["freedofs_t"],
                    setup["F_t"],
                    setup["ndof"],
                )
                cs.append(float(c_p))
            fd = (cs[0] - cs[1]) / (2 * h)
            worst = max(worst, abs(fd - dcx.flat[e]) / abs(dcx.flat[e]))
    return worst


def calibrate(mesh: str, iteration: str, rtols, device: str, nstage: int) -> None:
    """Print the calibration table for one mesh: autograd sensitivity error vs `rtol`,
    against the `REFERENCE_RTOL` MGCG solve.
    """
    nelx, nely = (int(v) for v in mesh.split("x"))
    with np.load(FIXTURES) as data:
        x = data[f"x_{mesh}_it{iteration}"]
        t = data[f"t_{mesh}_it{iteration}"]
    setup = mesh_setup(nelx, nely, device=device)
    setup["rtol"] = REFERENCE_RTOL
    ref = autograd_sensitivities(setup, x, t, nstage)

    print(
        f"\n=== {mesh} (ndof={setup['ndof']}), snapshot it{iteration}, "
        f"nStage={nstage}, device={device}, reference rtol={REFERENCE_RTOL:.0e} ==="
    )
    print(
        f"{'rtol':>8} {'fwd(min-max)':>14} {'bwd(min-max)':>14} {'dcx rel@act':>13}"
        f" {'dcx abs/peak':>13} {'dcxg rel@act':>13} {'dcxg abs/pk':>13}"
        f" {'dctg rel@act':>13} {'dctg abs/pk':>13}"
    )
    for rtol in rtols:
        setup["rtol"] = rtol
        got = autograd_sensitivities(setup, x, t, nstage)
        e_dcx = elementwise_errors(got["dcx"], ref["dcx"])
        e_dcxg = elementwise_errors(got["dcx_g"].ravel(), ref["dcx_g"].ravel())
        e_dctg = elementwise_errors(got["dct_g"].ravel(), ref["dct_g"].ravel())
        print(
            f"{rtol:>8.0e} "
            f"{min(got['fwd_iters']):>6d}-{max(got['fwd_iters']):<7d} "
            f"{min(got['bwd_iters']):>6d}-{max(got['bwd_iters']):<7d} "
            f"{e_dcx[0]:>13.2e} {e_dcx[1]:>13.2e} {e_dcxg[0]:>13.2e} {e_dcxg[1]:>13.2e}"
            f" {e_dctg[0]:>13.2e} {e_dctg[1]:>13.2e}"
        )


def main():
    rtols = args.rtols if args.rtols else DEFAULT_RTOLS
    for mesh in args.meshes:
        calibrate(mesh, args.iteration, rtols, args.device, args.nstage)

    # Finite-difference check on a small mesh with mid-range densities -- see
    # calibrate_cg_rtol.py's main() for why the near-binary fixture can't be used here.
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
