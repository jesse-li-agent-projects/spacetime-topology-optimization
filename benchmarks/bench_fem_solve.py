"""Phase 2 of `plans/archive/torch_port.md`: does GPU MGCG beat `scipy.spsolve` on the FEM solve?

Scope is the solve alone -- assemble plus solve for the SciPy cell, hierarchy build plus
CG for the torch cells -- not the surrounding optimization loop. Three cells, because
there is no direct-solver-on-GPU cell to fill: `spsolve` on CPU is the baseline, torch
MGCG on GPU is the candidate, and torch MGCG on CPU is a diagnostic control that makes a
negative result interpretable (a loss shared by both torch cells is algorithmic and a
better preconditioner might fix it; a loss only on GPU is the device being latency-bound
at these sizes, and it will not).

Everything the plan warns against is deliberately covered:

- **Never uniform density alone.** `x = volfrac` is the best-conditioned field the
  optimizer ever holds and it holds it only at iteration zero. Both fields are reported
  and the late near-binary one is what the verdict turns on.
- **Warm start is the real operating condition.** `move = tmove = 0.01` caps the design's
  per-iteration motion, so the previous `U` is an excellent initial guess -- an advantage
  a direct solver cannot exploit at all. Cold and warm cells solve the *same* system, one
  real `optimize.step` on from the snapshot, and differ only in the initial guess.
- **Batching.** The `nStage` gravity solves share a sparsity pattern and differ only in
  density, so they batch into one CG over an `(nStage, ndof)` right-hand side. That turns
  a latency-bound GPU problem into a throughput-bound one and has no CPU analogue.
- **Accuracy is asserted at every timed point**, against `spsolve` at the calibrated
  sensitivity tolerance. A timing at unmatched accuracy would be meaningless, and a fast
  solve at a loose tolerance is not a win.

Iteration counts are reported beside every time because they are what explains a result.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=["90x30", "180x60", "360x120"],
        help="meshes to benchmark",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=800,
        help="near-binary snapshot iteration (late = hardest)",
    )
    parser.add_argument(
        "--nstage", type=int, default=8, help="gravity stages (production value)"
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="timed repeats per configuration"
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="discarded warm-up repeats"
    )
    parser.add_argument(
        "--rtol", type=float, default=None, help="CG tolerance (default: calibrated)"
    )
    parser.add_argument(
        "--skip-cpu-cg",
        action="store_true",
        help="skip the torch-CG-on-CPU diagnostic control (it is the slowest cell)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

import time

import numpy as np
import torch

import sttopt.torch_fem as torch_fem
import sttopt.torch_mg as torch_mg
import tests.reference.fem as fem_ref
from benchmarks.calibrate_cg_rtol import (
    EMAX,
    EMIN,
    PENAL,
    RECOMMENDED_RTOL,
    SENSITIVITY_TOL,
    mesh_setup,
)
from tests.fixtures.generate_torch_port_designs import VOLFRAC, load_design


def design_pair(
    mesh: str, iteration: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The `(previous, current)` design one real optimizer iteration apart.

    Both halves come straight from the snapshot archive, which stores loops 799 and 800
    for exactly this purpose. Nothing here reconstructs or perturbs a design: whether the
    previous solution is a good initial guess is the question being measured, so
    manufacturing the pair would be assuming the answer. An earlier version did
    manufacture it -- one `optimize.step` from a profiling-only reconstructed `State` --
    and produced a step that moved `xPhys` by up to 0.996 with an initial residual 37-100x
    *worse* than a cold start. The real step moves it by 0.0067.

    The benchmark always times the *current* design; the previous one only supplies the
    warm start, so the cold and warm cells solve an identical system.

    :param mesh: `"NELXxNELY"`; a derived mesh upscales both halves alike.
    :param iteration: the *current* loop; its predecessor is `iteration - 1`.
    :return: `(x_prev, t_prev, x_cur, t_cur)`.
    """
    x_prev, t_prev = load_design(mesh, iteration - 1)
    x_cur, t_cur = load_design(mesh, iteration)
    return x_prev, t_prev, x_cur, t_cur


def stage_systems(
    setup: dict, x: np.ndarray, t: np.ndarray, nstage: int, beta_t: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    """The `nStage` gravity-stage systems, as `compliance.gravity_compliance` forms them.

    Both the operator and the right-hand side vary per stage: the stage density is
    `xtJoint = xPhys * t_mask(ti)`, and the load is that same field pushed through `C`.
    Batching has to carry both, which is why they are returned together.

    :param setup: `mesh_setup` output.
    :param x: `xPhys`, shape `(nely, nelx)`.
    :param t: `tPhys`, shape `(nely, nelx)`.
    :param nstage: number of deposition stages.
    :param beta_t: time-mask sharpness at the snapshot's continuation state.
    :return: `(densities, loads)`, shapes `(nstage, nel)` and `(nstage, ndof)`.
    """
    densities, loads = [], []
    for ti in np.linspace(0, 1, nstage + 1)[1:]:
        t_mask = 1.0 - 1.0 / (1.0 + np.exp(-beta_t * (t - ti)))
        xt = (x * t_mask).flatten()
        F = np.zeros(setup["ndof"])
        F[1::2] = -(setup["C"] @ xt)  # gravity acts in -y on each node's y-dof
        densities.append(xt)
        loads.append(F)
    return np.array(densities), np.array(loads)


def spsolve_time(setup: dict, densities: np.ndarray, loads: np.ndarray) -> tuple:
    """Time the SciPy baseline over a batch of systems, solved one at a time.

    Assembly is inside the timed region because it is part of what a solve costs today
    and the torch cells pay their own equivalent (the multigrid hierarchy build). Phase 0
    measured assembly at 11.4 ms against `spsolve`'s 139.7 ms, so this is a small and
    honest inclusion rather than a thumb on the scale.

    :param setup: `mesh_setup` output.
    :param densities: shape `(batch, nel)`.
    :param loads: right-hand sides, shape `(batch, ndof)`.
    :return: `(seconds_total, U)` with `U` of shape `(batch, ndof)`.
    """
    t0 = time.perf_counter()
    U = []
    for d, F in zip(densities, loads):
        K = fem_ref.assemble_stiffness(
            setup["KE"],
            d.reshape(setup["nely"], setup["nelx"]),
            EMIN,
            EMAX,
            PENAL,
            setup["edofMat"],
            setup["ndof"],
        )
        U.append(fem_ref.solve_fe(K, F, setup["freedofs"]))
    return time.perf_counter() - t0, np.array(U)


def mgcg_time(
    setup: dict, densities: torch.Tensor, F: torch.Tensor, *, rtol: float, x0=None
) -> tuple:
    """Time one torch MGCG call (hierarchy build included) and return `(s, U, n_iter)`.

    `torch.cuda.synchronize()` brackets the region on CUDA -- without it the launches
    return immediately and the measurement is of queueing, not of work.
    """
    device = densities.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    U, n_iter = torch_mg.solve(
        F,
        densities,
        setup["edofMat_t"],
        setup["KE_t"],
        EMIN,
        EMAX,
        PENAL,
        setup["mask"],
        setup["nelx"],
        setup["nely"],
        rtol=rtol,
        max_iter=2000,
        x0=x0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, U, n_iter


def accuracy(U_cg: torch.Tensor, U_ref: np.ndarray, setup: dict) -> float:
    """Max element-wise relative error of `ce = Ue^T KE Ue` -- the quantity the
    sensitivities are built from, and so the one the tolerance was calibrated against.

    Comparing `U` directly would be the wrong check: `U` is defined only up to the
    solver's error, whereas `ce` is what MMA actually reads, element by element.

    :param U_cg: CG displacements, shape `(batch, ndof)`.
    :param U_ref: `spsolve` displacements, same shape.
    :param setup: `mesh_setup` output.
    :return: the max over elements and batch members.
    """
    worst = 0.0
    KE, edof = setup["KE"], setup["edofMat"]
    for u_cg, u_ref in zip(U_cg.double().cpu().numpy(), np.atleast_2d(U_ref)):
        ce_cg = np.einsum("ij,jk,ik->i", u_cg[edof], KE, u_cg[edof])
        ce_ref = np.einsum("ij,jk,ik->i", u_ref[edof], KE, u_ref[edof])
        peak = np.abs(ce_ref).max()
        worst = max(worst, float(np.abs(ce_cg - ce_ref).max() / peak))
    return worst


def repeat_time(fn, repeats: int, warmup: int) -> float:
    """Median of `repeats` timings after discarding `warmup` -- median, not mean, so one
    scheduler hiccup does not set the number.
    """
    for _ in range(warmup):
        fn()
    return float(np.median([fn()[0] for _ in range(repeats)]))


class Table:
    """Accumulates and prints one row per benchmarked configuration."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, **row) -> None:
        self.rows.append(row)
        print(
            f"{row['mesh']:>8} {row['field']:>11} {row['cell']:>12} {row['start']:>5}"
            f" {row['batch']:>10} {row['ms']:>9.1f} {row['iters']:>6}"
            f" {row['ce_err']:>9.1e} {row['speedup']:>8}",
            flush=True,
        )

    def header(self) -> None:
        print(
            f"\n{'mesh':>8} {'field':>11} {'cell':>12} {'start':>5} {'batch':>10}"
            f" {'ms/solve':>9} {'iters':>6} {'ce_err':>9} {'speedup':>8}",
            flush=True,
        )


def benchmark_mesh(mesh: str, table: Table, rtol: float, opts) -> None:
    """Run every configuration for one mesh and add its rows to `table`."""
    nelx, nely = (int(v) for v in mesh.split("x"))
    x_prev, t_prev, x_bin, t_bin = design_pair(mesh, opts.iteration)
    # Uniform density is a conditioning control only, so it gets no warm-start row: the
    # optimizer holds `x = volfrac` at iteration zero alone, and there is no previous
    # iteration for it to have left a solution behind.
    fields = [
        ("uniform", np.full((nely, nelx), VOLFRAC), t_bin, ("cold",)),
        ("near-binary", x_bin, t_bin, ("cold", "warm")),
    ]

    devices = [("cuda", "MGCG-GPU")] if torch.cuda.is_available() else []
    if not opts.skip_cpu_cg:
        devices.append(("cpu", "MGCG-CPU"))

    cpu_setup = mesh_setup(nelx, nely)
    for field_name, x_cur, t_cur, starts in fields:
        # The warm start: what the previous iteration left behind. Not timed -- it is an
        # input to the warm cells, and the previous iteration already paid for it.
        d_prev, f_prev = stage_systems(cpu_setup, x_prev, t_prev, opts.nstage)
        _, U_prev = spsolve_time(cpu_setup, d_prev, f_prev)

        # The system every cell below actually solves.
        densities, loads = stage_systems(cpu_setup, x_cur, t_cur, opts.nstage)
        _, U_ref = spsolve_time(cpu_setup, densities, loads)

        base_ms = (
            1e3
            * repeat_time(
                lambda: spsolve_time(cpu_setup, densities, loads),
                opts.repeats,
                opts.warmup,
            )
            / opts.nstage
        )
        # `spsolve` cannot exploit a warm start at all, so one number covers both starts.
        for start in starts:
            table.add(
                mesh=mesh,
                field=field_name,
                cell="spsolve-CPU",
                start=start,
                batch="sequential",
                ms=base_ms,
                iters="-",
                ce_err=0.0,
                speedup="1.00x",
            )

        for device, cell in devices:
            setup = mesh_setup(nelx, nely, device=device)
            d_t = torch.tensor(densities, dtype=torch.float64, device=device)
            F_t = torch.tensor(loads, dtype=torch.float64, device=device)
            U_prev_t = torch.tensor(U_prev, dtype=torch.float64, device=device)

            for start in starts:
                x0_all = U_prev_t if start == "warm" else None
                for batch in ("sequential", "batched"):
                    if batch == "batched":
                        calls = [(d_t, F_t, x0_all)]
                    else:
                        calls = [
                            (d_t[i], F_t[i], None if x0_all is None else x0_all[i])
                            for i in range(opts.nstage)
                        ]

                    def run(calls=calls):
                        total, U, iters = 0.0, [], []
                        for d, F, x0 in calls:
                            s, u, n = mgcg_time(setup, d, F, rtol=rtol, x0=x0)
                            total += s
                            U.append(u.reshape(-1, setup["ndof"]))
                            iters.append(n)
                        return total, torch.cat(U), iters

                    try:
                        ms = 1e3 * repeat_time(run, opts.repeats, opts.warmup)
                        _, U_cg, iters = run()
                    except torch_fem.CGConvergenceError as exc:
                        print(f"  {mesh} {field_name} {cell} {start} {batch}: {exc}")
                        continue

                    err = accuracy(U_cg, U_ref, cpu_setup)
                    assert err < SENSITIVITY_TOL, (
                        f"{mesh}/{field_name}/{cell}/{start}/{batch}: ce error {err:.2e} "
                        f"exceeds the calibrated {SENSITIVITY_TOL:.0e}; this timing would "
                        "be at unmatched accuracy and is not comparable"
                    )
                    table.add(
                        mesh=mesh,
                        field=field_name,
                        cell=cell,
                        start=start,
                        batch=batch,
                        ms=ms / opts.nstage,
                        iters=f"{min(iters)}-{max(iters)}",
                        ce_err=err,
                        speedup=f"{base_ms / (ms / opts.nstage):.2f}x",
                    )


def main():
    rtol = args.rtol if args.rtol is not None else RECOMMENDED_RTOL
    print(f"CG rtol = {rtol:.0e} (calibrated; see benchmarks/calibrate_cg_rtol.py)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no CUDA device visible -- the GPU cell will be skipped")
    print(
        f"nStage={args.nstage}, repeats={args.repeats} (median), warmup={args.warmup}, "
        f"snapshot it{args.iteration:04d}"
    )
    print(
        "ms/solve is per single-stage solve: a batched cell's total divided by nStage, "
        "so every row is directly comparable."
    )

    table = Table()
    table.header()
    for mesh in args.meshes:
        benchmark_mesh(mesh, table, rtol, args)


if __name__ == "__main__":
    main()
