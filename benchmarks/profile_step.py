"""Profile `optimize.step` at production settings (180x60, nStage=8) on a realistic
near-binary design.

Ported for `plans/torch_port_part2.md` Phase 3.7 -- the NumPy/SciPy predecessor this
script used to profile (`fem.assemble_stiffness`, `scipy.sparse.linalg.spsolve`,
`conductivity.hotspot_constraint`, `mma.mmasub`) has been replaced end to end by torch
equivalents (`sttopt/torch_fem.py`/`torch_mg.py`/`torch_solve.py`'s matrix-free MGCG,
`conductivity.hotspot_value`, a torch `mma.mmasub`), so the patch points below target
those instead. Assembly has no separate cost any more -- the solve is matrix-free (part
1's `torch_fem.py`), so that row is reported as exactly 0 rather than measured.

Reports the same component breakdown as `plans/archive/torch_port.md` Phase 0's
NumPy baseline (spsolve/FEM, `mma.mmasub`, `hotspot_constraint`, assembly, everything
else) so the two are directly comparable; see `plans/torch_port_part2.md` Phase 3.7.

Profiles from a reconstructed steady-state `State` around the iteration-800 snapshot in
`tests/fixtures/torch_port_designs.npz` (near-binary, `beta_d`/`beta_t` continuation
saturated) -- never from `init_state`'s uniform `x = volfrac`, which is the easiest
conditioning the optimizer ever sees and not representative of production runtime.

Measurement approach: wall-clock timers installed by monkeypatching the exact
functions of interest (`compliance._solve_fe`/`_solve_fe_batched`,
`conductivity.hotspot_value`, `mma.mmasub`), with `torch.cuda.synchronize()` calls
bracketing each timed region so the numbers reflect actual device time rather than
kernel-launch return time. No file under `sttopt/` is modified.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup", type=int, default=1, help="step() calls to discard before timing"
    )
    parser.add_argument(
        "--iters", type=int, default=5, help="step() calls to time (after warmup)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if _cuda_available() else "cpu",
        help="torch device to run on",
    )
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    args = parse_args()

import time
from pathlib import Path

import numpy as np
import torch

import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.mma as mma
import sttopt.optimize as optimize

FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "torch_port_designs.npz"
)

# Matches tests/test_e2e_slow.py's reproduction of the thesis Chapter 4.4 experiment.
NELX, NELY = 180, 60
NSTAGE = 8
VOLFRAC = 0.5
THETA = 0.1
TCR = 0.8
TFIELD = 3
RMIN, LRMIN, RMIN_COND = 4.0, 2.0, 12.0

# beta_t and beta_d as of loop 800, per optimize.step's continuation schedules
# (beta_t += 5 every 30 loops while < 50; beta_d *= 2 every 50 loops, capped at
# beta_d_max=128). Both schedules saturate before loop 800 (beta_t at loop 350,
# beta_d at loop 350 too since 2**7 == 128 at loop 350) -- confirmed by simulating
# the exact update in optimize.step against BETA_INIT=1.0/beta_t0=10.0 for 800 loops.
LOOP = 800
BETA_T = 50.0
BETA_D = 128.0


def build_realistic_state(
    problem: optimize.Problem, x_snapshot: np.ndarray, t_snapshot: np.ndarray
) -> optimize.State:
    """Reconstruct a plausible iteration-800 `State` around a saved `(xPhys, tPhys)`
    snapshot, for profiling only -- not a claim that this is bit-identical to what a
    real run holds at loop 800. See the NumPy predecessor's identical docstring (git
    history) for why each approximation below doesn't affect the wall-clock costs
    measured here.
    """
    device, dtype = problem.device, problem.dtype
    x_t = torch.as_tensor(x_snapshot, device=device, dtype=dtype)
    t_t = torch.as_tensor(t_snapshot, device=device, dtype=dtype)
    xval = torch.cat([x_t.flatten(), t_t.flatten()])
    return optimize.State(
        x=x_t.clone(),
        xTilde=x_t.clone(),
        xPhys=x_t.clone(),
        t=t_t.clone(),
        tPhys=t_t.clone(),
        xold1=xval.clone(),
        xold2=xval.clone(),
        low=xval - 0.1,
        upp=xval + 0.1,
        loop=LOOP,
        beta_t=BETA_T,
        beta_d=BETA_D,
        factor=1.0,
        U=None,
    )


class _Timer:
    """Accumulates wall-clock time and call count across a monkeypatched function.
    Synchronizes CUDA (if the tensors involved are on a CUDA device) both before
    starting and after stopping the clock, so async kernel launches are attributed to
    the call that actually did the work rather than to whatever call happens to block
    on the result later.
    """

    def __init__(self, device: torch.device):
        self.total = 0.0
        self.calls = 0
        self._sync = device.type == "cuda"

    def wrap(self, fn):
        def wrapped(*a, **kw):
            if self._sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            if self._sync:
                torch.cuda.synchronize()
            self.total += time.perf_counter() - t0
            self.calls += 1
            return out

        return wrapped


def install_timers(device: torch.device) -> tuple[dict[str, _Timer], "callable"]:
    """Monkeypatch the suspects with timing wrappers; return the timers and a restore
    function. Patches `compliance._solve_fe`/`_solve_fe_batched` (the FEM solve --
    exactly one of the two runs per step depending on `Problem.batch_fem_solves`),
    `conductivity.hotspot_value`, and `mma.mmasub`.
    """
    timers = {
        "fem_solve": _Timer(device),
        "hotspot": _Timer(device),
        "mmasub": _Timer(device),
    }

    orig_solve_fe = compliance._solve_fe
    orig_solve_fe_batched = compliance._solve_fe_batched
    orig_hotspot = conductivity.hotspot_value
    orig_mmasub = mma.mmasub

    compliance._solve_fe = timers["fem_solve"].wrap(orig_solve_fe)
    compliance._solve_fe_batched = timers["fem_solve"].wrap(orig_solve_fe_batched)
    conductivity.hotspot_value = timers["hotspot"].wrap(orig_hotspot)
    mma.mmasub = timers["mmasub"].wrap(orig_mmasub)

    def restore():
        compliance._solve_fe = orig_solve_fe
        compliance._solve_fe_batched = orig_solve_fe_batched
        conductivity.hotspot_value = orig_hotspot
        mma.mmasub = orig_mmasub

    return timers, restore


def main():
    device = torch.device(args.device)
    problem = optimize.build_problem(
        NELX,
        NELY,
        NSTAGE,
        VOLFRAC,
        THETA,
        TCR,
        TFIELD,
        RMIN,
        LRMIN,
        RMIN_COND,
        device=device,
        dtype=torch.float64,
    )

    data = np.load(FIXTURES)
    x0 = data["x_180x60_it0800"]
    t0 = data["t_180x60_it0800"]
    state = build_realistic_state(problem, x0, t0)

    n_solves_per_step = 1 + NSTAGE

    print(f"device: {device}")
    print(f"mesh: {NELX}x{NELY}, ndof={problem.ndof}")
    print(f"batch_fem_solves: {problem.batch_fem_solves}")
    print(f"solves/step (if unbatched): {n_solves_per_step}")
    print(f"npairs (hotspot neighbor list): {problem.e1.shape[0]}")
    print(f"warmup steps: {args.warmup}, timed steps: {args.iters}")
    print()

    # Warm-up: exercise all code paths (kernel compilation caches, allocator, CG's
    # data-dependent iteration count) without timing, per the task's instruction to
    # discard the first iteration(s).
    for _ in range(args.warmup):
        state, _ = optimize.step(problem, state)

    timers, restore = install_timers(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(args.iters):
        state, _ = optimize.step(problem, state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total = time.perf_counter() - t_start
    restore()

    n = args.iters
    print(f"Total wall time for {n} step() calls: {t_total:.3f} s")
    print(f"Per-step wall time: {t_total / n * 1000:.1f} ms")
    print()

    print("Targeted wall-clock breakdown (summed over all timed steps):")
    header = f"{'suspect':<22}{'calls':>8}{'calls/step':>12}{'total(s)':>12}{'per-call(ms)':>14}{'%/step':>9}"
    print(header)
    print("-" * len(header))
    for name in ["fem_solve", "mmasub", "hotspot"]:
        tm = timers[name]
        per_call_ms = (tm.total / tm.calls * 1000) if tm.calls else float("nan")
        calls_per_step = tm.calls / n
        pct = tm.total / t_total * 100
        print(
            f"{name:<22}{tm.calls:>8}{calls_per_step:>12.1f}{tm.total:>12.4f}{per_call_ms:>14.3f}{pct:>9.1f}"
        )
    accounted = sum(timers[name].total for name in ["fem_solve", "mmasub", "hotspot"])
    other = t_total - accounted
    print(
        f"{'assembly (matrix-free)':<22}{'--':>8}{'--':>12}{0.0:>12.4f}{'--':>14}{0.0:>9.1f}"
    )
    print(
        f"{'everything else':<22}{'--':>8}{'--':>12}{other:>12.4f}{'--':>14}{other / t_total * 100:>9.1f}"
    )
    print()
    print(
        f"*** fem_solve: {timers['fem_solve'].total / timers['fem_solve'].calls * 1000:.3f} ms/call "
        f"({timers['fem_solve'].total / timers['fem_solve'].calls:.6f} s/call) at {NELX}x{NELY} "
        f"on device {device}, near-binary (loop {LOOP}) design ***"
    )


if __name__ == "__main__":
    main()
