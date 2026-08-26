"""Profile `optimize.step` at production settings (180x60, nStage=8) on a realistic
near-binary design, for `plans/torch_port.md` Phase 0.

Answers: where does `step` spend its time, and -- separately -- how much of
`fem.solve_fe` is `scipy.sparse.linalg.spsolve` itself versus the
`K[np.ix_(freedofs, freedofs)]` boundary-condition reindex. The `spsolve` number this
prints is the baseline Phase 2's GPU CG benchmark must beat.

Profiles from a reconstructed steady-state `State` around the iteration-800 snapshot in
`tests/fixtures/torch_port_designs.npz` (near-binary, `beta_d`/`beta_t` continuation
saturated) -- never from `init_state`'s uniform `x = volfrac`, which is the easiest
conditioning the optimizer ever sees and not representative of production runtime.

Measurement approach: `cProfile` for the overall call-graph breakdown, plus targeted
wall-clock timers installed by monkeypatching the exact functions/methods of interest
(`scipy.sparse.linalg.spsolve`, `sp.csr_matrix.__getitem__`, `fem.assemble_stiffness`,
`conductivity.hotspot_constraint`, `mma.mmasub`) for the duration of the profiled run,
then restored. No file under `sttopt/` is modified.
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
        "--profile-out",
        type=str,
        default=None,
        help="optional path to dump raw cProfile stats (pstats format)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sttopt.conductivity as conductivity
import sttopt.fem as fem
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
    real run holds at loop 800.

    The fixture stores only `xPhys`/`tPhys` (see `generate_torch_port_designs.py`), not
    the full `State` (raw `x`/`t`, `xTilde`, MMA history). Reconstruction:

    - `xTilde = xPhys`: at `beta_d = 128` the Heaviside projection is so steep that a
      filtered field consistent with this near-binary `xPhys` is itself near-binary,
      i.e. close to `xPhys`. This also gives a realistic `dx` (the projection
      derivative): near-zero almost everywhere, which is the correct late-iteration
      behaviour, not a profiling artefact.
    - raw `x`, `t` (unfiltered MMA output) are approximated by `xPhys`, `tPhys` too. Only
      used for move-limit bounds and MMA's `xold1`/`xold2`; wrong by a small amount but
      does not change `mmasub`'s asymptotic cost (dense algebra over fixed-size arrays).
    - `xold1`/`xold2`: both set to the same `[x; t]` vector (a converged design changes
      little step to step, so this is close to reality; exact values don't affect
      mmasub's runtime).
    - `low`/`upp`: arbitrary arrays of the right shape, tight around `xval`. mmasub's
      cost does not depend on these values either.
    - `factor = 1.0`: the hotspot-constraint rescaling constant. Affects only the
      constraint's *value*, not `hotspot_constraint`'s cost.

    None of these approximations affect the wall-clock costs measured here, since every
    suspect's cost is shape/algorithm-driven (assembly, spsolve, the pair reductions,
    mmasub's dense algebra), not value-driven. They would matter for a correctness
    check, which this script is not.
    """
    nel = problem.nelx * problem.nely
    xval = np.concatenate([x_snapshot.flatten(), t_snapshot.flatten()])
    return optimize.State(
        x=x_snapshot.copy(),
        xTilde=x_snapshot.copy(),
        xPhys=x_snapshot.copy(),
        t=t_snapshot.copy(),
        tPhys=t_snapshot.copy(),
        xold1=xval.copy(),
        xold2=xval.copy(),
        low=xval - 0.1,
        upp=xval + 0.1,
        loop=LOOP,
        beta_t=BETA_T,
        beta_d=BETA_D,
        factor=1.0,
    )


class _Timer:
    """Accumulates wall-clock time and call count across a monkeypatched function."""

    def __init__(self):
        self.total = 0.0
        self.calls = 0

    def wrap(self, fn):
        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            self.total += time.perf_counter() - t0
            self.calls += 1
            return out

        return wrapped


def install_timers() -> tuple[dict[str, _Timer], "callable"]:
    """Monkeypatch the suspects with timing wrappers; return the timers and a restore
    function. Patches module-level functions (`fem.assemble_stiffness`,
    `conductivity.hotspot_constraint`, `mma.mmasub`) plus `scipy.sparse.linalg.spsolve`
    and `sp.csr_matrix.__getitem__` (the latter to isolate `K[np.ix_(...)]`'s cost --
    `fem.solve_fe` is the only place in `optimize.step`'s call graph that fancy-indexes
    a sparse matrix by a `(row_idx, col_idx)` tuple, so a class-wide patch is safe here).
    """
    timers = {
        "assemble_stiffness": _Timer(),
        "hotspot_constraint": _Timer(),
        "mmasub": _Timer(),
        "spsolve": _Timer(),
        "ix_reindex": _Timer(),
    }

    orig_assemble = fem.assemble_stiffness
    orig_hotspot = conductivity.hotspot_constraint
    orig_mmasub = mma.mmasub
    orig_spsolve = spla.spsolve
    orig_getitem = sp.csr_matrix.__getitem__

    fem.assemble_stiffness = timers["assemble_stiffness"].wrap(orig_assemble)
    conductivity.hotspot_constraint = timers["hotspot_constraint"].wrap(orig_hotspot)
    mma.mmasub = timers["mmasub"].wrap(orig_mmasub)
    spla.spsolve = timers["spsolve"].wrap(orig_spsolve)

    ix_timer = timers["ix_reindex"]

    def timed_getitem(self, key):
        t0 = time.perf_counter()
        out = orig_getitem(self, key)
        ix_timer.total += time.perf_counter() - t0
        ix_timer.calls += 1
        return out

    sp.csr_matrix.__getitem__ = timed_getitem

    def restore():
        fem.assemble_stiffness = orig_assemble
        conductivity.hotspot_constraint = orig_hotspot
        mma.mmasub = orig_mmasub
        spla.spsolve = orig_spsolve
        sp.csr_matrix.__getitem__ = orig_getitem

    return timers, restore


def main():
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
    )

    data = np.load(FIXTURES)
    x0 = data["x_180x60_it0800"]
    t0 = data["t_180x60_it0800"]
    state = build_realistic_state(problem, x0, t0)

    n_solves_per_step = 1 + NSTAGE

    print(f"mesh: {NELX}x{NELY}, ndof={problem.ndof}, nStage={NSTAGE}")
    print(f"solves/step: {n_solves_per_step}")
    print(f"npairs (hotspot neighbor list): {problem.e1.shape[0]}")
    print(f"warmup steps: {args.warmup}, timed steps: {args.iters}")
    print()

    # Warm-up: exercise all code paths (branch caches, allocator, page cache) without
    # timing, per the task's instruction to discard the first iteration(s).
    for _ in range(args.warmup):
        state, _ = optimize.step(problem, state)

    timers, restore = install_timers()
    profiler = cProfile.Profile()
    t_start = time.perf_counter()
    profiler.enable()
    for _ in range(args.iters):
        state, _ = optimize.step(problem, state)
    profiler.disable()
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
    for name in [
        "assemble_stiffness",
        "spsolve",
        "ix_reindex",
        "hotspot_constraint",
        "mmasub",
    ]:
        tm = timers[name]
        per_call_ms = (tm.total / tm.calls * 1000) if tm.calls else float("nan")
        calls_per_step = tm.calls / n
        pct = tm.total / t_total * 100
        print(
            f"{name:<22}{tm.calls:>8}{calls_per_step:>12.1f}{tm.total:>12.4f}{per_call_ms:>14.3f}{pct:>9.1f}"
        )
    print()

    solve_fe_total = timers["spsolve"].total + timers["ix_reindex"].total
    print(
        f"fem.solve_fe total (spsolve + ix_reindex only, excludes assemble): "
        f"{solve_fe_total:.4f} s ({solve_fe_total / t_total * 100:.1f}% of step time)"
    )
    print(
        f"  of which spsolve:    {timers['spsolve'].total:.4f} s "
        f"({timers['spsolve'].total / solve_fe_total * 100:.1f}%)"
    )
    print(
        f"  of which ix_reindex: {timers['ix_reindex'].total:.4f} s "
        f"({timers['ix_reindex'].total / solve_fe_total * 100:.1f}%)"
    )
    print()
    print(
        f"*** spsolve: {timers['spsolve'].total / timers['spsolve'].calls * 1000:.3f} ms/call "
        f"({timers['spsolve'].total / timers['spsolve'].calls:.6f} s/call) at {NELX}x{NELY} "
        f"on a near-binary (loop {LOOP}) design ***"
    )
    print()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    buf = io.StringIO()
    stats_out = pstats.Stats(profiler, stream=buf)
    stats_out.sort_stats("cumulative")
    stats_out.print_stats(25)
    print("cProfile, top 25 by cumulative time:")
    print(buf.getvalue())

    if args.profile_out:
        stats.dump_stats(args.profile_out)
        print(f"raw cProfile stats written to {args.profile_out}")


if __name__ == "__main__":
    main()
