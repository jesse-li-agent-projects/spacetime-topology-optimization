"""Command-line entry point for the space-time topology optimization loop -- the
argparse equivalent of `conductivity_estimation_stto_main.m`'s hardcoded constants at
the top of that script (`nelx`, `nely`, `nloop`, ... through `beta_max`), with the same
default values (the original *full-scale* script's constants, not the smaller ones the
fixture harness/tests use for speed).

Drives `optimize.build_problem`/`init_state`/`step` directly (not `optimize.run`) so it
can print per-iteration progress, matching the MATLAB source's `disp` line
(`conductivity_estimation_stto_main.m`, ~line 537): `It.: {loop:4d} Obj.: {obj:10.4f}
Vol.: {vol:6.3f} Tm.: {tru_max:7.3f}`. This is a long-running production script (800
iterations at 180x60 is not something to run in this sandbox -- see the repo's sandbox
rules) so console progress is the useful signal here, not MATLAB's periodic live-figure
updates (`mod(loop,10)==0`), which this port doesn't replicate.

At the end, saves one PNG combining `viz.combination_plot`/`viz.stage_boundary_plot` on
one `Axes` (via `stage_boundary_plot`'s `combination_coords=True`, needed because the two
functions' native coordinate frames don't coincide -- see `viz.py`'s docstring) -- more
useful as a saved artifact than MATLAB's separate two-figure/two-window layout. Reproduces
the main script's specific *final* plot recipe (lines ~581-594): `XPhys` binarized
(`>0.5 -> 1, else 0`), colored by `T1 = (1-K_est)*XPhys`, with stage-boundary lines
overlaid. MATLAB captures `XPhys`/computes `K_est` from the *last iteration's pre-update*
density/time fields (`XPhys=xPhys(:)` at the top of that iteration's hotspot block, before
that same iteration's own MMA step) -- so this uses `prev_state` (the state entering the
final `step` call), not the post-loop `state` (which is one MMA update ahead of what MATLAB
plots). `IterationRecord` doesn't carry `K_est`, so it's recomputed here via
`conductivity.estimated_conductivity` on `prev_state`.

The per-iteration console line mirrors MATLAB's `disp` (~line 537) exactly, including
which quantities it reads: "Obj." is the *full* MMA objective (`f0val` = whole-structure
compliance + Theta-weighted per-stage gravity, matching MATLAB's `obj` variable, which is
never reassigned between `f0val = obj` and the `disp` call) -- NOT `IterationRecord.obj`
(whole-structure compliance alone, MATLAB's `objf(loop)`, a different variable). "Vol." is
this iteration's *post-update* `xPhys` (MATLAB's `disp` runs after the MMA step
reassigns `xPhys`, so it measures the density feeding the *next* iteration) -- NOT
`IterationRecord.vol` (the pre-update value MATLAB's own `vol(loop)` array stores,
computed earlier in the same iteration).
"""

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Space-time topology optimization with a conductivity-based "
        "overheating (hotspot) constraint."
    )
    parser.add_argument("--nelx", type=int, default=180, help="mesh elements in x")
    parser.add_argument("--nely", type=int, default=60, help="mesh elements in y")
    parser.add_argument(
        "--nloop", type=int, default=800, help="number of MMA iterations"
    )
    parser.add_argument("--nStage", type=int, default=8, help="number of print stages")
    parser.add_argument(
        "--volfrac", type=float, default=0.5, help="volume fraction budget"
    )
    parser.add_argument(
        "--Theta", type=float, default=0.1, help="gravity-compliance objective weight"
    )
    parser.add_argument(
        "--Tcr", type=float, default=0.8, help="hotspot constraint threshold"
    )
    parser.add_argument(
        "--tfield",
        type=int,
        default=3,
        choices=(1, 2, 3),
        help="initial time-field variant: 1=top-left corner, 2=left edge, 3=bottom-left corner "
        "(coerced to a TimeField by build_problem)",
    )
    parser.add_argument("--rmin", type=float, default=4.0, help="density filter radius")
    parser.add_argument(
        "--lrmin", type=float, default=2.0, help="time-field continuity filter radius"
    )
    parser.add_argument(
        "--rmin-cond",
        dest="rmin_cond",
        type=float,
        default=12.0,
        help="conductivity neighborhood radius (unrelated to --rmin despite the MATLAB "
        "source reusing the name `rmin` for both -- see the port plan)",
    )
    parser.add_argument(
        "--beta-max",
        dest="beta_max",
        type=float,
        default=128.0,
        help="Heaviside projection sharpness cap",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("./plot"),
        help="directory to save the final combination+boundary plot into",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    import sttopt.conductivity as conductivity
    import sttopt.optimize as optimize
    import sttopt.viz as viz

    # Fail fast on an unwritable output dir, before spending nloop iterations of compute.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problem = optimize.build_problem(
        args.nelx,
        args.nely,
        args.nStage,
        args.volfrac,
        args.Theta,
        args.Tcr,
        args.tfield,
        args.rmin,
        args.lrmin,
        args.rmin_cond,
        beta_max=args.beta_max,
    )
    state = optimize.init_state(problem, beta=1.0)

    prev_state = state  # state entering the final `step` call -- see module docstring
    for _ in range(args.nloop):
        prev_state = state
        state, record = optimize.step(problem, state)
        print(
            f"It.: {state.loop:4d} Obj.: {record.f0val:10.4f} "
            f"Vol.: {state.xPhys.mean():6.3f} Tm.: {record.tru_max:7.3f}"
        )

    K_est = conductivity.estimated_conductivity(
        prev_state.xPhys,
        prev_state.tPhys,
        problem.e1,
        problem.e2,
        problem.w,
        problem.q,
        problem.rouf,
    ).reshape(problem.nely, problem.nelx)
    XPhys = (prev_state.xPhys > 0.5).astype(float)
    T1 = (1 - K_est) * XPhys

    ax = viz.combination_plot(XPhys, T1, eps=1.0e-1)
    viz.stage_boundary_plot(
        prev_state.tPhys, args.nStage, ax=ax, combination_coords=True
    )

    out_path = args.output_dir / "final_structure.png"
    ax.figure.savefig(out_path, dpi=150)
    print(f"Saved final structure plot to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
