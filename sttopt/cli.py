"""Command-line entry point for the space-time topology optimization loop -- the
argparse equivalent of `conductivity_estimation_stto_main.m`'s hardcoded constants at
the top of that script (`nelx`, `nely`, `nloop`, ... through `beta_d_max`), with the same
default values (the original *full-scale* script's constants, not the smaller ones the
fixture harness/tests use for speed).

Drives `optimize.build_problem`/`init_state`/`step` directly (not `optimize.run`) so it
can print per-iteration progress, matching the MATLAB source's `disp` line. This is a
long-running production script (800 iterations at 180x60 is not something to run in
this sandbox -- see the repo's sandbox rules), so console progress is the useful signal
here, not MATLAB's periodic live-figure updates, which this port doesn't replicate. The
printed "Obj."/"Vol." read the current iteration's full MMA objective and post-update
density -- not `IterationRecord.obj`/`.vol` (see that class's field comments), which are
different, pre-update quantities MATLAB's own `disp` line doesn't read either.

At the end, saves one PNG combining `viz.combination_plot`/`viz.stage_boundary_plot` on
one `Axes` (via `stage_boundary_plot`'s `combination_coords=True`, needed because the two
functions' native coordinate frames don't coincide -- see `viz.py`'s docstring) -- more
useful as a saved artifact than MATLAB's separate two-figure/two-window layout. Reproduces
the main script's final plot recipe: `XPhys` binarized (`>0.5 -> 1, else 0`), colored by
`hotspot_severity = (1-K_est)*XPhys` -- not a print time, despite feeding
`combination_plot`'s color-by slot -- with stage-boundary lines overlaid. MATLAB computes
this from the *pre-update* density/time fields entering the last iteration, not the
post-loop state (one MMA update further along) -- so this uses `prev_state` (the state
entering the final `step` call), recomputing `K_est` via
`conductivity.estimated_conductivity` since `IterationRecord` doesn't carry it.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sttopt.run_config import RunConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Space-time topology optimization with a conductivity-based "
        "overheating (hotspot) constraint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "default.json",
        help="path to a RunConfig JSON file (e.g. a previous run's output/<tag>/"
        "config.json). Fields not listed below "
        "(nelx, nely, volfrac, nStage, Theta, Tcr, tfield, rmin, lrmin, rmin_cond, "
        "beta_d_max, and the rest of build_problem's hyperparameters) are settable "
        "only through this file. CLI flags below override values loaded from it",
    )
    parser.add_argument(
        "--nloop", type=int, default=None, help="number of MMA iterations"
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="run identifier; artefacts (progress snapshots, final design, final plot, "
        "config.json) are saved under output/<tag>/",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device to run on, e.g. 'cpu' or 'cuda:0' (default: CUDA if "
        "available, else CPU)",
    )
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> "RunConfig":
    """
    Build the `RunConfig` for a run: load `--config` if given, then apply any
    explicitly-passed CLI flags on top.

    :param args: parsed CLI arguments (`parse_args`'s return value)
    :return: the resolved `RunConfig`
    """
    import json

    from sttopt.run_config import RunConfig

    config = RunConfig.from_dict(json.loads(args.config.read_text()))
    for field in ("nloop", "tag", "device"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    return config


def main(args: argparse.Namespace) -> None:
    import json

    import numpy as np

    import sttopt.conductivity as conductivity
    import sttopt.optimize as optimize
    import sttopt.torch_util as torch_util
    import sttopt.viz as viz

    config = resolve_config(args)

    # Fail fast on an unwritable output dir, before spending nloop iterations of compute.
    output_dir = Path("output") / config.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    problem = optimize.build_problem(
        config.nelx,
        config.nely,
        config.nStage,
        config.volfrac,
        config.Theta,
        config.Tcr,
        config.tfield,
        config.rmin,
        config.lrmin,
        config.rmin_cond,
        beta_d_max=config.beta_d_max,
        Emin=config.Emin,
        Emax=config.Emax,
        nu=config.nu,
        penal=config.penal,
        eta=config.eta,
        p=config.p,
        q=config.q,
        r=config.r,
        rouf=config.rouf,
        a0=config.a0,
        mma_c=config.mma_c,
        move=config.move,
        tmove=config.tmove,
        device=config.device,
        batch_fem_solves=config.batch_fem_solves,
    )
    state = optimize.init_state(problem, beta_d=1.0)

    prev_state = state  # state entering the final `step` call -- see module docstring
    record = None  # unset if --nloop 0 (no `step` calls)
    for _ in range(args.nloop):
        prev_state = state
        state, record = optimize.step(problem, state)
        print(
            f"It.: {state.loop:4d} Obj.: {record.f0val:10.4f} "
            f"Vol.: {state.xPhys.mean():6.3f} Tm.: {record.tru_max:7.3f}"
        )
        if state.loop % 50 == 0:
            np.savez(
                output_dir / f"design_it{state.loop:04d}.npz",
                x=torch_util.to_numpy(state.x),
                t=torch_util.to_numpy(state.t),
            )

    # nan if --nloop 0 (no `step` calls, so no IterationRecord to read these from)
    f0val = record.f0val if record is not None else float("nan")
    vol = record.vol if record is not None else float("nan")
    tru_max = record.tru_max if record is not None else float("nan")
    np.savez(
        output_dir / "final_design.npz",
        loop=state.loop,
        x=torch_util.to_numpy(state.x),
        xTilde=torch_util.to_numpy(state.xTilde),
        xPhys=torch_util.to_numpy(state.xPhys),
        t=torch_util.to_numpy(state.t),
        tPhys=torch_util.to_numpy(state.tPhys),
        f0val=f0val,
        vol=vol,
        tru_max=tru_max,
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
    XPhys = (prev_state.xPhys > 0.5).to(problem.dtype)
    # Same quantity as hotspot_constraint's internal T_val (conductivity.py), density-masked
    # for display -- not a print time, despite feeding combination_plot's "timing" slot.
    hotspot_severity = (1 - K_est) * XPhys

    # Tensor boundary: `viz` takes plain arrays -- see sttopt/torch_util.py.
    XPhys = torch_util.to_numpy(XPhys)
    hotspot_severity = torch_util.to_numpy(hotspot_severity)
    tPhys = torch_util.to_numpy(prev_state.tPhys)

    ax = viz.combination_plot(XPhys, hotspot_severity, eps=1.0e-1)
    viz.stage_boundary_plot(tPhys, config.nStage, ax=ax, combination_coords=True)

    out_path = output_dir / "final_structure.png"
    ax.figure.savefig(out_path, dpi=150)
    print(f"Saved final structure plot to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
