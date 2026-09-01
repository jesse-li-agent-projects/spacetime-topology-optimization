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
        help="path to a RunConfig JSON file (e.g. a previous run's output/<tag>/"
        "config.json). Fields (nelx, nely, nloop, volfrac, nStage, Theta, Tcr, "
        "print_base, rmin, lrmin, rmin_cond, beta_d_max, and the rest of "
        "build_problem's hyperparameters) are settable only through this file",
    )
    parser.add_argument(
        "--tag",
        help="run identifier; artefacts (progress snapshots, final design, "
        "config.json) are saved under output/<tag>/",
    )
    parser.add_argument(
        "--tag-force",
        action="store_true",
        help="delete an existing output/<tag> directory before this run, instead of "
        "erroring out",
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
    Build the `RunConfig` for a run: load `--config`. `--tag`/`--tag-force`/`--device`
    are run bookkeeping -- they control where/whether a run's artefacts land, not the
    optimization itself -- so they're never part of `RunConfig`; read them directly off
    `args` instead.

    :param args: parsed CLI arguments (`parse_args`'s return value)
    :return: the resolved `RunConfig`
    """
    import json

    from sttopt.run_config import RunConfig

    config = RunConfig.from_dict(json.loads(args.config.read_text()))
    return config


def main(args: argparse.Namespace) -> None:
    import json
    import shutil

    import numpy as np

    import sttopt.optimize as optimize
    import sttopt.torch_util as torch_util

    config = resolve_config(args)

    assert config.nloop > 0, "Number of iterations must be positive"

    # Fail fast on an unwritable/clobbered output dir, before spending nloop
    # iterations of compute.
    output_dir = Path("output") / args.tag
    if output_dir.exists():
        if args.tag_force:
            shutil.rmtree(output_dir)
        else:
            raise SystemExit(
                f"[error] {output_dir} already exists. Use --tag-force to overwrite, "
                f"or pick a different --tag."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    problem = optimize.build_problem(config, device=args.device)
    state = optimize.init_state(problem, beta_d=1.0)

    for _ in range(config.nloop):
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

    np.savez(
        output_dir / "final_design.npz",
        loop=state.loop,
        x=torch_util.to_numpy(state.x),
        xTilde=torch_util.to_numpy(state.xTilde),
        xPhys=torch_util.to_numpy(state.xPhys),
        t=torch_util.to_numpy(state.t),
        tPhys=torch_util.to_numpy(state.tPhys),
        f0val=record.f0val,
        vol=record.vol,
        tru_max=record.tru_max,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
