"""Command-line entry point for the space-time topology optimization loop -- the
argparse equivalent of `conductivity_estimation_stto_main.m`'s hardcoded constants at
the top of that script (`nelx`, `nely`, `nloop`, ... through `beta_d_max`), with the same
default values (the original *full-scale* script's constants, not the smaller ones the
fixture harness/tests use for speed).

Drives `stto.build_problem`/`init_state`/`step` directly (not `stto.run`) so it
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
        "--snapshot-every",
        type=int,
        default=50,
        help="save x/t/xPhys/tPhys every this many iterations (default: 50)",
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
    import time

    import numpy as np

    import sttopt.stto as stto
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

    problem = stto.build_problem(config, device=args.device)
    state = stto.init_state(problem, beta_d=1.0)

    log = (output_dir / "iterations.jsonl").open("w")
    start = time.perf_counter()
    for _ in range(config.nloop):
        state, record = stto.step(problem, state)
        xPhys, tPhys = stto.physical_fields(problem, state.x, state.t, state.beta_d)
        diag = record.diagnostics
        print(
            f"It.: {state.loop:4d} Obj.: {record.f:10.4f} "
            f"Vol.: {xPhys.mean():6.3f} Tm.: {record.tru_max:7.3f} "
            f"Unif.: {record.uniformity:8.5f} c: {record.obj:9.3f} "
            f"TmTrue: {diag['true_max']:6.3f} neff: {diag['n_eff']:7.1f}"
        )
        entry = dict(
            loop=state.loop,
            elapsed=time.perf_counter() - start,
            f=record.f,
            obj=record.obj,
            vol=float(xPhys.mean()),
            tru_max=record.tru_max,
            uniformity=record.uniformity,
            roughness=record.roughness,
            roughness_weight=record.roughness_weight,
            g=record.g.tolist(),
            lam=record.lam.tolist(),
            **diag,
        )
        log.write(json.dumps(entry) + "\n")
        log.flush()
        if state.loop % args.snapshot_every == 0:
            np.savez_compressed(
                output_dir / f"design_it{state.loop:04d}.npz",
                x=torch_util.to_numpy(state.x),
                t=torch_util.to_numpy(state.t),
                xPhys=torch_util.to_numpy(xPhys),
                tPhys=torch_util.to_numpy(tPhys),
            )
    log.close()

    np.savez(
        output_dir / "final_design.npz",
        loop=state.loop,
        x=torch_util.to_numpy(state.x),
        xPhys=torch_util.to_numpy(xPhys),
        t=torch_util.to_numpy(state.t),
        tPhys=torch_util.to_numpy(tPhys),
        f=record.f,
        vol=record.vol,
        tru_max=record.tru_max,
        uniformity=record.uniformity,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
