"""Command-line entry point for fixed-geometry fabrication-sequence optimization
(`seqopt`). Modeled on `stto_cli.py`: drives `seqopt.build_problem`/`init_state`/`step`
directly so it can print per-iteration progress.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import Float

    import sttopt.seqopt as seqopt
    from sttopt.run_config import SeqRunConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-geometry fabrication-sequence optimization: optimize the "
        "print-time field alone for a prescribed geometry."
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        required=True,
        help="path to an .npz geometry file holding an xPhys key (see "
        "sttopt.geometry_builders)",
    )
    parser.add_argument(
        "--non-binarized",
        action="store_true",
        help="waive the binary-geometry check and use --geometry's xPhys as given",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to a SeqRunConfig JSON file (e.g. configs/seq_default.json)",
    )
    parser.add_argument(
        "--tag",
        help="run identifier; artefacts are saved under output/<tag>/",
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
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=50,
        help="save a design_it*.npz time-field checkpoint every N iterations; 0 saves "
        "none (default: %(default)s)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="append a diagnostics record to iterations.jsonl every N iterations; 0 "
        "logs none (default: %(default)s)",
    )
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> "SeqRunConfig":
    import json

    from sttopt.run_config import SeqRunConfig

    return SeqRunConfig.from_dict(json.loads(args.config.read_text()))


def iteration_diagnostics(
    record: "seqopt.IterationRecord",
    step: "Float[np.ndarray, ' n']",
    tmove: float,
) -> dict:
    """One iteration's line of `iterations.jsonl`: the objective terms, the
    sawtooth diagnostics, plus what the trust region actually did with them.

    Read `true_cv`, not `uniformity`, when judging a run: the two differ by whatever the
    transverse sawtooth is padding, so `uniformity` alone can improve while the field
    degrades (`seqopt.IterationRecord`).

    The trust-region fields are the point of logging every iteration. `step_max` against
    `tmove` says whether the configured move limit binds at all, `move_frac` says for
    how many variables, and the asymptote widths say how far MMA's own adaptation has
    opened or closed since -- an oscillating run tightens its asymptotes and shrinks its
    steps, which the objective trace alone does not distinguish from convergence.

    :param record: the iteration's `seqopt.IterationRecord`
    :param step: `t_new - t_old`, flattened
    :param tmove: the configured per-iteration move limit, to measure saturation against
    :return: the JSON-serializable record for one log line
    """
    import numpy as np

    absolute = np.abs(step)
    width = record.upp - record.low
    return {
        "f": record.f,
        "hotspot": record.hotspot,
        "uniformity": record.uniformity,
        "tru_max": record.tru_max,
        "roughness": record.roughness,
        "true_cv": record.true_cv,
        "sawtooth": record.sawtooth,
        "sawtooth_raw": record.sawtooth_raw,
        "roughness_weight": record.roughness_weight,
        "step_max": float(absolute.max()),
        "step_mean": float(absolute.mean()),
        "move_frac": float((absolute >= 0.999 * tmove).mean()),
        "asymptote_width_min": float(width.min()),
        "asymptote_width_median": float(np.median(width)),
        "asymptote_width_max": float(width.max()),
        # `mma.trust_region_params` lets the asymptotes relax to 20x their initial
        # distance, so this ceiling -- not `tmove` -- is what a monotonically-moving
        # run actually ends up limited by.
        "asymptote_at_ceiling": float((width >= 0.999 * 2 * 20 * tmove).mean()),
        "grad_absmax": float(np.abs(record.df).max()),
        "g_max": float(record.g.max()),
        "lam_max": float(record.lam.max()),
    }


def main(args: argparse.Namespace) -> None:
    import json
    import shutil

    import numpy as np
    import torch

    import sttopt.geometry as geometry
    import sttopt.seqopt as seqopt
    import sttopt.torch_util as torch_util

    config = resolve_config(args)
    assert config.nloop > 0, "Number of iterations must be positive"
    xPhys = geometry.load_geometry(args.geometry, binary=not args.non_binarized)

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
    (output_dir / "seq_config.json").write_text(json.dumps(config.to_dict(), indent=2))

    problem = seqopt.build_problem(config, xPhys, device=args.device)
    # build_problem may have dropped solid that can't reach the build plate, so save
    # its geometry rather than the loaded one: the artefact should be what ran.
    xPhys = torch_util.to_numpy(problem.xPhys)
    np.savez(output_dir / "geometry.npz", xPhys=xPhys)
    state = seqopt.init_state(problem)

    log_path = output_dir / "iterations.jsonl"
    with open(log_path, "w") as log:
        for _ in range(config.nloop):
            previous_t = state.t
            state, record = seqopt.step(problem, state)
            print(
                f"It.: {state.loop:4d} f: {record.f:10.4f} "
                f"hot: {record.hotspot:8.5f} unif: {record.uniformity:8.5f} "
                f"Tm.: {record.tru_max:7.3f} rough: {record.roughness:7.4f} "
                f"true_cv: {record.true_cv:8.5f} saw: {record.sawtooth:7.4f}"
            )
            if args.log_every and state.loop % args.log_every == 0:
                step = torch_util.to_numpy(
                    (state.t - previous_t).flatten().to(torch.float64)
                )
                entry = {"loop": state.loop} | iteration_diagnostics(
                    record, step, config.tmove
                )
                log.write(json.dumps(entry) + "\n")
                log.flush()
            if args.snapshot_every and state.loop % args.snapshot_every == 0:
                np.savez(
                    output_dir / f"design_it{state.loop:04d}.npz",
                    t=torch_util.to_numpy(state.t),
                    tPhys=torch_util.to_numpy(
                        seqopt.physical_timefield(problem, state.t)
                    ),
                )

    np.savez(
        output_dir / "final_design.npz",
        loop=state.loop,
        xPhys=xPhys,
        # `tPhys` is the name viz.py reads, so the plotting path works unchanged; `t`
        # is the raw design variable, which differs from it only under a time filter
        # but is the field a sawtooth diagnosis has to look at when one is configured.
        tPhys=torch_util.to_numpy(seqopt.physical_timefield(problem, state.t)),
        t=torch_util.to_numpy(state.t),
        f=record.f,
        hotspot=record.hotspot,
        uniformity=record.uniformity,
        roughness=record.roughness,
        tru_max=record.tru_max,
        true_cv=record.true_cv,
        sawtooth=record.sawtooth,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
