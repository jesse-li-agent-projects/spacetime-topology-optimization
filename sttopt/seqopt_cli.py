"""Command-line entry point for fixed-geometry fabrication-sequence optimization
(`seqopt`). Modeled on `stto_cli.py`: drives `seqopt.build_problem`/`init_state`/`step`
directly so it can print per-iteration progress.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> "SeqRunConfig":
    import json

    from sttopt.run_config import SeqRunConfig

    return SeqRunConfig.from_dict(json.loads(args.config.read_text()))


def main(args: argparse.Namespace) -> None:
    import json
    import shutil

    import numpy as np

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

    for _ in range(config.nloop):
        state, record = seqopt.step(problem, state)
        print(
            f"It.: {state.loop:4d} f: {record.f:10.4f} "
            f"hot: {record.hotspot:8.5f} unif: {record.uniformity:8.5f} "
            f"Tm.: {record.tru_max:7.3f}"
        )
        if state.loop % 50 == 0:
            np.savez(
                output_dir / f"design_it{state.loop:04d}.npz",
                t=torch_util.to_numpy(state.t),
            )

    np.savez(
        output_dir / "final_design.npz",
        loop=state.loop,
        xPhys=xPhys,
        # tPhys == t here (seqopt does not filter the time field); written under the
        # name viz.py reads so the plotting path works unchanged. No separate "t" key.
        tPhys=torch_util.to_numpy(state.t),
        f=record.f,
        hotspot=record.hotspot,
        uniformity=record.uniformity,
        tru_max=record.tru_max,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
