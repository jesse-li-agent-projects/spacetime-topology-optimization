"""Command-line entry point for `checks.check_design` on a saved `stto` design, for runs
saved before `stto_cli` checked its own final design, or for a snapshot."""

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a saved stto design, binarized: print start and support "
        "(hard), and the constraint values and compliance (reported). Writes "
        "<design>_checks.json beside the design."
    )
    parser.add_argument(
        "design",
        type=Path,
        help="a run directory's final_design.npz or design_it*.npz; the run's "
        "config.json is read from the same directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device to run on, e.g. 'cpu' or 'cuda:0' (default: CUDA if "
        "available, else CPU)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

import json

import numpy as np

import sttopt.checks as checks
import sttopt.stto as stto
import sttopt.torch_util as torch_util
from sttopt.run_config import RunConfig


def main(args: argparse.Namespace) -> None:
    config = RunConfig.from_dict(
        json.loads((args.design.parent / "config.json").read_text())
    )
    problem = stto.build_problem(config, device=args.device)
    with np.load(args.design) as data:
        if "loop" not in data:
            raise SystemExit(
                f"{args.design}: no 'loop' key, so the settings it was optimized under are unknown; snapshots saved before they carried one cannot be checked"
            )
        loop = int(data["loop"])
        x, t = (
            torch_util.to_tensor(data[k], problem.device, problem.dtype)
            for k in ("x", "t")
        )
    report = checks.check_design(problem, x, t, loop)
    checks.report_path(args.design).write_text(json.dumps(report, indent=2))
    print(checks.summary(report))


if __name__ == "__main__":
    main(args)
