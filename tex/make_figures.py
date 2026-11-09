"""Builds the figures `tex/main.tex` includes, as PDFs in `tex/figures/`. Each figure is a
module under `tex/figures/` exposing `build(out_dir, **options)`; add new ones to FIGURES.

    python -m tex.make_figures [NAME ...] [--out DIR] [--device D] [--rerun]   # from the repo root

Figures that depend on optimization runs make the missing ones (under `output/`) first.
"""

import argparse
from pathlib import Path

FIGURES = ("hotspot_weights", "seq_vs_sim")
# Options each figure's `build` accepts, besides `out_dir`
RUN_OPTIONS = {"seq_vs_sim": ("device", "rerun")}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the figures tex/main.tex includes."
    )
    parser.add_argument(
        "names",
        nargs="*",
        choices=FIGURES,
        help="figures to build (default: all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "figures",
        help="output directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for optimization runs (default: CUDA if available)",
    )
    parser.add_argument(
        "--rerun", action="store_true", help="redo the optimization runs even if cached"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

import importlib


def main(args: argparse.Namespace) -> None:
    for name in args.names or FIGURES:
        module = importlib.import_module(f"tex.figures.{name}")
        options = {k: getattr(args, k) for k in RUN_OPTIONS.get(name, ())}
        print(f"== {name}")
        module.build(args.out, **options)


if __name__ == "__main__":
    main(args)
