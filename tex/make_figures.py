"""Builds the figures `tex/main.tex` includes, as PDFs in `tex/figures/`. Each figure is a
module under `tex/figures/` exposing `build(out_dir, **options)`; add new ones to FIGURES.

    python -m tex.make_figures [NAME ...] [--out DIR] [--device D]   # from the repo root

Figures that depend on optimization runs read them from `output/`; the commands that make
them are in the `configs/` README each figure module names. Nothing here optimizes.
"""

import argparse
from pathlib import Path

FIGURES = (
    "cantilever_setup",
    "hotspot_weights",
    "outside_domain",
    "seq_vs_sim",
    "tcr_sweep",
    "time_formulations",
)
# Options each figure's `build` accepts, besides `out_dir`
OPTIONS = {
    "cantilever_setup": ("device",),
    "seq_vs_sim": ("device",),
    "tcr_sweep": ("device",),
    "time_formulations": ("device",),
}


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
        help="torch device the designs are read on (default: CUDA if available)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

import importlib


def main(args: argparse.Namespace) -> None:
    for name in args.names or FIGURES:
        module = importlib.import_module(f"tex.figures.{name}")
        options = {k: getattr(args, k) for k in OPTIONS.get(name, ())}
        print(f"== {name}")
        module.build(args.out, **options)


if __name__ == "__main__":
    main(args)
