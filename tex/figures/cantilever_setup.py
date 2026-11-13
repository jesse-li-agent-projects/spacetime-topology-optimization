"""The cantilever's problem setup, for `tex/main.tex`: the domain, its clamped edge, the
tip load and the build plate, read from the problem the cantilever figures use. Run from
the repo root, via `python -m tex.make_figures`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from sttopt.run_config import LaplaceRunConfig
from tex.figures import designs
from tex.figures.style import TEXT_WIDTH_IN

CONFIG = Path("configs/laplace_default.json")


def build(out_dir: Path, *, device: str | None = None) -> None:
    config = LaplaceRunConfig.from_dict(json.loads(CONFIG.read_text()))
    problem = designs.evaluator(config, device)
    nelx, nely = config.nelx, config.nely
    # Margins around the plot box, in inches, left and right equal so the box centres
    # under the caption; they hold the plate and clamp on the left and the tip load below
    side, bottom, top = 0.3, 0.25, 0.03
    box_w = 0.5 * TEXT_WIDTH_IN
    box_h = box_w * nely / nelx
    w, h = box_w + 2 * side, box_h + bottom + top
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((side / w, bottom / h, box_w / w, box_h / h))
    designs.draw_setup(ax, problem)
    fig.savefig(out_dir / "cantilever_setup.pdf")
    plt.close(fig)
