"""Hotspot severity across a sweep of the maximum acceptable severity `T_cr` on the
cantilever, for `tex/main.tex`, with the sequential design for reference: one panel per
design, all on one absolute colour scale, void gray.

Reads the runs in `output/tcr_C105_*`, made as in `configs/tcr_sweep/README.md`. Run from
the repo root, via `python -m tex.make_figures`.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from sttopt.run_config import LaplaceRunConfig
from tex.figures import designs
from tex.figures.style import TEXT_WIDTH_IN

CONFIG = Path("configs/laplace_default.json")
README = "configs/tcr_sweep/README.md"
REFERENCE = "tcr_C105_sequential"
# Loosest first, so the designs depart from the sequential one in order
TCRS = (0.7, 0.6, 0.55, 0.5, 0.45)
# Magnitude, so one hue light to dark; void in the neutral gray the other figures use
CMAP = matplotlib.colormaps["Reds"].with_extremes(bad=designs.VOID_GRAY)


def _draw(ax, d: designs.Design, norm: Normalize) -> None:
    """Severity over the solid part (`xPhys >= 0.5`), void gray."""
    nely, nelx = d.xPhys.shape
    shown = (d.xPhys >= 0.5) & np.isfinite(d.severity)
    ax.imshow(
        np.where(shown, d.severity, np.nan),
        cmap=CMAP,
        norm=norm,
        extent=(0, nelx, -nely, 0),
    )
    ax.set_xlim(0, nelx)
    ax.set_ylim(-nely, 0)
    ax.set_axis_off()


def build(out_dir: Path, *, device: str | None = None) -> None:
    config = LaplaceRunConfig.from_dict(json.loads(CONFIG.read_text()))
    problem = designs.evaluator(config, device)
    reference = designs.evaluate(problem, designs.run_dir(REFERENCE, README))
    panels = [(f"Sequential, compliance {reference.compliance:.1f}", reference)]
    for Tcr in TCRS:
        d = designs.evaluate(problem, designs.run_dir(f"tcr_C105_T{Tcr:.2f}", README))
        # Rounded first, so a change that rounds to zero does not print as -0.0
        change = round(100 * (d.compliance / reference.compliance - 1), 1) + 0.0
        panels.append(
            (f"$T_{{cr}} = {Tcr:g}$, compliance {d.compliance:.1f} ({change:+.1f}%)", d)
        )
    norm = Normalize(0, max(d.max_severity for _, d in panels))

    rows = -(-len(panels) // 2)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.21 * rows),
        layout="constrained",
    )
    for ax, (title, d) in zip(axes.flat, panels):
        _draw(ax, d, norm)
        ax.set_title(
            f"{title},\nhighest hotspot severity {d.max_severity:.2f}", fontsize=7
        )
    fig.colorbar(
        ScalarMappable(norm, CMAP),
        ax=axes,
        label="hotspot severity $S_e$",
        extend="min",
        shrink=0.6,
        aspect=20,
    )
    fig.savefig(out_dir / "tcr_sweep.pdf")
    plt.close(fig)
