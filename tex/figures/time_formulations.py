"""The three time-field formulations on S03bT06: the time field, with the density and the
hotspot severity beneath, from the full simultaneous optimization (left) and from
optimizing only the time field on the Laplace run's grayscale density (right).

Reads `output/tf_S03bT06_*`, made as in `configs/time_formulations/README.md`. Every design
is read with the Laplace cell's hotspot, whichever formulation produced it. Run from the
repo root, via `python -m tex.make_figures`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from sttopt import run_config
from sttopt.run_config import LaplaceRunConfig
from tex.figures import designs
from tex.figures.style import TEXT_WIDTH_IN

CONFIG = Path("configs/time_formulations/laplace.json")
README = "configs/time_formulations/README.md"
FORMULATIONS = (("laplace", "Laplace"), ("heat", "Heat"), ("direct", "Direct"))
GROUPS = (("", "Simultaneous"), ("_fixed", "Density fixed"))


def _caption(d: designs.Design, reference: designs.Design | None) -> str:
    lines = [
        f"max $S_e$ {d.max_severity:.2f}",
        f"{d.local_minima} min, {d.saddles} saddles",
    ]
    if reference is not None:
        lines.insert(
            0,
            f"$c$ {d.compliance:.0f} ({100 * (d.compliance / reference.compliance - 1):+.1f}%)",
        )
    return "\n".join(lines)


def build(out_dir: Path, *, device: str | None = None) -> None:
    config = LaplaceRunConfig.from_dict(json.loads(CONFIG.read_text()))
    problem = designs.evaluator(config, device)
    Tcr = run_config.final_value(config.Tcr)
    runs = {
        (suffix, key): designs.evaluate(
            problem, designs.run_dir(f"tf_S03bT06_{key}{suffix}", README)
        )
        for suffix, _ in GROUPS
        for key, _ in FORMULATIONS
    }
    reference = runs[("", "laplace")]
    norm = designs.severity_norm(list(runs.values()), Tcr)

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.58))
    # Three formulations per group, a gap between the groups, and a colorbar column
    gs = fig.add_gridspec(
        3,
        8,
        width_ratios=[1, 1, 1, 0.15, 1, 1, 1, 0.1],
        left=0.01,
        right=0.86,
        top=0.88,
        bottom=0.13,
        wspace=0.08,
        hspace=0.12,
    )
    for (suffix, group), first in zip(GROUPS, (0, 4)):
        fig.text(
            (gs[0, first].get_position(fig).x0 + gs[0, first + 2].get_position(fig).x1)
            / 2,
            0.985,
            group,
            ha="center",
            va="top",
            fontsize=9,
        )
        for col, (key, name) in enumerate(FORMULATIONS):
            d = runs[(suffix, key)]
            density, time, severity = (
                fig.add_subplot(gs[r, first + col]) for r in range(3)
            )
            designs.draw_density(density, d)
            designs.draw_time(time, d)
            designs.draw_severity(severity, d, Tcr, norm)
            density.set_title(name, fontsize=8)
            severity.text(
                0.5,
                -0.04,
                _caption(d, reference if not suffix else None),
                transform=severity.transAxes,
                ha="center",
                va="top",
                fontsize=5.5,
            )
    for row, mappable, label in (
        (0, ScalarMappable(Normalize(0, 1), "gray_r"), "density"),
        (1, ScalarMappable(Normalize(0, 1), "viridis"), "time field"),
        (
            2,
            ScalarMappable(norm, designs.SEVERITY_CMAP),
            "excess severity $S_e - T_{cr}$",
        ),
    ):
        fig.colorbar(
            mappable,
            cax=fig.add_subplot(gs[row, 7]),
            label=label,
            extend="min" if row == 2 else "neither",
        )
    fig.savefig(out_dir / "time_formulations.pdf")
    plt.close(fig)
