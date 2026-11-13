"""Sequential against simultaneous optimization of the density and the time field, for
`tex/main.tex`: density, time field and hotspot severity of each design, severity read on
the grayscale design as optimized.

Reads the runs in `output/tex_seq_vs_sim_{plain_to,sequential,simultaneous}`, made as in
`configs/seq_vs_sim/README.md`. Run from the repo root, via `python -m tex.make_figures`.
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

CONFIG = Path("configs/laplace_default.json")
README = "configs/seq_vs_sim/README.md"
TAGS = {
    "sequential": "tex_seq_vs_sim_sequential",
    "simultaneous": "tex_seq_vs_sim_simultaneous",
}


def _title(label: str, d: designs.Design, reference: designs.Design | None) -> str:
    c = f"compliance {d.compliance:.1f}"
    if reference is not None:
        c += f" ({100 * (d.compliance / reference.compliance - 1):.1f}% higher)"
    return f"{label}, {c},\nhighest hotspot severity {d.max_severity:.2f}"


def build(out_dir: Path, *, device: str | None = None) -> None:
    config = LaplaceRunConfig.from_dict(json.loads(CONFIG.read_text()))
    problem = designs.evaluator(config, device)
    Tcr = run_config.final_value(config.Tcr)
    seq, sim = (
        designs.evaluate(problem, designs.run_dir(TAGS[name], README))
        for name in ("sequential", "simultaneous")
    )
    columns = (
        (seq, _title("Sequential", seq, None)),
        (sim, _title("Simultaneous", sim, seq)),
    )
    norm = designs.severity_norm([seq, sim], Tcr)

    fig, axes = plt.subplots(
        3, 2, figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.5), layout="constrained"
    )
    for ax_col, (d, title) in zip(axes.T, columns):
        density, time, severity = ax_col
        designs.draw_density(density, d)
        designs.draw_time(time, d)
        designs.draw_severity(severity, d, Tcr, norm)
        density.set_title(title, fontsize=7)
    for row, mappable, label in (
        (0, ScalarMappable(Normalize(0, 1), "gray_r"), "density"),
        (1, ScalarMappable(Normalize(0, 1), "viridis"), "time field"),
        (
            2,
            ScalarMappable(norm, designs.SEVERITY_CMAP),
            "excess severity\n$S_e - T_{cr}$",
        ),
    ):
        fig.colorbar(
            mappable,
            ax=axes[row],
            label=label,
            extend="min" if row == 2 else "neither",
            shrink=0.9,
            aspect=10,
        )
    fig.savefig(out_dir / "seq_vs_sim.pdf")
    plt.close(fig)
