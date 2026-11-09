"""Matplotlib style shared by the figures in `tex/main.tex`; importing it applies it."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT_WIDTH_IN = 6.3  # a4paper, 2.5 cm margins

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#1a1a19",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    }
)
