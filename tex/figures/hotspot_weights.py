"""The hotspot's angular weight lambda against kappa, and its cooling credit F against k,
as PDFs for `tex/main.tex`. Curves come from `sttopt.conductivity` itself.

    python -m tex.figures.hotspot_weights [--out DIR]   # from the repo root
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent,
        help="directory to write the PDFs to (default: this script's)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sttopt.conductivity as conductivity

# One blue ramp, light -> dark along each parameter's schedule.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
KAPPAS = [0.0, 0.5, 1.0, 2.37]
SHARPNESSES = [1.0, 2.0, 4.0, 8.0]
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


def angular_weight(
    theta: np.ndarray, kappa: float
) -> np.ndarray:  # theta: (n,) radians from straight behind the front
    """`lambda` at `|grad t| >> g0`, for a neighbour `theta` away from straight behind."""
    theta_t = torch.as_tensor(theta, dtype=torch.float64)
    dirs = torch.stack([torch.sin(theta_t), -torch.cos(theta_t)], dim=1)
    g_dot_dir = -torch.cos(theta_t)  # unit gradient along +y
    return conductivity._lobe(g_dot_dir, 1.0, dirs, kappa, 0.0).numpy()


def cooling_credit(dt_over_tau: np.ndarray, k: float) -> np.ndarray:
    """`F(dt)` for `dt` in units of the cooling time."""
    t = torch.as_tensor(dt_over_tau, dtype=torch.float64)
    a = torch.zeros(len(t), dtype=torch.long)
    b = torch.arange(1, len(t) + 1)
    return conductivity._cooling_credit(
        torch.cat([t.new_zeros(1), -t]), a, b, 1.0, k
    ).numpy()


def lambda_figure(path: Path) -> None:
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 2.5))
    ax = fig.add_axes([0.08, 0.17, 0.5, 0.76])
    polar = fig.add_axes([0.64, 0.08, 0.34, 0.86], projection="polar")
    theta = np.linspace(0, np.pi, 361)
    # Polar: the print direction points up, so the lobe hangs below the element.
    phi = np.concatenate([-np.pi / 2 + theta, -np.pi / 2 - theta[::-1]])
    for kappa, color in zip(KAPPAS, RAMP):
        lam = angular_weight(theta, kappa)
        ax.plot(np.degrees(theta), lam, color=color, label=rf"$\kappa = {kappa:g}$")
        polar.plot(phi, np.concatenate([lam, lam[::-1]]), color=color)
    # Das2023 Eq. (3.5): linear in the angle over the semicircle behind the front.
    linear = np.clip(1 - theta / (np.pi / 2), 0, None)
    style = dict(color="#1a1a19", ls="--", lw=1.1)
    ax.plot(np.degrees(theta), linear, label="Das (2023)", **style)
    polar.plot(phi, np.concatenate([linear, linear[::-1]]), **style)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel(r"angle $\theta$ between $\hat d_{ej}$ and $-\nabla t$ (deg)")
    ax.set_ylabel(r"angular weight $\lambda_{ej}$")
    for x, text in [(0, "behind"), (90, "on the front"), (180, "ahead")]:
        ax.annotate(
            text,
            (x, 1.05),
            xytext=(0, 2),
            textcoords="offset points",
            ha={0: "left", 180: "right"}.get(x, "center"),
            va="bottom",
            fontsize=8,
            color="#52514e",
        )
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), borderaxespad=0.2)
    polar.set_ylim(0, 1.05)
    polar.set_yticks([0.5, 1.0])
    polar.set_yticklabels(["0.5", "1"], fontsize=7)
    polar.set_xticks([])
    polar.grid(True)
    polar.annotate(
        "",
        xy=(np.pi / 2, 0.75),
        xytext=(np.pi / 2, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#1a1a19", lw=0.9),
    )
    polar.annotate(
        r"$\nabla t$", (np.pi / 2, 0.82), ha="center", va="bottom", fontsize=9
    )
    polar.plot([0], [0], "o", color="#1a1a19", ms=3)
    polar.annotate(r"$e$", (0, 0), xytext=(4, -9), textcoords="offset points")
    fig.savefig(path)
    plt.close(fig)


def credit_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.62, 2.3))
    dt = np.linspace(-3, 5, 801)
    for k, color in zip(SHARPNESSES, RAMP):
        ax.plot(dt, cooling_credit(dt, k), color=color, label=rf"$k = {k:g}$")
    ax.axvline(0, color="#52514e", lw=0.6, ls=":")
    ax.set_xlim(dt[0], dt[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\Delta t / \tau$  ($> 0$: neighbour printed earlier)")
    ax.set_ylabel(r"cooling credit $F(\Delta t)$")
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    args.out.mkdir(parents=True, exist_ok=True)
    lambda_figure(args.out / "angular_weight_kappa.pdf")
    credit_figure(args.out / "cooling_credit_k.pdf")
