"""Summarize sweep runs and plot dynamics.

python3 analyze.tmp.py <group name> <run> [<run> ...]
Writes runs/analysis/<group>_dynamics.png, <group>_designs.png, and appends to summary.tsv.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sttopt.compliance as compliance
import sttopt.stto as stto
import sttopt.torch_util as torch_util
from sttopt.run_config import RunConfig, final_value

ROOT = Path("/tmp/claude-1001/d210ea4f/runs")
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)
COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

group, runs = sys.argv[1], sys.argv[2:]
logs, finals = {}, {}
for run in runs:
    d = ROOT / "output" / run
    lines = (d / "iterations.jsonl").read_text().splitlines()
    logs[run] = [json.loads(l) for l in lines if l.strip()]
    config = RunConfig.from_dict(json.loads((d / "config.json").read_text()))
    fin = d / "final_design.npz"
    if not fin.exists():
        continue
    z = np.load(fin)
    problem = stto.build_problem(config)

    def comp(arr):
        x = torch_util.to_tensor(arr, device=problem.device, dtype=problem.dtype)
        c, _ = compliance.whole_compliance(
            x,
            problem.KE,
            problem.edofMat,
            config.Emin,
            config.Emax,
            final_value(config.penal),
            problem.freedofs,
            problem.F,
            problem.ndof,
        )
        return float(c)

    xPhys, tPhys = z["xPhys"], z["tPhys"]
    xt = torch_util.to_tensor(xPhys, device=problem.device, dtype=problem.dtype)
    tt_ = torch_util.to_tensor(tPhys, device=problem.device, dtype=problem.dtype)
    K = stto.estimated_conductivity(problem, xt, tt_)
    fin_mask = torch.isfinite(K)
    xb = (xPhys > 0.5).astype(float)
    xbt = torch_util.to_tensor(xb, device=problem.device, dtype=problem.dtype)
    Kb = stto.estimated_conductivity(problem, xbt, tt_)
    fb = torch.isfinite(Kb)
    sev_bin = float(((1 - Kb) * xbt.flatten())[fb].max())
    sev_map = torch.where(
        fin_mask, (1 - K) * xt.flatten() ** config.r, torch.full_like(K, np.nan)
    )
    last = logs[run][-1]
    c_series = np.array([e["obj"] for e in logs[run]])
    within2 = int(np.argmax(np.abs(c_series - c_series[-1]) <= 0.02 * c_series[-1]) + 1)
    over = [e["loop"] for e in logs[run] if e["true_max"] > last["Tcr"] + 0.005]
    tail = logs[run][-50:]
    log_text = (ROOT / "logs" / f"{run}.log").read_text()
    finals[run] = dict(
        c=comp(xPhys),
        c_bin=comp(xb),
        true_max=last["true_max"],
        true_max_bin=sev_bin,
        # Stability over the last 50 iterations: an end-of-run value alone hides the
        # sawtooth a stale calibration offset leaves behind.
        tm_tail_max=max(e["true_max"] for e in tail),
        tm_tail_min=min(e["true_max"] for e in tail),
        c_tail_max=max(e["obj"] for e in tail),
        subsolv_warnings=log_text.count("subsolv:"),
        unif=last["uniformity"],
        rough=last["roughness"],
        grey=last["grey"],
        n_eff=last["n_eff"],
        it_c_within2pct=within2,
        last_it_over_Tcr=(over[-1] if over else 0),
        s_per_it=last["elapsed"] / last["loop"],
        xPhys=xPhys,
        tPhys=tPhys,
        sev=torch_util.to_numpy(sev_map).reshape(config.nely, config.nelx),
    )

with open(OUT / "summary.tsv", "a") as f:
    for run, m in finals.items():
        row = {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
        f.write(run + "\t" + json.dumps(row) + "\n")
        print(
            run,
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
        )

# Dynamics: small multiples sharing the iteration axis, one color per run in fixed order.
panels = [
    ("obj", "compliance c", "log"),
    ("true_max", "true max severity", "linear"),
    ("lam_hot", "hotspot multiplier (lam)", "symlog"),
    ("n_eff", "hotspot n_eff", "log"),
    ("grey", "grey fraction", "linear"),
    ("uniformity", "uniformity (unweighted)", "log"),
    ("dx_mean", "mean |dx| per step", "log"),
    ("beta_d", "beta_d", "log"),
]
fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.0 * len(panels)), sharex=True)
for i, run in enumerate(runs):
    L = logs[run]
    it = [e["loop"] for e in L]
    for ax, (key, label, scale) in zip(axes, panels):
        y = [e["lam"][-1] for e in L] if key == "lam_hot" else [e[key] for e in L]
        ax.plot(it, y, color=COLORS[i % len(COLORS)], lw=1.2, label=run)
for ax, (key, label, scale) in zip(axes, panels):
    ax.set_yscale(scale)
    ax.set_title(label, loc="left", fontsize=9, color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    if key == "true_max":
        tcr = [e["Tcr"] for e in logs[runs[0]]]
        ax.plot(
            [e["loop"] for e in logs[runs[0]]],
            tcr,
            color=MUTED,
            lw=0.8,
            ls="-",
            label="Tcr (" + runs[0] + ")",
        )
        ax.set_ylim(0.5, 1.05)
    if key == "obj":
        ax.set_ylim(150, 1200)
        ax.axhline(200, color=MUTED, lw=0.8)
axes[0].legend(fontsize=8, ncol=2, frameon=False)
axes[-1].set_xlabel("iteration", color=MUTED)
fig.tight_layout()
fig.savefig(OUT / f"{group}_dynamics.png", dpi=110)

# Final designs: density with print-time contours, and severity map, per run.
done = [r for r in runs if r in finals]
if done:
    fig, axes = plt.subplots(len(done), 2, figsize=(12, 2.3 * len(done)), squeeze=False)
    for row, run in zip(axes, done):
        m = finals[run]
        row[0].imshow(1 - m["xPhys"], cmap="gray", vmin=0, vmax=1)
        tm = np.where(m["xPhys"] > 0.5, m["tPhys"], np.nan)
        row[0].contour(
            tm, levels=np.linspace(0, 1, 17), colors="#2a78d6", linewidths=0.6
        )
        row[0].set_title(
            f"{run}: c={m['c']:.1f} (bin {m['c_bin']:.1f}) unif={m['unif']:.3f}",
            fontsize=9,
            loc="left",
        )
        im = row[1].imshow(
            np.where(m["xPhys"] > 0.5, m["sev"], np.nan), cmap="viridis", vmin=0, vmax=1
        )
        row[1].set_title(
            f"severity: true max {m['true_max']:.3f} (bin {m['true_max_bin']:.3f})",
            fontsize=9,
            loc="left",
        )
        for a in row:
            a.set_xticks([])
            a.set_yticks([])
    fig.colorbar(im, ax=axes[:, 1].tolist(), fraction=0.02)
    fig.savefig(OUT / f"{group}_designs.png", dpi=110)
