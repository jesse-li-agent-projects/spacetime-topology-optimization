"""Sequential against simultaneous optimization of the density and the time field, for
`tex/main.tex`: density, time field and hotspot severity of each design, severity read on
the grayscale design as optimized.

- Simultaneous: `stto_laplace` on `configs/laplace_default.json`.
- Sequential: a compliance-only plain TO on the same schedule, then the time field
  (diffusivity and wall data, same parameterization and severity as the simultaneous
  run) optimized against the hotspot with the density held fixed.

The runs are cached under `output/tex_seq_vs_sim_*`; `build(rerun=True)` redoes them.
Run from the repo root, via `python -m tex.make_figures`.
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.stto_laplace as stto_laplace
import sttopt.torch_util as torch_util
import sttopt.viz as viz
from sttopt import run_config
from sttopt.run_config import LaplaceRunConfig
from tex.figures.style import TEXT_WIDTH_IN

BASE_CONFIG = Path("configs/laplace_default.json")
OUTPUT = Path("output")
TAGS = {
    name: f"tex_seq_vs_sim_{name}"
    for name in ("plain_to", "sequential", "simultaneous")
}
VOID_GRAY = "0.85"


def _run_dir(name: str) -> Path:
    return OUTPUT / TAGS[name]


def _run_stto_laplace(name: str, config: dict, device: str | None) -> None:
    """Runs `stto_laplace_cli` on `config` into `_run_dir(name)`, replacing it."""
    OUTPUT.mkdir(exist_ok=True)
    config_path = OUTPUT / f"{TAGS[name]}_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    cmd = [
        sys.executable,
        "-m",
        "sttopt.stto_laplace_cli",
        "--config",
        str(config_path),
    ]
    cmd += ["--tag", TAGS[name], "--tag-force"]
    if device is not None:
        cmd += ["--device", device]
    subprocess.run(cmd, check=True)


def _run_sequential(base: dict, device: str | None) -> None:
    """Optimizes the time field alone on the plain TO's final density.

    The raw density variable is reset to the plain TO's after every step, so the
    physical density is held (projection pinned at the plain TO's final sharpness)
    while the time field's variables and MMA run as in the simultaneous run.
    """
    plain = LaplaceRunConfig.from_dict(
        json.loads((_run_dir("plain_to") / "config.json").read_text())
    )
    config = LaplaceRunConfig.from_dict(
        {
            **base,
            "beta_d_schedule": {
                "points": [[0, run_config.final_value(plain.beta_d_schedule)]],
                "mode": "linear",
            },
            "hotspot_shaping_weight": 0.0,  # acts on the density only
        }
    )
    problem = stto_laplace.build_problem(config, device=device)
    with np.load(_run_dir("plain_to") / "final_design.npz") as data:
        x = torch_util.to_tensor(data["x"], problem.device, problem.dtype)
    state = dataclasses.replace(stto_laplace.init_state(problem), x=x)
    for it in range(config.nloop):
        state, record = stto_laplace.step(problem, state)
        state = dataclasses.replace(state, x=x)
        print(f"sequential it {it:4d}  Tm {record.tru_max:7.3f}  c {record.obj:9.3f}")
    with torch.no_grad():
        xPhys, laplace = stto_laplace.physical_fields(
            problem,
            state.x,
            state.mu,
            state.wall,
            state.beta_d,
            state.loop,
            state.laplace,
        )
    out = _run_dir("sequential")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    np.savez_compressed(
        out / "final_design.npz",
        x=torch_util.to_numpy(state.x),
        mu=torch_util.to_numpy(state.mu),
        wall=torch_util.to_numpy(state.wall),
        xPhys=torch_util.to_numpy(xPhys),
        tPhys=torch_util.to_numpy(laplace.tPhys),
        loop=state.loop,
    )


def run_all(device: str | None, rerun: bool) -> None:
    """Makes whichever of the three runs is missing (all of them under `rerun`)."""
    base = json.loads(BASE_CONFIG.read_text())
    plain_to = {
        **base,
        "hotspot_weight": 0.0,
        "uniformity_weight": 0.0,
        "hotspot_shaping_weight": 0.0,
    }
    stale = rerun
    for name, run in (
        ("plain_to", lambda: _run_stto_laplace("plain_to", plain_to, device)),
        ("sequential", lambda: _run_sequential(base, device)),
        ("simultaneous", lambda: _run_stto_laplace("simultaneous", base, device)),
    ):
        if stale or not (_run_dir(name) / "final_design.npz").exists():
            run()
            # What follows the plain TO is built on it
            stale = stale or name == "plain_to"


def _evaluate(problem: stto_laplace.Problem, design: Path) -> dict:
    """The design's fields, compliance and severity, all on the density as optimized."""
    config = problem.config
    with np.load(design / "final_design.npz") as data:
        xPhys, tPhys = data["xPhys"], data["tPhys"]
    x_t, t_t = (
        torch_util.to_tensor(a, problem.device, problem.dtype) for a in (xPhys, tPhys)
    )
    with torch.no_grad():
        c, _, _ = compliance.batched_whole_and_gravity_compliance(
            x_t,
            torch.clamp(t_t, max=1),
            problem.KE,
            problem.edofMat,
            config.Emin,
            config.Emax,
            run_config.final_value(config.penal),
            problem.freedofs,
            problem.F,
            problem.ndof,
            problem.C,
            run_config.final_value(config.beta_t_schedule),
            [float(ti) for ti in np.linspace(0, 1, config.nStage + 1)[1:]],
        )
        K = problem.terms.estimated_conductivity(x_t, t_t)
        S = conductivity.severity(
            K, x_t, run_config.final_value(config.r), problem.terms.hotspot.void_switch
        ).reshape(xPhys.shape)
    S = torch_util.to_numpy(S)
    return dict(
        xPhys=xPhys,
        tPhys=tPhys,
        compliance=float(c),
        severity=S,
        max_severity=float(S.max()),
    )


def _title(label: str, result: dict, reference: dict | None) -> str:
    c = f"compliance {result['compliance']:.1f}"
    if reference is not None:
        c += f" ({100 * (result['compliance'] / reference['compliance'] - 1):.1f}% higher)"
    return f"{label}, {c},\nhighest hotspot severity {result['max_severity']:.2f}"


def figure(path: Path, device: str | None) -> None:
    config = LaplaceRunConfig.from_dict(json.loads(BASE_CONFIG.read_text()))
    problem = stto_laplace.build_problem(config, device=device)
    Tcr = run_config.final_value(config.Tcr)
    seq = _evaluate(problem, _run_dir("sequential"))
    sim = _evaluate(problem, _run_dir("simultaneous"))
    columns = (
        (seq, _title("Sequential", seq, None)),
        (sim, _title("Simultaneous", sim, seq)),
    )

    # Red above Tcr, blue below, white at it; the low side clips, as the high side is the
    # quantity of interest
    top = max(seq["max_severity"], sim["max_severity"])
    sev_norm = TwoSlopeNorm(vcenter=Tcr, vmin=Tcr - (top - Tcr), vmax=top)
    sev_cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad=VOID_GRAY)

    nely, nelx = seq["xPhys"].shape
    fig, axes = plt.subplots(
        3, 2, figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.5), layout="constrained"
    )
    for ax_col, (res, title) in zip(axes.T, columns):
        shown = res["xPhys"] >= 0.5
        density, time, severity = ax_col
        density.imshow(
            res["xPhys"], cmap="gray_r", vmin=0, vmax=1, extent=(0, nelx, -nely, 0)
        )
        viz.timefield_filled_contour_plot(
            res["xPhys"], res["tPhys"], 20, show_void=True, colorbar=False, ax=time
        )
        severity.imshow(
            np.where(shown & np.isfinite(res["severity"]), res["severity"], np.nan),
            cmap=sev_cmap,
            norm=sev_norm,
            extent=(0, nelx, -nely, 0),
        )
        density.set_title(title, fontsize=7)
        time.set_title("")
        for ax in ax_col:
            ax.set_xlim(0, nelx)
            ax.set_ylim(-nely, 0)
            ax.set_axis_off()
    for row, mappable, label in (
        (0, ScalarMappable(matplotlib.colors.Normalize(0, 1), "gray_r"), "density"),
        (
            1,
            ScalarMappable(
                matplotlib.colors.Normalize(*viz.TIMEFIELD_RANGE), "viridis"
            ),
            "time field",
        ),
        (2, ScalarMappable(sev_norm, sev_cmap), "hotspot severity"),
    ):
        fig.colorbar(
            mappable,
            ax=axes[row],
            label=label,
            extend="min" if row == 2 else "neither",
            shrink=0.9,
            aspect=10,
        )
    fig.savefig(path)
    plt.close(fig)


def build(out_dir: Path, *, device: str | None = None, rerun: bool = False) -> None:
    run_all(device, rerun)
    figure(out_dir / "seq_vs_sim.pdf", device)
