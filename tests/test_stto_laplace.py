"""`stto_laplace`: the harmonic-time-field variant's loop, end to end on a small mesh,
and its CLI."""

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import sttopt.checks as checks
import sttopt.run_config as run_config
import sttopt.stto_laplace as stto_laplace
import sttopt.stto_laplace_cli as stto_laplace_cli
from sttopt.run_config import LaplaceRunConfig

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "laplace_default.json"
NITER = 12


def _config(nelx=30, nely=10, **overrides):
    base = LaplaceRunConfig.from_dict(json.loads(CONFIG_PATH.read_text()))
    h = base.element_size_m
    mesh = dict(nelx=nelx, width_m=nelx * h, height_m=nely * h, nloop=NITER)
    return dataclasses.replace(base, **(mesh | overrides))


@pytest.fixture(scope="module")
def smoke_run():
    with warnings.catch_warnings():
        # The wall group's gradient changes sign from step to step, so MMA tightens
        # its asymptotes, and the subproblem's last barrier level hits its Newton cap
        warnings.filterwarnings("ignore", "subsolv", RuntimeWarning)
        return stto_laplace.run(_config(), device="cpu")


def test_default_config_loads_with_a_plate_base():
    config = LaplaceRunConfig.from_dict(json.loads(CONFIG_PATH.read_text()))
    assert config.print_base in ("edge", "bottom_edge")


def test_the_start_is_the_ramp(smoke_run):
    """Uniform `chi` and the ramp wall data give a linear `t` away from the plate, up to
    the wall filter rounding the ramp's kinks at the far corners (measured 0.011)."""
    tPhys = smoke_run.tPhys_traj[0].numpy()
    expected = (np.arange(30) + 0.5)[None].repeat(10, axis=0)
    np.testing.assert_allclose(tPhys, expected / expected.max(), atol=0.02)
    # Exact next to the plate, where the filter reflects about the plate corners
    np.testing.assert_allclose(
        tPhys[:, :5], (expected / expected.max())[:, :5], atol=1e-3
    )


def test_smoke_run_stays_finite(smoke_run):
    for record in smoke_run.records:
        assert np.isfinite(record.f)
        assert np.isfinite(record.g).all() and np.isfinite(record.dg).all()
        assert np.isfinite(record.df).all()
    assert all(torch.isfinite(t).all() for t in smoke_run.tPhys_traj)


def test_smoke_run_lowers_the_objective_or_the_infeasibility(smoke_run):
    first, last = smoke_run.records[0], smoke_run.records[-1]
    assert last.f < first.f or max(last.g.max(), 0) < max(first.g.max(), 0)


def test_design_stays_on_the_unit_box(smoke_run):
    state = smoke_run.state
    for v in (state.x, state.mu, state.wall):
        assert float(v.min()) >= 0 and float(v.max()) <= 1


def test_the_wall_data_is_optimized(smoke_run):
    """`wall` receives sensitivities and moves, so it is a live design group."""
    first, last = smoke_run.wall_traj[0], smoke_run.wall_traj[-1]
    assert not torch.allclose(first, last)


def test_warm_starts_carry_no_graph(smoke_run):
    state = smoke_run.state
    assert not state.U.requires_grad
    assert all(v is None or not v.requires_grad for v in state.laplace)


def test_check_design_reports_the_domain_as_well(smoke_run):
    problem = stto_laplace.build_problem(_config(), device="cpu")
    with warnings.catch_warnings():
        # A short run's design may fail support; this test is about the report
        warnings.simplefilter("ignore", UserWarning)
        report = stto_laplace.check_design(problem, smoke_run.state)
    assert {"start", "support", "saddles", "domain", "constraints"} <= report.keys()
    assert report["start"]["passed"]
    assert report["passed"] == report["support"]["passed"]
    assert "earliest node off the plate" in checks.summary(report)
    assert "over the domain:" in checks.summary(report)


def test_cli_writes_the_run_artefacts(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(nloop=3).to_dict()))
    monkeypatch.chdir(tmp_path)
    args = ["--config", str(config_path), "--tag", "smoke", "--snapshot-every", "2"]
    stto_laplace_cli.main(stto_laplace_cli.parse_args(args + ["--device", "cpu"]))
    out = tmp_path / "output" / "smoke"
    lines = [
        json.loads(line) for line in (out / "iterations.jsonl").read_text().splitlines()
    ]
    assert len(lines) == 3
    assert {
        "local_minima",
        "saddles",
        "domain_local_minima",
        "domain_saddles",
    } <= lines[0].keys()
    assert {"log_chi_roughness", "wall_peaks", "wall_dips"} <= lines[-1].keys()
    design = np.load(out / "final_design.npz")
    assert {"x", "mu", "wall", "xPhys", "tPhys"} <= set(design.files)
    report = json.loads(checks.report_path(out / "final_design.npz").read_text())
    assert report["loop"] == 3


@pytest.mark.filterwarnings("ignore:subsolv:RuntimeWarning")
def test_a_step_stays_finite_when_void_is_later_than_the_part():
    """As in `test_stto_heat`: `t` in the void beyond the part exceeds 1."""
    config = _config()
    problem = stto_laplace.build_problem(config, device="cpu")
    state = stto_laplace.init_state(problem)
    x = torch.full_like(state.x, 1e-3)
    x[:, :8] = 1.0
    loop = 200  # penal ramps 2 -> 3 over 150-350
    state = dataclasses.replace(
        state,
        x=x,
        loop=loop,
        beta_d=run_config.weight_at(config.beta_d_schedule, loop),
        beta_t=run_config.weight_at(config.beta_t_schedule, loop),
    )
    assert run_config.weight_at(config.penal, loop) % 1 != 0
    _, sol = stto_laplace.physical_fields(
        problem, state.x, state.mu, state.wall, state.beta_d
    )
    assert float(sol.tPhys.max()) > 1  # non-vacuous
    _, record = stto_laplace.step(problem, state)
    assert np.isfinite(record.f) and np.isfinite(record.g).all()
    assert np.isfinite(record.dg).all()
