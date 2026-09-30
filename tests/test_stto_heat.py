"""`stto_heat`: the virtual-heat variant's optimization loop, end to end on a small
mesh, and its CLI."""

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import sttopt.checks as checks
import sttopt.run_config as run_config
import sttopt.stto_heat as stto_heat
import sttopt.stto_heat_cli as stto_heat_cli
from sttopt.run_config import HeatRunConfig

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "heat_default.json"
NITER = 12


def _config(nelx=30, nely=10, **overrides):
    base = HeatRunConfig.from_dict(json.loads(CONFIG_PATH.read_text()))
    h = base.element_size_m
    mesh = dict(nelx=nelx, width_m=nelx * h, height_m=nely * h, nloop=NITER)
    return dataclasses.replace(base, **(mesh | overrides))


@pytest.fixture(scope="module")
def smoke_run():
    return stto_heat.run(_config(), device="cpu")


def test_default_config_loads_with_a_plate_base():
    config = HeatRunConfig.from_dict(json.loads(CONFIG_PATH.read_text()))
    assert config.print_base in ("edge", "bottom_edge")


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
    for v in (state.x, state.mu):
        assert float(v.min()) >= 0 and float(v.max()) <= 1


def test_warm_starts_carry_no_graph(smoke_run):
    state = smoke_run.state
    assert not state.U.requires_grad
    assert all(v is None or not v.requires_grad for v in state.heat)


def test_the_time_field_starts_at_the_plate(smoke_run):
    """The plate column of `tPhys` is the earliest one, as an edge plate requires."""
    tPhys = smoke_run.tPhys_traj[-1]
    column_min = tPhys.min(dim=0).values
    assert int(column_min.argmin()) == 0


def test_check_design_reports_in_the_shared_layout(smoke_run):
    problem = stto_heat.build_problem(_config(), device="cpu")
    state = smoke_run.state
    with warnings.catch_warnings():
        # A short run's design may fail support; this test is about the report
        warnings.simplefilter("ignore", UserWarning)
        report = stto_heat.check_design(problem, state.x, state.mu, state.loop)
    assert {"start", "support", "saddles", "constraints", "compliance"} <= report.keys()
    assert report["start"]["passed"]
    assert "earliest node off the plate" in checks.summary(report)


def test_cli_writes_the_run_artefacts(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(nloop=3).to_dict()))
    monkeypatch.chdir(tmp_path)
    stto_heat_cli.main(
        stto_heat_cli.parse_args(
            [
                "--config",
                str(config_path),
                "--tag",
                "smoke",
                "--snapshot-every",
                "2",
                "--device",
                "cpu",
            ]
        )
    )
    out = tmp_path / "output" / "smoke"
    lines = [
        json.loads(line) for line in (out / "iterations.jsonl").read_text().splitlines()
    ]
    assert len(lines) == 3
    assert {"local_minima", "saddles"} <= lines[0].keys()
    assert "log_chi_roughness" in lines[-1]
    design = np.load(out / "final_design.npz")
    assert {"x", "mu", "xPhys", "tPhys"} <= set(design.files)
    assert (
        json.loads(checks.report_path(out / "final_design.npz").read_text())["loop"]
        == 3
    )


@pytest.mark.filterwarnings("ignore:subsolv:RuntimeWarning")
def test_a_step_stays_finite_when_void_is_later_than_the_part():
    """A solid block at the plate and void beyond it: `t` in the void exceeds 1, the
    part's maximum. At a non-integer `penal`, the gravity stages must still be finite."""
    config = _config()
    problem = stto_heat.build_problem(config, device="cpu")
    state = stto_heat.init_state(problem)
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
    _, heat = stto_heat.physical_fields(problem, state.x, state.mu, state.beta_d)
    assert float(heat.tPhys.max()) > 1  # non-vacuous
    _, record = stto_heat.step(problem, state)
    assert np.isfinite(record.f) and np.isfinite(record.g).all()
    assert np.isfinite(record.dg).all()
