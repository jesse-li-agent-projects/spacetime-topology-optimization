"""`stto_laplace`: the harmonic-time-field variant's loop, end to end on a small mesh,
and its CLI."""

import dataclasses
import math
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import sttopt.checks as checks
import sttopt.load_cases as load_cases
import sttopt.run_config as run_config
import sttopt.stto_laplace as stto_laplace
import sttopt.stto_laplace_cli as stto_laplace_cli
import sttopt.virtual_heat as virtual_heat
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


# Three iterations leave the design grey, so binarizing moves the hotspot row a lot.
@pytest.mark.filterwarnings("ignore:binarizing raised the hotspot row:UserWarning")
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
    loop = 150  # where the default penal is fractional
    state = dataclasses.replace(
        state,
        x=x,
        loop=loop,
        beta_d=run_config.weight_at(config.beta_d_schedule, loop),
        beta_t=run_config.weight_at(config.beta_t_schedule, loop),
    )
    assert run_config.weight_at(config.penal, loop) % 1 != 0
    _, sol = stto_laplace.physical_fields(
        problem, state.x, state.mu, state.wall, state.beta_d, loop
    )
    assert float(sol.tPhys.max()) > 1  # non-vacuous
    _, record = stto_laplace.step(problem, state)
    assert np.isfinite(record.f) and np.isfinite(record.g).all()
    assert np.isfinite(record.dg).all()


@pytest.mark.filterwarnings("ignore:subsolv:RuntimeWarning")
def test_field_move_bounds_the_time_field_variables_apart_from_the_density():
    move, field_move = 0.01, 0.002
    problem = stto_laplace.build_problem(
        _config(move=move, field_move=field_move), device="cpu"
    )
    state = stto_laplace.init_state(problem)
    for _ in range(3):
        state, record = stto_laplace.step(problem, state)
    diag = record.diagnostics
    assert diag["dx_max"] <= move + 1e-12
    assert max(diag["dmu_max"], diag["dwall_max"]) <= field_move + 1e-12
    assert diag["dx_max"] > field_move  # non-vacuous
    assert diag["dtphys_max"] > 0


@pytest.mark.filterwarnings("ignore:subsolv:RuntimeWarning")
def test_a_density_exponent_routes_time_through_the_material():
    """With `chi` following the density, a void gap in a bar slows time across it, so
    the far side is later than with `chi` blind to the density."""
    nelx, nely = 30, 10
    x = torch.full((nely, nelx), 1e-3, dtype=torch.float64)
    x[:, :12] = 1.0  # a block on the plate, a void gap, then another block
    x[:, 16:] = 1.0
    t_far = {}
    for e in (0.0, 1.0):
        problem = stto_laplace.build_problem(
            _config(nelx, nely, chi_density_exponent=e), device="cpu"
        )
        state = stto_laplace.init_state(problem)
        state = dataclasses.replace(state, x=x)
        _, sol = stto_laplace.physical_fields(
            problem, state.x, state.mu, state.wall, math.inf, 0
        )
        t_far[e] = float(sol.tPhys[:, 20:].mean())
        _, record = stto_laplace.step(problem, state)
        assert np.isfinite(record.f) and np.isfinite(record.dg).all()
    assert t_far[1.0] != t_far[0.0]


@pytest.mark.parametrize("sensitivity", [True, False])
def test_the_density_sensitivity_through_chi_can_be_cut(sensitivity):
    """With `chi_density_sensitivity` off, `t` still follows the density but no
    gradient reaches it through `chi`."""
    nelx, nely = 30, 10
    config = _config(
        nelx, nely, chi_density_exponent=4.0, chi_density_sensitivity=sensitivity
    )
    problem = stto_laplace.build_problem(config, device="cpu")
    state = stto_laplace.init_state(problem)
    xPhys = torch.rand(nely, nelx, dtype=torch.float64).clamp(0.1).requires_grad_()
    sol = virtual_heat.laplace_time_field(
        config,
        problem.mesh,
        xPhys,
        state.mu.clone().requires_grad_(),  # so `t` has a graph either way
        problem.wall_filter @ state.wall,
        density_exponent=4.0,
    )
    (grad,) = torch.autograd.grad(sol.tPhys.sum(), xPhys, allow_unused=True)
    assert (grad is not None and float(grad.abs().max()) > 0) == sensitivity


def _shaping_gradient(x, Tcr=0.6):
    """`FieldTerms.hotspot_shaping` and its density gradient on a 60x20 mesh printed
    column by column from the left, under capped-ratio shielding."""
    nely, nelx = x.shape
    base = _config(nelx, nely)
    config = _config(
        nelx,
        nely,
        hotspot_shielding="capped_ratio",
        hotspot_shaping_weight=1.0,
        rmin_cond_m=4 * base.element_size_m,
        Tcr=Tcr,
    )
    terms = stto_laplace.build_problem(config, device="cpu").terms
    t = torch.linspace(0, 1, nelx, dtype=torch.float64).expand(nely, nelx)
    x = x.clone().requires_grad_(True)
    shaping = terms.hotspot_shaping(x, t, 0)
    (grad,) = torch.autograd.grad(shaping, x)
    return float(shaping.detach()), grad


def test_hotspot_shaping_pushes_a_violating_bar_out_and_its_void_support_in():
    x = torch.zeros(20, 60, dtype=torch.float64)
    x[:, 40:44] = 0.5  # a grey bar with nothing printed before it
    _, grad = _shaping_gradient(x)
    rows = slice(8, 12)
    assert (grad[rows, 40] > 0).all()  # R1
    assert (grad[rows, 38] < 0).all()  # R2, on void


def test_hotspot_shaping_is_quiet_on_a_bar_whose_solid_version_passes():
    x = torch.zeros(20, 60, dtype=torch.float64)
    x[:, :40] = 1.0
    x[:, 40:44] = 0.5  # the same bar, on solid support
    _, quiet = _shaping_gradient(x)
    x[:, :40] = 0.0
    _, loud = _shaping_gradient(x)
    rows = slice(8, 12)
    assert quiet[rows, 36:44].abs().max() < 1e-3 * loud[rows, 40].abs().min()


def test_hotspot_shaping_needs_a_relative_shielding():
    with pytest.raises(ValueError, match="hotspot_shaping_weight"):
        stto_laplace.build_problem(
            _config(hotspot_shielding="absolute", hotspot_shaping_weight=1.0),
            device="cpu",
        )


@pytest.mark.filterwarnings("ignore:subsolv:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:binarizing raised the hotspot row:UserWarning")
def test_an_l_bracket_runs_on_its_domain_alone():
    n = 24
    base = _config()
    config = _config(
        n,
        n,
        load_case="l_bracket_corner",
        print_base="right_edge",
        rmin_cond_m=4 * base.element_size_m,
        nloop=3,
    )
    result = stto_laplace.run(config, device="cpu")
    active = torch.as_tensor(
        load_cases.active_mask(load_cases.LoadCase(config.load_case), n, n)
    )
    for xPhys in result.xPhys_traj:
        assert float(xPhys[~active].abs().max()) == 0
    first = result.records[0]
    # Of the domain, not the mesh
    assert first.vol == pytest.approx(float(result.xPhys_traj[0].sum() / active.sum()))
    assert np.isfinite(first.df).all() and np.isfinite(first.dg).all()
    problem = stto_laplace.build_problem(config, device="cpu")
    report = stto_laplace.check_design(problem, result.state)
    assert "constraints" in report
