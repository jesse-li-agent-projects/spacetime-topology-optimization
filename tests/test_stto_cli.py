"""Smoke test for sttopt.stto_cli. Calls parse_args/main directly (not via subprocess) with
tiny overrides -- per the repo's sandbox rules, nothing near production scale
(180x60x800) is run here, and per the plan's Phase 9 guidance this phase gets the
lightest testing budget of the whole port.
"""

import dataclasses
import json
import re

import numpy as np
import pytest

import sttopt.checks_cli as checks_cli
import sttopt.stto_cli as stto_cli
import sttopt.stto as stto
from conftest import ELEMENT_M, default_run_config

# A few iterations do not make a printable design, so the final checks warn; they are
# tested in `test_checks.py`.
pytestmark = pytest.mark.filterwarnings("ignore:print (start|support):UserWarning")

# The mesh, nStage and the radii are config-file-only (not CLI flags), so this
# fixture's overrides for them go through --config rather than argv.
_FIXTURE_CONFIG = default_run_config(
    nelx=7,
    nely=5,
    nStage=2,
    rmin_m=2 * ELEMENT_M,
    lrmin_m=2 * ELEMENT_M,
    rmin_cond_m=3 * ELEMENT_M,
    nloop=2,
    # Per-step volume change scales with `move`; the obj/vol test needs it clear of
    # what the printed "Vol." can resolve.
    move=0.03,
    # The default's lower early penalty moves the volume less per step.
    penal=3.0,
    # Set here rather than left to the default, which has been `false`: without the
    # stage volume constraints the global volume constraint pins the mean from the first
    # iteration, and a volume that no longer moves within an iteration is one the
    # obj/vol test cannot read either quantity from. `move` alone does not recover it.
    enable_stage_volume=True,
)


def _argv(tmp_path, tag):
    config_path = tmp_path / "fixture_config.json"
    config_path.write_text(json.dumps(_FIXTURE_CONFIG.to_dict()))
    return [
        "--config",
        str(config_path),
        "--tag",
        tag,
    ]


def test_cli_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = stto_cli.parse_args(_argv(tmp_path, "smoke"))

    stto_cli.main(args)

    assert (tmp_path / "output" / "smoke" / "final_design.npz").exists()


def test_checks_cli_reproduces_the_runs_own_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stto_cli.main(stto_cli.parse_args(_argv(tmp_path, "checked")))
    report = tmp_path / "output" / "checked" / "final_design_checks.json"
    written = json.loads(report.read_text())
    report.unlink()

    checks_cli.main(checks_cli.parse_args([str(report.with_name("final_design.npz"))]))

    rerun = json.loads(report.read_text())
    # The physics differs in the last bits between two runs on the GPU.
    physics = ("constraints", "compliance")
    assert {k: v for k, v in rerun.items() if k not in physics} == {
        k: v for k, v in written.items() if k not in physics
    }
    assert rerun["compliance"] == pytest.approx(written["compliance"], rel=1e-12)
    assert rerun["constraints"].keys() == written["constraints"].keys()
    for name, values in written["constraints"].items():
        np.testing.assert_allclose(rerun["constraints"][name], values, rtol=1e-12)


def _reference_run(config):
    """Independently drives the same optimize loop main() does, for comparison against
    what main() actually prints -- catches a regression to the wrong MATLAB quantity
    (see the Phase 9 review: stto_cli.py originally printed IterationRecord.obj/.vol, which
    are NOT what MATLAB's disp actually prints; see stto_cli.py's module docstring).
    """
    problem = stto.build_problem(config)
    state = stto.init_state(problem)
    records, states = [], []
    for _ in range(config.nloop):
        state, record = stto.step(problem, state)
        records.append(record)
        states.append(state)
    return problem, state, records, states


def test_cli_prints_full_objective_and_post_update_volume(
    capsys, tmp_path, monkeypatch
):
    """
    "Obj." must print the full MMA objective (`IterationRecord.f`), not the
    whole-structure compliance (`.obj`); "Vol." must print this iteration's post-update
    volume, not the pre-update `.vol`.
    """
    monkeypatch.chdir(tmp_path)
    args = stto_cli.parse_args(_argv(tmp_path, "obj_vol"))
    problem, _, records, states = _reference_run(stto_cli.resolve_config(args))

    stto_cli.main(args)
    out = capsys.readouterr().out

    it_lines = [line for line in out.splitlines() if line.startswith("It.:")]
    assert len(it_lines) == _FIXTURE_CONFIG.nloop
    # Printed rounding: "Obj." is %10.4f, "Vol." is %6.3f.
    obj_rounding, vol_rounding = 5e-5, 5e-4
    obj_rtol, vol_atol = 1e-6, 6e-4
    for line, record, state in zip(it_lines, records, states):
        obj = float(re.search(r"Obj\.:\s*([\d.-]+)", line).group(1))
        vol = float(re.search(r"Vol\.:\s*([\d.-]+)", line).group(1))
        xPhys, _ = stto.physical_fields(problem, state.x, state.t, state.beta_d)
        post_update_vol = float(xPhys.mean())
        np.testing.assert_allclose(obj, record.f, rtol=obj_rtol)
        np.testing.assert_allclose(vol, post_update_vol, atol=vol_atol, rtol=0)
        # Guard against vacuous passes: the wrong quantity is only certain to fail the
        # checks above if it sits further off than their tolerance plus the rounding.
        assert abs(record.f - record.obj) > obj_rtol * abs(record.f) + obj_rounding
        assert abs(post_update_vol - record.vol) > vol_atol + vol_rounding


def test_cli_logs_scheduled_continuation(tmp_path, monkeypatch):
    """Every scheduled setting reaches the iteration it is scheduled for, and the
    per-iteration log records it."""
    monkeypatch.chdir(tmp_path)
    config = dataclasses.replace(
        _FIXTURE_CONFIG,
        nloop=3,
        hotspot_aggregation="logsumexp",
        hotspot_beta={"points": [[0, 4.0], [2, 16.0]], "mode": "log"},
        penal={"points": [[0, 1.0], [2, 3.0]]},
        Tcr={"points": [[0, 5.0], [2, 0.8]]},
        beta_d_schedule={"points": [[0, 1.0], [2, 4.0]], "mode": "log"},
        beta_t_schedule=20.0,
    )
    config_path = tmp_path / "scheduled.json"
    config_path.write_text(json.dumps(config.to_dict()))
    stto_cli.main(stto_cli.parse_args(["--config", str(config_path), "--tag", "sched"]))

    lines = (tmp_path / "output" / "sched" / "iterations.jsonl").read_text()
    log = [json.loads(line) for line in lines.splitlines()]
    assert [e["loop"] for e in log] == [0, 1, 2]
    np.testing.assert_allclose([e["hotspot_beta"] for e in log], [4.0, 8.0, 16.0])
    np.testing.assert_allclose([e["penal"] for e in log], [1.0, 2.0, 3.0])
    np.testing.assert_allclose([e["Tcr"] for e in log], [5.0, 2.9, 0.8])
    np.testing.assert_allclose([e["beta_d"] for e in log], [1.0, 2.0, 4.0])
    assert all(e["beta_t"] == 20.0 for e in log)


def test_resume_reproduces_an_uninterrupted_run(tmp_path, monkeypatch):
    """A run stopped after a checkpoint and resumed must end where an uninterrupted
    run ends, with one log entry per iteration."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "resume_config.json"
    config_path.write_text(
        json.dumps(dataclasses.replace(_FIXTURE_CONFIG, nloop=5).to_dict())
    )

    def run(tag, *extra):
        argv = ["--tag", tag, "--snapshot-every", "2", *extra]
        stto_cli.main(stto_cli.parse_args(argv))

    run("whole", "--config", str(config_path))

    real_step = stto.step
    calls = []

    def stopping_step(problem, state):
        # Iteration 3 runs after the checkpoint at iteration 2, and must be run again.
        if len(calls) == 4:
            raise KeyboardInterrupt
        calls.append(state.loop)
        return real_step(problem, state)

    monkeypatch.setattr(stto, "step", stopping_step)
    with pytest.raises(KeyboardInterrupt):
        run("resumed", "--config", str(config_path))
    monkeypatch.setattr(stto, "step", real_step)
    run("resumed", "--resume")

    whole, resumed = (
        np.load(tmp_path / "output" / tag / "final_design.npz")
        for tag in ("whole", "resumed")
    )
    # Two runs of one config already differ in the last bits from iteration 0 on, and
    # the difference grows with the iterations; measured up to ~40 eps relative.
    eps = np.finfo(float).eps
    tol = dict(rtol=128 * eps, atol=8 * eps)
    for key in whole.files:
        np.testing.assert_allclose(resumed[key], whole[key], err_msg=key, **tol)

    # The hottest element's position is an argmax, which those bits can move between
    # near-tied elements; `true_max` checks its value.
    skipped = {"elapsed", "hot_row", "hot_col"}

    def log(tag):
        lines = (tmp_path / "output" / tag / "iterations.jsonl").read_text()
        return [
            {k: v for k, v in json.loads(line).items() if k not in skipped}
            for line in lines.splitlines()
        ]

    resumed_log, whole_log = log("resumed"), log("whole")
    assert [e.keys() for e in resumed_log] == [e.keys() for e in whole_log]
    for r, w in zip(resumed_log, whole_log):
        for key in w:
            np.testing.assert_allclose(r[key], w[key], err_msg=key, **tol)
