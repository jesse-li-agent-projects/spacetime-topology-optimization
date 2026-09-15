"""Smoke test for sttopt.stto_cli. Calls parse_args/main directly (not via subprocess) with
tiny overrides -- per the repo's sandbox rules, nothing near production scale
(180x60x800) is run here, and per the plan's Phase 9 guidance this phase gets the
lightest testing budget of the whole port.
"""

import dataclasses
import json
import re
from pathlib import Path

import numpy as np

import sttopt.stto_cli as stto_cli
import sttopt.stto as stto
from sttopt.run_config import RunConfig

# nelx/nely/nStage/rmin/lrmin/rmin_cond are config-file-only (not CLI flags), so this
# fixture's overrides for them go through --config rather than argv.
_DEFAULT_CONFIG = RunConfig.from_dict(
    json.loads((Path(__file__).parent.parent / "configs" / "default.json").read_text())
)
_FIXTURE_CONFIG = dataclasses.replace(
    _DEFAULT_CONFIG,
    nelx=7,
    nely=5,
    nStage=2,
    rmin=2,
    lrmin=2,
    rmin_cond=3,
    nloop=2,
    # Per-step volume change scales with `move`; the obj/vol test needs it clear of
    # what the printed "Vol." can resolve.
    move=0.03,
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


def _reference_run(config):
    """Independently drives the same optimize loop main() does, for comparison against
    what main() actually prints -- catches a regression to the wrong MATLAB quantity
    (see the Phase 9 review: stto_cli.py originally printed IterationRecord.obj/.vol, which
    are NOT what MATLAB's disp actually prints; see stto_cli.py's module docstring).
    """
    problem = stto.build_problem(config)
    state = stto.init_state(problem, beta_d=1.0)
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
        hotspot_aggregation="logsumexp_severity",
        hotspot_beta={"points": [[1, 4.0], [3, 16.0]], "log": True},
        penal={"points": [[1, 1.0], [3, 3.0]]},
        Tcr={"points": [[1, 5.0], [3, 0.8]]},
        beta_d_schedule={"points": [[1, 1.0], [3, 4.0]], "log": True},
        beta_t_schedule=20.0,
    )
    config_path = tmp_path / "scheduled.json"
    config_path.write_text(json.dumps(config.to_dict()))
    stto_cli.main(stto_cli.parse_args(["--config", str(config_path), "--tag", "sched"]))

    lines = (tmp_path / "output" / "sched" / "iterations.jsonl").read_text()
    log = [json.loads(line) for line in lines.splitlines()]
    assert [e["loop"] for e in log] == [1, 2, 3]
    np.testing.assert_allclose([e["hotspot_beta"] for e in log], [4.0, 8.0, 16.0])
    np.testing.assert_allclose([e["penal"] for e in log], [1.0, 2.0, 3.0])
    np.testing.assert_allclose([e["Tcr"] for e in log], [5.0, 2.9, 0.8])
    np.testing.assert_allclose([e["beta_d"] for e in log], [1.0, 2.0, 4.0])
    assert all(e["beta_t"] == 20.0 for e in log)
    # A change in the aggregate's sharpness refreshes its calibration.
    assert all(e["recalibrated"] for e in log[1:])
