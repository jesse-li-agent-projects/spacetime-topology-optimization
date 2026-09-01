"""Smoke test for sttopt.cli. Calls parse_args/main directly (not via subprocess) with
tiny overrides -- per the repo's sandbox rules, nothing near production scale
(180x60x800) is run here, and per the plan's Phase 9 guidance this phase gets the
lightest testing budget of the whole port.
"""

import dataclasses
import json
import re
from pathlib import Path

import numpy as np

import sttopt.cli as cli
import sttopt.optimize as optimize
from sttopt.run_config import RunConfig

# nelx/nely/nStage/rmin/lrmin/rmin_cond are config-file-only (not CLI flags), so this
# fixture's overrides for them go through --config rather than argv.
_DEFAULT_CONFIG = RunConfig.from_dict(
    json.loads((Path(__file__).parent.parent / "configs" / "default.json").read_text())
)
_FIXTURE_CONFIG = dataclasses.replace(
    _DEFAULT_CONFIG, nelx=7, nely=5, nStage=2, rmin=2, lrmin=2, rmin_cond=3, nloop=2
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
    args = cli.parse_args(_argv(tmp_path, "smoke"))

    cli.main(args)

    assert (tmp_path / "output" / "smoke" / "final_design.npz").exists()


def _reference_run(config):
    """Independently drives the same optimize loop main() does, for comparison against
    what main() actually prints -- catches a regression to the wrong MATLAB quantity
    (see the Phase 9 review: cli.py originally printed IterationRecord.obj/.vol, which
    are NOT what MATLAB's disp actually prints; see cli.py's module docstring).
    """
    problem = optimize.build_problem(config)
    state = optimize.init_state(problem, beta_d=1.0)
    records, states = [], []
    for _ in range(config.nloop):
        state, record = optimize.step(problem, state)
        records.append(record)
        states.append(state)
    return problem, state, records, states


def test_cli_prints_full_objective_and_post_update_volume(
    capsys, tmp_path, monkeypatch
):
    """Regression test for the Phase 9 review findings: "Obj." must be the full MMA
    objective (f0val), not whole-structure-compliance-only (IterationRecord.obj); "Vol."
    must be this iteration's post-update xPhys, not IterationRecord.vol (pre-update).
    """
    monkeypatch.chdir(tmp_path)
    args = cli.parse_args(_argv(tmp_path, "obj_vol"))
    _, _, records, states = _reference_run(cli.resolve_config(args))

    cli.main(args)
    out = capsys.readouterr().out

    it_lines = [line for line in out.splitlines() if line.startswith("It.:")]
    assert len(it_lines) == _FIXTURE_CONFIG.nloop
    for line, record, state in zip(it_lines, records, states):
        obj = float(re.search(r"Obj\.:\s*([\d.-]+)", line).group(1))
        vol = float(re.search(r"Vol\.:\s*([\d.-]+)", line).group(1))
        np.testing.assert_allclose(obj, record.f0val, rtol=1e-6)
        # atol matches the printed field's own rounding (%6.3f -> max rounding error
        # 5e-4); rtol=1e-6 alone would be far tighter than the print format supports.
        np.testing.assert_allclose(vol, float(state.xPhys.mean()), atol=6e-4, rtol=0)
        # Guard against vacuous passes: f0val/.obj and pre-/post-update volume must
        # actually differ here, or this test wouldn't catch printing the wrong one.
        assert abs(record.f0val - record.obj) > 1e-3
        assert abs(float(state.xPhys.mean()) - record.vol) > 1e-3
