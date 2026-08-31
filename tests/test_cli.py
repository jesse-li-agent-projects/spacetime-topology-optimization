"""Smoke test for sttopt.cli. Calls parse_args/main directly (not via subprocess) with
tiny overrides -- per the repo's sandbox rules, nothing near production scale
(180x60x800) is run here, and per the plan's Phase 9 guidance this phase gets the
lightest testing budget of the whole port.
"""

import json
import re

import numpy as np

import sttopt.cli as cli
import sttopt.conductivity as conductivity
import sttopt.optimize as optimize
import sttopt.torch_util as torch_util
import sttopt.viz as viz
from sttopt.run_config import RunConfig

# nStage/rmin/lrmin/rmin_cond are config-file-only (not CLI flags), so this fixture's
# overrides for them go through --config rather than argv.
_FIXTURE_CONFIG = RunConfig(nStage=2, rmin=2, lrmin=2, rmin_cond=3)


def _argv(tmp_path, tag):
    config_path = tmp_path / "fixture_config.json"
    config_path.write_text(json.dumps(_FIXTURE_CONFIG.to_dict()))
    return [
        "--config",
        str(config_path),
        "--nelx",
        "7",
        "--nely",
        "5",
        "--nloop",
        "2",
        "--tag",
        tag,
    ]


def test_cli_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = cli.parse_args(_argv(tmp_path, "smoke"))

    cli.main(args)

    assert (tmp_path / "output" / "smoke" / "final_structure.png").exists()
    assert (tmp_path / "output" / "smoke" / "final_design.npz").exists()


def _reference_run(config):
    """Independently drives the same optimize loop main() does, for comparison against
    what main() actually prints/plots -- catches a regression to the wrong MATLAB
    quantity (see the Phase 9 review: cli.py originally printed IterationRecord.obj/.vol,
    which are NOT what MATLAB's disp actually prints; see cli.py's module docstring).
    """
    problem = optimize.build_problem(
        config.nelx,
        config.nely,
        config.nStage,
        config.volfrac,
        config.Theta,
        config.Tcr,
        config.tfield,
        config.rmin,
        config.lrmin,
        config.rmin_cond,
        beta_d_max=config.beta_d_max,
    )
    state = optimize.init_state(problem, beta_d=1.0)
    prev_state, records, states = state, [], []
    for _ in range(config.nloop):
        prev_state = state
        state, record = optimize.step(problem, state)
        records.append(record)
        states.append(state)
    return problem, prev_state, state, records, states


def test_cli_prints_full_objective_and_post_update_volume(
    capsys, tmp_path, monkeypatch
):
    """Regression test for the Phase 9 review findings: "Obj." must be the full MMA
    objective (f0val), not whole-structure-compliance-only (IterationRecord.obj); "Vol."
    must be this iteration's post-update xPhys, not IterationRecord.vol (pre-update).
    """
    monkeypatch.chdir(tmp_path)
    args = cli.parse_args(_argv(tmp_path, "obj_vol"))
    _, _, _, records, states = _reference_run(cli.resolve_config(args))

    cli.main(args)
    out = capsys.readouterr().out

    it_lines = [line for line in out.splitlines() if line.startswith("It.:")]
    assert len(it_lines) == args.nloop
    for line, record, state in zip(it_lines, records, states):
        obj = float(re.search(r"Obj\.:\s*([\d.-]+)", line).group(1))
        vol = float(re.search(r"Vol\.:\s*([\d.-]+)", line).group(1))
        np.testing.assert_allclose(obj, record.f0val, rtol=1e-6)
        # atol matches the printed field's own rounding (%6.3f -> max rounding error
        # 5e-4); rtol=1e-6 alone would be far tighter than the print format supports.
        np.testing.assert_allclose(vol, state.xPhys.mean(), atol=6e-4, rtol=0)
        # Guard against vacuous passes: f0val/.obj and pre-/post-update volume must
        # actually differ here, or this test wouldn't catch printing the wrong one.
        assert abs(record.f0val - record.obj) > 1e-3
        assert abs(state.xPhys.mean() - record.vol) > 1e-3


def test_cli_closing_plot_uses_pre_final_update_state(tmp_path, monkeypatch):
    """Regression test: the closing plot's XPhys/K_est must come from the state
    *entering* the final iteration (prev_state), matching MATLAB's XPhys=xPhys(:)
    captured at the top of the last loop iteration -- not the post-update state (which
    is one MMA step ahead of what MATLAB actually plots).
    """
    monkeypatch.chdir(tmp_path)
    args = cli.parse_args(_argv(tmp_path, "plot_state"))
    problem, prev_state, final_state, _, _ = _reference_run(cli.resolve_config(args))

    def _expected_T1(state):
        """The color field cli.py's closing plot computes from a given state -- a
        continuous quantity (unlike the eps=0.5-binarized XPhys, which can coincide
        between prev_state/final_state by rounding even when the underlying densities
        differ, as they do on this fixture -- checked and confirmed not a reliable
        discriminator here).
        """
        K_est = conductivity.estimated_conductivity(
            state.xPhys,
            state.tPhys,
            problem.e1,
            problem.e2,
            problem.w,
            problem.q,
            problem.rouf,
        ).reshape(problem.nely, problem.nelx)
        XPhys = (state.xPhys > 0.5).to(problem.dtype)
        return torch_util.to_numpy((1 - K_est) * XPhys)

    prev_T1 = _expected_T1(prev_state)
    final_T1 = _expected_T1(final_state)
    assert not np.allclose(
        prev_T1, final_T1
    ), "prev_state/final_state's T1 must differ here, or this test can't distinguish them"

    # cli.py does `import sttopt.viz as viz` *inside* main(), so `cli.viz` isn't a module
    # attribute -- patch the real sttopt.viz module (the same object that local import
    # binds to) instead.
    captured = {}
    original = viz.combination_plot

    def spy(xPhys, tPhys, eps, **kwargs):
        captured["T1"] = tPhys
        return original(xPhys, tPhys, eps, **kwargs)

    monkeypatch.setattr(viz, "combination_plot", spy)
    cli.main(args)

    np.testing.assert_allclose(captured["T1"], prev_T1, rtol=1e-8)
    assert not np.allclose(captured["T1"], final_T1)
