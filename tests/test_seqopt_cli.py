"""Smoke test for sttopt.seqopt_cli, mirroring tests/test_stto_cli.py."""

import json

import numpy as np

import sttopt.seqopt_cli as seqopt_cli
from conftest import default_seq_run_config

_FIXTURE_CONFIG = default_seq_run_config(
    lrmin=1.5, rmin_cond=2.5, nStage=2, nloop=2, tmove=0.05
)


def _geometry() -> np.ndarray:
    xPhys = np.zeros((5, 7))
    xPhys[-2:, :] = 1.0
    xPhys[:, 0] = 1.0
    return xPhys


def _argv(tmp_path, tag):
    config_path = tmp_path / "fixture_seq_config.json"
    config_path.write_text(json.dumps(_FIXTURE_CONFIG.to_dict()))
    geometry_path = tmp_path / "geometry_in.npz"
    np.savez(geometry_path, xPhys=_geometry())
    return [
        "--geometry",
        str(geometry_path),
        "--config",
        str(config_path),
        "--tag",
        tag,
    ]


def test_cli_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = seqopt_cli.parse_args(_argv(tmp_path, "smoke"))

    seqopt_cli.main(args)

    run_dir = tmp_path / "output" / "smoke"
    assert (run_dir / "final_design.npz").exists()
    assert (run_dir / "seq_config.json").exists()
    assert (run_dir / "geometry.npz").exists()

    design = np.load(run_dir / "final_design.npz")
    assert "xPhys" in design
    assert "tPhys" in design
    assert design["xPhys"].shape == design["tPhys"].shape == (5, 7)


def test_cli_logs_one_diagnostics_record_per_iteration(tmp_path, monkeypatch):
    """Every iteration is logged by default, so a run can be checked for oscillation
    afterwards without rerunning it."""
    monkeypatch.chdir(tmp_path)
    args = seqopt_cli.parse_args(_argv(tmp_path, "logged"))

    seqopt_cli.main(args)

    log = (tmp_path / "output" / "logged" / "iterations.jsonl").read_text()
    lines = log.splitlines()
    assert len(lines) == _FIXTURE_CONFIG.nloop
    entries = [json.loads(line) for line in lines]
    assert [e["loop"] for e in entries] == [1, 2]
    for entry in entries:
        # The trust-region fields are why this is logged per iteration at all.
        assert entry["step_max"] <= _FIXTURE_CONFIG.tmove + 1e-12
        assert 0.0 <= entry["move_frac"] <= 1.0
        assert entry["asymptote_width_min"] > 0.0


def test_cli_snapshot_and_log_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = seqopt_cli.parse_args(
        _argv(tmp_path, "quiet") + ["--snapshot-every", "0", "--log-every", "0"]
    )

    seqopt_cli.main(args)

    run_dir = tmp_path / "output" / "quiet"
    assert (run_dir / "iterations.jsonl").read_text() == ""
    assert not list(run_dir.glob("design_it*.npz"))


def test_cli_tag_exists_without_force_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output" / "taken").mkdir(parents=True)
    args = seqopt_cli.parse_args(_argv(tmp_path, "taken"))
    try:
        seqopt_cli.main(args)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "already exists" in str(e)
