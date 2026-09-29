import json
import warnings

import numpy as np
import pytest
import torch
from conftest import default_run_config

from sttopt import checks, stto, timefield, torch_util


def test_unsupported_passes_a_field_printed_outward_from_the_base():
    solid = np.ones((3, 4), dtype=bool)
    t = np.add.outer(np.arange(3), np.arange(4)).astype(float)
    assert not checks.unsupported(solid, t, np.array([0])).any()


def test_unsupported_flags_a_pit_and_an_island():
    t = np.add.outer(np.arange(4), np.arange(5)).astype(float)
    t[2, 2] = -1.0  # printed before every neighbor
    solid = np.ones((4, 5), dtype=bool)
    solid[:, 3] = False  # column 4 is cut off from the base
    flagged = checks.unsupported(solid, t, np.array([0]))
    assert flagged[2, 2]
    assert flagged[:, 4].any()
    assert flagged[:, :3].sum() == 1


def test_unsupported_counts_a_diagonal_neighbor():
    solid = np.eye(3, dtype=bool)
    t = np.diag([0.0, 1.0, 2.0])
    assert not checks.unsupported(solid, t, np.array([0])).any()


def _printable_design(problem: stto.Problem) -> tuple[torch.Tensor, torch.Tensor]:
    """Solid everywhere, printed outward from the base: passes both hard checks."""
    config = problem.config
    t = timefield.init_timefield(
        config.nelx, config.nely, timefield.TimeField[config.print_base.upper()]
    )
    to_tensor = lambda a: torch_util.to_tensor(a, problem.device, problem.dtype)
    return to_tensor(np.ones_like(t)), to_tensor(t)


@pytest.fixture(scope="module")
def problem():
    # No time filter, so the raw time field is the physical one up to scale.
    return stto.build_problem(
        default_run_config(nelx=30, nely=10, time_filter_rmin_m=0.0), device="cpu"
    )


def test_check_design_passes_a_printable_design(problem):
    x, t = _printable_design(problem)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = checks.check_design(problem, x, t, problem.config.nloop)
    assert report["passed"]
    assert report["volume_fraction"] == 1.0
    assert "hotspot" in report["constraints"]
    json.dumps(report, allow_nan=False)


def test_check_design_warns_on_a_pit(problem):
    x, t = _printable_design(problem)
    t[5, 15] = 0.0
    with pytest.warns(UserWarning, match="print support"):
        report = checks.check_design(problem, x, t, problem.config.nloop)
    assert not report["passed"]
    assert report["start"]["passed"]
    assert report["support"]["unsupported_at"] == [[5, 15]]


def test_check_design_warns_on_a_late_start(problem):
    x, t = _printable_design(problem)
    t.view(-1)[problem.Nei] = 0.01
    with pytest.warns(UserWarning, match="print start"):
        report = checks.check_design(problem, x, t, problem.config.nloop)
    assert not report["passed"]
    assert report["support"]["passed"]


def test_check_design_reads_the_binarized_design(problem):
    """Uniform grey reads as all solid above `eta` and all void below it."""
    _, t = _printable_design(problem)
    eta = problem.config.eta
    above = checks.check_design(problem, torch.full_like(t, eta + 0.1), t, 0)
    assert above["volume_fraction"] == 1.0 and above["passed"]
    with pytest.warns(UserWarning, match="print start"):
        below = checks.check_design(problem, torch.full_like(t, eta - 0.1), t, 0)
    assert below["volume_fraction"] == 0.0
    assert below["start"]["solid_base_elements"] == 0
