"""`--fixed-density`: a run that optimizes only its time field on another run's density,
including across time-field formulations."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import sttopt.stto_cli as stto_cli
import sttopt.stto_heat_cli as stto_heat_cli
import sttopt.stto_laplace_cli as stto_laplace_cli
from sttopt.run_config import HeatRunConfig, LaplaceRunConfig, RunConfig

CONFIGS = Path(__file__).parent.parent / "configs"
NELX, NELY, NITER = 30, 10, 4

# A few iterations leave the design grey, the print start unsettled (`stto`) and MMA's
# subproblem unconverged
pytestmark = [
    pytest.mark.filterwarnings("ignore:subsolv:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:binarizing raised the hotspot row:UserWarning"),
    pytest.mark.filterwarnings("ignore:print start:UserWarning"),
]


def _small(cls, path: str, **overrides):
    base = cls.from_dict(json.loads((CONFIGS / path).read_text()))
    h = base.element_size_m
    mesh = dict(nelx=NELX, width_m=NELX * h, height_m=NELY * h, nloop=NITER)
    return dataclasses.replace(base, **(mesh | overrides))


def _run(cli, config, tag, *extra):
    Path("config.json").write_text(json.dumps(config.to_dict()))
    args = ["--config", "config.json", "--tag", tag, "--device", "cpu", *extra]
    cli.main(cli.parse_args(args))
    return np.load(Path("output") / tag / "final_design.npz")


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _small(LaplaceRunConfig, "laplace_default.json")
    return _run(stto_laplace_cli, config, "source")


@pytest.mark.parametrize(
    "cli, cls, path",
    [
        (stto_laplace_cli, LaplaceRunConfig, "laplace_default.json"),
        (stto_heat_cli, HeatRunConfig, "heat_default.json"),
        (stto_cli, RunConfig, "default.json"),
    ],
)
def test_the_density_is_the_source_s_whichever_formulation_follows(
    source, cli, cls, path
):
    config = _small(cls, path)
    held = _run(cli, config, "held", "--fixed-density", "output/source")
    assert np.allclose(held["xPhys"], source["xPhys"], atol=1e-8)
    assert np.array_equal(held["x"], source["x"])


def test_a_config_that_filters_the_density_differently_is_rejected(source):
    config = _small(LaplaceRunConfig, "laplace_default.json", rmin_m=2 * 0.004)
    with pytest.raises(SystemExit, match="physical density"):
        _run(stto_laplace_cli, config, "held", "--fixed-density", "output/source")
