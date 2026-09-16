"""Round-trip and unknown-key tests for `RunConfig`/`SeqRunConfig`'s shared
`_ConfigMixin` JSON plumbing."""

import json

import pytest

from conftest import default_run_config, default_seq_run_config
from sttopt.run_config import (
    CosineSchedule,
    PiecewiseSchedule,
    RunConfig,
    SeqRunConfig,
    weight_at,
)


@pytest.mark.parametrize(
    "config",
    [default_run_config(), default_seq_run_config()],
    ids=["RunConfig", "SeqRunConfig"],
)
def test_to_dict_from_dict_round_trips(config):
    assert type(config).from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "cls,config",
    [(RunConfig, default_run_config()), (SeqRunConfig, default_seq_run_config())],
)
def test_from_dict_warns_and_drops_unknown_keys(cls, config):
    d = config.to_dict()
    d["not_a_real_field"] = 123
    with pytest.warns(UserWarning, match="not_a_real_field"):
        got = cls.from_dict(d)
    assert got == config


def test_cosine_schedule_decays_between_its_endpoints():
    """`decay_iterations` is the number of iterations still decaying, so a 300-iteration
    decay of an 800-iteration run reads as written."""
    schedule = CosineSchedule(initial=1.0, decay_iterations=300, final=0.02)
    assert schedule.at(0) == 1.0
    assert schedule.at(300) == pytest.approx(0.02)
    assert schedule.at(150) == pytest.approx(0.51)  # half a cosine's midpoint


def test_cosine_schedule_holds_final_past_the_decay():
    """The decay is allowed to finish well before the run does, so every later
    iteration has to stay at `final` rather than overshoot below it."""
    schedule = CosineSchedule(initial=1.0, decay_iterations=300, final=0.02)
    assert schedule.at(800) == pytest.approx(0.02)
    assert schedule.at(10_000) == pytest.approx(0.02)


def test_cosine_schedule_decreases_monotonically():
    values = [
        CosineSchedule(initial=1.0, decay_iterations=300, final=0.02).at(loop)
        for loop in range(301)
    ]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


def test_cosine_schedule_rejects_a_zero_length_decay():
    """The failure is in `decay_iterations` itself, not in any run length it is
    compared against."""
    with pytest.raises(ValueError, match="decay_iterations must be at least 1"):
        CosineSchedule(initial=1.0, decay_iterations=0, final=0.02)


@pytest.mark.parametrize(
    "cls,make_config",
    [(RunConfig, default_run_config), (SeqRunConfig, default_seq_run_config)],
)
def test_config_round_trips_a_cosine_weight(cls, make_config):
    """JSON has no way to name a `CosineSchedule`, so a mapping in the field's place is
    one -- and it has to survive the round trip a run directory's config JSON depends
    on."""
    scheduled = make_config(
        roughness_weight={"initial": 1.0, "decay_iterations": 300, "final": 0.02}
    )
    assert scheduled.roughness_weight == CosineSchedule(
        initial=1.0, decay_iterations=300, final=0.02
    )

    revived = cls.from_dict(json.loads(json.dumps(scheduled.to_dict())))
    assert revived == scheduled
    assert weight_at(revived.roughness_weight, 0) == 1.0
    assert weight_at(revived.roughness_weight, 800) == pytest.approx(0.02)


def test_weight_at_passes_a_bare_number_through():
    """A plain number means a constant, so an unscheduled config needs no special
    casing at the call site."""
    assert weight_at(0.2, 0) == 0.2
    assert weight_at(0.2, 10_000) == 0.2


def test_seq_config_round_trips_a_constant_weight_as_a_number():
    """The simple form must stay simple: a config written with a number reads back as
    that number, so an artefact does not gain a schedule it never had."""
    config = default_seq_run_config(roughness_weight=0.2)
    assert config.to_dict()["roughness_weight"] == 0.2
    revived = SeqRunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert revived.roughness_weight == 0.2


def test_piecewise_schedule_interpolates_and_holds_outside_its_points():
    schedule = PiecewiseSchedule(points=[[100, 2.0], [300, 0.8]])
    assert schedule.at(0) == 2.0
    assert schedule.at(200) == pytest.approx(1.4)
    assert schedule.at(10_000) == 0.8


def test_piecewise_schedule_log_interpolates_geometrically():
    schedule = PiecewiseSchedule(points=[[0, 1.0], [100, 100.0]], mode="log")
    assert schedule.at(50) == pytest.approx(10.0)


def test_piecewise_step_schedule_holds_each_value_until_the_next_point():
    """`step` does not interpolate: the value jumps at a point's own iteration, which is
    what the projection sharpness ramps have always done."""
    schedule = PiecewiseSchedule(points=[[0, 1.0], [50, 2.0], [100, 4.0]], mode="step")
    assert [schedule.at(loop) for loop in (0, 49, 50, 99, 100, 10_000)] == [
        1.0,
        1.0,
        2.0,
        2.0,
        4.0,
        4.0,
    ]


def test_piecewise_schedule_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="not a valid Interpolation"):
        PiecewiseSchedule(points=[[0, 1.0]], mode="quadratic")


def test_piecewise_schedule_rejects_unsorted_points():
    with pytest.raises(ValueError, match="sorted by iteration"):
        PiecewiseSchedule(points=[[300, 0.8], [100, 2.0]])


def test_config_round_trips_a_piecewise_schedule():
    """A mapping with `points` is a `PiecewiseSchedule`, in any schedulable field."""
    config = default_run_config(
        Tcr={"points": [[0, 5.0], [200, 0.8]]},
        hotspot_beta={"points": [[0, 4.0], [400, 32.0]], "mode": "log"},
    )
    assert isinstance(config.Tcr, PiecewiseSchedule)
    revived = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert revived == config
    assert weight_at(revived.hotspot_beta, 400) == pytest.approx(32.0)
