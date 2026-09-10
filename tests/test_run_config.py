"""Round-trip and unknown-key tests for `RunConfig`/`SeqRunConfig`'s shared
`_ConfigMixin` JSON plumbing."""

import json

import pytest

from conftest import default_run_config, default_seq_run_config
from sttopt.run_config import CosineSchedule, RunConfig, SeqRunConfig, weight_at


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
    """`decay_iterations` is the last iteration still decaying, so a 300-iteration
    decay of an 800-iteration run reads as written."""
    schedule = CosineSchedule(initial=1.0, decay_iterations=300, final=0.02)
    assert schedule.at(1) == 1.0
    assert schedule.at(301) == pytest.approx(0.02)
    assert schedule.at(151) == pytest.approx(0.51)  # half a cosine's midpoint


def test_cosine_schedule_holds_final_past_the_decay():
    """The decay is allowed to finish well before the run does, so every later
    iteration has to stay at `final` rather than overshoot below it."""
    schedule = CosineSchedule(initial=1.0, decay_iterations=300, final=0.02)
    assert schedule.at(800) == pytest.approx(0.02)
    assert schedule.at(10_000) == pytest.approx(0.02)


def test_cosine_schedule_decreases_monotonically():
    values = [
        CosineSchedule(initial=1.0, decay_iterations=300, final=0.02).at(loop)
        for loop in range(1, 302)
    ]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


def test_cosine_schedule_rejects_a_zero_length_decay():
    """The failure is in `decay_iterations` itself, not in any run length it is
    compared against."""
    with pytest.raises(ValueError, match="decay_iterations must be at least 1"):
        CosineSchedule(initial=1.0, decay_iterations=0, final=0.02)


def test_seq_config_round_trips_a_cosine_weight():
    """JSON has no way to name a `CosineSchedule`, so a mapping in the field's place is
    one -- and it has to survive the round trip a run directory's `seq_config.json`
    depends on."""
    scheduled = default_seq_run_config(
        roughness_weight={"initial": 1.0, "decay_iterations": 300, "final": 0.02}
    )
    assert scheduled.roughness_weight == CosineSchedule(
        initial=1.0, decay_iterations=300, final=0.02
    )

    revived = SeqRunConfig.from_dict(json.loads(json.dumps(scheduled.to_dict())))
    assert revived == scheduled
    assert weight_at(revived.roughness_weight, 1) == 1.0
    assert weight_at(revived.roughness_weight, 800) == pytest.approx(0.02)


def test_weight_at_passes_a_bare_number_through():
    """A plain number means a constant, so an unscheduled config needs no special
    casing at the call site."""
    assert weight_at(0.2, 1) == 0.2
    assert weight_at(0.2, 10_000) == 0.2


def test_seq_config_round_trips_a_constant_weight_as_a_number():
    """The simple form must stay simple: a config written with a number reads back as
    that number, so an artefact does not gain a schedule it never had."""
    config = default_seq_run_config(roughness_weight=0.2)
    assert config.to_dict()["roughness_weight"] == 0.2
    revived = SeqRunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert revived.roughness_weight == 0.2
