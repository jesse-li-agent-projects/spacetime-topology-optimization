"""Round-trip and unknown-key tests for `RunConfig`/`SeqRunConfig`'s shared
`_ConfigMixin` JSON plumbing."""

import pytest

from conftest import default_run_config, default_seq_run_config
from sttopt.run_config import RunConfig, SeqRunConfig


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
