"""Tests for sttopt.timefield against MATLAB fixtures (see conftest.py, conventions.md)."""

import pytest

import sttopt.timefield as timefield
from conftest import assert_close, load_fixture


def test_timefield_variants():
    fx = load_fixture("timefield")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])

    tfield1 = timefield.timefield_corner(nelx, nely)
    tfield2 = timefield.timefield_edge(nelx, nely)
    tfield3 = timefield.timefield_opposite_corner(nelx, nely)

    assert tfield1.shape == fx["tfield1"].shape == (nely, nelx)
    assert tfield2.shape == fx["tfield2"].shape == (nely, nelx)
    assert tfield3.shape == fx["tfield3"].shape == (nely, nelx)
    assert_close(tfield1, fx["tfield1"], tier="algebraic")
    assert_close(tfield2, fx["tfield2"], tier="algebraic")
    assert_close(tfield3, fx["tfield3"], tier="algebraic")


def test_init_timefield_dispatch():
    fx = load_fixture("timefield")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])

    assert_close(
        timefield.init_timefield(nelx, nely, 1), fx["tfield1"], tier="algebraic"
    )
    assert_close(
        timefield.init_timefield(nelx, nely, 2), fx["tfield2"], tier="algebraic"
    )
    assert_close(
        timefield.init_timefield(nelx, nely, 3), fx["tfield3"], tier="algebraic"
    )

    with pytest.raises(ValueError):
        timefield.init_timefield(nelx, nely, 4)
