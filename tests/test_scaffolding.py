"""Smoke test that the package and test infrastructure are wired up correctly."""

import sttopt
from conftest import assert_close, e2e_rtol


def test_package_importable():
    assert sttopt.__doc__


def test_assert_close_tiers():
    assert_close(1.0, 1.0 + 1e-11, tier="algebraic")
    assert_close(1.0, 1.0 + 1e-7, tier="solved")


def test_e2e_rtol_grows_with_iteration():
    assert e2e_rtol(1) < e2e_rtol(5) < e2e_rtol(100)
