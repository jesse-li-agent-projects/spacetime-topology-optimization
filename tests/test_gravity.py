"""Tests for sttopt.gravity against MATLAB fixtures (see conftest.py, conventions.md)."""

import sttopt.gravity as gravity
from conftest import assert_close, load_fixture


def test_gravity_load_matrix():
    fx = load_fixture("gravity")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])

    C = gravity.gravity_load_matrix(nelx, nely)
    expected = fx["C"].toarray()  # sparse fixture, small problem size -- densify for comparison only

    assert C.shape == expected.shape == ((nelx + 1) * (nely + 1), nelx * nely)
    assert_close(C.toarray(), expected, tier="algebraic")
