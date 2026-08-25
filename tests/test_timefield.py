"""Tests for sttopt.timefield against MATLAB fixtures (see conftest.py, conventions.md)."""

import numpy as np
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


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (4, 4), (2, 3)])
def test_timefield_variants_span_0_to_1(nelx, nely):
    """Each variant is a normalized field: from first principles (not the fixture,
    which only exercises nelx=7, nely=5) it must span exactly [0, 1], hitting 0 at the
    corner/edge it's defined from and 1 at the opposite extreme -- for square and
    non-square grids alike, since `_corner_distance_grid`'s docstring warns non-square
    grids change the field's shape, not just its scale."""
    tfield_corner = timefield.timefield_corner(nelx, nely)
    tfield_edge = timefield.timefield_edge(nelx, nely)
    tfield_opposite = timefield.timefield_opposite_corner(nelx, nely)

    for tfield in (tfield_corner, tfield_edge, tfield_opposite):
        assert tfield.min() == 0.0
        assert tfield.max() == 1.0

    # Pin down *where* 0/1 land, not just that they occur somewhere -- catches an
    # accidental x/y axis swap that could still leave min/max at 0/1 in the wrong place.
    assert np.unravel_index(tfield_corner.argmin(), tfield_corner.shape) == (0, 0)
    assert np.unravel_index(tfield_opposite.argmin(), tfield_opposite.shape) == (
        nely - 1,
        0,
    )
    assert np.array_equal(tfield_edge[0], np.linspace(0, 1, nelx))


@pytest.mark.parametrize("nelx,nely", [(1, 5), (5, 1), (1, 1)])
def test_degenerate_mesh_rejected(nelx, nely):
    """`nelx < 2` or `nely < 2` degenerates the corner variants to a divide-by-zero and
    the edge variant to a constant field instead of spanning [0, 1] -- rejected outright
    rather than silently producing nan/a wrong-shaped field."""
    with pytest.raises(ValueError):
        timefield.timefield_corner(nelx, nely)
    with pytest.raises(ValueError):
        timefield.timefield_edge(nelx, nely)
    with pytest.raises(ValueError):
        timefield.timefield_opposite_corner(nelx, nely)
    with pytest.raises(ValueError):
        timefield.init_timefield(nelx, nely, timefield.TimeField.CORNER)
