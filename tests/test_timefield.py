"""Tests for sttopt.timefield: golden-regression fixture checks (see conftest.py,
conventions.md)."""

import numpy as np
import pytest

import sttopt.timefield as timefield
from conftest import assert_close, load_fixture_npz

_VARIANTS = [
    (timefield.TimeField.CORNER, "tfield1"),
    (timefield.TimeField.EDGE, "tfield2"),
    (timefield.TimeField.OPPOSITE_CORNER, "tfield3"),
]


@pytest.mark.parametrize("variant,key", _VARIANTS)
def test_timefield_variant_matches_fixture(variant, key):
    """`init_timefield` is the only entry point, so this covers both the field itself and
    the dispatch mapping each `TimeField` to the right one."""
    fx = load_fixture_npz("timefield")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])

    got = timefield.init_timefield(nelx, nely, variant)
    assert got.shape == fx[key].shape == (nely, nelx)
    assert_close(got, fx[key], tier="algebraic")

    # A bare int dispatches identically -- cli.py passes `--tfield` straight through.
    assert np.array_equal(got, timefield.init_timefield(nelx, nely, int(variant)))


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        timefield.init_timefield(7, 5, 4)


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (4, 4), (2, 3)])
def test_timefield_variants_span_0_to_1(nelx, nely):
    """Each variant is a normalized field: from first principles (not the fixture,
    which only exercises nelx=7, nely=5) it must span exactly [0, 1], hitting 0 at the
    corner/edge it's defined from and 1 at the opposite extreme -- for square and
    non-square grids alike, since `_corner_distance_grid`'s docstring warns non-square
    grids change the field's shape, not just its scale."""
    tfield_corner = timefield.init_timefield(nelx, nely, timefield.TimeField.CORNER)
    tfield_edge = timefield.init_timefield(nelx, nely, timefield.TimeField.EDGE)
    tfield_opposite = timefield.init_timefield(
        nelx, nely, timefield.TimeField.OPPOSITE_CORNER
    )

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


@pytest.mark.parametrize("nelx,nely", [(1, 5), (5, 1)])
@pytest.mark.parametrize("variant", list(timefield.TimeField))
def test_lone_one_mesh_is_finite(nelx, nely, variant):
    """A lone-1 mesh is well-defined -- finite everywhere, though not necessarily
    spanning [0, 1] (see the module docstring). Only `nelx == nely == 1` degenerates,
    and rejecting that one is `optimize.build_problem`'s job, not this module's."""
    assert np.all(np.isfinite(timefield.init_timefield(nelx, nely, variant)))
