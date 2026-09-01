"""Tests for sttopt.timefield: golden-regression fixture checks (see conftest.py,
conventions.md)."""

import numpy as np
import pytest
import torch

import sttopt.timefield as timefield
import sttopt.torch_util as torch_util
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

    # A bare int dispatches identically -- optimize.build_problem resolves
    # RunConfig.print_base's string to a TimeField member before this point.
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


def test_uniform_ramp_has_zero_gradient_spread():
    """A linear ramp has the same gradient magnitude everywhere -- the ideal the penalty
    drives toward -- so its spread is zero regardless of the ramp's direction or slope."""
    ny, nx = 9, 11
    ys, xs = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    for ramp in (0.3 * xs, 0.7 * ys, 0.2 * xs - 0.5 * ys):
        assert timefield.gradient_magnitude_std(ramp) == pytest.approx(0.0, abs=1e-6)


def test_gradient_magnitude_std_matches_numpy_interior_reference():
    """Value check against an independent NumPy computation over the interior, pinning
    both the central-difference stencil and which elements are counted."""
    rng = np.random.default_rng(0)
    field = rng.random((8, 6))
    gy, gx = np.gradient(field)  # 2nd-order central in the interior
    magnitude = np.sqrt(gx[1:-1, 1:-1] ** 2 + gy[1:-1, 1:-1] ** 2)

    value = timefield.gradient_magnitude_std(torch.from_numpy(field))
    assert float(value) == pytest.approx(magnitude.std(ddof=1), rel=1e-9)


def test_gradient_magnitude_std_is_differentiable_at_a_flat_field():
    """A constant field makes every gradient vanish, where an unregularized sqrt would
    return a NaN derivative -- the field the optimizer would drive toward."""
    flat = torch.full((7, 7), 0.4, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(timefield.gradient_magnitude_std(flat), flat)
    assert torch.all(torch.isfinite(grad))


@pytest.mark.parametrize("nelx,nely", [(2, 9), (9, 2), (3, 3)])
def test_gradient_magnitude_std_degenerate_interior(nelx, nely):
    """Fewer than two interior elements leaves no spread to measure: zero, not NaN."""
    field = torch.rand((nely, nelx), dtype=torch.float64)
    assert float(timefield.gradient_magnitude_std(field)) == 0.0


def test_gradient_magnitude_std_sensitivity_matches_finite_differences():
    """Autograd's gradient of the penalty against central differences of its own value
    -- the penalty enters the objective through autograd, so this is what the optimizer
    actually descends."""
    ny, nx = 6, 7
    rng = np.random.default_rng(2)
    field = rng.random((ny, nx))
    leaf = torch.from_numpy(field).requires_grad_(True)
    (grad,) = torch.autograd.grad(timefield.gradient_magnitude_std(leaf), leaf)

    h = 1e-6
    fd = np.zeros_like(field)
    for j in range(ny):
        for i in range(nx):
            plus, minus = field.copy(), field.copy()
            plus[j, i] += h
            minus[j, i] -= h
            fd[j, i] = float(
                timefield.gradient_magnitude_std(torch.from_numpy(plus))
                - timefield.gradient_magnitude_std(torch.from_numpy(minus))
            ) / (2 * h)

    np.testing.assert_allclose(torch_util.to_numpy(grad), fd, rtol=1e-5, atol=1e-8)
    # Non-vacuity, and a check that the border is not silently frozen out of the
    # gradient: a border element still enters an interior element's stencil.
    assert np.abs(fd).max() > 1e-3
    assert np.abs(fd[0]).max() > 1e-3
