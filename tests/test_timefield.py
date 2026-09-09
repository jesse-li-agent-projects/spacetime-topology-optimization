"""Tests for sttopt.timefield: golden-regression fixture checks (see conftest.py,
conventions.md)."""

import itertools

import numpy as np
import pytest
import torch

import sttopt.geometry as geometry
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

    # A bare int dispatches identically -- stto.build_problem resolves
    # RunConfig.print_base's string to a TimeField member before this point.
    assert np.array_equal(got, timefield.init_timefield(nelx, nely, int(variant)))


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        timefield.init_timefield(7, 5, 5)


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
    and rejecting that one is `stto.build_problem`'s job, not this module's."""
    assert np.all(np.isfinite(timefield.init_timefield(nelx, nely, variant)))


def test_uniform_ramp_has_zero_gradient_spread():
    """A linear ramp has the same gradient magnitude everywhere -- the ideal the penalty
    drives toward -- so its spread is zero regardless of the ramp's direction or slope.
    """
    ny, nx = 9, 11
    ys, xs = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    for ramp in (0.3 * xs, 0.7 * ys, 0.2 * xs - 0.5 * ys):
        assert timefield.gradient_magnitude_std(ramp) == pytest.approx(0.0, abs=1e-6)


def test_gradient_magnitude_std_matches_numpy_gauss_reference():
    """Value check against an independent NumPy computation, pinning both the Q4
    Gauss-point stencil and which samples are counted."""
    rng = np.random.default_rng(0)
    field = rng.random((8, 6))
    a = 3.0**-0.5
    lo_lo, lo_hi, hi_lo, hi_hi = (
        field[:-1, :-1],
        field[:-1, 1:],
        field[1:, :-1],
        field[1:, 1:],
    )
    samples = []
    for eta, xi in itertools.product((-a, a), (-a, a)):
        gx = (1 - eta) / 2 * (lo_hi - lo_lo) + (1 + eta) / 2 * (hi_hi - hi_lo)
        gy = (1 - xi) / 2 * (hi_lo - lo_lo) + (1 + xi) / 2 * (hi_hi - lo_hi)
        samples.append(np.sqrt(gx**2 + gy**2))
    magnitude = np.stack(samples)

    value = timefield.gradient_magnitude_std(torch.from_numpy(field))
    assert float(value) == pytest.approx(magnitude.std(ddof=1), rel=1e-9)


def test_gradient_stencil_only_null_space_is_the_constant_field():
    """The stencil must see every high-frequency mode: a checkerboard it is blind to is
    an arbitrarily jagged field that scores as well as a smooth one (PR #91). Full 2x2
    quadrature is what keeps the `(-1)^(i+j)` hourglass mode visible in particular,
    which a single evaluation at the cell centre would miss.
    """
    ny, nx = 8, 9
    i, j = np.indices((ny, nx))
    checkerboards = {
        "(-1)^j": (-1.0) ** j,
        "(-1)^i": (-1.0) ** i,
        "(-1)^(i+j)": (-1.0) ** (i + j),
    }
    for name, mode in checkerboards.items():
        dt_dx, dt_dy = timefield._q4_gauss_gradient(torch.from_numpy(mode))
        largest = max(float(dt_dx.abs().max()), float(dt_dy.abs().max()))
        assert largest > 1.0, f"{name} is invisible to the stencil"

    constant = torch.ones((ny, nx), dtype=torch.float64)
    dt_dx, dt_dy = timefield._q4_gauss_gradient(constant)
    assert float(dt_dx.abs().max()) == 0.0
    assert float(dt_dy.abs().max()) == 0.0


def test_gradient_stencil_is_exact_on_a_linear_field():
    """A Q4 gradient reproduces a linear field's gradient exactly at every Gauss point,
    so a constant-thickness sweep scores zero spread however it is oriented."""
    ny, nx = 6, 7
    i, j = np.indices((ny, nx))
    field = torch.from_numpy(3.0 * j - 2.0 * i)
    dt_dx, dt_dy = timefield._q4_gauss_gradient(field)
    assert float(dt_dx.min()) == pytest.approx(3.0, abs=1e-12)
    assert float(dt_dx.max()) == pytest.approx(3.0, abs=1e-12)
    assert float(dt_dy.min()) == pytest.approx(-2.0, abs=1e-12)
    assert float(dt_dy.max()) == pytest.approx(-2.0, abs=1e-12)


def test_interpolate_to_gauss_is_a_partition_of_unity():
    """The weights are the same shape functions as the gradient, so a uniform field
    interpolates to itself -- otherwise a weight would not match the sample it
    weights."""
    ones = torch.ones((5, 6), dtype=torch.float64)
    assert float((timefield.interpolate_to_gauss(ones) - 1.0).abs().max()) == 0.0


def test_gradient_magnitude_elements_covers_every_element():
    """The scattered plotting field has no blank border: every element sits at some
    cell's corner, so every element gets a value."""
    ny, nx = 5, 6
    i, j = np.indices((ny, nx))
    per_element = timefield.gradient_magnitude_elements(
        torch.from_numpy(3.0 * j + 0.0 * i)
    )
    assert per_element.shape == (ny, nx)
    assert torch.all(torch.isfinite(per_element))
    # A linear field has one gradient magnitude everywhere, border included.
    assert float((per_element - 3.0).abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_gradient_magnitude_std_is_differentiable_at_a_flat_field():
    """A constant field makes every gradient vanish, where an unregularized sqrt would
    return a NaN derivative -- the field the optimizer would drive toward."""
    flat = torch.full((7, 7), 0.4, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(timefield.gradient_magnitude_std(flat), flat)
    assert torch.all(torch.isfinite(grad))


@pytest.mark.parametrize("nelx,nely", [(1, 9), (9, 1), (1, 1)])
def test_gradient_magnitude_std_degenerate_mesh(nelx, nely):
    """A mesh one element wide in either direction has no cell to put Gauss points in,
    leaving no spread to measure: zero, not NaN. Two elements in each direction is
    already enough for a cell, so nothing larger is degenerate."""
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


# --- BOTTOM_EDGE -----------------------------------------------------------------


def test_bottom_edge_range_and_orientation():
    nelx, nely = 6, 9
    field = timefield.init_timefield(nelx, nely, timefield.TimeField.BOTTOM_EDGE)
    assert field.shape == (nely, nelx)
    np.testing.assert_allclose(field[-1], 0.0)  # bottom row: t=0
    np.testing.assert_allclose(field[0], 1.0)  # top row: t=1
    # Monotonically decreasing bottom-to-top, constant across each row.
    assert np.all(np.diff(field[:, 0]) < 0)
    for row in field:
        np.testing.assert_allclose(row, row[0])


# --- base_elements -----------------------------------------------------------------


@pytest.mark.parametrize(
    "variant,expected",
    [
        (timefield.TimeField.CORNER, np.array([0])),
        (timefield.TimeField.EDGE, np.arange(5) * 7),
        (timefield.TimeField.OPPOSITE_CORNER, np.arange(5) * 7),
        (timefield.TimeField.BOTTOM_EDGE, 4 * 7 + np.arange(7)),
    ],
)
def test_base_elements_all_variants(variant, expected):
    np.testing.assert_array_equal(timefield.base_elements(7, 5, variant), expected)


def test_base_elements_unknown_variant_rejected():
    with pytest.raises(ValueError):
        timefield.base_elements(7, 5, 5)


# --- init_geodesic_timefield -------------------------------------------------------


def test_geodesic_timefield_zero_at_base_spans_0_to_1():
    xPhys = np.ones((6, 8))
    base = timefield.base_elements(8, 6, timefield.TimeField.BOTTOM_EDGE)
    field = timefield.init_geodesic_timefield(xPhys, base)
    assert field.shape == (6, 8)
    np.testing.assert_allclose(field.flatten()[base], 0.0)
    assert field.max() == pytest.approx(1.0)
    assert np.all(np.isfinite(field))


def test_geodesic_timefield_does_not_tunnel_across_a_void_gap():
    """The point of measuring through material: two solid arms separated by a void
    slot must not inherit each other's times across it. The far arm's tip is the last
    thing printed even though it is one element away from a much earlier arm."""
    nely, nelx = 5, 11
    xPhys = np.zeros((nely, nelx))
    xPhys[-1, :] = 1.0  # solid base row (build plate)
    xPhys[:, 0] = 1.0  # left arm, straight up from the plate
    xPhys[0, :] = 1.0  # top arm, reachable only the long way round via the left arm
    base = timefield.base_elements(nelx, nely, timefield.TimeField.BOTTOM_EDGE)

    field = timefield.init_geodesic_timefield(xPhys, base)

    # The top arm's far end is the farthest point along material, so it prints last --
    # not the ~2 elements' worth of time its vertical neighbours in the base row have.
    assert field[0, -1] == pytest.approx(1.0)
    assert field[0, -1] > field[-1, -1] + 0.5


def test_geodesic_timefield_void_grows_outward_from_the_material():
    """Every void element prints after the solid it grows from -- the nearest one in
    the traversal, not every solid it happens to touch: a void pocket beside both an
    early and a late arm follows the early one.
    """
    nely, nelx = 8, 6
    xPhys = np.zeros((nely, nelx))
    xPhys[-1, :] = 1.0  # build-plate row
    xPhys[:, 0] = 1.0  # a tall column, so the part has extent to normalize by
    base = timefield.base_elements(nelx, nely, timefield.TimeField.BOTTOM_EDGE)

    field = timefield.init_geodesic_timefield(xPhys, base).flatten()
    solid = geometry.solid_mask(xPhys)
    src, dst, _ = geometry.neighbor_pairs(nelx, nely)

    touching_solid = solid[src] & ~solid[dst]
    for void in np.unique(dst[touching_solid]):
        neighbors = src[touching_solid & (dst == void)]
        assert field[void] > field[neighbors].min()

    assert field.max() <= 1.0


def test_geodesic_timefield_solid_times_ignore_surrounding_void():
    """Normalization is by the largest *solid* distance, so padding the bounding box
    with empty space leaves the part's own times unchanged."""

    def bottom_anchored(nely: int) -> np.ndarray:
        xPhys = np.zeros((nely, 6))
        xPhys[-1, :] = 1.0
        xPhys[-3:, 0] = 1.0
        return xPhys

    field = timefield.init_geodesic_timefield(
        bottom_anchored(4),
        timefield.base_elements(6, 4, timefield.TimeField.BOTTOM_EDGE),
    )
    padded = timefield.init_geodesic_timefield(
        bottom_anchored(10),
        timefield.base_elements(6, 10, timefield.TimeField.BOTTOM_EDGE),
    )

    solid = bottom_anchored(4) > 0.5
    np.testing.assert_allclose(field[-3:][solid[-3:]], padded[-3:][solid[-3:]])
    assert padded.max() <= 1.0


def test_geodesic_timefield_rejects_a_part_with_no_geodesic_extent():
    """A part that is nothing but its own print-start elements deposits entirely at
    t=0: there is no sequence to optimize, and no scale to normalize by."""
    xPhys = np.zeros((4, 5))
    xPhys[-1, :] = 1.0  # solid exactly on the build plate, nowhere else
    base = timefield.base_elements(5, 4, timefield.TimeField.BOTTOM_EDGE)

    with pytest.raises(ValueError, match="no geodesic extent"):
        timefield.init_geodesic_timefield(xPhys, base)


def test_geodesic_timefield_rejects_solid_with_no_path_to_the_plate():
    nely, nelx = 5, 11
    xPhys = np.zeros((nely, nelx))
    xPhys[-1, :] = 1.0
    xPhys[0, nelx // 2] = 1.0  # an island with no solid path to the plate
    base = timefield.base_elements(nelx, nely, timefield.TimeField.BOTTOM_EDGE)

    with pytest.raises(ValueError, match="no path of material"):
        timefield.init_geodesic_timefield(xPhys, base)


# --- uniformity_penalty -------------------------------------------------------------


def test_uniformity_penalty_dispatches_to_gradient_cv():
    field = torch.rand((7, 6), dtype=torch.float64)
    direct = timefield._gradient_cv(field, None)
    via_dispatch = timefield.uniformity_penalty(
        field, timefield.UniformityMetric.GRADIENT_CV
    )
    assert float(direct) == pytest.approx(float(via_dispatch))


def test_uniformity_penalty_unknown_metric_rejected():
    field = torch.rand((5, 5), dtype=torch.float64)
    with pytest.raises(ValueError):
        timefield.uniformity_penalty(field, "not_a_real_metric")


def test_gradient_cv_weighting_changes_the_answer():
    """A masked-off region with a deliberately different gradient must be excluded by
    the weighting, not merely down-weighted to irrelevance -- so weighted and
    unweighted spreads differ."""
    ny, nx = 10, 10
    ys, xs = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    field = 0.05 * xs  # uniform gradient over the left region
    field[:, nx // 2 :] = 0.05 * (nx // 2) + 0.5 * (xs[:, nx // 2 :] - nx // 2)  # steep

    weights = torch.zeros((ny, nx), dtype=torch.float64)
    weights[:, : nx // 2] = 1.0  # "part" is only the uniform-gradient half

    unweighted = timefield.uniformity_penalty(
        field, timefield.UniformityMetric.GRADIENT_CV
    )
    weighted = timefield.uniformity_penalty(
        field, timefield.UniformityMetric.GRADIENT_CV, weights=weights
    )
    assert float(weighted) < float(unweighted)
    assert float(weighted) == pytest.approx(0.0, abs=1e-6)


def test_gradient_cv_zero_weight_returns_zero():
    field = torch.rand((6, 6), dtype=torch.float64)
    weights = torch.zeros((6, 6), dtype=torch.float64)
    assert (
        float(
            timefield.uniformity_penalty(
                field, timefield.UniformityMetric.GRADIENT_CV, weights=weights
            )
        )
        == 0.0
    )


def test_gradient_cv_zero_mean_gradient_returns_zero():
    field = torch.full((6, 6), 0.3, dtype=torch.float64)  # constant -> zero gradient
    assert (
        float(
            timefield.uniformity_penalty(field, timefield.UniformityMetric.GRADIENT_CV)
        )
        == 0.0
    )


def test_gradient_cv_is_blind_to_a_sawtooth_across_the_print_direction():
    """The stencil sees a `(-1)^j` sawtooth in `dt/dx`, but only as an alternating sign,
    and `|grad t|` squares that away: on a print sweeping in y, such a sawtooth of any
    amplitude leaves the coefficient of variation exactly unchanged.

    Pinned because it is the reason `roughness` is in the objective and not only in the
    log (PR #92): a mode this measure cannot see is one the optimizer may spend freely.
    """
    ny, nx = 20, 24
    ramp = np.tile(np.linspace(0.0, 1.0, ny)[:, None], (1, nx))
    _, j = np.indices((ny, nx))
    reference = float(timefield._gradient_cv(torch.from_numpy(ramp), None))

    for amplitude in (1e-4, 1e-2):
        field = torch.from_numpy(ramp + amplitude * (-1.0) ** j)
        assert float(timefield._gradient_cv(field, None)) == pytest.approx(
            reference, abs=1e-12
        )
        assert float(timefield.roughness(field)) == pytest.approx(amplitude, rel=1e-9)


def test_roughness_penalizes_padding_thin_layers_with_a_sawtooth():
    """The sawtooth above is not merely free to the uniformity penalty but rewarded:
    modulating its amplitude thickens locally thin layers in quadrature, so `_gradient_cv`
    improves while the field gets jaggier. `roughness` is what moves the other way.
    """
    ny, nx = 30, 40
    _, j = np.indices((ny, nx))
    rate = (0.6 + 0.8 * j / (nx - 1)) / (ny - 1)  # layer thickness varies across x
    ramp = np.cumsum(rate, axis=0) - rate
    padding = np.sqrt(np.clip(rate.max() ** 2 - rate**2, 0.0, None)) / 2

    plain = torch.from_numpy(ramp)
    padded = torch.from_numpy(ramp + padding * (-1.0) ** j)

    assert float(timefield._gradient_cv(padded, None)) < float(
        timefield._gradient_cv(plain, None)
    )
    assert float(timefield.roughness(padded)) > float(timefield.roughness(plain))
