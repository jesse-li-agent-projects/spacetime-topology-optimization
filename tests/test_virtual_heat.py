"""`virtual_heat`: the maps from the design fields to `tPhys` of both variants."""

import dataclasses
import warnings

import numpy as np
import pytest
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sttopt.checks as checks
import sttopt.fem as fem
import sttopt.timefield as timefield
import sttopt.virtual_heat as vh
from conftest import virtual_heat_config
from sttopt.run_config import (
    HeatRunConfig,
    HeatTimeMap,
    LaplaceRunConfig,
    LaplaceTimeMap,
)

torch = pytest.importorskip("torch")

CPU, F64 = torch.device("cpu"), torch.float64


def _mesh(nelx, nely, base="edge"):
    return vh.ScalarMesh.build(nelx, nely, base, CPU, F64)


def _full(nelx, nely, value):
    return torch.full((nely, nelx), value, dtype=F64)


def _node_rings(nelx, nely):
    """The 8-ring of every node, padded with the node itself at the border."""
    idx = np.arange((nelx + 1) * (nely + 1)).reshape(nely + 1, nelx + 1)
    padded = np.pad(idx, 1, mode="edge")
    rings = [
        padded[1 + di : 2 + di + nely, 1 + dj : 2 + dj + nelx]
        for di, dj in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]
    ]
    return np.stack(rings, axis=-1).reshape(-1, 8)  # in order around the node


# -- Wu2025 Fig. 2.6(e) ----------------------------------------------------------------


@pytest.mark.parametrize("beta_wu", [0.1, 1.0, 10.0])
def test_one_minus_reproduces_wu_fig_2_6e(beta_wu):
    """A 20 x 100 column with the plate at the bottom and Wu's uniform `chi = 0.1`.
    Wu's `l_c` is the longest path, 100; ours is `sqrt(20 * 100)`, and our `chi` is 1
    at `mu = 1/2`, so `beta = 10 * beta_wu * (l_c / 100)**2 = 2 beta_wu` gives the same
    decay rate `k = sqrt(beta_wu / 0.1) / 100`."""
    mesh = _mesh(20, 100, "bottom_edge")
    config = virtual_heat_config(
        HeatRunConfig,
        nelx=20,
        nely=100,
        print_base="bottom_edge",
        drain_beta=2 * beta_wu,
        time_map="one_minus",
        heat_cg_rtol=1e-12,
    )
    T = vh.heat_time_field(
        config, mesh, _full(20, 100, 1.0), _full(20, 100, 0.5)
    ).primary
    column = T.reshape(101, 21).numpy()[::-1, 10]  # bottom to top, x = 10
    y = np.arange(101.0)
    k = np.sqrt(beta_wu / 0.1) / 100
    analytic = np.cosh(k * (100 - y)) / np.cosh(k * 100)
    np.testing.assert_allclose(column, analytic, atol=2e-4)
    # The two curves of the figure that read clearly, at y = 100
    read_off = {0.1: 0.648, 1.0: 0.085}
    if beta_wu in read_off:
        assert column[-1] == pytest.approx(read_off[beta_wu], abs=0.01)


# -- Variant 2: the wall data and the harmonic field ------------------------------------


@pytest.mark.parametrize("base", ["edge", "bottom_edge"])
def test_ramp_init_reproduces_the_ramp(base):
    nelx, nely = 30, 10
    mesh = _mesh(nelx, nely, base)
    wall = vh.ramp_wall(mesh)
    # Inside the box with room above it, so the data can rise above the ramp
    assert wall.min() >= 0 and float(wall.max()) == pytest.approx(0.5)
    config = virtual_heat_config(
        LaplaceRunConfig, nelx=nelx, nely=nely, print_base=base, laplace_cg_rtol=1e-12
    )
    xPhys = _full(nelx, nely, 0.5)
    tPhys = vh.laplace_time_field(
        config, mesh, xPhys, _full(nelx, nely, 0.5), wall
    ).tPhys
    rows, cols = np.mgrid[:nely, :nelx]
    distance = cols + 0.5 if base == "edge" else nely - rows - 0.5
    np.testing.assert_allclose(tPhys.numpy(), distance / distance.max(), atol=1e-12)


def _unimodal(rng, n):
    """Random wall data on `[0, 1]` with one peak: the smaller of a rising and a
    falling running sum."""
    rising = np.cumsum(rng.uniform(0, 1, n))
    falling = np.cumsum(rng.uniform(0, 1, n)[::-1])[::-1]
    b = np.minimum(rising, falling)
    return torch.tensor(b / b.max())


def _harmonic(seed, nelx=24, nely=16, contrast=1e3, smooth=False, unimodal=True):
    """A harmonic field with random wall data (one peak, or independent per node) and
    a random `mu` spanning `[0, 1]`: independent per element, or smoothed over about
    two elements."""
    rng = np.random.default_rng(seed)
    mesh = _mesh(nelx, nely)
    config = virtual_heat_config(
        LaplaceRunConfig,
        nelx=nelx,
        nely=nely,
        chi_contrast=contrast,
        laplace_cg_rtol=1e-12,
    )
    if smooth:
        s = ndi.gaussian_filter(rng.standard_normal((nely, nelx)), 2.0, mode="nearest")
        mu = torch.tensor(0.5 + 0.5 * s / np.abs(s).max())
    else:
        mu = torch.tensor(rng.uniform(0, 1, (nely, nelx)))
    n_arc = len(mesh.arc)
    wall = _unimodal(rng, n_arc) if unimodal else torch.tensor(rng.uniform(0, 1, n_arc))
    u = vh.laplace_time_field(config, mesh, _full(nelx, nely, 0.5), mu, wall).primary
    return mesh, u.numpy()


@pytest.mark.parametrize("seed", range(5))
def test_harmonic_field_has_no_interior_extremum_on_the_nodes(seed):
    """For any wall data, not only data with one peak."""
    mesh, u = _harmonic(seed, unimodal=False)
    interior = np.zeros((mesh.nely + 1, mesh.nelx + 1), bool)
    interior[1:-1, 1:-1] = True
    ring = u[_node_rings(mesh.nelx, mesh.nely)]
    strict_max = (u[:, None] > ring).all(axis=1) & interior.ravel()
    strict_min = (u[:, None] < ring).all(axis=1) & interior.ravel()
    assert not strict_max.any() and not strict_min.any()


def saddle_count(u, nelx, nely):
    """Interior nodes of the nodal field `u` that are discrete saddles."""
    interior = np.zeros((nely + 1, nelx + 1), bool)
    interior[1:-1, 1:-1] = True
    return int(checks.saddles(u.reshape(nely + 1, nelx + 1), interior).sum())


@pytest.mark.parametrize("seed", range(5))
def test_harmonic_field_with_unimodal_walls_has_no_interior_saddle(seed):
    """The 2D theorem (Alessandrini & Magnanini) leaves no interior critical point for
    unimodal wall data. Here with a smooth `chi` at the full contrast."""
    mesh, u = _harmonic(seed, smooth=True)
    assert saddle_count(u, mesh.nelx, mesh.nely) == 0


def test_two_wall_peaks_give_an_interior_saddle():
    """Where two branches of the print merge: the saddle between two peaks, which the
    optimized wall data may now have."""
    nelx, nely = 24, 16
    mesh = _mesh(nelx, nely, "bottom_edge")
    config = virtual_heat_config(
        LaplaceRunConfig,
        nelx=nelx,
        nely=nely,
        print_base="bottom_edge",
        laplace_cg_rtol=1e-12,
    )
    # Along the arc: up the left wall, across the top, down the right wall
    k = np.arange(len(mesh.arc))
    peaks = np.exp(-(((k - 0.3 * len(k)) / 4) ** 2)) + np.exp(
        -(((k - 0.7 * len(k)) / 4) ** 2)
    )
    wall = torch.tensor(0.2 + 0.8 * peaks)
    u = vh.laplace_time_field(
        config, mesh, _full(nelx, nely, 0.5), _full(nelx, nely, 0.5), wall
    ).primary
    assert saddle_count(u.numpy(), nelx, nely) >= 1


@pytest.mark.xfail(
    strict=True,
    reason="the theorem is continuous; element-scale chi jumps of ~100x or more give discrete saddles (plans/virtual_heat_timefield.md, Phase 4)",
)
@pytest.mark.parametrize("seed", range(3))
def test_harmonic_field_with_element_scale_chi_has_no_interior_saddle(seed):
    mesh, u = _harmonic(seed)
    assert saddle_count(u, mesh.nelx, mesh.nely) == 0


# -- Every option: finite gradients, and the gradient against finite differences --------

OPTIONS = [(HeatRunConfig, m) for m in HeatTimeMap] + [
    (LaplaceRunConfig, m) for m in LaplaceTimeMap
]


def _time_field(cls, time_map, nelx, nely, drain_beta, rtol=1e-13):
    """`(mesh, f)`: `f(xPhys, mu, wall) -> tPhys` for one variant and map, every solve
    at `rtol`."""
    mesh = _mesh(nelx, nely)
    tight = dict(poisson_cg_rtol=rtol, time_map=time_map.value)
    if cls is HeatRunConfig:
        tight |= dict(heat_cg_rtol=rtol, drain_beta=drain_beta)
    else:
        tight |= dict(laplace_cg_rtol=rtol)
    config = virtual_heat_config(cls, nelx=nelx, nely=nely, **tight)
    if cls is HeatRunConfig:
        return mesh, lambda x, mu, wall: vh.heat_time_field(config, mesh, x, mu).tPhys
    return (
        mesh,
        lambda x, mu, wall: vh.laplace_time_field(config, mesh, x, mu, wall).tPhys,
    )


@pytest.mark.parametrize("cls,time_map", OPTIONS)
def test_gradients_are_finite_in_deep_void(cls, time_map):
    """At a mesh the V-cycle coarsens and a production tolerance, so `T` deep in the
    void is below the solve error (and can be negative) -- as in a real run."""
    nelx, nely = 90, 30
    mesh, f = _time_field(cls, time_map, nelx, nely, drain_beta=49.0, rtol=1e-8)
    x = torch.full((nely, nelx), 1e-9, dtype=F64)
    x[:, :10] = 1.0  # a solid block at the plate, and void everywhere beyond it
    x.requires_grad_()
    mu = _full(nelx, nely, 0.5).requires_grad_()
    wall = vh.ramp_wall(mesh).requires_grad_()
    tPhys = f(x, mu, wall)
    assert torch.isfinite(tPhys).all()
    (tPhys * torch.rand_like(tPhys)).sum().backward()
    for leaf in (x, mu, wall):
        assert leaf.grad is None or torch.isfinite(leaf.grad).all()


@pytest.mark.parametrize("cls,time_map", OPTIONS)
def test_gradient_matches_finite_differences(cls, time_map, monkeypatch):
    """The scales that are treated as constants by design (the normalization, the
    `neg_log` floor) are frozen, so finite differences
    see the same function autograd differentiates."""
    monkeypatch.setattr(vh, "normalize", lambda t, xPhys: t / 0.7)
    monkeypatch.setattr(vh, "NEG_LOG_FLOOR", 0.0)
    nelx, nely = 10, 7  # coarse enough for the dense coarse solve alone: exact
    rng = np.random.default_rng(1)
    mesh, f = _time_field(cls, time_map, nelx, nely, drain_beta=9.0)
    x = torch.tensor(rng.uniform(0.3, 0.9, (nely, nelx))).requires_grad_()
    mu = torch.tensor(rng.uniform(0.2, 0.8, (nely, nelx))).requires_grad_()
    wall0 = vh.ramp_wall(mesh)
    wall = (wall0 + torch.tensor(rng.uniform(0, 0.3, wall0.shape))).requires_grad_()
    w = torch.tensor(rng.uniform(-1, 1, (nely, nelx)))

    def L(*leaves):
        return (w * f(*leaves)).sum()

    leaves = [x, mu, wall]
    L(*leaves).backward()
    for i, leaf in enumerate(leaves):
        if leaf.grad is None:  # e.g. the heat map does not read the wall data
            continue
        d = torch.tensor(rng.standard_normal(leaf.shape))
        eps = 1e-6
        with torch.no_grad():
            plus = [v + eps * d if j == i else v for j, v in enumerate(leaves)]
            minus = [v - eps * d if j == i else v for j, v in enumerate(leaves)]
            fd = (L(*plus) - L(*minus)) / (2 * eps)
        assert float((leaf.grad * d).sum()) == pytest.approx(
            float(fd), rel=1e-5, abs=1e-9
        )


# -- The normalization ------------------------------------------------------------------


def test_normalization_reads_the_whole_domain_on_a_uniform_start():
    t = torch.arange(12.0, dtype=F64).reshape(3, 4)
    assert float(vh.normalize(t, _full(4, 3, 0.5)).max()) == 1.0


def test_normalization_reads_the_solid_of_a_binary_design():
    t = torch.arange(12.0, dtype=F64).reshape(3, 4)
    x = torch.zeros(3, 4, dtype=F64)
    x[0] = 1.0  # t up to 3 on the solid; 11 in the void
    np.testing.assert_allclose(vh.normalize(t, x).numpy(), t.numpy() / 3)


def test_normalization_holds_when_every_density_is_below_the_projection_threshold():
    """No element reaches `eta`, but the set is relative to the largest density, so it
    is still the densest elements."""
    t = torch.arange(12.0, dtype=F64).reshape(3, 4)
    x = torch.full((3, 4), 0.05, dtype=F64)
    x[1, :2] = 0.2  # t = 4, 5
    np.testing.assert_allclose(vh.normalize(t, x).numpy(), t.numpy() / 5)


def test_plate_nodes_reject_a_point_base():
    with pytest.raises(ValueError, match="plate"):
        vh.plate_nodes(4, 3, "corner")


def test_config_placeholder_changes_nothing_it_does_not_name():
    """Guard for the helper above: only the named fields differ from the defaults."""
    a = virtual_heat_config(HeatRunConfig, nelx=20, nely=10)
    b = dataclasses.replace(a, drain_beta=3.0)
    assert {k for k in a.to_dict() if a.to_dict()[k] != b.to_dict()[k]} == {
        "drain_beta"
    }


# -- Distance recovery and the drain's margin -------------------------------------------
# Binary geometries at uniform mu. The reference is the geodesic distance through the
# solid on a raster 4x finer (`timefield._solid_geodesic`), averaged onto the elements.

FINE = 4
DRAIN_BETA = (
    25.0  # sqrt(beta) = 5, the largest value of the Phase 4 sweep that keeps the margin
)


def _rectangle(nx, ny, s):
    return np.ones((ny * s, nx * s), bool)


def _annulus(nx, ny, s):
    """A ring on a base strip at the bottom edge."""
    yy, xx = np.mgrid[: ny * s, : nx * s]
    x, y = (xx + 0.5) / s, ny - (yy + 0.5) / s
    r = np.hypot(x - nx / 2, y - ny / 2)
    return ((r < 0.45 * ny) & (r > 0.25 * ny)) | (y < 0.1 * ny)


def _worm(nx, ny, s):
    """A band around a sine, from the plate at the left edge."""
    yy, xx = np.mgrid[: ny * s, : nx * s]
    x, y = (xx + 0.5) / s, ny - (yy + 0.5) / s
    centre = ny / 2 + 0.3 * ny * np.sin(2 * np.pi * 1.5 * x / nx)
    return np.abs(y - centre) < 0.12 * ny


def _streets(nx, ny, s):
    """Streets 4 elements wide around void blocks: many holes."""
    yy, xx = np.mgrid[: ny * s, : nx * s]
    x, y = (xx + 0.5) / s, (yy + 0.5) / s
    return (np.mod(x, 20) < 4) | (np.mod(y, 20) < 4) | (x > nx - 4) | (y > ny - 4)


GEOMETRIES = {
    "rectangle": (_rectangle, 120, 40, "edge"),
    "annulus": (_annulus, 80, 80, "bottom_edge"),
    "worm": (_worm, 180, 60, "edge"),
    "streets": (_streets, 180, 60, "edge"),
}

# Measured at DRAIN_BETA (max / mean error over the part, of a field normalized to 1):
# neg_log up to 0.156 / 0.049 (annulus), poisson up to 0.089 / 0.023 (streets). The
# tolerances are 1.6-1.7x those.
RECOVERY_TOLERANCE = {"neg_log": (0.25, 0.08), "poisson": (0.15, 0.04)}


def _reference_distance(build, nx, ny, base):
    solid = build(nx, ny, FINE).ravel()
    tf = timefield.TimeField[base.upper()]
    base_f = timefield.base_elements(nx * FINE, ny * FINE, tf)
    d = timefield._solid_geodesic(solid, base_f[solid[base_f]], nx * FINE, ny * FINE)
    d = np.where(solid, d / FINE, np.nan).reshape(ny, FINE, nx, FINE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # void elements: all nan
        return np.nanmean(d, axis=(1, 3))


def _geometry_config(geometry, **overrides):
    _, nx, ny, base = GEOMETRIES[geometry]
    return virtual_heat_config(
        HeatRunConfig,
        nelx=nx,
        nely=ny,
        print_base=base,
        drain_beta=DRAIN_BETA,
        heat_cg_rtol=1e-8,
        poisson_cg_rtol=1e-8,
        **overrides,
    )


@pytest.mark.parametrize("time_map", ["neg_log", "poisson"])
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_distance_recovery(geometry, time_map):
    build, nx, ny, base = GEOMETRIES[geometry]
    solid = build(nx, ny, 1)
    ref = _reference_distance(build, nx, ny, base)
    ref = ref / np.nanmax(ref[solid])
    config = _geometry_config(geometry, time_map=time_map)
    xPhys = torch.tensor(solid, dtype=F64)
    mu = torch.full_like(xPhys, 0.5)
    tPhys = vh.heat_time_field(config, _mesh(nx, ny, base), xPhys, mu).tPhys
    err = np.abs(tPhys.numpy() - ref)[solid]
    max_tol, mean_tol = RECOVERY_TOLERANCE[time_map]
    assert err.max() <= max_tol and err.mean() <= mean_tol, (err.max(), err.mean())


def test_drain_keeps_T_on_the_part_well_above_the_solve_error():
    """On the geometry with the least margin in the sweep: min T on the part was 1e3
    times the absolute CG error at 1e-8 against a direct solve. The requirement is
    100x."""
    build, nx, ny, base = GEOMETRIES["streets"]
    solid = build(nx, ny, 1)
    mesh = _mesh(nx, ny, base)
    config = _geometry_config("streets", time_map="one_minus")
    xPhys = torch.tensor(solid, dtype=F64)
    mu = torch.full_like(xPhys, 0.5)
    T = vh.heat_time_field(config, mesh, xPhys, mu).primary.numpy()

    chi = np.clip(solid.ravel().astype(float), vh.DENSITY_FLOOR, None)
    e = mesh.edof.numpy()
    drain = DRAIN_BETA / mesh.unit_length**2 * np.eye(4) / 4
    vals = chi[:, None, None] * fem.diffusion_KE() + drain
    rows, cols = np.repeat(e, 4, 1).ravel(), np.tile(e, (1, 4)).ravel()
    K = sp.csr_matrix((vals.ravel(), (rows, cols)), shape=(mesh.ndof,) * 2)
    fixed = mesh.plate.numpy()
    direct = fixed.astype(float)
    free = ~fixed
    rhs = -(K[free][:, fixed] @ direct[fixed])
    direct[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)

    part = np.zeros(mesh.ndof, bool)
    part[e[solid.ravel()].ravel()] = True
    assert T[part].min() >= 100 * np.abs(T - direct)[part].max()
