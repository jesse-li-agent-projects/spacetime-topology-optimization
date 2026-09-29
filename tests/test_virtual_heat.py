"""`virtual_heat`: the maps from the design fields to `tPhys` of both variants."""

import dataclasses

import numpy as np
import pytest
import scipy.ndimage as ndi

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
    a, c = vh.ramp_wall_coefficients(mesh)
    assert (
        a.min() >= 0
        and a.max() <= 0.5 + 1e-12
        and c.min() >= 0
        and c.max() <= 0.5 + 1e-12
    )
    config = virtual_heat_config(
        LaplaceRunConfig, nelx=nelx, nely=nely, print_base=base, laplace_cg_rtol=1e-12
    )
    xPhys = _full(nelx, nely, 0.5)
    tPhys = vh.laplace_time_field(
        config, mesh, xPhys, _full(nelx, nely, 0.5), a, c
    ).tPhys
    rows, cols = np.mgrid[:nely, :nelx]
    distance = cols + 0.5 if base == "edge" else nely - rows - 0.5
    np.testing.assert_allclose(tPhys.numpy(), distance / distance.max(), atol=1e-12)


def test_wall_data_is_unimodal_for_random_coefficients():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(3, 60))
        a, c = (torch.tensor(rng.uniform(0, 1, n)) for _ in range(2))
        b = vh.unimodal_wall_data(a, c, 0.1).numpy()
        signs = np.sign(np.diff(b))
        signs = signs[signs != 0]
        assert np.count_nonzero(np.diff(signs)) <= 1  # up, then down, at most once
        assert b.min() >= 0


def _harmonic(seed, nelx=24, nely=16, contrast=1e3, smooth=False):
    """A harmonic field with random unimodal wall data and a random `mu` spanning
    `[0, 1]`: independent per element, or smoothed over about two elements."""
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
    a, c = (torch.tensor(rng.uniform(0, 1, len(mesh.arc))) for _ in range(2))
    u = vh.laplace_time_field(config, mesh, _full(nelx, nely, 0.5), mu, a, c).primary
    return mesh, u.numpy()


@pytest.mark.parametrize("seed", range(5))
def test_harmonic_field_has_no_interior_extremum_on_the_nodes(seed):
    mesh, u = _harmonic(seed)
    interior = np.zeros((mesh.nely + 1, mesh.nelx + 1), bool)
    interior[1:-1, 1:-1] = True
    ring = u[_node_rings(mesh.nelx, mesh.nely)]
    strict_max = (u[:, None] > ring).all(axis=1) & interior.ravel()
    strict_min = (u[:, None] < ring).all(axis=1) & interior.ravel()
    assert not strict_max.any() and not strict_min.any()


def saddle_count(u, nelx, nely):
    """Interior nodes where `u - u_node` changes sign 4 or more times around the 8-ring."""
    interior = np.zeros((nely + 1, nelx + 1), bool)
    interior[1:-1, 1:-1] = True
    ring = u[_node_rings(nelx, nely)] - u[:, None]
    # A tie (e.g. along an iso-line of a linear field) is no sign, not a change
    signs = np.where(np.abs(ring) > 1e-9 * np.ptp(u), np.sign(ring), 0)
    count = 0
    for n in np.flatnonzero(interior.ravel()):
        s = signs[n][signs[n] != 0]
        count += np.count_nonzero(s != np.roll(s, 1)) >= 4
    return count


@pytest.mark.parametrize("seed", range(5))
def test_harmonic_field_with_unimodal_walls_has_no_interior_saddle(seed):
    """The 2D theorem (Alessandrini & Magnanini) leaves no interior critical point for
    unimodal wall data. Here with a smooth `chi` at the full contrast."""
    mesh, u = _harmonic(seed, smooth=True)
    assert saddle_count(u, mesh.nelx, mesh.nely) == 0


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


def _time_field(cls, time_map, nelx, nely, drain_beta):
    """`(mesh, f)`: `f(xPhys, mu, a, c) -> tPhys` for one variant and map, solved
    tightly."""
    mesh = _mesh(nelx, nely)
    tight = dict(poisson_cg_rtol=1e-13, time_map=time_map.value)
    if cls is HeatRunConfig:
        tight |= dict(heat_cg_rtol=1e-13, drain_beta=drain_beta)
    else:
        tight |= dict(laplace_cg_rtol=1e-13)
    config = virtual_heat_config(cls, nelx=nelx, nely=nely, **tight)
    if cls is HeatRunConfig:
        return mesh, lambda x, mu, a, c: vh.heat_time_field(config, mesh, x, mu).tPhys
    return (
        mesh,
        lambda x, mu, a, c: vh.laplace_time_field(config, mesh, x, mu, a, c).tPhys,
    )


@pytest.mark.parametrize("cls,time_map", OPTIONS)
def test_gradients_are_finite_in_deep_void(cls, time_map):
    nelx, nely = 30, 12
    mesh, f = _time_field(cls, time_map, nelx, nely, drain_beta=49.0)
    x = torch.full((nely, nelx), 1e-9, dtype=F64)
    x[:, :10] = 1.0  # a solid block at the plate, and void everywhere beyond it
    x.requires_grad_()
    mu = _full(nelx, nely, 0.5).requires_grad_()
    a0, c0 = vh.ramp_wall_coefficients(mesh)
    a, c = a0.clone().requires_grad_(), c0.clone().requires_grad_()
    tPhys = f(x, mu, a, c)
    assert torch.isfinite(tPhys).all()
    (tPhys * torch.rand_like(tPhys)).sum().backward()
    for leaf in (x, mu, a, c):
        assert leaf.grad is None or torch.isfinite(leaf.grad).all()


@pytest.mark.parametrize("cls,time_map", OPTIONS)
def test_gradient_matches_finite_differences(cls, time_map, monkeypatch):
    """The scales that are treated as constants by design (the normalization, the
    `neg_log` floor, the direction field's `eps`) are frozen, so finite differences
    see the same function autograd differentiates."""
    monkeypatch.setattr(vh, "normalize", lambda t, xPhys: t / 0.7)
    monkeypatch.setattr(vh, "NEG_LOG_FLOOR", 0.0)
    monkeypatch.setattr(timefield, "NORMAL_EPS", 0.0)
    nelx, nely = 10, 7  # coarse enough for the dense coarse solve alone: exact
    rng = np.random.default_rng(1)
    mesh, f = _time_field(cls, time_map, nelx, nely, drain_beta=9.0)
    x = torch.tensor(rng.uniform(0.3, 0.9, (nely, nelx))).requires_grad_()
    mu = torch.tensor(rng.uniform(0.2, 0.8, (nely, nelx))).requires_grad_()
    a0, c0 = vh.ramp_wall_coefficients(mesh)
    a = (a0 + torch.tensor(rng.uniform(0, 0.3, a0.shape))).requires_grad_()
    c = (c0 + torch.tensor(rng.uniform(0, 0.3, c0.shape))).requires_grad_()
    w = torch.tensor(rng.uniform(-1, 1, (nely, nelx)))

    def L(*leaves):
        return (w * f(*leaves)).sum()

    leaves = [x, mu, a, c]
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
