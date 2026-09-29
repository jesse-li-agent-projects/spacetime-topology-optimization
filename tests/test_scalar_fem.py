"""The scalar (one dof per node) diffusion-reaction solve that the virtual-heat time
fields rest on: manufactured-solution convergence for each boundary-condition set, the
M-matrix property and the discrete maximum principle, multigrid-CG against a direct
solve, the adjoint, and 1D analytic profiles.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sttopt.fem as fem

torch = pytest.importorskip("torch")

import sttopt.torch_fem as torch_fem  # noqa: E402
import sttopt.torch_solve as torch_solve  # noqa: E402

KE = torch.tensor(fem.diffusion_KE(), dtype=torch.float64)
ME = torch.tensor(fem.lumped_mass_ME(), dtype=torch.float64)
TIGHT = 1e-12  # CG tolerance far below every discretization error measured here


class Mesh:
    """A `nelx` x `nely` scalar mesh of element size `h`, with node and element-centre
    coordinates (x right, y up, origin at the bottom-left corner)."""

    def __init__(self, nelx, nely, h=1.0):
        self.nelx, self.nely, self.h = nelx, nely, h
        self.edof = torch.tensor(fem.element_dof_map(nelx, nely, 1), dtype=torch.int64)
        self.ndof = (nelx + 1) * (nely + 1)
        rows, cols = np.mgrid[: nely + 1, : nelx + 1]
        self.x, self.y = (cols * h).ravel(), ((nely - rows) * h).ravel()
        erows, ecols = np.mgrid[:nely, :nelx]
        self.xc, self.yc = ((ecols + 0.5) * h).ravel(), (
            (nely - erows - 0.5) * h
        ).ravel()
        # Nodal share of the domain, in element areas: the lumped load's quadrature
        self.area = np.bincount(
            self.edof.numpy().ravel(),
            minlength=self.ndof,
            weights=np.full(4 * nelx * nely, 0.25),
        )

    def mask(self, fixed):
        """Free-dof mask, `fixed` a boolean per node."""
        return torch.tensor(~fixed)

    def assemble(self, chi, reaction=None):
        """The global sparse matrix, for the direct solve and the M-matrix check."""
        e = self.edof.numpy()
        vals = chi[:, None, None] * fem.diffusion_KE()
        if reaction is not None:
            vals = vals + reaction[:, None, None] * fem.lumped_mass_ME()
        rows = np.repeat(e, 4, axis=1).ravel()
        cols = np.tile(e, (1, 4)).ravel()
        return sp.csr_matrix((vals.ravel(), (rows, cols)), shape=(self.ndof,) * 2)


def _t(a):
    return torch.as_tensor(a, dtype=torch.float64)


def _solve(mesh, chi, F, g, fixed, reaction=None, rtol=TIGHT):
    return torch_solve.lifted_femsolve(
        _t(chi),
        _t(F),
        _t(g),
        mesh.edof,
        KE,
        mesh.mask(fixed),
        mesh.nelx,
        mesh.nely,
        reaction=None if reaction is None else _t(reaction),
        ME=None if reaction is None else ME,
        rtol=rtol,
    )


def _direct(mesh, chi, F, g, fixed, reaction=None):
    K = mesh.assemble(chi, reaction)
    free = ~fixed
    u = np.where(fixed, g, 0.0)
    rhs = F - K @ u
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs[free])
    return u


# -- Manufactured solutions ----------------------------------------------------------
# chi = exp(s), s = 0.5 sin(pi x) cos(pi y), sampled at element centres, on the unit
# square; f = -div(chi grad u) + alpha u = -chi (lap u + grad s . grad u) + alpha u.

PI = np.pi


def _s(x, y):
    return 0.5 * np.sin(PI * x) * np.cos(PI * y)


def _grad_s(x, y):
    return 0.5 * PI * np.cos(PI * x) * np.cos(PI * y), -0.5 * PI * np.sin(
        PI * x
    ) * np.sin(PI * y)


def _all_dirichlet(x, y):
    u = np.sin(PI * x) * np.sin(PI * y) + x * y
    ux = PI * np.cos(PI * x) * np.sin(PI * y) + y
    uy = PI * np.sin(PI * x) * np.cos(PI * y) + x
    lap = -2 * PI**2 * np.sin(PI * x) * np.sin(PI * y)
    return u, ux, uy, lap


def _plate_neumann_drain(x, y):
    """Zero normal derivative on x = 1, y = 0 and y = 1."""
    cx, sx, cy, sy = np.cos(PI * x), np.sin(PI * x), np.cos(PI * y), np.sin(PI * y)
    u = 1 + 0.5 * cx + 0.3 * cx * cy
    ux = -0.5 * PI * sx - 0.3 * PI * sx * cy
    uy = -0.3 * PI * cx * sy
    lap = -0.5 * PI**2 * cx - 0.6 * PI**2 * cx * cy
    return u, ux, uy, lap


def _zero_plate_mixed(x, y):
    """Zero on x = 0, zero normal derivative on y = 0."""
    s2, c2, cy, sy = (
        np.sin(PI * x / 2),
        np.cos(PI * x / 2),
        np.cos(PI * y),
        np.sin(PI * y),
    )
    u = s2 * (1 + 0.5 * cy)
    ux = PI / 2 * c2 * (1 + 0.5 * cy)
    uy = -0.5 * PI * s2 * sy
    lap = -(PI**2) / 4 * s2 * (1 + 0.5 * cy) - 0.5 * PI**2 * s2 * cy
    return u, ux, uy, lap


ALPHA = 4.0  # drain of the Variant 1 set, in 1 / length**2

BC_SETS = {
    # name: (solution, drain, fixed(x, y))
    "all_dirichlet": (
        _all_dirichlet,
        0.0,
        lambda x, y: (x == 0) | (x == 1) | (y == 0) | (y == 1),
    ),
    "plate_neumann_drain": (_plate_neumann_drain, ALPHA, lambda x, y: x == 0),
    "zero_plate_mixed": (
        _zero_plate_mixed,
        0.0,
        lambda x, y: (x == 0) | (x == 1) | (y == 1),
    ),
}


def _mms_error(name, n):
    solution, alpha, fixed_at = BC_SETS[name]
    mesh = Mesh(n, n, 1.0 / n)
    u, ux, uy, lap = solution(mesh.x, mesh.y)
    sx, sy = _grad_s(mesh.x, mesh.y)
    f = -np.exp(_s(mesh.x, mesh.y)) * (lap + sx * ux + sy * uy) + alpha * u
    chi = np.exp(_s(mesh.xc, mesh.yc))
    h2 = mesh.h**2
    reaction = np.full(n * n, alpha * h2) if alpha else None
    fixed = fixed_at(np.round(mesh.x, 12), np.round(mesh.y, 12))
    got = _solve(mesh, chi, h2 * mesh.area * f, u, fixed, reaction).numpy()
    return np.sqrt(np.sum(h2 * mesh.area * (got - u) ** 2))


@pytest.mark.parametrize("name", BC_SETS)
def test_manufactured_solution_converges_at_second_order(name):
    errors = [_mms_error(name, n) for n in (16, 32, 64)]
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all(rates > 1.9), (errors, rates)


# -- The M-matrix property and the discrete maximum principle ------------------------


def _log_uniform(rng, n, contrast):
    return np.exp(rng.uniform(0, np.log(contrast), n))


@pytest.mark.parametrize("contrast", [1e3, 1e6])
@pytest.mark.parametrize("drain", [0.0, 0.5])
def test_assembled_matrix_is_an_m_matrix(contrast, drain):
    rng = np.random.default_rng(1)
    mesh = Mesh(12, 9)
    chi = _log_uniform(rng, 12 * 9, contrast)
    K = mesh.assemble(chi, np.full(12 * 9, drain) if drain else None).tocoo()
    off = K.row != K.col
    assert np.all(K.data[off] <= 0)
    # Row sums: zero for pure diffusion, the drain's share otherwise
    np.testing.assert_allclose(
        np.asarray(K.sum(axis=1)).ravel(), drain * mesh.area, atol=1e-12 * contrast
    )


def _node_neighbours(nelx, nely):
    """The 8-ring of every node, as index arrays padded with the node itself."""
    idx = np.arange((nelx + 1) * (nely + 1)).reshape(nely + 1, nelx + 1)
    padded = np.pad(idx, 1, mode="edge")
    rings = [
        padded[1 + di : 1 + di + nely + 1, 1 + dj : 1 + dj + nelx + 1]
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if (di, dj) != (0, 0)
    ]
    return np.stack(rings, axis=-1).reshape(-1, 8)


def _strict_extrema(values, rings, candidates):
    """Nodes of `candidates` strictly above, or strictly below, their whole ring."""
    ring = values[rings]
    above = (values[:, None] > ring).all(axis=1) & candidates
    below = (values[:, None] < ring).all(axis=1) & candidates
    return np.flatnonzero(above), np.flatnonzero(below)


def _harmonic_field(contrast, nelx=24, nely=16):
    """Harmonic `t` with random wall data and random `chi`."""
    rng = np.random.default_rng(2)
    mesh = Mesh(nelx, nely)
    chi = _log_uniform(rng, nelx * nely, contrast)
    boundary = (mesh.x == 0) | (mesh.x == nelx) | (mesh.y == 0) | (mesh.y == nely)
    g = np.where(boundary, rng.uniform(0, 1, mesh.ndof), 0.0)
    return mesh, boundary, _solve(mesh, chi, np.zeros(mesh.ndof), g, boundary).numpy()


@pytest.mark.parametrize("contrast", [1e3, 1e6])
def test_laplace_field_has_no_interior_extremum_on_the_nodes(contrast):
    """Every interior node is a positive-weight average of its ring, so none is a
    strict extremum."""
    mesh, boundary, t = _harmonic_field(contrast)
    rings = _node_neighbours(mesh.nelx, mesh.nely)
    above, below = _strict_extrema(t, rings, ~boundary)
    assert above.size == 0 and below.size == 0, (above, below)


@pytest.mark.xfail(
    strict=True,
    reason="an element mean is not a positive-weight average of its neighbours' means, so element-scale chi contrast can make one a strict extremum (plans/virtual_heat_timefield.md, Phase 3)",
)
@pytest.mark.parametrize("contrast", [1e3, 1e6])
def test_laplace_field_has_no_interior_extremum_on_the_element_means(contrast):
    """The element means are the values the rest of the code reads."""
    mesh, _, t = _harmonic_field(contrast)
    nelx, nely = mesh.nelx, mesh.nely
    means = t[mesh.edof.numpy()].mean(axis=1)
    interior = np.zeros((nely, nelx), bool)
    interior[1:-1, 1:-1] = True
    above, below = _strict_extrema(
        means, _node_neighbours(nelx - 1, nely - 1), interior.ravel()
    )
    assert above.size == 0 and below.size == 0, (above, below)


@pytest.mark.parametrize("contrast", [1e3, 1e6])
def test_drained_field_has_no_maximum_off_the_plate(contrast):
    """With a drain, `T` is subharmonic: no node or element mean off the plate is a
    strict maximum, the zero-flux walls included."""
    rng = np.random.default_rng(3)
    nelx, nely = 24, 16
    mesh = Mesh(nelx, nely)
    chi = _log_uniform(rng, nelx * nely, contrast)
    plate = mesh.x == 0
    T = _solve(
        mesh,
        chi,
        np.zeros(mesh.ndof),
        plate.astype(float),
        plate,
        np.full(nelx * nely, 0.05),
    ).numpy()

    above, _ = _strict_extrema(T, _node_neighbours(nelx, nely), ~plate)
    assert above.size == 0, above

    means = T[mesh.edof.numpy()].mean(axis=1)
    off_plate = np.ones((nely, nelx), bool)
    off_plate[:, 0] = False
    above, _ = _strict_extrema(
        means, _node_neighbours(nelx - 1, nely - 1), off_plate.ravel()
    )
    assert above.size == 0, above


# -- Multigrid-CG against a direct solve ---------------------------------------------


@pytest.mark.parametrize("contrast", [1.0, 1e3, 1e6])
@pytest.mark.parametrize("drain", [0.0, 0.05])
def test_multigrid_cg_matches_a_direct_solve(contrast, drain):
    """On a mesh large enough to coarsen, so the V-cycle is in the loop."""
    rng = np.random.default_rng(4)
    nelx, nely = 90, 30
    mesh = Mesh(nelx, nely)
    chi = _log_uniform(rng, nelx * nely, contrast)
    plate = mesh.x == 0
    reaction = np.full(nelx * nely, drain) if drain else None
    F = rng.uniform(-1, 1, mesh.ndof)
    g = plate * rng.uniform(0, 1, mesh.ndof)
    if not drain:  # a pure-Neumann remainder would leave nothing to pin; fix a wall
        plate = plate | (mesh.x == nelx)
        g = plate * rng.uniform(0, 1, mesh.ndof)
    got = _solve(mesh, chi, F, g, plate, reaction, rtol=1e-11).numpy()
    want = _direct(mesh, chi, F, g, plate, reaction)
    assert np.abs(got - want).max() <= 1e-7 * np.abs(want).max()


def test_batched_right_hand_sides_match_single_solves():
    rng = np.random.default_rng(5)
    mesh = Mesh(12, 9)
    chi = _log_uniform(rng, 12 * 9, 1e3)
    plate = mesh.x == 0
    F = rng.uniform(-1, 1, (3, mesh.ndof))
    g = plate * rng.uniform(0, 1, (3, mesh.ndof))
    reaction = np.full(12 * 9, 0.1)
    batched = _solve(mesh, chi, F, g, plate, reaction).numpy()
    for i in range(3):
        single = _solve(mesh, chi, F[i], g[i], plate, reaction).numpy()
        np.testing.assert_allclose(batched[i], single, rtol=1e-10, atol=1e-12)


def test_a_warm_start_at_the_solution_needs_no_iteration():
    rng = np.random.default_rng(6)
    mesh = Mesh(90, 30)
    chi = _log_uniform(rng, 90 * 30, 1e3)
    plate = mesh.x == 0
    reaction = _t(np.full(90 * 30, 0.05))
    args = (_t(chi), _t(np.zeros(mesh.ndof)), _t(plate.astype(float)), mesh.edof, KE)
    kwargs = dict(reaction=reaction, ME=ME, rtol=1e-8)
    T = torch_solve.lifted_femsolve(*args, mesh.mask(plate), 90, 30, **kwargs)
    info = {}
    torch_solve.lifted_femsolve(
        *args, mesh.mask(plate), 90, 30, x0=T * mesh.mask(plate), info=info, **kwargs
    )
    assert info["forward_n_iter"] == 0


# -- The adjoint ---------------------------------------------------------------------


def test_gradcheck_through_chi_drain_dirichlet_values_and_rhs():
    """A mesh small enough to be solved by the dense coarse factorization alone, so
    the solve is exact and finite differences are clean."""
    rng = np.random.default_rng(7)
    mesh = Mesh(5, 4)
    fixed = mesh.x == 0
    fixed[-1] = True  # and one more wall node, so the Dirichlet data varies
    mask = mesh.mask(fixed)
    leaves = [
        _t(_log_uniform(rng, 20, 10.0)).requires_grad_(),
        _t(rng.uniform(0.1, 1.0, 20)).requires_grad_(),
        _t(rng.uniform(-1, 1, mesh.ndof)).requires_grad_(),
        _t(rng.uniform(0, 1, mesh.ndof)).requires_grad_(),
    ]
    w = _t(rng.uniform(-1, 1, mesh.ndof))

    def L(chi, reaction, F, g):
        u = torch_solve.lifted_femsolve(
            chi, F, g, mesh.edof, KE, mask, 5, 4, reaction=reaction, ME=ME, rtol=1e-13
        )
        return (w * u).sum()

    assert torch.autograd.gradcheck(L, leaves, eps=1e-6, atol=1e-7, rtol=1e-5)


def test_gradient_matches_finite_differences_through_the_v_cycle():
    """At a mesh that coarsens, along random directions, at the production tolerance's
    order of accuracy."""
    rng = np.random.default_rng(8)
    nelx, nely = 60, 30
    mesh = Mesh(nelx, nely)
    plate = mesh.x == 0
    mask = mesh.mask(plate)
    chi = _t(_log_uniform(rng, nelx * nely, 1e3)).requires_grad_()
    g = _t(plate * 1.0).requires_grad_()
    reaction = _t(np.full(nelx * nely, 0.05))
    w = _t(rng.uniform(0, 1, mesh.ndof))
    F = _t(np.zeros(mesh.ndof))

    def L(chi, g):
        T = torch_solve.lifted_femsolve(
            chi,
            F,
            g,
            mesh.edof,
            KE,
            mask,
            nelx,
            nely,
            reaction=reaction,
            ME=ME,
            rtol=1e-12,
        )
        return (w * T).sum()

    L(chi, g).backward()
    for leaf, grad in ((chi, chi.grad), (g, g.grad)):
        d = _t(rng.standard_normal(leaf.shape)) * (leaf.detach().abs() + 1e-3)
        eps = 1e-6
        with torch.no_grad():
            fd = (
                L(
                    chi + eps * d if leaf is chi else chi,
                    g + eps * d if leaf is g else g,
                )
                - L(
                    chi - eps * d if leaf is chi else chi,
                    g - eps * d if leaf is g else g,
                )
            ) / (2 * eps)
        assert float((grad * d).sum()) == pytest.approx(float(fd), rel=1e-6)


# -- 1D analytic profiles ------------------------------------------------------------


def test_drain_with_zero_flux_gives_the_cosh_profile():
    """`T = 1` at x = 0, zero flux elsewhere, uniform `chi`: `T = cosh(k (L - x)) /
    cosh(k L)` with `k = sqrt(alpha)`, to second order in h."""
    L, k = 1.0, 3.0

    def error(n):
        mesh = Mesh(n, 2, L / n)
        plate = mesh.x == 0
        reaction = np.full(2 * n, k**2 * mesh.h**2)
        T = _solve(
            mesh, np.ones(2 * n), np.zeros(mesh.ndof), plate * 1.0, plate, reaction
        )
        exact = np.cosh(k * (L - mesh.x)) / np.cosh(k * L)
        return np.abs(T.numpy() - exact).max()

    errors = [error(n) for n in (20, 40, 80)]
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    assert errors[-1] < 1e-3 and np.all(rates > 1.9), (errors, rates)


def test_laplace_between_two_plates_is_linear():
    """Exact at any resolution, since a bilinear element reproduces a linear field."""
    mesh = Mesh(17, 5)
    fixed = (mesh.x == 0) | (mesh.x == 17)
    rng = np.random.default_rng(9)
    chi = np.full(17 * 5, 2.5)
    t = _solve(mesh, chi, np.zeros(mesh.ndof), mesh.x / 17 * fixed, fixed).numpy()
    np.testing.assert_allclose(t, mesh.x / 17, atol=1e-11)
    del rng
