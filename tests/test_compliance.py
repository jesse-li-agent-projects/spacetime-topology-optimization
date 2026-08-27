"""Tests for sttopt.compliance against golden-regression fixtures, finite-difference
checks, and closed-form elasticity solutions.

See conftest.py/conventions.md for fixture format and tolerance policy.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import sttopt.compliance as compliance
import sttopt.fem as fem
import sttopt.gravity as gravity
import sttopt.torch_util as torch_util
from conftest import assert_close, load_fixture_npz, point_load_problem, tt, tti


def test_whole_compliance_matches_fixture():
    fx = load_fixture_npz("compliance")
    e2e = load_fixture_npz("e2e")
    nelx, nely, nloop = int(fx["nelx"]), int(fx["nely"]), e2e["xPhys_traj"].shape[2] - 1
    Emin, Emax, penal = 1e-9, 1.0, 3

    KE = tt(fem.plane_stress_KE(nu=0.3))
    edofMat = tti(fem.element_dof_map(nelx, nely))
    F, freedofs, ndof = point_load_problem(nelx, nely)
    F, freedofs = tt(F), tti(freedofs)

    for k in range(nloop):
        xPhys = tt(e2e["xPhys_traj"][:, :, k])
        c, dcx = compliance.whole_compliance(
            xPhys, KE, edofMat, Emin, Emax, penal, freedofs, F, ndof
        )
        assert_close(c, fx["c_whole_all"][k], tier="solved")
        assert_close(dcx, fx["dcx_whole_all"][:, :, k], tier="solved")


def test_gravity_compliance_matches_fixture():
    fx = load_fixture_npz("compliance")
    e2e = load_fixture_npz("e2e")
    grav = load_fixture_npz("gravity")
    nelx, nely, nStage = int(fx["nelx"]), int(fx["nely"]), int(fx["nStage"])
    nloop = e2e["xPhys_traj"].shape[2] - 1
    Emin, Emax, penal = 1e-9, 1.0, 3
    beta_t = 10.0  # beta_t=10 fixed for loop=1..3: loop%30==0 never triggers (see generate_fixtures.py)

    KE = tt(fem.plane_stress_KE(nu=0.3))
    edofMat = tti(fem.element_dof_map(nelx, nely))
    _, freedofs, ndof = point_load_problem(nelx, nely)
    freedofs = tti(freedofs)
    C = torch_util.csr_to_tensor(sp.csr_matrix(grav["C"]), "cpu", torch.float64)

    tP = np.linspace(0, 1, nStage + 1)

    for k in range(nloop):
        xPhys = tt(e2e["xPhys_traj"][:, :, k])
        tPhys = tt(e2e["tPhys_traj"][:, :, k])
        for i in range(nStage):
            ti = tP[i + 1]
            c, dcx, dct = compliance.gravity_compliance(
                xPhys,
                tPhys,
                KE,
                edofMat,
                Emin,
                Emax,
                penal,
                ti,
                C,
                beta_t,
                freedofs,
                ndof,
            )
            assert_close(c, fx["c_grav_all"][k, i], tier="solved")
            assert_close(dcx, fx["dcx_grav_all"][:, i, k], tier="solved")
            assert_close(dct, fx["dct_grav_all"][:, i, k], tier="solved")


# --- Closed-form elasticity checks (pure Python, no MATLAB fixture) ---
#
# Node/dof numbering and the consistent-nodal-force helper mirror test_fem.py's
# closed-form patch tests; see that file's docstrings for the underlying elasticity
# reasoning (Q4 elements are exact for constant-strain fields).


def _x_dofs(nodes) -> np.ndarray:
    return 2 * np.asarray(nodes)


def _y_dofs(nodes) -> np.ndarray:
    return 2 * np.asarray(nodes) + 1


def _dofs(nodes) -> np.ndarray:
    return np.stack([_x_dofs(nodes), _y_dofs(nodes)], axis=-1).ravel()


def _add_edge_traction(F, nodes, traction):
    tx, ty = traction
    n = len(nodes)
    for i, node in enumerate(nodes):
        weight = 0.5 if i in (0, n - 1) else 1.0
        F[2 * node] += weight * tx
        F[2 * node + 1] += weight * ty


@pytest.mark.parametrize("Emax", [1.0, 3.7])
@pytest.mark.parametrize("t", [1.0, 2.5])
def test_whole_compliance_axial_bar_patch(t, Emax):
    """whole_compliance's `c` against the exact closed-form bar-in-tension compliance.

    Roller BCs plus a uniform edge traction reproduce a bar in uniaxial tension --
    constant strain, exact for a Q4 element (test_fem.py's `test_uniaxial_tension_patch`
    checks the same setup's displacement field directly). Unlike that test, this checks
    compliance.py's own `ce`/`simp`/`sum(...)` path against elasticity theory, rather
    than only against the MATLAB fixture (matches-fixture tests above) or against itself
    (FD tests below) -- neither of which would catch a bug shared with the MATLAB source.

    `t`/`Emax` are swept off 1.0 so the closed form's `t**2` and `1/Emax` are actually
    exercised; at unit values a missing factor of either is invisible (see PR #15).
    """
    nelx, nely = 7, 5  # asymmetric, per conventions.md
    Emin, penal = 1e-9 * Emax, 3
    ndof = 2 * (nelx + 1) * (nely + 1)
    xPhys = np.ones((nely, nelx))
    KE = fem.plane_stress_KE(nu=0.3)
    edofMat = fem.element_dof_map(nelx, nely)

    # Rollers: pin the x=0 and y=0 lines, matching the tension field's zeros there.
    nodes = fem.node_grid(nelx, nely)
    fixeddofs = np.concatenate([_x_dofs(nodes[:, 0]), _y_dofs(nodes[0, :])])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    F = np.zeros(ndof)
    _add_edge_traction(F, nodes[:, -1], (t, 0.0))

    c, dcx = compliance.whole_compliance(
        tt(xPhys), tt(KE), tti(edofMat), Emin, Emax, penal, tti(freedofs), tt(F), ndof
    )

    # Unit elements/thickness: length L = nelx, cross-section area A = nely * 1.
    # xPhys == 1 everywhere -> E == Emax regardless of penal (Emin cancels exactly).
    L, A = nelx, nely
    c_analytic = t**2 * L * A / Emax  # = P^2 L / (E A), with resultant P = t * A
    np.testing.assert_allclose(c, c_analytic, rtol=1e-9)


def _cantilever_beam_compliance(
    nely: int, P: float = 1.0, Emax: float = 1.0
) -> tuple[float, float]:
    """whole_compliance's `c` (and the Timoshenko-beam prediction) for a slender
    cantilever, at a given through-thickness element count.

    Full clamp at x=0, tip load P distributed as consistent nodal forces across the
    x=nelx edge (a concentrated corner load would be singular under mesh refinement,
    and off-centroid, so it isn't a fair comparison). c == F^T U == P * tip deflection,
    so `c_analytic` is quadratic in P -- it has units of work, not of deflection.

    Bending (Euler-Bernoulli) plus the Timoshenko shear correction. That correction is
    small at L/H == 20 but load-bearing: dropping it, or using a shear coefficient other
    than k == 5/6, fails the tests below (see PR #15).

    L/H == 20 is held fixed as `nely` grows, so this isn't exact even in the limit:
    the full clamp is stiffer than beam theory's idealized support, and standard
    full-integration Q4 elements are mildly over-stiff in bending (shear locking) --
    both push c_FEM below c_analytic by an amount that shrinks with mesh refinement
    (see `test_whole_compliance_cantilever_beam_converges`) but never exactly vanishes.
    """
    nelx = 20 * nely  # L/H == 20
    nu = 0.3
    Emin, penal = 1e-9 * Emax, 3
    ndof = 2 * (nelx + 1) * (nely + 1)
    xPhys = np.ones((nely, nelx))
    KE = fem.plane_stress_KE(nu)
    edofMat = fem.element_dof_map(nelx, nely)

    nodes = fem.node_grid(nelx, nely)
    fixeddofs = _dofs(nodes[:, 0])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    F = np.zeros(ndof)
    _add_edge_traction(F, nodes[:, -1], (0.0, -P / nely))

    c, dcx = compliance.whole_compliance(
        tt(xPhys), tt(KE), tti(edofMat), Emin, Emax, penal, tti(freedofs), tt(F), ndof
    )

    L, H = nelx, nely
    I = H**3 / 12
    delta_bend = P * L**3 / (3 * Emax * I)
    shear_ratio = (3 / 5) * (1 + nu) * (H / L) ** 2  # Timoshenko, rectangular k = 5/6
    c_analytic = P * delta_bend * (1 + shear_ratio)
    return c, c_analytic


# nely -> max acceptable |c_FEM - c_analytic| / c_analytic, each with headroom above the
# measured error at that resolution (-2.03%, -0.58%, -0.21%, -0.12%) -- not guessed.
_CANTILEVER_MAX_PCT_ERROR = {5: 0.025, 10: 0.008, 20: 0.003, 40: 0.0016}


@pytest.mark.parametrize("nely", sorted(_CANTILEVER_MAX_PCT_ERROR))
def test_whole_compliance_cantilever_beam(nely):
    c, c_analytic = _cantilever_beam_compliance(nely)
    assert (
        c < c_analytic
    ), "expected a stiffer-clamp/shear-locking bias below c_analytic"
    np.testing.assert_allclose(c, c_analytic, rtol=_CANTILEVER_MAX_PCT_ERROR[nely])


def test_whole_compliance_cantilever_beam_converges():
    """The FEM/beam-theory gap should shrink monotonically as the mesh refines.

    A stronger, less arbitrary check than any single resolution's tolerance in
    `test_whole_compliance_cantilever_beam`: a real discretization error shrinks with
    refinement, so a bug that merely happens to land within one level's tolerance band
    would still fail this test.
    """
    errors = [
        abs(c - c_analytic) / c_analytic
        for c, c_analytic in (
            _cantilever_beam_compliance(nely)
            for nely in sorted(_CANTILEVER_MAX_PCT_ERROR)
        )
    ]
    assert errors == sorted(errors, reverse=True), f"errors not decreasing: {errors}"


@pytest.mark.parametrize("P, Emax", [(2.5, 1.0), (1.0, 3.7), (2.5, 3.7), (0.4, 0.25)])
def test_whole_compliance_cantilever_beam_dimensional_consistency(P, Emax):
    """`c_analytic` must track `c` when the load and modulus leave 1.0.

    `c` is quadratic in the load and inverse in the modulus, so a closed form short a
    factor of either still agrees at P == Emax == 1.0 -- which is exactly how a missing
    `P` survived the tolerance and convergence tests above (PR #15). Comparing the
    *relative* error across (P, Emax) pins the scaling without re-deriving the formula.
    """
    # rtol is set by sparse-solve roundoff across the differently-scaled stiffness
    # matrices (~1e-8), not by physics: a missing P or Emax factor is an O(1) miss.
    c0, a0 = _cantilever_beam_compliance(10)
    c, a = _cantilever_beam_compliance(10, P=P, Emax=Emax)
    np.testing.assert_allclose((c - a) / a, (c0 - a0) / a0, rtol=1e-6)
    np.testing.assert_allclose(c, c0 * P**2 / Emax, rtol=1e-6)


def test_whole_compliance_scales_as_load_squared_and_inverse_modulus():
    """`c == F^T K^-1 F` is exactly quadratic in load and inverse in modulus.

    Checked directly on `whole_compliance` at a non-uniform density, so it is
    independent of any beam/bar closed form and of `penal` -- this is the scaling law
    the analytic tests above lean on.
    """
    nelx, nely = 6, 4
    penal = 3
    KE = tt(fem.plane_stress_KE(nu=0.3))
    edofMat = tti(fem.element_dof_map(nelx, nely))
    F, freedofs, ndof = point_load_problem(nelx, nely)
    F, freedofs = tt(F), tti(freedofs)
    xPhys = tt(_random_field(np.random.default_rng(3), nely, nelx))

    def c_of(alpha, Emax):
        c, _ = compliance.whole_compliance(
            xPhys, KE, edofMat, 1e-9 * Emax, Emax, penal, freedofs, alpha * F, ndof
        )
        return c

    c0 = c_of(1.0, 1.0)
    for alpha in (2.0, 0.25, 3.7):
        np.testing.assert_allclose(c_of(alpha, 1.0), c0 * alpha**2, rtol=1e-9)
    for Emax in (2.0, 0.25, 3.7):
        np.testing.assert_allclose(c_of(1.0, Emax), c0 / Emax, rtol=1e-9)


def _gravity_cantilever_compliance(
    nelx: int,
    nely: int,
    Emax: float,
    tPhys: np.ndarray | None = None,
    ti: float = 0.5,
    beta_t: float = 10.0,
    w_scale: float = 1.0,
) -> float:
    """`gravity_compliance`'s `c` for a fully-clamped, self-weight-loaded cantilever.

    `tPhys`/`ti`/`beta_t` default to a fully built structure: `tPhys` all-zero makes
    every element exactly "active" for any `ti` > 0 (`time_mask`'s sigmoid is exactly
    1, not just approximately -- `num` cancels to 0 in that case), so this isolates
    the self-weight load path with no sigmoid-softening error. `w_scale` rescales
    `gravity.gravity_load_matrix`'s load; see
    `test_gravity_compliance_partial_build_matches_truncated_mesh` for why.
    """
    Emin, penal = 1e-9 * Emax, 3
    ndof = 2 * (nelx + 1) * (nely + 1)
    xPhys = np.ones((nely, nelx))
    if tPhys is None:
        tPhys = np.zeros((nely, nelx))
    KE = fem.plane_stress_KE(nu=0.3)
    edofMat = fem.element_dof_map(nelx, nely)

    fixeddofs = _dofs(fem.node_grid(nelx, nely)[:, 0])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    C = gravity.gravity_load_matrix(nelx, nely) * w_scale

    c, _, _ = compliance.gravity_compliance(
        tt(xPhys),
        tt(tPhys),
        tt(KE),
        tti(edofMat),
        Emin,
        Emax,
        penal,
        ti,
        torch_util.csr_to_tensor(C, "cpu", torch.float64),
        beta_t,
        tti(freedofs),
        ndof,
    )
    return c


def _self_weight_cantilever_analytic(nelx: int, nely: int, Emax: float) -> float:
    """Euler-Bernoulli compliance of a fully built, uniformly self-loaded cantilever.

    Shear is deliberately left uncorrected here, unlike `_cantilever_beam_compliance`'s
    Timoshenko term for the tip-load case: at this test's L/H == 10 slenderness, the
    FEM/bending-only gap plateaus at shear's fixed relative contribution instead of
    shrinking to zero under mesh refinement (measured -1.74%, +0.59%, +1.22%, +1.38%
    at nely = 4, 8, 16, 32) -- which is why the tolerance below is a flat bound rather
    than a per-resolution shrinking one. Adding the UDL shear term
    `(4/3)(1+nu)(H/L)**2` does not rescue a convergence check: the residual plateaus
    near -0.30% rather than vanishing (see PR #16).
    """
    # `gravity.py`'s `fe = 1/(nelx*nely)` normalizes TOTAL self-weight to 1 at full
    # density regardless of mesh size, so the load per unit length is w = 1/nelx.
    L, H, w = nelx, nely, 1.0 / nelx
    I = H**3 / 12
    return w**2 * L**5 / (20 * Emax * I)


# headroom above the measured 1.74% (see helper docstring)
_SELF_WEIGHT_MAX_PCT_ERROR = 0.025


@pytest.mark.parametrize("Emax", [1.0, 2.5])
@pytest.mark.parametrize("nely", [4, 8, 16])
def test_gravity_compliance_self_weight_cantilever(nely, Emax):
    # L/H == 10: reasonably slender, but (per helper docstring) not slender enough for
    # the uncorrected shear bias to vanish -- absorbed into the flat tolerance instead.
    nelx = 10 * nely
    c = _gravity_cantilever_compliance(nelx, nely, Emax)
    c_analytic = _self_weight_cantilever_analytic(nelx, nely, Emax)
    np.testing.assert_allclose(c, c_analytic, rtol=_SELF_WEIGHT_MAX_PCT_ERROR)


def test_gravity_compliance_scales_as_1_over_Emax():
    """Self-weight load doesn't depend on `Emax` (only stiffness does), so `c` should
    scale as exactly `1/Emax` -- a stronger, exact check than the beam-theory
    tolerance above, isolating the `Emax` path from bending-theory approximation
    error."""
    nelx, nely = 80, 8
    c1 = _gravity_cantilever_compliance(nelx, nely, Emax=1.0)
    c2 = _gravity_cantilever_compliance(nelx, nely, Emax=3.7)
    np.testing.assert_allclose(c1, c2 * 3.7, rtol=1e-8)


def test_gravity_compliance_partial_build_matches_truncated_mesh():
    """`gravity_compliance`'s `tPhys`/`ti` path: a cantilever built up column-by-column
    along its length, stopped partway through the build, should match the self-weight
    compliance of a *shorter* cantilever built to full density over just the built
    portion.

    Compares two independent FEM solves rather than layering a second closed-form
    approximation on top of the beam-theory one above: the full mesh's build-order
    field (`tPhys` linear in x) with `ti` set between column `m - 1`'s and column
    `m`'s `tPhys` -- away from either, so a sharp `beta_t` leaves the sigmoid's
    transition zone with negligible weight on any column -- against an actual
    `m`-column mesh built at full density.

    `gravity.gravity_load_matrix` normalizes total weight by its own `nelx * nely`, so
    at equal `nely` the truncated mesh's per-element weight is `nelx / m` times the
    full mesh's; `w_scale = m / nelx` cancels that. Skipping the rescale would make
    the truncated mesh's `c` larger by exactly `(nelx / m) ** 2`, compliance being
    quadratic in load -- a load-convention mismatch, not a bug. (At unequal `nely` the
    matching factor would be `(m * nely_trunc) / (nelx * nely_full)`.)
    """
    nely, nelx, m = 8, 80, 40
    # sharp: sigmoid transition width is far below the 1/nelx column spacing
    beta_t = 1000.0
    tPhys = np.tile(np.arange(nelx) / nelx, (nely, 1))
    ti = (m - 0.5) / nelx

    c_partial = _gravity_cantilever_compliance(
        nelx, nely, 2.5, tPhys=tPhys, ti=ti, beta_t=beta_t
    )
    c_truncated = _gravity_cantilever_compliance(m, nely, 2.5, w_scale=m / nelx)
    np.testing.assert_allclose(c_partial, c_truncated, rtol=1e-4)


# --- Finite-difference internal-consistency checks (pure Python, no MATLAB fixture) ---


def _random_field(rng, nely, nelx):
    """Asymmetric density/time fields, kept away from the [0,1] boundary (near-void/near-solid
    densities make the stiffness matrix ill-conditioned; see `well_conditioned` below for the
    stronger safeguard the gravity FD test needs)."""
    return rng.uniform(0.2, 0.8, size=(nely, nelx))


def test_whole_compliance_fd_dcx():
    nelx, nely = 6, 4
    Emin, Emax, penal = 1e-9, 1.0, 3
    KE = tt(fem.plane_stress_KE(nu=0.3))
    edofMat = tti(fem.element_dof_map(nelx, nely))
    F, freedofs, ndof = point_load_problem(nelx, nely)
    F, freedofs = tt(F), tti(freedofs)
    h = 1e-6

    rng = np.random.default_rng(0)
    for seed in range(5):
        xPhys = _random_field(rng, nely, nelx)
        _, dcx = compliance.whole_compliance(
            tt(xPhys), KE, edofMat, Emin, Emax, penal, freedofs, F, ndof
        )
        fd = np.zeros_like(xPhys)
        for j in range(nely):
            for i in range(nelx):
                xp = xPhys.copy()
                xp[j, i] += h
                xm = xPhys.copy()
                xm[j, i] -= h
                cp, _ = compliance.whole_compliance(
                    tt(xp), KE, edofMat, Emin, Emax, penal, freedofs, F, ndof
                )
                cm, _ = compliance.whole_compliance(
                    tt(xm), KE, edofMat, Emin, Emax, penal, freedofs, F, ndof
                )
                fd[j, i] = (cp - cm) / (2 * h)
        np.testing.assert_allclose(dcx, fd, rtol=1e-4, atol=1e-6)


def test_gravity_compliance_fd_dcx_and_dct():
    nelx, nely = 6, 4
    Emin, Emax, penal = 1e-9, 1.0, 3
    beta_t = 10.0
    ti = 0.5
    KE_np = fem.plane_stress_KE(nu=0.3)
    edofMat_np = fem.element_dof_map(nelx, nely)
    KE, edofMat = tt(KE_np), tti(edofMat_np)
    _, freedofs_np, ndof = point_load_problem(nelx, nely)
    freedofs = tti(freedofs_np)

    C_np = gravity.gravity_load_matrix(nelx, nely)
    C = torch_util.csr_to_tensor(C_np, "cpu", torch.float64)
    h = 1e-4

    # Self-weight-only loading (no external point load) on a small random mesh can
    # land near a mechanism -- K_free's condition number spans 1e4 to 1e8 across
    # random draws below, and FD noise blows up (verified by an h-convergence sweep)
    # long before the analytic gradient does. Reject ill-conditioned draws rather than
    # loosen tolerances to paper over them: this is a numerical-conditioning artifact
    # of the FD probe, not evidence about the analytic formula.
    def well_conditioned(xPhys, tPhys):
        ft = compliance.time_mask(tt(tPhys), ti, beta_t).numpy()
        K = fem.assemble_stiffness(
            KE_np, xPhys * ft, Emin, Emax, penal, edofMat_np, ndof
        )
        Kfree = K[np.ix_(freedofs_np, freedofs_np)].toarray()
        return np.linalg.cond(Kfree) < 1e5

    rng = np.random.default_rng(1)
    accepted = 0
    tries = 0
    max_tries = 200
    while accepted < 5:
        assert (
            tries < max_tries
        ), f"couldn't find 5 well-conditioned draws in {max_tries} tries"
        xPhys = _random_field(rng, nely, nelx)
        tPhys = _random_field(rng, nely, nelx)
        tries += 1
        if not well_conditioned(xPhys, tPhys):
            continue
        accepted += 1

        _, dcx, dct = compliance.gravity_compliance(
            tt(xPhys),
            tt(tPhys),
            KE,
            edofMat,
            Emin,
            Emax,
            penal,
            ti,
            C,
            beta_t,
            freedofs,
            ndof,
        )

        fd_x = np.zeros(nely * nelx)
        fd_t = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            # C-order element -> (row, col), per conventions.md
            j, i = e // nelx, e % nelx

            xp = xPhys.copy()
            xp[j, i] += h
            xm = xPhys.copy()
            xm[j, i] -= h
            cp, _, _ = compliance.gravity_compliance(
                tt(xp),
                tt(tPhys),
                KE,
                edofMat,
                Emin,
                Emax,
                penal,
                ti,
                C,
                beta_t,
                freedofs,
                ndof,
            )
            cm, _, _ = compliance.gravity_compliance(
                tt(xm),
                tt(tPhys),
                KE,
                edofMat,
                Emin,
                Emax,
                penal,
                ti,
                C,
                beta_t,
                freedofs,
                ndof,
            )
            fd_x[e] = (cp - cm) / (2 * h)

            tp = tPhys.copy()
            tp[j, i] += h
            tm = tPhys.copy()
            tm[j, i] -= h
            cp, _, _ = compliance.gravity_compliance(
                tt(xPhys),
                tt(tp),
                KE,
                edofMat,
                Emin,
                Emax,
                penal,
                ti,
                C,
                beta_t,
                freedofs,
                ndof,
            )
            cm, _, _ = compliance.gravity_compliance(
                tt(xPhys),
                tt(tm),
                KE,
                edofMat,
                Emin,
                Emax,
                penal,
                ti,
                C,
                beta_t,
                freedofs,
                ndof,
            )
            fd_t[e] = (cp - cm) / (2 * h)

        np.testing.assert_allclose(dcx, fd_x, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(dct, fd_t, rtol=1e-3, atol=1e-6)


def test_time_mask_derivative_matches_fd():
    rng = np.random.default_rng(2)
    tPhys = tt(rng.uniform(0.05, 0.95, size=(4, 6)))
    ti, beta_t = 0.4, 10.0
    h = 1e-6
    analytic = compliance.time_mask_derivative(tPhys, ti, beta_t)
    fd = (
        compliance.time_mask(tPhys + h, ti, beta_t)
        - compliance.time_mask(tPhys - h, ti, beta_t)
    ) / (2 * h)
    np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=1e-8)
