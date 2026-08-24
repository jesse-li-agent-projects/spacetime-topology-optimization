"""Tests for sttopt.constraints against MATLAB fixtures and finite-difference checks.

See conftest.py/conventions.md for fixture format and tolerance policy.

xTilde/dx reconstruction (needed for the density-sensitivity chain rule, constraints (1)
and (4) only -- (2)/(3) are pure functions of tPhys and need no Heaviside term): the MATLAB
main loop only updates `xTilde` at the *end* of each iteration (`xTilde(:) = (H*s(:))./Hs`,
`s` = the density half of that iteration's raw MMA output `xmma`), so the `xTilde` used by
iteration `k`'s constraint block is:
  - k=0: the initial uniform field `xTilde = volfrac` (`generate_fixtures.m`'s init, before
    the loop runs at all).
  - k>=1: `H @ s_{k-1}.flatten('F') / Hs`, where `s_{k-1}` is the density half of
    `xmma_all[:, k-1]` (`mma.mat`, saved for *every* loop iteration, unlike the single-shot
    `xval_1`/`fval_1`/etc. "loop==1" snapshot also in that fixture).
This makes all 3 fixture iterations reconstructable (not just iteration 0): `xmma_all`
gives an exact, non-shaky path to `xTilde`/`dx` at every iteration, so no subset-testing
fallback was needed.
"""

import numpy as np

import sttopt.compliance as compliance
import sttopt.constraints as constraints
import sttopt.filters as filters
from conftest import assert_close, load_fixture

NELX, NELY = 7, 5
RMIN = LRMIN = 2
BETA, ETA = 1.0, 0.5
ROU = 10.0


def _reconstruct_xTilde_traj(xmma_all, H, Hs, nelx, nely, volfrac, nloop):
    """xTilde at the start of each iteration k=0..nloop-1; see module docstring."""
    nel = nelx * nely
    traj = [np.full((nely, nelx), volfrac)]
    for k in range(nloop - 1):
        xTilde = (H @ xmma_all[:nel, k]) / Hs
        traj.append(xTilde.reshape((nely, nelx), order="F"))
    return traj


def test_constraints_match_fixture():
    fx = load_fixture("constraints")
    e2e = load_fixture("e2e")
    mma = load_fixture("mma")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])
    nStage, volfrac, tfield = int(fx["nStage"]), float(fx["volfrac"]), int(fx["tfield"])
    nloop = e2e["xPhys_traj"].shape[2] - 1
    nel = nelx * nely
    # Nei = 1:nely (0-indexed: 0..nely-1), not the tfield==1 singleton
    assert tfield == 3

    H, Hs = filters.density_filter(nelx, nely, RMIN)
    L = filters.continuity_filter(nelx, nely, LRMIN)
    Nei = np.arange(nely)

    xTilde_traj = _reconstruct_xTilde_traj(
        mma["xmma_all"], H, Hs, nelx, nely, volfrac, nloop
    )

    for k in range(nloop):
        xPhys = e2e["xPhys_traj"][:, :, k]
        tPhys = e2e["tPhys_traj"][:, :, k]
        xTilde = xTilde_traj[k]
        dx = filters.heaviside_projection_derivative(xTilde, BETA, ETA)

        fval_all = fx["fval_all"][:, k]
        dfdx_all = fx["dfdx_all"][:, :, k]

        # (1) global volume
        fval, dfx, dft = constraints.global_volume_fraction(xPhys, dx, H, Hs, volfrac)
        assert_close(fval, fval_all[0], tier="algebraic")
        assert_close(dfx, dfdx_all[0, :nel], tier="algebraic")
        assert_close(dft, dfdx_all[0, nel:], tier="algebraic")

        # (2) time-field continuity
        fval, dfx, dft = constraints.time_field_continuity(tPhys, L, H, Hs)
        assert_close(fval, fval_all[1], tier="algebraic")
        assert_close(dfx, dfdx_all[1, :nel], tier="algebraic")
        assert_close(dft, dfdx_all[1, nel:], tier="algebraic")

        # (3) start-point
        fval, dfx, dft = constraints.start_point(tPhys, Nei, H, Hs)
        assert_close(fval, fval_all[2 : 2 + nely], tier="algebraic")
        assert_close(dfx, dfdx_all[2 : 2 + nely, :nel], tier="algebraic")
        assert_close(dft, dfdx_all[2 : 2 + nely, nel:], tier="algebraic")

        # (4) per-stage volume, upper/lower interleaved starting at row 2+nely
        base = 2 + nely
        for i in range(1, nStage + 1):
            fu, fl, dfx, dft = constraints.stage_volume_bounds(
                xPhys, tPhys, dx, H, Hs, i, nStage, volfrac, ROU
            )
            row_u, row_l = base + 2 * (i - 1), base + 2 * (i - 1) + 1
            assert_close(fu, fval_all[row_u], tier="algebraic")
            assert_close(fl, fval_all[row_l], tier="algebraic")
            assert_close(dfx, dfdx_all[row_u, :nel], tier="algebraic")
            assert_close(dft, dfdx_all[row_u, nel:], tier="algebraic")
            assert_close(-dfx, dfdx_all[row_l, :nel], tier="algebraic")
            assert_close(-dft, dfdx_all[row_l, nel:], tier="algebraic")


# --- Finite-difference internal-consistency checks (pure Python, no MATLAB fixture) ---
# Full chain: raw design variable -> density filter (H/Hs) -> [Heaviside, density only] ->
# constraint value. Small asymmetric mesh per conventions.md.


def _xPhys_of(x_raw, H, Hs, nely, nelx, beta, eta):
    xTilde = (H @ x_raw.flatten(order="F") / Hs).reshape((nely, nelx), order="F")
    return filters.heaviside_projection(xTilde, beta, eta), xTilde


def _tPhys_of(t_raw, H, Hs, nely, nelx):
    return (H @ t_raw.flatten(order="F") / Hs).reshape((nely, nelx), order="F")


def test_stage_volume_bounds_fd():
    nelx, nely = 6, 4
    volfrac = 0.4
    stage, nStage = 2, 3
    h = 1e-6
    H, Hs = filters.density_filter(nelx, nely, RMIN)

    rng = np.random.default_rng(0)
    for seed in range(5):
        x_raw = rng.uniform(0.2, 0.8, size=(nely, nelx))
        t_raw = rng.uniform(0.05, 0.95, size=(nely, nelx))

        xPhys, xTilde = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
        tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
        dx = filters.heaviside_projection_derivative(xTilde, BETA, ETA)

        _, _, dfx, dft = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, H, Hs, stage, nStage, volfrac, ROU
        )
        # Guard against a vacuously-passing FD check (e.g. `ft` saturated near 0/1
        # everywhere, making `dft` ~atol-sized regardless of correctness).
        assert np.abs(dfx).max() > 1e-3
        assert np.abs(dft).max() > 1e-3

        def fval_upper_of(x_raw, t_raw):
            xPhys, _ = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
            tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
            fu, _, _, _ = constraints.stage_volume_bounds(
                xPhys, tPhys, dx, H, Hs, stage, nStage, volfrac, ROU
            )
            return fu

        fd_x = np.zeros(nely * nelx)
        fd_t = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            j, i = e % nely, e // nely
            xp, xm = x_raw.copy(), x_raw.copy()
            xp[j, i] += h
            xm[j, i] -= h
            fd_x[e] = (fval_upper_of(xp, t_raw) - fval_upper_of(xm, t_raw)) / (2 * h)

            tp, tm = t_raw.copy(), t_raw.copy()
            tp[j, i] += h
            tm[j, i] -= h
            fd_t[e] = (fval_upper_of(x_raw, tp) - fval_upper_of(x_raw, tm)) / (2 * h)

        np.testing.assert_allclose(dfx, fd_x, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(dft, fd_t, rtol=1e-4, atol=1e-6)


def test_global_volume_fraction_fd():
    nelx, nely = 6, 4
    volfrac = 0.4
    h = 1e-6
    H, Hs = filters.density_filter(nelx, nely, RMIN)

    rng = np.random.default_rng(2)
    for seed in range(5):
        x_raw = rng.uniform(0.2, 0.8, size=(nely, nelx))

        xPhys, xTilde = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
        dx = filters.heaviside_projection_derivative(xTilde, BETA, ETA)

        _, dfx, dft = constraints.global_volume_fraction(xPhys, dx, H, Hs, volfrac)
        np.testing.assert_allclose(dft, 0.0)
        assert np.abs(dfx).max() > 1e-3  # guard against a vacuous atol-only pass

        def fval_of(x_raw):
            xPhys, _ = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
            fval, _, _ = constraints.global_volume_fraction(xPhys, dx, H, Hs, volfrac)
            return fval

        fd_x = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            j, i = e % nely, e // nely
            xp, xm = x_raw.copy(), x_raw.copy()
            xp[j, i] += h
            xm[j, i] -= h
            fd_x[e] = (fval_of(xp) - fval_of(xm)) / (2 * h)

        np.testing.assert_allclose(dfx, fd_x, rtol=1e-5, atol=1e-8)


def test_start_point_fd():
    nelx, nely = 6, 4
    h = 1e-6
    H, Hs = filters.density_filter(nelx, nely, RMIN)
    Nei = np.arange(nely)

    rng = np.random.default_rng(3)
    for seed in range(5):
        t_raw = rng.uniform(0.05, 0.95, size=(nely, nelx))
        tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)

        fval, dfx, dft = constraints.start_point(tPhys, Nei, H, Hs)
        np.testing.assert_allclose(dfx, 0.0)
        assert np.abs(dft).max() > 1e-3  # guard against a vacuous atol-only pass

        def fval_of(t_raw):
            tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
            fval, _, _ = constraints.start_point(tPhys, Nei, H, Hs)
            return fval

        fd_t = np.zeros((len(Nei), nely * nelx))
        for e in range(nely * nelx):
            j, i = e % nely, e // nely
            tp, tm = t_raw.copy(), t_raw.copy()
            tp[j, i] += h
            tm[j, i] -= h
            fd_t[:, e] = (fval_of(tp) - fval_of(tm)) / (2 * h)

        np.testing.assert_allclose(dft, fd_t, rtol=1e-5, atol=1e-8)


def test_time_field_continuity_fd():
    nelx, nely = 6, 4
    h = 1e-6
    H, Hs = filters.density_filter(nelx, nely, RMIN)
    L = filters.continuity_filter(nelx, nely, LRMIN)

    rng = np.random.default_rng(1)
    for seed in range(5):
        t_raw = rng.uniform(0.05, 0.95, size=(nely, nelx))
        tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)

        _, dfx, dft = constraints.time_field_continuity(tPhys, L, H, Hs)
        np.testing.assert_allclose(dfx, 0.0)
        assert np.abs(dft).max() > 1e-3  # guard against a vacuous atol-only pass

        def fval_of(t_raw):
            tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
            fval, _, _ = constraints.time_field_continuity(tPhys, L, H, Hs)
            return fval

        fd_t = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            j, i = e % nely, e // nely
            tp, tm = t_raw.copy(), t_raw.copy()
            tp[j, i] += h
            tm[j, i] -= h
            fd_t[e] = (fval_of(tp) - fval_of(tm)) / (2 * h)

        np.testing.assert_allclose(dft, fd_t, rtol=1e-5, atol=1e-8)


# --- Layout-based valid/invalid checks -----------------------------------------------
# These test known good/bad *designs* against first-principles expectations (not the
# exact regularization constants baked into the constraint formulas, which are
# implementation detail and shouldn't be encoded into a well-designed test).


def test_global_volume_fraction_uniform_layouts():
    """Uniform xPhys at 0, 1, and volfrac itself are exact closed-form cases regardless
    of volfrac's value."""
    nely, nelx = 4, 6
    dx = np.ones((nely, nelx))  # dfx isn't checked here, only fval
    H, Hs = filters.density_filter(nelx, nely, RMIN)
    for volfrac in (0.2, 0.35, 0.6):
        for c, expected in (
            (0.0, -1.0),
            (1.0, 1.0 / volfrac - 1.0),
            (volfrac, 0.0),
        ):
            xPhys = np.full((nely, nelx), c)
            fval, _, _ = constraints.global_volume_fraction(xPhys, dx, H, Hs, volfrac)
            np.testing.assert_allclose(fval, expected, rtol=1e-9, atol=1e-9)


def _element_grid(nely, nelx):
    """(row, col) grid matching the (nely, nelx) array shape, i.e. row=y, col=x."""
    jj, ii = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    return jj, ii


def test_time_field_continuity_layouts():
    """Smooth monotonic time fields should satisfy the continuity constraint far better
    than a checkerboard -- and a checkerboard confined to a small local patch inside an
    otherwise-smooth field should still register as clearly worse than the smooth
    baseline, not get washed out by the surrounding smooth region."""
    nely, nelx = 4, 6
    H, Hs = filters.density_filter(nelx, nely, RMIN)
    L = filters.continuity_filter(nelx, nely, LRMIN)
    jj, ii = _element_grid(nely, nelx)

    def fval_of(tPhys):
        fval, _, _ = constraints.time_field_continuity(tPhys, L, H, Hs)
        return fval

    grad_x = ii / (nelx - 1)
    grad_y = jj / (nely - 1)
    grad_diag = (ii + jj) / (nelx - 1 + nely - 1)
    checkerboard = ((ii + jj) % 2).astype(float)

    fval_smooth = max(fval_of(g) for g in (grad_x, grad_y, grad_diag))
    fval_checker = fval_of(checkerboard)

    local = grad_x.copy()
    local[1:3, 1:3] = ((ii[1:3, 1:3] + jj[1:3, 1:3]) % 2).astype(float)
    fval_local = fval_of(local)

    assert fval_smooth < fval_local < fval_checker


def _distance_field(Nei, nely, nelx):
    """Normalized Manhattan-distance-to-nearest-Nei-cell field, in element-index space
    (0-indexed, Fortran/column-major numbering matching `conventions.md`)."""
    nel = nely * nelx
    e = np.arange(nel)
    j_of_e, i_of_e = e % nely, e // nely
    coords = np.stack([j_of_e, i_of_e], axis=1)
    nei_coords = coords[Nei]
    dists = np.min(
        np.abs(coords[:, None, :] - nei_coords[None, :, :]).sum(axis=2), axis=1
    )
    maxdist = dists.max()
    field = dists / maxdist if maxdist > 0 else dists
    return field.reshape((nely, nelx), order="F")


def test_start_point_layouts():
    """Nei shapes beyond the two existing tfield conventions (single corner, whole
    column) should work generically. A tPhys built as distance-to-nearest-Nei-cell is a
    physically sensible print-time field that satisfies the constraint at Nei; the same
    kind of field anchored elsewhere should violate it."""
    nely, nelx = 4, 6
    nel = nely * nelx
    H, Hs = filters.density_filter(nelx, nely, RMIN)

    cases = [
        (np.array([0]), np.array([nel - 1])),  # single corner vs. opposite corner
        (np.array([nel // 2]), np.array([0])),  # interior singleton, no tfield analog
        (np.array([0, nel - 1]), np.array([nel // 2])),  # scattered opposite corners
    ]
    for Nei, wrong_anchor in cases:
        tPhys_valid = _distance_field(Nei, nely, nelx)
        fval, dfx, dft = constraints.start_point(tPhys_valid, Nei, H, Hs)
        assert fval.shape == (len(Nei),)
        assert dfx.shape == (len(Nei), nel)
        assert dft.shape == (len(Nei), nel)
        assert np.all(fval < 1e-6)  # satisfied: Nei cells sit at the field's minimum

        tPhys_invalid = _distance_field(wrong_anchor, nely, nelx)
        fval_wrong, _, _ = constraints.start_point(tPhys_invalid, Nei, H, Hs)
        assert np.all(fval_wrong > 0.05)  # violated: Nei isn't where printing starts


def test_stage_volume_bounds_xPhys_weighting():
    """The per-stage budget must actually weight by xPhys, not just threshold tPhys:
    concentrating density in the already-printed half vs. the not-yet-printed half
    should give very different `deposited` amounts for the *same* time field, and an
    even split should land close to satisfying the budget."""
    nely, nelx = 4, 6
    volfrac, nStage, stage = 0.4, 2, 1  # ti = 0.5, budget = 0.5
    H, Hs = filters.density_filter(nelx, nely, RMIN)
    dx = np.ones((nely, nelx))

    tPhys = np.empty((nely, nelx))
    tPhys[:, :3] = 0.1  # clearly printed by ti=0.5
    tPhys[:, 3:] = 0.9  # clearly not yet printed

    def fval_upper_of(xPhys):
        fu, _, _, _ = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, H, Hs, stage, nStage, volfrac, ROU
        )
        return fu

    xPhys_printed_heavy = np.where(tPhys < 0.5, 0.8, 0.1)
    xPhys_notyet_heavy = np.where(tPhys < 0.5, 0.1, 0.8)
    xPhys_equal = np.full((nely, nelx), volfrac)

    fu_printed = fval_upper_of(xPhys_printed_heavy)
    fu_equal = fval_upper_of(xPhys_equal)
    fu_notyet = fval_upper_of(xPhys_notyet_heavy)

    assert fu_printed > fu_equal > fu_notyet
    assert fu_printed > 0.05  # over-budget: violated
    assert abs(fu_equal) < 0.05  # even split roughly meets the budget
    assert fu_notyet < -0.05  # well under budget: comfortably satisfied


def test_stage_volume_bounds_rou_sharpness():
    """Elements sitting just past the stage cutoff should count for less as `rou`
    sharpens (a sharper mask approaches a hard 0/1 cutoff); softer `rou` gives them
    more partial credit, making the constraint looser."""
    nely, nelx = 4, 6
    volfrac, nStage, stage = 0.4, 2, 1  # ti = 0.5
    H, Hs = filters.density_filter(nelx, nely, RMIN)
    dx = np.ones((nely, nelx))
    xPhys = np.full((nely, nelx), volfrac)

    tPhys = np.empty((nely, nelx))
    tPhys[:, :2] = 0.1  # clearly printed
    tPhys[:, 2:4] = 0.55  # just past the ti=0.5 cutoff -- should be excluded
    tPhys[:, 4:] = 0.9  # clearly not yet printed

    def fval_upper_of(rou):
        fu, _, _, _ = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, H, Hs, stage, nStage, volfrac, rou
        )
        return fu

    fus = [fval_upper_of(rou) for rou in (1, 3, 10, 30, 100)]
    assert all(a > b for a, b in zip(fus, fus[1:]))  # strictly looser as rou softens
