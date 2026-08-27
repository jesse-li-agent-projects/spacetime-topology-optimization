"""Tests for sttopt.constraints against golden-regression fixtures and
finite-difference checks. See conftest.py/conventions.md for fixture format and
tolerance policy.
"""

import numpy as np
import torch

import sttopt.constraints as constraints
import sttopt.filters as filters
import sttopt.torch_util as torch_util
from conftest import assert_close, load_fixture_npz, tt, tti

NELX, NELY = 7, 5
RMIN = LRMIN = 2
BETA, ETA = 1.0, 0.5
ROU = 10.0


def _tensor_filter(nelx, nely, rmin):
    """`filters.density_filter`, converted to the tensor form `constraints.py` now
    expects (the filter *builder* itself stays NumPy/SciPy, per the plan)."""
    H, Hs = filters.density_filter(nelx, nely, rmin)
    return torch_util.csr_to_tensor(H, "cpu", torch.float64), tt(Hs)


def test_constraints_match_fixture():
    fx = load_fixture_npz("constraints")
    e2e = load_fixture_npz("e2e")
    nelx, nely = int(fx["nelx"]), int(fx["nely"])
    nStage, volfrac, tfield = int(fx["nStage"]), float(fx["volfrac"]), int(fx["tfield"])
    nloop = e2e["xPhys_traj"].shape[2] - 1
    nel = nelx * nely
    # Nei = 0..nely-1 (0-indexed), not the tfield==1 singleton
    assert tfield == 3

    H, Hs = _tensor_filter(nelx, nely, RMIN)
    L = torch_util.csr_to_tensor(
        filters.continuity_filter(nelx, nely, LRMIN), "cpu", torch.float64
    )
    # column 0 (all rows), per conventions.md's C-order element enumeration
    Nei = tti(np.arange(nely) * nelx)

    for k in range(nloop):
        xPhys = tt(e2e["xPhys_traj"][:, :, k])
        tPhys = tt(e2e["tPhys_traj"][:, :, k])
        dx = tt(e2e["dx_all"][:, :, k])

        fval_all = fx["fval_all"][:, k]
        dfdx_all = fx["dfdx_all"][:, :, k]
        dfdx_x = dfdx_all[:, :nel]
        dfdx_t = dfdx_all[:, nel:]

        # (1) global volume
        fval, dfx, dft = constraints.global_volume_fraction(xPhys, dx, H, Hs, volfrac)
        assert_close(fval, fval_all[0], tier="algebraic")
        assert_close(dfx, dfdx_x[0], tier="algebraic")
        assert_close(dft, dfdx_t[0], tier="algebraic")

        # (2) time-field continuity
        fval, dfx, dft = constraints.time_field_continuity(tPhys, L, H, Hs)
        assert_close(fval, fval_all[1], tier="algebraic")
        assert_close(dfx, dfdx_x[1], tier="algebraic")
        assert_close(dft, dfdx_t[1], tier="algebraic")

        # (3) start-point
        fval, dfx, dft = constraints.start_point(tPhys, Nei, H, Hs)
        assert_close(fval, fval_all[2 : 2 + nely], tier="algebraic")
        assert_close(dfx, dfdx_x[2 : 2 + nely], tier="algebraic")
        assert_close(dft, dfdx_t[2 : 2 + nely], tier="algebraic")

        # (4) per-stage volume, upper/lower interleaved starting at row 2+nely
        base = 2 + nely
        for i, t_stage in enumerate(np.linspace(0, 1, nStage + 1)[1:]):
            fu, fl, dfx, dft = constraints.stage_volume_bounds(
                xPhys, tPhys, dx, H, Hs, t_stage, volfrac, ROU
            )
            row_u, row_l = base + 2 * i, base + 2 * i + 1
            assert_close(fu, fval_all[row_u], tier="algebraic")
            assert_close(fl, fval_all[row_l], tier="algebraic")
            assert_close(dfx, dfdx_x[row_u], tier="algebraic")
            assert_close(dft, dfdx_t[row_u], tier="algebraic")
            assert_close(-dfx, dfdx_x[row_l], tier="algebraic")
            assert_close(-dft, dfdx_t[row_l], tier="algebraic")


# --- Finite-difference internal-consistency checks (pure Python, no MATLAB fixture) ---
# Full chain: raw design variable -> density filter (H/Hs) -> [Heaviside, density only] ->
# constraint value. Small asymmetric mesh per conventions.md.


def _xPhys_of(x_raw, H, Hs, nely, nelx, beta, eta):
    xTilde = (H @ x_raw.flatten() / Hs).reshape((nely, nelx))
    return filters.heaviside_projection(xTilde, beta, eta), xTilde


def _tPhys_of(t_raw, H, Hs, nely, nelx):
    return (H @ t_raw.flatten() / Hs).reshape((nely, nelx))


def test_stage_volume_bounds_fd():
    nelx, nely = 6, 4
    volfrac = 0.4
    t_stage = 2 / 3  # stage 2 of 3
    h = 1e-6
    H, Hs = _tensor_filter(nelx, nely, RMIN)

    rng = np.random.default_rng(0)
    for seed in range(5):
        x_raw = tt(rng.uniform(0.2, 0.8, size=(nely, nelx)))
        t_raw = tt(rng.uniform(0.05, 0.95, size=(nely, nelx)))

        xPhys, xTilde = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
        tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
        dx = filters.heaviside_projection_derivative(xTilde, BETA, ETA)

        _, _, dfx, dft = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, H, Hs, t_stage, volfrac, ROU
        )
        # Guard against a vacuously-passing FD check (e.g. `ft` saturated near 0/1
        # everywhere, making `dft` ~atol-sized regardless of correctness).
        assert dfx.abs().max() > 1e-3
        assert dft.abs().max() > 1e-3

        def fval_upper_of(x_raw, t_raw):
            xPhys, _ = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
            tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
            fu, _, _, _ = constraints.stage_volume_bounds(
                xPhys, tPhys, dx, H, Hs, t_stage, volfrac, ROU
            )
            return fu

        def fval_lower_of(x_raw, t_raw):
            xPhys, _ = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
            tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
            _, fl, _, _ = constraints.stage_volume_bounds(
                xPhys, tPhys, dx, H, Hs, t_stage, volfrac, ROU
            )
            return fl

        fd_x_upper = np.zeros(nely * nelx)
        fd_t_upper = np.zeros(nely * nelx)
        fd_x_lower = np.zeros(nely * nelx)
        fd_t_lower = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            j, i = e // nelx, e % nelx
            xp, xm = x_raw.clone(), x_raw.clone()
            xp[j, i] += h
            xm[j, i] -= h
            fd_x_upper[e] = (fval_upper_of(xp, t_raw) - fval_upper_of(xm, t_raw)) / (
                2 * h
            )
            fd_x_lower[e] = (fval_lower_of(xp, t_raw) - fval_lower_of(xm, t_raw)) / (
                2 * h
            )

            tp, tm = t_raw.clone(), t_raw.clone()
            tp[j, i] += h
            tm[j, i] -= h
            fd_t_upper[e] = (fval_upper_of(x_raw, tp) - fval_upper_of(x_raw, tm)) / (
                2 * h
            )
            fd_t_lower[e] = (fval_lower_of(x_raw, tp) - fval_lower_of(x_raw, tm)) / (
                2 * h
            )

        np.testing.assert_allclose(dfx, fd_x_upper, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(dft, fd_t_upper, rtol=1e-4, atol=1e-6)
        # fval_lower's sensitivity rows are documented as exactly -dfx, -dft --
        # check that against its own independent FD pass, not just algebraically.
        np.testing.assert_allclose(-dfx, fd_x_lower, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(-dft, fd_t_lower, rtol=1e-4, atol=1e-6)


def test_global_volume_fraction_fd():
    nelx, nely = 6, 4
    volfrac = 0.4
    h = 1e-6
    H, Hs = _tensor_filter(nelx, nely, RMIN)

    rng = np.random.default_rng(2)
    for seed in range(5):
        x_raw = tt(rng.uniform(0.2, 0.8, size=(nely, nelx)))

        xPhys, xTilde = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
        dx = filters.heaviside_projection_derivative(xTilde, BETA, ETA)

        _, dfx, dft = constraints.global_volume_fraction(xPhys, dx, H, Hs, volfrac)
        np.testing.assert_allclose(dft, 0.0)
        assert dfx.abs().max() > 1e-3  # guard against a vacuous atol-only pass

        def fval_of(x_raw):
            xPhys, _ = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
            fval, _, _ = constraints.global_volume_fraction(xPhys, dx, H, Hs, volfrac)
            return fval

        fd_x = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            j, i = e // nelx, e % nelx
            xp, xm = x_raw.clone(), x_raw.clone()
            xp[j, i] += h
            xm[j, i] -= h
            fd_x[e] = (fval_of(xp) - fval_of(xm)) / (2 * h)

        np.testing.assert_allclose(dfx, fd_x, rtol=1e-5, atol=1e-8)


def test_start_point_fd():
    nelx, nely = 6, 4
    h = 1e-6
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    Nei = tti(np.arange(nely) * nelx)

    rng = np.random.default_rng(3)
    for seed in range(5):
        t_raw = tt(rng.uniform(0.05, 0.95, size=(nely, nelx)))
        tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)

        fval, dfx, dft = constraints.start_point(tPhys, Nei, H, Hs)
        np.testing.assert_allclose(dfx, 0.0)
        assert dft.abs().max() > 1e-3  # guard against a vacuous atol-only pass

        def fval_of(t_raw):
            tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
            fval, _, _ = constraints.start_point(tPhys, Nei, H, Hs)
            return fval

        fd_t = np.zeros((len(Nei), nely * nelx))
        for e in range(nely * nelx):
            j, i = e // nelx, e % nelx
            tp, tm = t_raw.clone(), t_raw.clone()
            tp[j, i] += h
            tm[j, i] -= h
            fd_t[:, e] = (fval_of(tp) - fval_of(tm)) / (2 * h)

        np.testing.assert_allclose(dft, fd_t, rtol=1e-5, atol=1e-8)


def test_time_field_continuity_fd():
    nelx, nely = 6, 4
    h = 1e-6
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    L = torch_util.csr_to_tensor(
        filters.continuity_filter(nelx, nely, LRMIN), "cpu", torch.float64
    )

    rng = np.random.default_rng(1)
    for seed in range(5):
        t_raw = tt(rng.uniform(0.05, 0.95, size=(nely, nelx)))
        tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)

        _, dfx, dft = constraints.time_field_continuity(tPhys, L, H, Hs)
        np.testing.assert_allclose(dfx, 0.0)
        assert dft.abs().max() > 1e-3  # guard against a vacuous atol-only pass

        def fval_of(t_raw):
            tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
            fval, _, _ = constraints.time_field_continuity(tPhys, L, H, Hs)
            return fval

        fd_t = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            j, i = e // nelx, e % nelx
            tp, tm = t_raw.clone(), t_raw.clone()
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
    dx = torch.ones(nely, nelx, dtype=torch.float64)  # dfx isn't checked here
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    for volfrac in (0.2, 0.35, 0.6):
        for c, expected in (
            (0.0, -1.0),
            (1.0, 1.0 / volfrac - 1.0),
            (volfrac, 0.0),
        ):
            xPhys = torch.full((nely, nelx), c, dtype=torch.float64)
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
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    L = torch_util.csr_to_tensor(
        filters.continuity_filter(nelx, nely, LRMIN), "cpu", torch.float64
    )
    jj, ii = _element_grid(nely, nelx)

    def fval_of(tPhys):
        fval, _, _ = constraints.time_field_continuity(tt(tPhys), L, H, Hs)
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
    (0-indexed, C/row-major numbering matching `conventions.md`)."""
    nel = nely * nelx
    e = np.arange(nel)
    j_of_e, i_of_e = e // nelx, e % nelx
    coords = np.stack([j_of_e, i_of_e], axis=1)
    nei_coords = coords[Nei]
    dists = np.min(
        np.abs(coords[:, None, :] - nei_coords[None, :, :]).sum(axis=2), axis=1
    )
    maxdist = dists.max()
    field = dists / maxdist if maxdist > 0 else dists
    return field.reshape((nely, nelx))


def test_start_point_layouts():
    """Nei shapes beyond the two existing tfield conventions (single corner, whole
    column) should work generically. A tPhys built as distance-to-nearest-Nei-cell is a
    physically sensible print-time field that satisfies the constraint at Nei; the same
    kind of field anchored elsewhere should violate it."""
    nely, nelx = 4, 6
    nel = nely * nelx
    H, Hs = _tensor_filter(nelx, nely, RMIN)

    cases = [
        (np.array([0]), np.array([nel - 1])),  # single corner vs. opposite corner
        (np.array([nel // 2]), np.array([0])),  # interior singleton, no tfield analog
        (np.array([0, nel - 1]), np.array([nel // 2])),  # scattered opposite corners
    ]
    for Nei, wrong_anchor in cases:
        tPhys_valid = tt(_distance_field(Nei, nely, nelx))
        fval, dfx, dft = constraints.start_point(tPhys_valid, tti(Nei), H, Hs)
        assert fval.shape == (len(Nei),)
        assert dfx.shape == (len(Nei), nel)
        assert dft.shape == (len(Nei), nel)
        assert torch.all(fval < 1e-6)  # satisfied: Nei cells sit at the field's minimum

        tPhys_invalid = tt(_distance_field(wrong_anchor, nely, nelx))
        fval_wrong, _, _ = constraints.start_point(tPhys_invalid, tti(Nei), H, Hs)
        assert torch.all(fval_wrong > 0.05)  # violated: Nei isn't where printing starts


def test_stage_volume_bounds_xPhys_weighting():
    """The per-stage budget must actually weight by xPhys, not just threshold tPhys:
    concentrating density in the already-printed half vs. the not-yet-printed half
    should give very different `deposited` amounts for the *same* time field, and an
    even split should land close to satisfying the budget."""
    nely, nelx = 4, 6
    volfrac, t_stage = 0.4, 0.5  # halfway through the build
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    dx = torch.ones(nely, nelx, dtype=torch.float64)

    tPhys = np.empty((nely, nelx))
    tPhys[:, :3] = 0.1  # clearly printed by ti=0.5
    tPhys[:, 3:] = 0.9  # clearly not yet printed
    tPhys = tt(tPhys)

    def fval_upper_of(xPhys):
        fu, _, _, _ = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, H, Hs, t_stage, volfrac, ROU
        )
        return fu

    def fval_lower_of(xPhys):
        _, fl, _, _ = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, H, Hs, t_stage, volfrac, ROU
        )
        return fl

    xPhys_printed_heavy = torch.where(tPhys < 0.5, 0.8, 0.1)
    xPhys_notyet_heavy = torch.where(tPhys < 0.5, 0.1, 0.8)
    xPhys_equal = torch.full((nely, nelx), volfrac, dtype=torch.float64)

    fu_printed = fval_upper_of(xPhys_printed_heavy)
    fu_equal = fval_upper_of(xPhys_equal)
    fu_notyet = fval_upper_of(xPhys_notyet_heavy)

    assert fu_printed > fu_equal > fu_notyet
    assert fu_printed > 0.05  # over-budget: violated
    assert abs(fu_equal) < 0.05  # even split roughly meets the budget
    assert fu_notyet < -0.05  # well under budget: comfortably satisfied

    # Mirror image on fval_lower: deposited *below* budget violates the lower bound,
    # so the ordering (and which cases violate/satisfy) flips.
    fl_printed = fval_lower_of(xPhys_printed_heavy)
    fl_equal = fval_lower_of(xPhys_equal)
    fl_notyet = fval_lower_of(xPhys_notyet_heavy)

    assert fl_printed < fl_equal < fl_notyet
    assert fl_notyet > 0.05  # well under budget: violates the lower bound
    assert abs(fl_equal) < 0.05  # even split roughly meets the budget
    assert fl_printed < -0.05  # over-budget: comfortably satisfies the lower bound


def test_stage_volume_bounds_beta_t_sharpness():
    """Elements sitting just past the stage cutoff should count for less as `beta_t`
    sharpens (a sharper mask approaches a hard 0/1 cutoff); softer `beta_t` gives them
    more partial credit, making the constraint looser."""
    nely, nelx = 4, 6
    volfrac, t_stage = 0.4, 0.5  # halfway through the build
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    dx = torch.ones(nely, nelx, dtype=torch.float64)
    xPhys = torch.full((nely, nelx), volfrac, dtype=torch.float64)

    tPhys = np.empty((nely, nelx))
    tPhys[:, :2] = 0.1  # clearly printed
    tPhys[:, 2:4] = 0.55  # just past the ti=0.5 cutoff -- should be excluded
    tPhys[:, 4:] = 0.9  # clearly not yet printed
    tPhys = tt(tPhys)

    def fval_upper_of(beta_t):
        fu, _, _, _ = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, H, Hs, t_stage, volfrac, beta_t
        )
        return fu

    def fval_lower_of(beta_t):
        _, fl, _, _ = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, H, Hs, t_stage, volfrac, beta_t
        )
        return fl

    fus = [fval_upper_of(beta_t) for beta_t in (1, 3, 10, 30, 100)]
    assert all(a > b for a, b in zip(fus, fus[1:]))  # strictly looser as beta_t softens

    # Mirror image: fval_lower should also strictly loosen (increase) as beta_t softens.
    fls = [fval_lower_of(beta_t) for beta_t in (1, 3, 10, 30, 100)]
    assert all(a < b for a, b in zip(fls, fls[1:]))


def test_stage_volume_bounds_lower_has_slack_margin():
    """fval_lower must be a genuinely tighter constraint than -fval_upper, not just its
    negation -- MMA's one-sided inequalities need a slack margin between the two so an
    exact-equality deposited==budget solution isn't simultaneously "just satisfying" and
    "just violating" both bounds. Don't assume the margin's exact size (implementation
    detail); just check it's present and strictly positive."""
    nely, nelx = 4, 6
    volfrac, t_stage = 0.4, 0.5
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    dx = torch.ones(nely, nelx, dtype=torch.float64)

    rng = np.random.default_rng(4)
    xPhys = tt(rng.uniform(0.1, 0.9, size=(nely, nelx)))
    tPhys = tt(rng.uniform(0.05, 0.95, size=(nely, nelx)))

    fu, fl, _, _ = constraints.stage_volume_bounds(
        xPhys, tPhys, dx, H, Hs, t_stage, volfrac, ROU
    )
    assert fl < -fu - 1e-8


# --- Phase 3.4 (plans/torch_port_part2.md): autograd sensitivities against hand-derived --
#
# Each `..._value` function takes only `xPhys`/`tPhys`; these tests rebuild the same
# filter(+Heaviside, for density) chain `optimize.step` threads from raw `x`/`t`
# leaves, so `torch.autograd.grad` reproduces exactly the `H @ (... * dx / Hs)` rows
# the hand-derived functions return -- `algebraic` tier throughout (no FE solve).


def _filtered_leaves(nelx, nely, H, Hs, rng):
    """Fresh `x`/`t` leaves and their filtered (density: + Heaviside) fields, matching
    `optimize.step`'s construction."""
    x = tt(rng.uniform(0.1, 0.9, size=(nely, nelx))).requires_grad_(True)
    t = tt(rng.uniform(0.05, 0.95, size=(nely, nelx))).requires_grad_(True)
    xTilde = ((H @ x.flatten()) / Hs).reshape(nely, nelx)
    xPhys = filters.heaviside_projection(xTilde, BETA, ETA)
    tPhys = ((H @ t.flatten()) / Hs).reshape(nely, nelx)
    return x, t, xPhys, tPhys


def test_global_volume_fraction_value_matches_hand_derived():
    nelx, nely, volfrac = 6, 4, 0.4
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    rng = np.random.default_rng(50)
    x, t, xPhys, tPhys = _filtered_leaves(nelx, nely, H, Hs, rng)
    dx = filters.heaviside_projection_derivative(
        ((H @ x.detach().flatten()) / Hs).reshape(nely, nelx), BETA, ETA
    )

    fv_ref, dfx_ref, dft_ref = constraints.global_volume_fraction(
        xPhys.detach(), dx, H, Hs, volfrac
    )
    fv = constraints.global_volume_fraction_value(xPhys, volfrac)
    dfx, dft = torch.autograd.grad(fv, (x, t), allow_unused=True)

    assert_close(fv.detach(), fv_ref, tier="algebraic")
    assert_close(dfx.flatten(), dfx_ref, tier="algebraic")
    assert dft is None  # no time-field dependence, matching dft_ref's all-zero row
    assert torch.all(dft_ref == 0.0)


def test_time_field_continuity_value_matches_hand_derived():
    nelx, nely = 6, 4
    L = torch_util.csr_to_tensor(
        filters.continuity_filter(nelx, nely, LRMIN), "cpu", torch.float64
    )
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    rng = np.random.default_rng(51)
    x, t, xPhys, tPhys = _filtered_leaves(nelx, nely, H, Hs, rng)

    fv_ref, dfx_ref, dft_ref = constraints.time_field_continuity(
        tPhys.detach(), L, H, Hs
    )
    fv = constraints.time_field_continuity_value(tPhys, L)
    dfx, dft = torch.autograd.grad(fv, (x, t), allow_unused=True)

    assert_close(fv.detach(), fv_ref, tier="algebraic")
    assert dfx is None  # no density dependence, matching dfx_ref's all-zero row
    assert torch.all(dfx_ref == 0.0)
    assert_close(dft.flatten(), dft_ref, tier="algebraic")


def test_start_point_value_matches_hand_derived():
    nelx, nely = 6, 4
    Nei = torch.arange(nely) * nelx
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    rng = np.random.default_rng(52)
    x, t, xPhys, tPhys = _filtered_leaves(nelx, nely, H, Hs, rng)

    fv_ref, dfx_ref, dft_ref = constraints.start_point(tPhys.detach(), Nei, H, Hs)
    fv = constraints.start_point_value(tPhys, Nei)
    for k in range(len(Nei)):
        dfx, dft = torch.autograd.grad(
            fv[k], (x, t), retain_graph=True, allow_unused=True
        )
        assert_close(fv[k].detach(), fv_ref[k], tier="algebraic")
        assert dfx is None  # no density dependence, matching dfx_ref's all-zero row
        assert torch.all(dfx_ref[k] == 0.0)
        assert_close(dft.flatten(), dft_ref[k], tier="algebraic")


def test_stage_volume_bounds_value_matches_hand_derived():
    nelx, nely, volfrac, t_stage = 6, 4, 0.4, 0.5
    H, Hs = _tensor_filter(nelx, nely, RMIN)
    rng = np.random.default_rng(53)
    x, t, xPhys, tPhys = _filtered_leaves(nelx, nely, H, Hs, rng)
    dx = filters.heaviside_projection_derivative(
        ((H @ x.detach().flatten()) / Hs).reshape(nely, nelx), BETA, ETA
    )

    fu_ref, fl_ref, dfx_ref, dft_ref = constraints.stage_volume_bounds(
        xPhys.detach(), tPhys.detach(), dx, H, Hs, t_stage, volfrac, ROU
    )
    fu = constraints.stage_volume_bounds_value(xPhys, tPhys, t_stage, volfrac, ROU)
    dfx, dft = torch.autograd.grad(fu, (x, t))
    fl = -fu - 1.0e-5

    assert_close(fu.detach(), fu_ref, tier="algebraic")
    assert_close(fl.detach(), fl_ref, tier="algebraic")
    assert_close(dfx.flatten(), dfx_ref, tier="algebraic")
    assert_close(dft.flatten(), dft_ref, tier="algebraic")
    # The lower row's sensitivity is the upper's explicit negation, not a second
    # autograd call (plans/torch_port_part2.md Phase 3.4).
    assert_close(-dfx.flatten(), -dfx_ref, tier="algebraic")
    assert_close(-dft.flatten(), -dft_ref, tier="algebraic")
