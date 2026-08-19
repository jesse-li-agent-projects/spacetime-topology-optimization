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
