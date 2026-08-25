"""Tests for sttopt.conductivity against MATLAB fixtures and finite-difference checks.

See conftest.py/conventions.md for fixture format and tolerance policy, and
test_constraints.py's module docstring for the xTilde-trajectory-reconstruction
pattern reused here for the hotspot-constraint fixture test.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import sttopt.conductivity as conductivity
import sttopt.filters as filters
from conftest import assert_close, load_fixture, load_fixture_npz

NELX, NELY = 7, 5
RMIN_COND = 3
BETA, ETA = 1.0, 0.5
P, Q, R, ROUF = 25, 3, 0.05, 100
TCR = 0.8


def _reconstruct_xTilde_traj(xmma_all, H, Hs, nelx, nely, volfrac, nloop):
    """xTilde at the start of each iteration k=0..nloop-1 (see test_constraints.py)."""
    nel = nelx * nely
    traj = [np.full((nely, nelx), volfrac)]
    for k in range(nloop - 1):
        xTilde = (H @ xmma_all[:nel, k]) / Hs
        traj.append(xTilde.reshape((nely, nelx), order="F"))
    return traj


def test_we_equals_w_el():
    """Trap 3: WE{i}(j) (the weight neighbor E1(j) assigns back to i) always equals
    w_el{i}(j) itself, since the neighbor-weight structure is symmetric. Confirmed
    directly against the fixture before relying on the simplification elsewhere.
    """
    fx = load_fixture("conductivity_neighbors")
    w_lookup = {
        (int(a), int(b)): float(v)
        for a, b, v in zip(fx["coo_e1"], fx["coo_e2"], fx["coo_w"])
    }
    we_lookup = {
        (int(a), int(b)): float(v)
        for a, b, v in zip(fx["coo_we1"], fx["coo_we2"], fx["coo_we"])
    }
    assert set(w_lookup) == set(we_lookup)
    for key, w_val in w_lookup.items():
        assert we_lookup[key] == w_val


def test_neighbor_weights_match_fixture():
    fx = load_fixture("conductivity_neighbors")
    nelx, nely, rmin_cond = int(fx["nelx"]), int(fx["nely"]), float(fx["rmin_cond"])
    assert (nelx, nely, rmin_cond) == (NELX, NELY, RMIN_COND)

    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    got = {(int(a), int(b)): float(v) for a, b, v in zip(e1, e2, w)}
    expected = {
        (int(a) - 1, int(b) - 1): float(v)
        for a, b, v in zip(fx["coo_e1"], fx["coo_e2"], fx["coo_w"])
    }
    assert set(got) == set(expected)
    for key, w_val in expected.items():
        assert_close(got[key], w_val, tier="algebraic")


def test_K_est_matches_fixture():
    fx = load_fixture("conductivity")
    nfx = load_fixture("conductivity_neighbors")
    e2e = load_fixture("e2e")
    nloop = e2e["xPhys_traj"].shape[2] - 1

    e1, e2, w = conductivity.neighbor_weights(NELX, NELY, RMIN_COND)
    # sanity: same neighbor structure as the fixture (also checked in its own test)
    assert e1.shape == nfx["coo_e1"].shape

    for k in range(nloop):
        xPhys = e2e["xPhys_traj"][:, :, k]
        tPhys = e2e["tPhys_traj"][:, :, k]
        K_est = conductivity.estimated_conductivity(xPhys, tPhys, e1, e2, w, Q, ROUF)
        assert_close(K_est, fx["K_est_all"][:, k], tier="algebraic")


def test_hotspot_constraint_matches_fixture():
    fx = load_fixture("conductivity")
    e2e = load_fixture("e2e")
    mma = load_fixture("mma")
    constraints_fx = load_fixture("constraints")
    nelx, nely = int(constraints_fx["nelx"]), int(constraints_fx["nely"])
    volfrac = float(constraints_fx["volfrac"])
    nloop = e2e["xPhys_traj"].shape[2] - 1
    assert (nelx, nely) == (NELX, NELY)

    factor_all = fx["factor_all"]
    assert np.all(factor_all == 1.0), "expected factor==1 for all 3 fixture iterations"

    RMIN = 2  # density filter radius used throughout the fixture harness
    H, Hs = filters.density_filter(nelx, nely, RMIN)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)

    xTilde_traj = _reconstruct_xTilde_traj(
        mma["xmma_all"], H, Hs, nelx, nely, volfrac, nloop
    )

    # The fixture's tPhys (seeded from tfield3, a corner-distance field, then MMA-perturbed)
    # never happens to put two DISTINCT elements at an exact tie -- confirmed here so the
    # report doesn't overclaim fixture coverage of the distinct-element-tie branch (that
    # branch is exercised only by the synthetic `_at_ties` test below). Corner-distance
    # fields (tfield 1/3) are tie-free by construction (irrational-ish Euclidean distances
    # rarely coincide), but tfield 2 (`timefield_edge`, a linear ramp constant down each
    # column) is NOT: at a realistic 180x60 mesh it produces structural off-diagonal ties
    # on ~5% of neighbor pairs, surviving density filtering -- so the distinct-element-tie
    # branch is far from a synthetic-only corner case for that timefield choice.
    for k in range(nloop):
        tflat = e2e["tPhys_traj"][:, :, k].flatten(order="F")
        off_diag = e1 != e2
        assert np.sum(tflat[e1[off_diag]] == tflat[e2[off_diag]]) == 0

    for k in range(nloop):
        xPhys = e2e["xPhys_traj"][:, :, k]
        tPhys = e2e["tPhys_traj"][:, :, k]
        xTilde = xTilde_traj[k]
        dx = filters.heaviside_projection_derivative(xTilde, BETA, ETA)
        factor = float(factor_all[k])

        fval, df1, dt1, _, _ = conductivity.hotspot_constraint(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, TCR, P, Q, R, ROUF
        )

        # numer/tru_max are algebraically recoverable from fval (fval = factor*numer/Tcr - 1);
        # with factor==1 here this is one independent check (not three), since numer/tru_max
        # coincide and both invert the same relation -- df1/dt1 are the real second check.
        numer = (fval + 1) * TCR / factor
        tru_max = factor * numer

        assert_close(numer, fx["numer_all"][k], tier="algebraic")
        assert_close(tru_max, fx["tru_max_all"][k], tier="algebraic")
        assert_close(df1, fx["df1_all"][:, k], tier="algebraic")
        assert_close(dt1, fx["dt1_all"][:, k], tier="algebraic")


def test_K_est_matches_golden_scenes():
    """Regression fixture, not a MATLAB cross-check: 4 synthetic overhang scenes (bottom_up,
    left_right, sharp_sigmoid, soft_sigmoid -- different build directions / sigmoid
    sharpness) at a 168x120 grid, visually validated via conductivity_viz.tmp.py before
    their K_est output was frozen as golden values here.

    Only xPhys/tPhys/rouf/K_est are stored in the fixture -- e1/e2/w are recomputed from
    (nelx, nely, rmin_cond) rather than persisted, since at this resolution the COO arrays
    are ~744 MB (pair count scales ~rmin_cond**2 * nelx*nely) versus ~224 KB for the rest.
    """
    fx = load_fixture_npz("conductivity_golden_scenes")
    nelx, nely, rmin_cond = int(fx["nelx"]), int(fx["nely"]), float(fx["rmin_cond"])
    q = int(fx["q"])
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)

    for name in ["bottom_up", "left_right", "sharp_sigmoid", "soft_sigmoid"]:
        xPhys = fx[f"{name}_xPhys"]
        tPhys = fx[f"{name}_tPhys"]
        rouf = float(fx[f"{name}_rouf"])
        K_est = conductivity.estimated_conductivity(xPhys, tPhys, e1, e2, w, q, rouf)
        assert_close(K_est, fx[f"{name}_K_est"], tier="algebraic")


# --- Finite-difference internal-consistency checks (pure Python, no MATLAB fixture) ---


def _xPhys_of(x_raw, H, Hs, nely, nelx, beta, eta):
    xTilde = (H @ x_raw.flatten(order="F") / Hs).reshape((nely, nelx), order="F")
    return filters.heaviside_projection(xTilde, beta, eta), xTilde


def _tPhys_of(t_raw, H, Hs, nely, nelx):
    return (H @ t_raw.flatten(order="F") / Hs).reshape((nely, nelx), order="F")


@pytest.mark.parametrize("factor", [1.0, 1.7])
def test_hotspot_constraint_fd_density(factor):
    """df1 (density sensitivity) vs. central-difference perturbation -- no known trap.
    `factor` is swept past the fixture's only-ever-observed value of 1.0: `fval`/`df1`
    are exactly affine/linear in `factor`, so this is a real, independent consistency
    check of the `factor` wiring, not just a repeat of the `factor==1` fixture test.
    """
    nelx, nely = 6, 4
    h = 1e-6
    RMIN = 2
    Tcr = 0.8
    H, Hs = filters.density_filter(nelx, nely, RMIN)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)

    rng = np.random.default_rng(0)
    for seed in range(5):
        x_raw = rng.uniform(0.2, 0.8, size=(nely, nelx))
        t_raw = rng.uniform(0.05, 0.95, size=(nely, nelx))

        xPhys, xTilde = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
        tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
        dx = filters.heaviside_projection_derivative(xTilde, BETA, ETA)

        _, df1, _, _, _ = conductivity.hotspot_constraint(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
        )
        # Guard well above the assert_allclose atol below, so this can't pass vacuously.
        assert np.abs(df1).max() > 1e-3

        def fval_of(x_raw, t_raw):
            xPhys, xTilde = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
            tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
            dx = filters.heaviside_projection_derivative(xTilde, BETA, ETA)
            fval, _, _, _, _ = conductivity.hotspot_constraint(
                xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
            )
            return fval

        fd_x = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            j, i = e % nely, e // nely
            xp, xm = x_raw.copy(), x_raw.copy()
            xp[j, i] += h
            xm[j, i] -= h
            fd_x[e] = (fval_of(xp, t_raw) - fval_of(xm, t_raw)) / (2 * h)

        print(
            f"factor={factor} seed={seed} max|df1|={np.abs(df1).max():.3e} "
            f"max|df1-fd|={np.abs(df1 - fd_x).max():.3e}"
        )
        np.testing.assert_allclose(df1, fd_x, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("factor", [1.0, 1.7])
def test_hotspot_constraint_fd_time_generic(factor):
    """dt1 (time sensitivity) vs. central-difference on a generic random field, where
    (verified empirically below) no two distinct elements' filtered `tPhys` values
    coincide, so the only exact tie any pair ever hits is each element's own trivial
    self-comparison (`tPhys[a] == tPhys[a]`). That structural self-tie's true
    contribution to the gradient is genuinely 0 (it's `sigmoid(rouf*(t_a - t_a)) =
    sigmoid(0)`, a constant independent of `t_a`, not a numerically-coincidental
    near-tie that a perturbation could shift) -- so `DFT_aa=0` there isn't an
    approximation, and FD matches analytic tightly, everywhere, for a field like this.
    See `test_hotspot_constraint_fd_time_at_ties` for the genuine (distinct-element)
    tie case, which -- since the `a == b` fix in `_pairwise_sigmoid_terms` -- now
    matches FD here too, rather than exhibiting the bounded discrepancy it used to.
    """
    nelx, nely = 6, 4
    RMIN = 2
    h = 1e-6
    Tcr = 0.8
    H, Hs = filters.density_filter(nelx, nely, RMIN)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)

    rng = np.random.default_rng(1)
    x_raw = rng.uniform(0.2, 0.8, size=(nely, nelx))
    t_raw = rng.uniform(0.05, 0.95, size=(nely, nelx))

    xPhys, xTilde = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
    tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
    dx = filters.heaviside_projection_derivative(xTilde, BETA, ETA)

    # Confirm the premise: no off-diagonal (distinct-element) exact ties in this field.
    tflat = tPhys.flatten(order="F")
    off_diag = e1 != e2
    assert np.sum(tflat[e1[off_diag]] == tflat[e2[off_diag]]) == 0

    _, _, dt1, _, _ = conductivity.hotspot_constraint(
        xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
    )
    assert np.abs(dt1).max() > 1e-3  # guard well above the atol below

    def fval_of(t_raw):
        tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
        fval, _, _, _, _ = conductivity.hotspot_constraint(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
        )
        return fval

    fd_t = np.zeros(nely * nelx)
    for e in range(nely * nelx):
        j, i = e % nely, e // nely
        tp, tm = t_raw.copy(), t_raw.copy()
        tp[j, i] += h
        tm[j, i] -= h
        fd_t[e] = (fval_of(tp) - fval_of(tm)) / (2 * h)

    # Tight everywhere: max observed |dt1 - fd| ~1e-9 at factor=1 (pure FD truncation
    # noise, scales with factor), no bounded tie-point residual -- confirms the
    # self-term handling is correct, not merely "close enough on average".
    print(
        f"factor={factor} max|dt1|={np.abs(dt1).max():.3e} "
        f"max|dt1-fd|={np.abs(dt1 - fd_t).max():.3e}"
    )
    np.testing.assert_allclose(dt1, fd_t, rtol=1e-4, atol=1e-7 * factor)


def test_hotspot_constraint_fd_time_at_ties():
    """dt1 vs. central-difference at a genuine tie between two DISTINCT elements (as
    opposed to the always-present, always-exact self-tie handled by the sibling
    `_generic` test above): forced via a uniform `tPhys` and an identity density filter
    (`H=I`, `Hs=1`) so `tPhys` can be perturbed directly and stays exactly uniform
    bit-for-bit (routing a uniform raw field through the *real* density filter, as in
    the sibling tests, does NOT reliably preserve exact equality -- H's matrix-multiply
    accumulation order and Hs's row-sum reduction can round differently in the last ULP
    -- so this test sidesteps the filter rather than relying on that).

    `_pairwise_sigmoid_terms` used to zero `DFT` on any value-tie `t[a] == t[b]`,
    ported verbatim from a MATLAB bug (`if TPhys(N_ele(o))==ti`, conflating "value tie"
    with "self-pair"; see conventions.md). That's now fixed to check `a == b` by index,
    so a genuine tie between distinct elements gets the ordinary `rouf/4` derivative
    instead of 0, and FD matches analytic tightly here too, not just in the generic case.

    Note: the MATLAB fixture's own `tPhys` trajectory never hits a distinct-element tie
    (checked in `test_hotspot_constraint_matches_fixture`), so this branch is validated
    only by this synthetic test, not cross-checked against MATLAB output. That's a
    property of the fixture's corner-distance timefield, though, not of ties being rare
    in general: `timefield_edge` (a linear ramp, constant down each column) produces
    ~5% of neighbor pairs as structural exact ties at a realistic 180x60 mesh, surviving
    density filtering -- so this isn't a hypothetical edge case for that timefield choice.
    Before the fix, the bug wasn't just "wrong at a measure-zero set": DFT was ~rouf/4 for
    near-ties and exactly 0 at exact ties, so dt1 was discontinuous in t -- a hole in the
    gradient field that an optimizer driving a symmetric design toward equal print times
    would walk straight into.
    """
    nelx, nely = 6, 4
    nel = nelx * nely
    # identity filter: tPhys IS the raw variable
    H, Hs = sp.eye(nel, format="csr"), np.ones(nel)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)
    factor, Tcr = 1.0, 0.8

    rng = np.random.default_rng(2)
    xPhys = rng.uniform(0.2, 0.8, size=(nely, nelx))
    tPhys = np.full((nely, nelx), 0.5)  # uniform -> every pair is exactly tied
    dx = np.ones((nely, nelx))  # unused by dt1; only scales df1

    # Confirm the premise: literally every neighbor pair (including all off-diagonal
    # ones) is an exact tie under a uniform field.
    tflat = tPhys.flatten(order="F")
    off_diag = e1 != e2
    assert np.sum(tflat[e1[off_diag]] == tflat[e2[off_diag]]) == off_diag.sum()

    _, _, dt1, _, _ = conductivity.hotspot_constraint(
        xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
    )
    # Guard well above the atol below: with the fix, ties no longer force dt1 to 0.
    assert np.abs(dt1).max() > 1e-3

    def fval_of(tPhys):
        fval, _, _, _, _ = conductivity.hotspot_constraint(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
        )
        return fval

    h = 1e-6
    fd_t = np.zeros(nely * nelx)
    for e in range(nely * nelx):
        j, i = e % nely, e // nely
        tp, tm = tPhys.copy(), tPhys.copy()
        tp[j, i] += h
        tm[j, i] -= h
        fd_t[e] = (fval_of(tp) - fval_of(tm)) / (2 * h)

    print(
        f"max|dt1|={np.abs(dt1).max():.3e} max|dt1-fd|={np.abs(dt1 - fd_t).max():.3e}"
    )
    np.testing.assert_allclose(dt1, fd_t, rtol=1e-4, atol=1e-7 * factor)
