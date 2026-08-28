"""Tests for sttopt.conductivity against golden-regression fixtures and
finite-difference checks. See conftest.py/conventions.md for fixture format and
tolerance policy.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import sttopt.conductivity as conductivity
import sttopt.filters as filters
import sttopt.torch_util as torch_util
import tests.reference.conductivity as conductivity_ref
from conftest import assert_close, load_fixture_npz, tt, tti

NELX, NELY = 7, 5
RMIN_COND = 3
BETA, ETA = 1.0, 0.5
P, Q, R, ROUF = 25, 3, 0.05, 100
TCR = 0.8


def _K_est(xPhys, tPhys, e1, e2, w, q, rouf):
    """`conductivity.estimated_conductivity`, wrapped to accept this file's plain-NumPy
    test inputs and hand back a NumPy array -- `conductivity.py` itself is torch now
    (Phase 3.2, plans/torch_port_part2.md), but the bulk of this file's assertions
    predate that and are cheaper to keep NumPy-side than to convert individually.
    """
    return (
        conductivity.estimated_conductivity(
            tt(xPhys), tt(tPhys), tti(e1), tti(e2), tt(w), q, rouf
        )
        .detach()
        .numpy()
    )


def _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, p, q, r, rouf):
    """`conductivity.hotspot_constraint`, wrapped the same way as `_K_est` above."""
    H_t = H if torch.is_tensor(H) else torch_util.csr_to_tensor(H, "cpu", torch.float64)
    res = conductivity_ref.hotspot_constraint(
        tt(xPhys),
        tt(tPhys),
        tti(e1),
        tti(e2),
        tt(w),
        tt(dx),
        H_t,
        tt(Hs),
        factor,
        Tcr,
        p,
        q,
        r,
        rouf,
    )
    return conductivity_ref.HotspotConstraintResult(
        res.fval,
        res.df1.detach().numpy(),
        res.dt1.detach().numpy(),
        res.numer,
        res.K_est.detach().numpy(),
    )


def test_neighbor_weights_match_fixture():
    fx = load_fixture_npz("conductivity_neighbors")
    nelx, nely, rmin_cond = int(fx["nelx"]), int(fx["nely"]), float(fx["rmin_cond"])
    assert (nelx, nely, rmin_cond) == (NELX, NELY, RMIN_COND)

    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    got = {(int(a), int(b)): float(v) for a, b, v in zip(e1, e2, w)}
    expected = {
        (int(a), int(b)): float(v) for a, b, v in zip(fx["e1"], fx["e2"], fx["w"])
    }
    assert set(got) == set(expected)
    for key, w_val in expected.items():
        assert_close(got[key], w_val, tier="algebraic")


def test_K_est_matches_fixture():
    fx = load_fixture_npz("conductivity")
    e2e = load_fixture_npz("e2e")
    nloop = e2e["xPhys_traj"].shape[2] - 1

    e1, e2, w = conductivity.neighbor_weights(NELX, NELY, RMIN_COND)

    for k in range(nloop):
        xPhys = e2e["xPhys_traj"][:, :, k]
        tPhys = e2e["tPhys_traj"][:, :, k]
        K_est = _K_est(xPhys, tPhys, e1, e2, w, Q, ROUF)
        assert_close(K_est, fx["K_est_all"][:, k], tier="algebraic")


def test_hotspot_constraint_matches_fixture():
    fx = load_fixture_npz("conductivity")
    e2e = load_fixture_npz("e2e")
    nelx, nely = int(e2e["nelx"]), int(e2e["nely"])
    nloop = e2e["xPhys_traj"].shape[2] - 1
    assert (nelx, nely) == (NELX, NELY)

    factor_all = fx["factor_all"]
    assert np.all(factor_all == 1.0), "expected factor==1 for all 3 fixture iterations"

    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)
    H, Hs = filters.density_filter(
        nelx, nely, 2
    )  # density filter radius used in the fixture

    # The fixture's tPhys never happens to put two DISTINCT elements at an exact tie --
    # confirmed here so the report doesn't overclaim fixture coverage of the
    # distinct-element-tie branch (exercised only by the synthetic `_at_ties` test
    # below). That's a property of this fixture's field, not of ties being rare in
    # general: a field with exact repeated values along a grid axis (e.g. a linear
    # ramp) produces structural off-diagonal ties on a nontrivial fraction of neighbor
    # pairs, surviving density filtering.
    for k in range(nloop):
        tflat = e2e["tPhys_traj"][:, :, k].flatten()
        off_diag = e1 != e2
        assert np.sum(tflat[e1[off_diag]] == tflat[e2[off_diag]]) == 0

    for k in range(nloop):
        xPhys = e2e["xPhys_traj"][:, :, k]
        tPhys = e2e["tPhys_traj"][:, :, k]
        dx = e2e["dx_all"][:, :, k]
        factor = float(factor_all[k])

        result = _hotspot(
            xPhys,
            tPhys,
            e1,
            e2,
            w,
            dx,
            H,
            Hs,
            factor,
            TCR,
            P,
            Q,
            R,
            ROUF,
        )

        assert_close(result.numer, fx["numer_all"][k], tier="algebraic")
        assert_close(result.df1, fx["df1_all"][:, k], tier="algebraic")
        assert_close(result.dt1, fx["dt1_all"][:, k], tier="algebraic")


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
        K_est = _K_est(xPhys, tPhys, e1, e2, w, q, rouf)
        assert_close(K_est, fx[f"{name}_K_est"], tier="algebraic")


# --- First-principles value checks (no MATLAB fixture, no self-regression) -----------
#
# Everything above compares K_est / the hotspot constraint either to a .mat fixture or
# to golden values frozen from this same Python; the FD tests below check gradients
# against the code's own fval. None of that can tell a smooth-but-wrong K_est from a
# right one. These pin the *values* against the definition the source implements
# (conductivity_estimation_stto_main.m lines ~360-395), restated here from the physics:
#
#   K_est[a] = sum_b W[a,b] * s(rouf*(t[a] - t[b])) * x[b]^q
#              -------------------------------------------------
#              sum_b W[a,b] * s(rouf*(t[a] - t[b]))
#
# a weighted average of neighbourhood density^q in which each neighbour `b` is weighted
# by (i) its geometric proximity W[a,b] and (ii) a logistic gate that opens as `b`'s
# deposition time falls below `a`'s. So K_est[a] measures how much *already-solidified*
# material surrounds `a` at the moment `a` is deposited: high K_est = well supported and
# able to conduct heat away, low K_est = overhanging into empty or still-molten space.
# `1 - K_est` is therefore the overheating risk the hotspot constraint bounds.
#
# Three consequences are used as the backbone of the tests below, and each would be
# broken by a different plausible bug:
#   * it is a convex combination of the x[b]^q, so it is bracketed by their min and max
#     (a broken normalizer breaks this);
#   * it depends on `t` only through differences, so it is invariant to a global time
#     shift (a missing/extra `t` term breaks this);
#   * the gate's *sign* -- earlier neighbours count, later ones don't -- is the whole
#     physical content, and a flipped sign leaves every one of the FD, fixture-shape and
#     bracketing checks intact.


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def _dense_neighbor_weights(nelx, nely, rmin_cond):
    """(nel, nel) rebuild of the documented weight rule, from its statement rather than
    from `neighbor_weights`' vectorized offset loop: `w = (rmin_cond - dist)/rmin_cond`
    for every pair within `rmin_cond`, self-pairs included (so `W[a,a] == 1`)."""
    nel = nelx * nely
    W = np.zeros((nel, nel))
    for a in range(nel):
        ia, ja = a % nelx, a // nelx
        for b in range(nel):
            ib, jb = b % nelx, b // nelx
            dist = np.hypot(ia - ib, ja - jb)
            if dist <= rmin_cond:
                W[a, b] = (rmin_cond - dist) / rmin_cond
    return W


def _reference_K_est(xPhys, tPhys, W, q, rouf):
    """Dense restatement of the formula in this section's header comment."""
    x = xPhys.flatten()
    t = tPhys.flatten()
    gate = W * _logistic(rouf * (t[:, None] - t[None, :]))
    return (gate @ x**q) / gate.sum(axis=1)


def test_K_est_matches_dense_reference():
    """The vectorized COO implementation against a dense restatement of the definition,
    on an asymmetric mesh at a radius that gives a nontrivial weight spread."""
    nelx, nely, rmin_cond, q, rouf = 5, 4, 2.5, 3.0, 40.0
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    W = _dense_neighbor_weights(nelx, nely, rmin_cond)

    rng = np.random.default_rng(11)
    for _ in range(3):
        xPhys = rng.uniform(0.05, 1.0, size=(nely, nelx))
        tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
        K_est = _K_est(xPhys, tPhys, e1, e2, w, q, rouf)
        np.testing.assert_allclose(
            K_est, _reference_K_est(xPhys, tPhys, W, q, rouf), rtol=1e-12, atol=1e-14
        )


def test_K_est_three_element_hard_gated_closed_form():
    """A hand-computable 1x3 strip, printed strictly left to right, with `rouf` large
    enough that every logistic gate is 0 or 1 to machine precision. At `rmin_cond = 2` on
    a single row the only weights are `W[a,a] = 1` and `W[a,a+-1] = 1/2`, so each
    element's average collapses to a closed form written out below by hand:

      * element 0 is deposited first. Its only open gate is its own (a self-pair gates at
        s(0) = 1/2 exactly), so it averages over nothing but itself:  K = x0^q.
        This is the physically load-bearing case -- the first material laid down is
        supported by nothing, and its estimated conductivity is its own density alone.
      * element 1 sees element 0 (already solid, weight 1/2, gate 1) and itself
        (weight 1, gate 1/2): both contribute 1/2, so  K = (x0^q + x1^q)/2.
      * element 2 likewise sees element 1 and itself:  K = (x1^q + x2^q)/2.
        Element 0 is out of range and must not appear -- if it does, `rmin_cond` is being
        applied as a square window rather than a radius.
    """
    nelx, nely, rmin_cond, q = 3, 1, 2.0, 3.0
    rouf = 1000.0  # rouf*dt = 400 at the closest pair: gates are 0/1 to ~1e-174
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)

    x0, x1, x2 = 0.3, 0.9, 0.5
    xPhys = np.array([[x0, x1, x2]])
    tPhys = np.array([[0.1, 0.5, 0.9]])

    K_est = _K_est(xPhys, tPhys, e1, e2, w, q, rouf)
    expected = np.array(
        [x0**q, (x0**q + x1**q) / 2, (x1**q + x2**q) / 2],
    )
    np.testing.assert_allclose(K_est, expected, rtol=1e-12, atol=1e-14)


def test_K_est_three_element_soft_gated_closed_form():
    """The same 1x3 strip at a finite `rouf`, where every gate is strictly between 0 and
    1 and the arithmetic no longer collapses -- written out term by term so the logistic
    argument's sign, scale and orientation are all pinned, not just its saturated limit.
    """
    nelx, nely, rmin_cond, q, rouf = 3, 1, 2.0, 3.0, 4.0
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)

    x = np.array([0.3, 0.9, 0.5])
    t = np.array([0.1, 0.5, 0.9])
    xPhys, tPhys = x[None, :], t[None, :]

    def gate(a, b):
        return _logistic(rouf * (t[a] - t[b]))

    K0 = (1.0 * gate(0, 0) * x[0] ** q + 0.5 * gate(0, 1) * x[1] ** q) / (
        1.0 * gate(0, 0) + 0.5 * gate(0, 1)
    )
    K1 = (
        0.5 * gate(1, 0) * x[0] ** q
        + 1.0 * gate(1, 1) * x[1] ** q
        + 0.5 * gate(1, 2) * x[2] ** q
    ) / (0.5 * gate(1, 0) + 1.0 * gate(1, 1) + 0.5 * gate(1, 2))
    K2 = (0.5 * gate(2, 1) * x[1] ** q + 1.0 * gate(2, 2) * x[2] ** q) / (
        0.5 * gate(2, 1) + 1.0 * gate(2, 2)
    )

    K_est = _K_est(xPhys, tPhys, e1, e2, w, q, rouf)
    np.testing.assert_allclose(K_est, [K0, K1, K2], rtol=1e-12, atol=1e-14)
    # All three gates genuinely partial: a saturated field would make this a repeat of
    # the hard-gated test above.
    assert 0.02 < gate(1, 0) < 0.98 and 0.02 < gate(0, 1) < 0.98


@pytest.mark.parametrize("c", [0.05, 0.4, 1.0])
def test_K_est_uniform_density_is_exactly_c_to_the_q(c):
    """A uniform density field makes every weighted average degenerate to its common
    value, so K_est == c**q at every element for *any* time field, `rouf` or mesh --
    the tightest closed form available, and one that fails immediately if the numerator
    and denominator are gated or weighted differently from each other."""
    nelx, nely, rmin_cond, q = 5, 4, 3.0, 3.0
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    rng = np.random.default_rng(12)
    xPhys = np.full((nely, nelx), c)
    for rouf in (0.0, 1.0, 100.0, 1e4):
        tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
        K_est = _K_est(xPhys, tPhys, e1, e2, w, q, rouf)
        np.testing.assert_allclose(K_est, c**q, rtol=1e-13, atol=1e-15)


def test_K_est_is_a_convex_combination_of_neighbour_densities():
    """K_est[a] is an average over a's neighbourhood with strictly positive weights, so
    it must lie inside the range of that neighbourhood's x**q -- never outside it, and
    in particular never outside [0, 1] for a density field in [0, 1]. A sign error in the
    gate, or a normalizer taken over the wrong index, breaks the bracket."""
    nelx, nely, rmin_cond, q, rouf = 6, 4, 2.5, 3.0, 100.0
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    W = _dense_neighbor_weights(nelx, nely, rmin_cond)
    in_range = W > 0

    rng = np.random.default_rng(13)
    for _ in range(5):
        xPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
        tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
        K_est = _K_est(xPhys, tPhys, e1, e2, w, q, rouf)
        xq = xPhys.flatten() ** q
        for a in range(nelx * nely):
            neigh = xq[in_range[a]]
            assert neigh.min() - 1e-12 <= K_est[a] <= neigh.max() + 1e-12
        assert np.all((K_est >= 0.0) & (K_est <= 1.0))


def test_K_est_invariant_to_global_time_shift():
    """Only *relative* deposition order matters: the gate reads t[a] - t[b], so shifting
    the whole schedule leaves K_est unchanged. Nothing else in the suite would notice a
    stray absolute-`t` term, because every fixture's time field is already normalized to
    roughly [0, 1]."""
    nelx, nely, rmin_cond, q, rouf = 5, 4, 3.0, 3.0, 50.0
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)

    rng = np.random.default_rng(14)
    xPhys = rng.uniform(0.1, 1.0, size=(nely, nelx))
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
    base = _K_est(xPhys, tPhys, e1, e2, w, q, rouf)
    for shift in (-5.0, -0.3, 0.75, 12.0):
        shifted = _K_est(xPhys, tPhys + shift, e1, e2, w, q, rouf)
        np.testing.assert_allclose(shifted, base, rtol=1e-11, atol=1e-13)


def test_K_est_rouf_zero_limit_ignores_build_order():
    """As `rouf -> 0` every gate opens to 1/2 regardless of deposition times, so K_est
    degenerates to the plain geometry-weighted neighbourhood average -- print order stops
    mattering entirely. This pins the gate's *scale*: `rouf` multiplying the time
    difference, not added to it or applied to something else."""
    nelx, nely, rmin_cond, q = 5, 4, 2.5, 3.0
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    W = _dense_neighbor_weights(nelx, nely, rmin_cond)

    rng = np.random.default_rng(15)
    xPhys = rng.uniform(0.1, 1.0, size=(nely, nelx))
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
    order_free = (W @ xPhys.flatten() ** q) / W.sum(axis=1)

    np.testing.assert_allclose(
        _K_est(xPhys, tPhys, e1, e2, w, q, 0.0),
        order_free,
        rtol=1e-13,
        atol=1e-15,
    )
    # ... and it approaches that limit continuously from a nonzero rouf.
    err = [
        np.abs(_K_est(xPhys, tPhys, e1, e2, w, q, rouf) - order_free).max()
        for rouf in (1.0, 0.1, 0.01)
    ]
    assert err[0] > err[1] > err[2] > 0


def test_K_est_rouf_infinite_limit_is_the_hard_build_order():
    """As `rouf -> inf` the gate becomes a hard step on deposition order: strictly
    earlier neighbours count fully, strictly later ones not at all, and exact ties (only
    the self-pair, for a tie-free schedule) count half. K_est converges to that
    discretely-computable value monotonically in `rouf`."""
    nelx, nely, rmin_cond, q = 5, 4, 2.5, 3.0
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    W = _dense_neighbor_weights(nelx, nely, rmin_cond)

    rng = np.random.default_rng(16)
    xPhys = rng.uniform(0.1, 1.0, size=(nely, nelx))
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
    t = tPhys.flatten()
    assert len(np.unique(t)) == t.size  # tie-free schedule: only self-pairs tie

    hard_gate = np.where(t[:, None] > t[None, :], 1.0, 0.0) + 0.5 * np.eye(t.size)
    gated = W * hard_gate
    limit = (gated @ xPhys.flatten() ** q) / gated.sum(axis=1)

    errs = [
        np.abs(_K_est(xPhys, tPhys, e1, e2, w, q, rouf) - limit).max()
        for rouf in (50.0, 500.0, 5000.0)
    ]
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 1e-9
    assert errs[0] > 1e-4  # non-vacuous: rouf=50 is genuinely far from the limit


def test_K_est_increases_when_an_element_is_printed_later():
    """The sign of the gate, isolated. A void element embedded in solid material has the
    lowest possible K_est of anything in the mesh; deferring its deposition lets more of
    that surrounding material solidify first, and every newly-solid neighbour raises the
    average. So K_est at that element must increase monotonically in its own print time,
    from its self-only value up to the full neighbourhood average.

    A flipped gate sign gives a perfectly smooth, correctly-bracketed, shift-invariant
    K_est that decreases here instead -- every other check in this file still passes.
    """
    nelx, nely, rmin_cond, q, rouf = 5, 5, 2.5, 3.0, 60.0
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)

    xPhys = np.ones((nely, nelx))
    xPhys[2, 2] = 0.05  # a void element in the middle of solid material
    centre = 2 * nelx + 2  # row-major index of (row 2, col 2)

    tPhys = np.full((nely, nelx), 0.5)
    tPhys += np.linspace(0, 0.01, tPhys.size).reshape(tPhys.shape)  # break exact ties

    K_of = []
    for own_t in (0.0, 0.25, 0.5, 0.75, 1.0):
        field = tPhys.copy()
        field[2, 2] = own_t
        K_of.append(_K_est(xPhys, field, e1, e2, w, q, rouf)[centre])

    assert all(a < b for a, b in zip(K_of, K_of[1:]))
    # Printed first, it is supported by nothing but itself; printed last, by everything.
    np.testing.assert_allclose(K_of[0], 0.05**q, rtol=1e-6)
    assert K_of[-1] > 0.9  # solid neighbourhood dominates once it is deposited last


def test_K_est_ranks_an_overhang_below_a_supported_element():
    """End-to-end sanity on a bottom-up build of a solid block: an element whose
    below-neighbours are void (an overhang) must score a lower K_est -- higher
    overheating risk -- than a geometrically identical element sitting on solid
    material. This is the entire reason the quantity exists."""
    nelx, nely, rmin_cond, q, rouf = 9, 7, 2.5, 3.0, 100.0
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)

    # Row 0 is the top of the mesh and row nely-1 the bottom (conventions.md), so a
    # bottom-up build deposits high row indices first.
    jj, _ = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    tPhys = (nely - 1 - jj) / (nely - 1)

    solid = np.ones((nely, nelx))
    overhang = np.ones((nely, nelx))
    overhang[4:, 5:] = 0.01  # material at rows 0-3, col 5+ juts out over empty space

    K_solid = _K_est(solid, tPhys, e1, e2, w, q, rouf)
    K_over = _K_est(overhang, tPhys, e1, e2, w, q, rouf)
    probe = 3 * nelx + 6  # (row 3, col 6): supported in `solid`, overhanging otherwise

    assert K_over[probe] < K_solid[probe] - 0.1
    assert K_solid[probe] > 0.9  # fully supported element: near-perfect conductivity


# --- hotspot_constraint value side, from first principles ----------------------------
#
# The constraint aggregates the per-element overheating severity
#
#   g[e] = (1 - K_est[e]) * x[e]**r
#
# -- risk, damped by density so that near-void elements (which cannot actually overheat,
# there being no material there) are discounted -- into the power mean of order p
#
#   numer = ( mean_e g[e]**p )**(1/p)
#
# and requires  factor * numer <= Tcr,  reported to MMA as  fval = factor*numer/Tcr - 1.
#
# The power mean is a smooth, differentiable stand-in for `max_e g[e]`, which it
# approaches from below as p grows; `factor` is the debiasing constant the main loop
# periodically re-derives as max_g/numer so that `factor*numer` tracks the true maximum
# rather than the (always smaller) p-mean. Everything below follows from that
# definition, and none of it re-derives the value from the code's own output.


def _hotspot_setup(nelx, nely, rmin_cond=2.5, rmin=2):
    H, Hs = filters.density_filter(nelx, nely, rmin)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    return H, Hs, e1, e2, w


@pytest.mark.parametrize("c", [0.2, 0.5, 0.85])
@pytest.mark.parametrize("factor", [1.0, 2.3])
def test_hotspot_constraint_uniform_density_closed_form(c, factor):
    """A uniform density field gives K_est == c**q everywhere (see the K_est tests), so
    every element's severity is the same number and the power mean collapses to it
    exactly, for any p, any time field and any mesh:

        numer = (1 - c**q) * c**r,  fval = factor*numer/Tcr - 1.

    This is the one point where the whole chain -- conductivity, the (1-K) severity, the
    x**r damping, the p-mean and the factor/Tcr scaling -- has a closed form, so it pins
    all of them at once. A missing `1 -`, a swapped q/r, or a mean taken over the wrong
    count all move this value.
    """
    nelx, nely, Tcr = 5, 4, 0.8
    H, Hs, e1, e2, w = _hotspot_setup(nelx, nely)
    rng = np.random.default_rng(21)

    xPhys = np.full((nely, nelx), c)
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
    dx = np.ones((nely, nelx))

    expected_numer = (1 - c**Q) * c**R
    for p in (2.0, 25.0, 60.0):
        res = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, p, Q, R, ROUF)
        np.testing.assert_allclose(res.numer, expected_numer, rtol=1e-12)
        np.testing.assert_allclose(
            res.fval, factor * expected_numer / Tcr - 1, rtol=1e-12
        )


def test_hotspot_constraint_solid_part_is_maximally_satisfied():
    """A fully solid part has K_est == 1 everywhere, so every element's severity is
    exactly 0 and the constraint sits at its floor of fval == -1 -- no overheating risk
    anywhere, whatever the print order. The `-1` is the offset MMA reads as "this
    constraint has all the slack there is".

    Values only; the sensitivities are separately specified by
    `test_hotspot_constraint_gradient_is_finite_for_a_solid_part`.
    """
    nelx, nely, Tcr = 5, 4, 0.8
    H, Hs, e1, e2, w = _hotspot_setup(nelx, nely)
    rng = np.random.default_rng(22)

    xPhys = np.ones((nely, nelx))
    dx = np.ones((nely, nelx))
    for _ in range(3):
        tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
        res = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, P, Q, R, ROUF)
        np.testing.assert_allclose(res.numer, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.fval, -1.0, atol=1e-12)


def test_hotspot_constraint_gradient_is_finite_for_a_solid_part():
    """A design with zero overheating severity everywhere is a perfectly ordinary point
    for MMA to visit -- fval is a well-defined -1 there -- so the constraint's
    sensitivities must be finite numbers, not NaN.

    `scale` contains `(sum_cond/nel)**(1/p - 1)`, and with `p > 1` the exponent is
    negative, so a zero total severity would divide by zero without the `sum_cond == 0`
    guard in `hotspot_constraint`. The limit is well-behaved: every entry of
    `cond_arr1`/`cond_arr2` carries a factor `(T_val*x**r)**(p-1)` that vanishes exactly
    when `T_val == 0` everywhere, so the true gradient is 0.
    """
    nelx, nely, Tcr = 5, 4, 0.8
    H, Hs, e1, e2, w = _hotspot_setup(nelx, nely)
    xPhys = np.ones((nely, nelx))
    tPhys = np.random.default_rng(27).uniform(0.0, 1.0, size=(nely, nelx))
    dx = np.ones((nely, nelx))

    res = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, P, Q, R, ROUF)
    assert np.all(np.isfinite(res.df1))
    assert np.all(np.isfinite(res.dt1))


def _severity(xPhys, tPhys, e1, e2, w, q, r, rouf):
    """Per-element severity g[e] = (1 - K_est[e]) * x[e]**r, from the definition."""
    K = _K_est(xPhys, tPhys, e1, e2, w, q, rouf)
    return (1 - K) * xPhys.flatten() ** r


def test_hotspot_constraint_numer_is_a_power_mean_bracketed_by_the_true_max():
    """The power mean of order p over `nel` nonnegative terms is bracketed by

        max_g / nel**(1/p)  <=  numer  <=  max_g

    -- it always *under*-reports the worst element, by a factor that shrinks as p grows.
    That systematic under-report is exactly what `factor` exists to correct, so pinning
    the bracket pins the quantity `factor` is defined against. A numer computed as a plain
    sum, or normalized by the wrong element count, leaves the bracket."""
    nelx, nely, Tcr = 6, 5, 0.8
    nel = nelx * nely
    H, Hs, e1, e2, w = _hotspot_setup(nelx, nely)
    rng = np.random.default_rng(23)

    for _ in range(4):
        xPhys = rng.uniform(0.15, 1.0, size=(nely, nelx))
        tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
        dx = np.ones((nely, nelx))
        max_g = _severity(xPhys, tPhys, e1, e2, w, Q, R, ROUF).max()

        for p in (2.0, 8.0, 25.0):
            res = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, p, Q, R, ROUF)
            assert max_g * nel ** (-1 / p) - 1e-12 <= res.numer <= max_g + 1e-12
        # Non-vacuous: at p=2 the bracket is genuinely wide, so the upper bound alone
        # isn't doing all the work.
        assert max_g * nel ** (-1 / 2.0) < 0.7 * max_g


def test_hotspot_constraint_numer_rises_to_the_true_max_with_p():
    """The p-mean is nondecreasing in p and converges to max_e g[e] as p -> inf. This is
    the sense in which the constraint is a smooth surrogate for a hard max; if `p` entered
    the exponent or the root the wrong way round, numer would move away from the maximum
    as p grew instead of toward it."""
    nelx, nely, Tcr = 6, 5, 0.8
    H, Hs, e1, e2, w = _hotspot_setup(nelx, nely)
    rng = np.random.default_rng(24)

    xPhys = rng.uniform(0.15, 1.0, size=(nely, nelx))
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
    dx = np.ones((nely, nelx))
    max_g = _severity(xPhys, tPhys, e1, e2, w, Q, R, ROUF).max()

    ps = [1.0, 2.0, 5.0, 15.0, 50.0, 200.0, 1000.0]
    numers = [
        _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, p, Q, R, ROUF).numer
        for p in ps
    ]
    assert all(a < b for a, b in zip(numers, numers[1:]))
    assert all(n <= max_g + 1e-12 for n in numers)  # approached from below, never above

    # Convergence is only algebraic -- the relative gap decays like nel**(1/p) - 1, so
    # even p=1000 is still ~0.4% short. Assert the decay, not an unreachable equality
    # (p can't simply be raised further: g**p underflows to 0 well before the gap
    # closes, which would send numer to 0 rather than to max_g).
    gaps = [(max_g - n) / max_g for n in numers]
    assert all(a > b for a, b in zip(gaps, gaps[1:]))
    assert gaps[-1] < 1e-2
    assert gaps[0] > 0.1  # non-vacuous: p=1 is genuinely far from the max


def test_hotspot_constraint_factor_refresh_recovers_the_true_max():
    """The main loop's periodic refresh sets `factor = max_g / numer`, and then reports
    `tru_max = factor * numer`. The point of that round trip is that tru_max equals the
    genuine worst-element severity -- the p-mean's under-report, cancelled exactly. This
    is the invariant the refresh exists to maintain; nothing outside the .mat fixtures
    checks it, and the fixture never fires the refresh at all (factor == 1 throughout its
    3 iterations)."""
    nelx, nely, Tcr = 6, 5, 0.8
    H, Hs, e1, e2, w = _hotspot_setup(nelx, nely)
    rng = np.random.default_rng(25)

    xPhys = rng.uniform(0.15, 1.0, size=(nely, nelx))
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
    dx = np.ones((nely, nelx))

    first = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, P, Q, R, ROUF)
    # The refresh reads K_est back off the result rather than recomputing it.
    np.testing.assert_allclose(
        first.K_est,
        _K_est(xPhys, tPhys, e1, e2, w, Q, ROUF),
        rtol=1e-13,
    )
    max_g = float(np.max((1 - first.K_est) * xPhys.flatten() ** R))
    assert max_g > first.numer  # the p-mean really is an under-report

    factor = max_g / first.numer
    refreshed = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF)
    # numer is factor-independent, so the refreshed call's tru_max is the true maximum.
    np.testing.assert_allclose(refreshed.numer, first.numer, rtol=1e-13)
    np.testing.assert_allclose(factor * refreshed.numer, max_g, rtol=1e-12)
    np.testing.assert_allclose(refreshed.fval, max_g / Tcr - 1, rtol=1e-12)


def test_hotspot_constraint_is_affine_in_factor_and_crosses_zero_at_Tcr():
    """fval = factor*numer/Tcr - 1 is affine in `factor` and crosses zero exactly when
    the scaled severity reaches Tcr -- i.e. the sign of fval is the feasible/infeasible
    verdict MMA acts on. Checked by construction: pick the `factor` that should put the
    constraint exactly on its boundary, and confirm it does."""
    nelx, nely, Tcr = 6, 5, 0.8
    H, Hs, e1, e2, w = _hotspot_setup(nelx, nely)
    rng = np.random.default_rng(26)

    xPhys = rng.uniform(0.15, 1.0, size=(nely, nelx))
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
    dx = np.ones((nely, nelx))

    def fval_at(factor):
        return _hotspot(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
        ).fval

    numer = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, P, Q, R, ROUF).numer
    boundary = Tcr / numer
    assert fval_at(boundary) == pytest.approx(0.0, abs=1e-12)
    assert fval_at(0.5 * boundary) == pytest.approx(-0.5, abs=1e-12)
    assert fval_at(2.0 * boundary) == pytest.approx(1.0, abs=1e-12)


def test_hotspot_constraint_orders_build_directions_on_a_non_uniform_design():
    """Build order must actually change the verdict, in the direction the quantity is
    defined to measure: depositing material *onto already-solid neighbours* is safer than
    depositing it onto empty space.

    On a part with a void notch in its lower right, printing bottom-up lays the
    overhanging material down after the void beneath it is already "past", so that
    material is surrounded by nothing and scores worse. Printing the same part top-down
    lays the overhang onto the dense material above it, and scores better.

    Note the direction is *not* a gravity argument -- K_est has no notion of up. It scores
    thermal shielding by already-solidified material and nothing else; keeping a print
    physically buildable is the job of the start-point and continuity constraints, not
    this one. Reading a gravity preference into this quantity would be reading in
    something it does not contain.

    The uniform-density control matters: with x constant, K_est is exactly x**q whatever
    the schedule, so build order becomes literally undetectable here. Any ordering claim
    only has content on a non-uniform design.
    """
    nelx, nely, Tcr = 9, 7, 0.8
    H, Hs, e1, e2, w = _hotspot_setup(nelx, nely)
    dx = np.ones((nely, nelx))

    jj, _ = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    bottom_up = (nely - 1 - jj) / (nely - 1)  # row nely-1 is the bottom: printed first
    top_down = jj / (nely - 1)

    uniform = np.full((nely, nelx), 0.9)
    overhang = np.full((nely, nelx), 0.9)
    overhang[4:, 5:] = 0.01  # void notch: rows 0-3 of cols 5+ jut out over nothing

    def numer_of(xPhys, tPhys):
        return _hotspot(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, P, Q, R, ROUF
        ).numer

    assert numer_of(overhang, bottom_up) > numer_of(overhang, top_down) + 1e-3
    np.testing.assert_allclose(
        numer_of(uniform, bottom_up), numer_of(uniform, top_down), rtol=1e-12
    )
    # The notched design is riskier than the solid-ish one under either schedule.
    assert numer_of(overhang, top_down) > numer_of(uniform, bottom_up)


# --- Finite-difference internal-consistency checks (pure Python, no MATLAB fixture) ---


def _xPhys_of(x_raw, H, Hs, nely, nelx, beta, eta):
    xTilde = (H @ x_raw.flatten() / Hs).reshape((nely, nelx))
    xPhys = filters.heaviside_projection(tt(xTilde), beta, eta).numpy()
    return xPhys, xTilde


def _tPhys_of(t_raw, H, Hs, nely, nelx):
    return (H @ t_raw.flatten() / Hs).reshape((nely, nelx))


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
        dx = filters.heaviside_projection_derivative(tt(xTilde), BETA, ETA)

        df1 = _hotspot(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
        ).df1
        # Guard well above the assert_allclose atol below, so this can't pass vacuously.
        assert np.abs(df1).max() > 1e-3

        def fval_of(x_raw, t_raw):
            xPhys, xTilde = _xPhys_of(x_raw, H, Hs, nely, nelx, BETA, ETA)
            tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
            dx = filters.heaviside_projection_derivative(tt(xTilde), BETA, ETA)
            return _hotspot(
                xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
            ).fval

        fd_x = np.zeros(nely * nelx)
        for e in range(nely * nelx):
            j, i = e // nelx, e % nelx
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
    dx = filters.heaviside_projection_derivative(tt(xTilde), BETA, ETA)

    # Confirm the premise: no off-diagonal (distinct-element) exact ties in this field.
    tflat = tPhys.flatten()
    off_diag = e1 != e2
    assert np.sum(tflat[e1[off_diag]] == tflat[e2[off_diag]]) == 0

    dt1 = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF).dt1
    assert np.abs(dt1).max() > 1e-3  # guard well above the atol below

    def fval_of(t_raw):
        tPhys = _tPhys_of(t_raw, H, Hs, nely, nelx)
        return _hotspot(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
        ).fval

    fd_t = np.zeros(nely * nelx)
    for e in range(nely * nelx):
        j, i = e // nelx, e % nelx
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
    property of the fixture's distance-based timefield, though, not of ties being rare
    in general: a field with exact repeated values along a grid axis (e.g. a linear
    ramp) produces structural exact ties on a nontrivial fraction of neighbor pairs,
    surviving density filtering -- not a hypothetical edge case for that kind of
    initialization. Before the fix, the bug wasn't just "wrong at a measure-zero set":
    DFT was ~rouf/4 for
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
    tflat = tPhys.flatten()
    off_diag = e1 != e2
    assert np.sum(tflat[e1[off_diag]] == tflat[e2[off_diag]]) == off_diag.sum()

    dt1 = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF).dt1
    # Guard well above the atol below: with the fix, ties no longer force dt1 to 0.
    assert np.abs(dt1).max() > 1e-3

    def fval_of(tPhys):
        return _hotspot(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, factor, Tcr, P, Q, R, ROUF
        ).fval

    h = 1e-6
    fd_t = np.zeros(nely * nelx)
    for e in range(nely * nelx):
        j, i = e // nelx, e % nelx
        tp, tm = tPhys.copy(), tPhys.copy()
        tp[j, i] += h
        tm[j, i] -= h
        fd_t[e] = (fval_of(tp) - fval_of(tm)) / (2 * h)

    print(
        f"max|dt1|={np.abs(dt1).max():.3e} max|dt1-fd|={np.abs(dt1 - fd_t).max():.3e}"
    )
    np.testing.assert_allclose(dt1, fd_t, rtol=1e-4, atol=1e-7 * factor)


def test_hotspot_constraint_finite_at_exact_zero_density():
    """df1/dt1 must stay finite when `xPhys` contains exact zeros, at the production
    `r=0.05, p=25` where `r*p - 1 = 0.25 > 0`.

    `xPhys` reaches exact 0 in normal operation: the Heaviside projection saturates
    `tanh` to +-1 in float64 for arguments beyond ~19, so its numerator cancels to
    exactly 0 for any `xTilde` below the threshold once `beta_d` is large -- not a rare
    edge case, since `beta_d` grows to a 128 cap over a production run's continuation.

    Before the fix, `cond_arr2`'s diagonal self-heating term multiplied `xa**(r-1)`
    (== +inf at xa == 0) by `Tsub_pow` (== 0 there), giving `nan` in df1. `dt1` never
    depended on that term, so it's checked here too only to confirm it stays clean.
    """
    nelx, nely = 12, 8
    RMIN = 2
    Tcr = 0.8
    H, Hs, e1, e2, w = _hotspot_setup(nelx, nely, RMIN_COND, RMIN)

    rng = np.random.default_rng(30)
    xPhys = (rng.uniform(size=(nely, nelx)) > 0.5).astype(float)
    assert np.any(xPhys == 0.0)  # premise: exact zeros are actually present
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
    dx = np.ones((nely, nelx))

    res = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, P, Q, R, ROUF)
    assert np.all(np.isfinite(res.df1))
    assert np.all(np.isfinite(res.dt1))


def test_hotspot_constraint_fd_density_near_binary():
    """df1 vs. central-difference at a near-binary-but-strictly-positive density field
    (0/1 hard field with the 0s replaced by a small epsilon), on a small mesh -- pins
    the combined `r*p-1` power as the correct derivative in the same near-singular
    regime as `test_hotspot_constraint_finite_at_exact_zero_density`, rather than merely
    "some finite number".

    Identity density filter and `dx=1` (as in `test_hotspot_constraint_fd_time_at_ties`
    for `tPhys`), so `xPhys` is the variable perturbed directly rather than routed
    through a real filter/Heaviside chain -- the FD step stays symmetric and the epsilon
    keeps every perturbed value strictly positive, so `x**(r*p-1)` (non-integer
    exponent) is never evaluated at a negative x.
    """
    nelx, nely = 8, 6
    nel = nelx * nely
    Tcr = 0.8
    h = 1e-7
    eps = 1e-3
    H, Hs = sp.eye(nel, format="csr"), np.ones(nel)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)

    rng = np.random.default_rng(31)
    xPhys = np.where(rng.uniform(size=(nely, nelx)) > 0.5, 1.0 - eps, eps)
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))
    dx = np.ones((nely, nelx))

    df1 = _hotspot(xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, P, Q, R, ROUF).df1
    assert np.all(np.isfinite(df1))
    assert np.abs(df1).max() > 1e-3  # guard well above the atol below

    def fval_of(xPhys):
        return _hotspot(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, Tcr, P, Q, R, ROUF
        ).fval

    fd_x = np.zeros(nel)
    for e in range(nel):
        j, i = e // nelx, e % nelx
        xp, xm = xPhys.copy(), xPhys.copy()
        xp[j, i] += h
        xm[j, i] -= h
        fd_x[e] = (fval_of(xp) - fval_of(xm)) / (2 * h)

    print(
        f"max|df1|={np.abs(df1).max():.3e} max|df1-fd|={np.abs(df1 - fd_x).max():.3e}"
    )
    np.testing.assert_allclose(df1, fd_x, rtol=1e-3, atol=1e-4)


# --- Phase 3.4 (plans/torch_port_part2.md): hotspot_value, autograd sensitivities ----
#
# hotspot_constraint's df1/dt1 are hand-derived; hotspot_value returns only (numer,
# K_est), differentiable end to end w.r.t. xPhys/tPhys, and its sensitivity is meant to
# come from autograd instead. These tests build the same H/Hs/dx chain by hand (the
# density filter's own adjoint, plus the Heaviside chain rule optimize.step's autograd
# path threads automatically) so the comparison lines up with hotspot_constraint's
# already-chained df1/dt1.


def test_hotspot_value_matches_hand_derived_sensitivities_on_fixture():
    """Autograd sensitivity of `hotspot_value`'s `numer` (`d(numer)/d(xPhys)`,
    `d(numer)/d(tPhys)`) against the hand-derived `df1`/`dt1`, at the fixture's 3
    snapshots -- `algebraic` tier (no FE solve involved). Reuses the fixture's own
    `dx` (the Heaviside-derivative field, `heaviside_projection_derivative`'s output
    for that snapshot) to finish the same `H @ (... * dx / Hs)` chain
    `hotspot_constraint`'s `df1` already is, rather than re-deriving `xPhys` from a raw
    `xTilde` the fixture doesn't store.
    """
    fx = load_fixture_npz("conductivity")
    e2e = load_fixture_npz("e2e")
    nelx, nely = int(e2e["nelx"]), int(e2e["nely"])
    nloop = e2e["xPhys_traj"].shape[2] - 1
    factor_all = fx["factor_all"]

    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)
    H, Hs = filters.density_filter(nelx, nely, 2)
    H_t = torch_util.csr_to_tensor(H, "cpu", torch.float64)
    Hs_t = tt(Hs)
    e1_t, e2_t, w_t = tti(e1), tti(e2), tt(w)

    for k in range(nloop):
        xPhys = tt(e2e["xPhys_traj"][:, :, k]).requires_grad_(True)
        tPhys = tt(e2e["tPhys_traj"][:, :, k]).requires_grad_(True)
        dx = tt(e2e["dx_all"][:, :, k]).flatten()
        factor = float(factor_all[k])

        numer, _ = conductivity.hotspot_value(
            xPhys, tPhys, e1_t, e2_t, w_t, P, Q, R, ROUF
        )
        fval = factor * numer / TCR - 1
        d_xPhys, d_tPhys = torch.autograd.grad(fval, (xPhys, tPhys))

        df1 = (H_t @ (d_xPhys.flatten() * dx / Hs_t)).detach().numpy()
        dt1 = (H_t @ (d_tPhys.flatten() / Hs_t)).detach().numpy()

        assert_close(df1, fx["df1_all"][:, k], tier="algebraic")
        assert_close(dt1, fx["dt1_all"][:, k], tier="algebraic")


def test_hotspot_value_fd_density_and_time():
    """`hotspot_value`'s autograd sensitivity against central differences, at a small
    mesh -- an oracle-free check independent of both `hotspot_value` and
    `hotspot_constraint`.
    """
    nelx, nely = 8, 6
    nel = nelx * nely
    h = 1e-6
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)
    e1, e2, w = tti(e1), tti(e2), tt(w)

    rng = np.random.default_rng(40)
    xPhys0 = rng.uniform(0.2, 0.9, size=(nely, nelx))
    tPhys0 = rng.uniform(0.0, 1.0, size=(nely, nelx))

    def numer_of(xPhys, tPhys):
        n, _ = conductivity.hotspot_value(
            tt(xPhys), tt(tPhys), e1, e2, w, P, Q, R, ROUF
        )
        return float(n)

    xPhys_t = tt(xPhys0).requires_grad_(True)
    tPhys_t = tt(tPhys0).requires_grad_(True)
    numer, _ = conductivity.hotspot_value(xPhys_t, tPhys_t, e1, e2, w, P, Q, R, ROUF)
    d_x, d_t = torch.autograd.grad(numer, (xPhys_t, tPhys_t))
    d_x, d_t = d_x.numpy().flatten(), d_t.numpy().flatten()

    fd_x = np.zeros(nel)
    fd_t = np.zeros(nel)
    for e in range(nel):
        j, i = e // nelx, e % nelx
        xp, xm = xPhys0.copy(), xPhys0.copy()
        xp[j, i] += h
        xm[j, i] -= h
        fd_x[e] = (numer_of(xp, tPhys0) - numer_of(xm, tPhys0)) / (2 * h)

        tp, tm = tPhys0.copy(), tPhys0.copy()
        tp[j, i] += h
        tm[j, i] -= h
        fd_t[e] = (numer_of(xPhys0, tp) - numer_of(xPhys0, tm)) / (2 * h)

    np.testing.assert_allclose(d_x, fd_x, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(d_t, fd_t, rtol=1e-4, atol=1e-7)


def test_hotspot_value_finite_at_exact_zero_density():
    """`hotspot_value`'s autograd gradient must stay finite at exact-zero `xPhys`
    (reached routinely once `beta_d` saturates), the autograd counterpart of
    `test_hotspot_constraint_finite_at_exact_zero_density`. Before the NaN-safe
    rewrite (`cond_p = T_val**p * x**(r*p)` instead of `(T_val * x**r)**p`), this
    reproduces Phase 0a's bug under autograd: `d(x**r)/dx` is `inf` at `x == 0`.
    """
    nelx, nely = 12, 8
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)
    e1, e2, w = tti(e1), tti(e2), tt(w)

    rng = np.random.default_rng(41)
    xPhys = (rng.uniform(size=(nely, nelx)) > 0.5).astype(float)
    assert np.any(xPhys == 0.0)
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))

    xPhys_t = tt(xPhys).requires_grad_(True)
    tPhys_t = tt(tPhys).requires_grad_(True)
    numer, _ = conductivity.hotspot_value(xPhys_t, tPhys_t, e1, e2, w, P, Q, R, ROUF)
    d_x, d_t = torch.autograd.grad(numer, (xPhys_t, tPhys_t))
    assert torch.all(torch.isfinite(d_x))
    assert torch.all(torch.isfinite(d_t))


def test_hotspot_value_naive_form_would_have_produced_nan():
    """Demonstrates the NaN-safe rewrite is load-bearing: the naive, algebraically
    equivalent `(T_val * x**r) ** p` form (the transliteration a mechanical port would
    produce) gives a `nan` gradient at the same exact-zero density where
    `hotspot_value`'s rewritten form stays finite -- proof the regression test above
    would actually have caught Phase 0a's bug resurrected under autograd, not merely
    that finite gradients happen to occur here.
    """
    nelx, nely = 12, 8
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)
    e1, e2, w = tti(e1), tti(e2), tt(w)

    rng = np.random.default_rng(41)
    xPhys = (rng.uniform(size=(nely, nelx)) > 0.5).astype(float)
    assert np.any(xPhys == 0.0)
    tPhys = rng.uniform(0.0, 1.0, size=(nely, nelx))

    x = tt(xPhys).flatten().requires_grad_(True)
    t = tt(tPhys).flatten()
    core = conductivity._conductivity_core(x, t, e1, e2, w, Q, ROUF)
    T_val = 1 - core.K_est
    cond_p = (T_val * x**R) ** P  # naive, pre-Phase-0a-fix form
    numer = (torch.sum(cond_p) / x.numel()) ** (1 / P)
    (d_x,) = torch.autograd.grad(numer, (x,))
    assert torch.any(torch.isnan(d_x))


def test_hotspot_value_fully_solid_part_gradient_is_finite():
    """`sum_cond == 0` (every element `T_val == 0`, i.e. a fully solid, maximally
    "cool" part) makes the naive `u**(1/p)` gradient diverge (`1/p - 1 < 0`);
    `_safe_pmean` gives it, and hence `hotspot_value`, a `0` gradient there instead --
    the analytic limit `hotspot_constraint`'s own `scale = 0.0 if sum_cond == 0` comment
    already derives for the hand-derived path.
    """
    nelx, nely = 6, 4
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND)
    e1, e2, w = tti(e1), tti(e2), tt(w)
    xPhys = torch.ones(nely, nelx, dtype=torch.float64, requires_grad=True)
    tPhys = torch.rand(nely, nelx, dtype=torch.float64)

    numer, K_est = conductivity.hotspot_value(xPhys, tPhys, e1, e2, w, P, Q, R, ROUF)
    assert float(numer.detach()) == 0.0
    assert torch.all(K_est == 1.0)
    (d_x,) = torch.autograd.grad(numer, (xPhys,))
    assert torch.all(torch.isfinite(d_x))
    assert torch.all(d_x == 0.0)
