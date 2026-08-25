"""Tests for `sttopt.filters`: fixture comparisons for H/Hs/L, FD checks for the Heaviside projection."""

import numpy as np
import pytest
from conftest import assert_close, load_fixture

from sttopt.filters import (
    continuity_filter,
    density_filter,
    heaviside_projection,
    heaviside_projection_derivative,
)

NELX, NELY = 7, 5
RMIN = LRMIN = 2


def test_density_filter_against_fixture():
    fx = load_fixture("filters")
    H, Hs = density_filter(NELX, NELY, RMIN)
    assert_close(H.toarray(), fx["H"].toarray(), tier="algebraic")
    assert_close(Hs, np.asarray(fx["Hs"].toarray()).flatten(), tier="algebraic")


def test_continuity_filter_against_fixture():
    fx = load_fixture("filters")
    L = continuity_filter(NELX, NELY, LRMIN)
    assert_close(L.toarray(), fx["L"].toarray(), tier="algebraic")


def test_continuity_filter_returns_sparse():
    # This only checks the *return type*, not that every intermediate stays sparse --
    # reading continuity_filter's implementation is how that's actually verified (it
    # never calls .toarray()/np.array() on an n x n object; see conventions.md's
    # eye(n)-L./M trap). A dense-then-resparsified implementation would still pass this.
    import scipy.sparse as sp

    L = continuity_filter(NELX, NELY, LRMIN)
    assert sp.issparse(L)


# --- continuity_filter properties (first-principles, no MATLAB fixture) ---------------
# `L @ t` is defined to be each element's own value minus the unweighted mean of its
# `lrmin`-neighbours (self excluded). Everything below follows from that definition
# alone, so these pin the operator's *meaning* rather than its fixture values -- the
# fixture test above only covers nelx=7, nely=5, lrmin=2.
#
# All the closed forms use lrmin=2, i.e. the 3x3 window minus self, which is the radius
# the reference loop and every fixture use. Neighbour counts: 8 in the interior, 5 along
# an edge, 3 at a corner.


def _column_index_field(nelx: int, nely: int) -> np.ndarray:
    """t[j, i] = i -- a field increasing by exactly 1 per column, constant down a column."""
    return np.broadcast_to(np.arange(nelx, dtype=float), (nely, nelx)).copy()


def _apply(L, field, nelx, nely):
    return (L @ field.flatten(order="F")).reshape((nely, nelx), order="F")


@pytest.mark.parametrize("c", [0.0, 1.0, -3.25, 0.4])
def test_continuity_filter_annihilates_constant_fields(c):
    """A uniform field disagrees with its neighbourhood nowhere, so `L @ t` must be
    exactly zero at *every* element -- including boundary ones, whose windows are
    truncated and whose row normalizer therefore differs. This is the penalty's floor:
    a single-shot print (all elements deposited at the same instant) is the smoothest
    time field there is, and must score zero, not merely "something small"."""
    nelx, nely = 6, 4
    L = continuity_filter(nelx, nely, LRMIN)
    out = _apply(L, np.full((nely, nelx), c), nelx, nely)
    np.testing.assert_allclose(out, 0.0, atol=1e-14)


def test_continuity_filter_on_linear_ramp_closed_form():
    """A ramp increasing by 1 per column is the canonical "perfectly continuous" build
    order (`timefield_edge`'s shape). Its 3x3 neighbourhood is symmetric about the centre
    everywhere except the first and last columns, so `L @ t` must vanish identically off
    those two columns, whatever the mesh size -- an operator that leaked a nonzero
    residual into the interior would penalize the very build order it exists to reward.

    On the two end columns the neighbourhood is one-sided and the residual is a fixed
    closed form: an edge element has 5 neighbours (2 in its own column, 3 in the adjacent
    one) so the mean sits 3/5 of a column away; a corner element has 3 neighbours (1 in
    its own column, 2 in the adjacent one) so the mean sits 2/3 of a column away. The
    sign is negative on the low-index end (the element is earlier than its neighbourhood)
    and mirrored on the high-index end.
    """
    for nelx, nely in [(6, 4), (5, 5), (7, 3)]:
        L = continuity_filter(nelx, nely, LRMIN)
        out = _apply(L, _column_index_field(nelx, nely), nelx, nely)

        np.testing.assert_allclose(out[:, 1:-1], 0.0, atol=1e-14)

        expected_first = np.full(nely, -3 / 5)
        expected_first[0] = expected_first[-1] = -2 / 3
        np.testing.assert_allclose(out[:, 0], expected_first, atol=1e-14)
        np.testing.assert_allclose(out[:, -1], -expected_first, atol=1e-14)


def test_continuity_filter_row_and_diagonal_ramps():
    """The same closed form must hold for a ramp down the rows (the operator has no
    preferred axis), and a diagonal ramp -- the sum of the two -- must vanish wherever
    both index directions are interior."""
    nelx, nely = 6, 5
    L = continuity_filter(nelx, nely, LRMIN)

    jj, ii = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    row_ramp = jj.astype(float)
    out_rows = _apply(L, row_ramp, nelx, nely)
    np.testing.assert_allclose(out_rows[1:-1, :], 0.0, atol=1e-14)
    expected_first = np.full(nelx, -3 / 5)
    expected_first[0] = expected_first[-1] = -2 / 3
    np.testing.assert_allclose(out_rows[0, :], expected_first, atol=1e-14)
    np.testing.assert_allclose(out_rows[-1, :], -expected_first, atol=1e-14)

    out_diag = _apply(L, (ii + jj).astype(float), nelx, nely)
    np.testing.assert_allclose(out_diag[1:-1, 1:-1], 0.0, atol=1e-14)


def test_continuity_filter_on_checkerboard_closed_form():
    """The worst case the penalty exists to reject. An interior element's 8 neighbours
    split 4/4 between the two checkerboard phases, so the neighbourhood mean is exactly
    1/2 and the residual is +-1/2 -- the largest magnitude any 0/1 field can produce, and
    an order of magnitude above the ramp's interior residual of 0."""
    nelx, nely = 6, 5
    L = continuity_filter(nelx, nely, LRMIN)
    jj, ii = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    phase = ((ii + jj) % 2).astype(float)

    out = _apply(L, phase, nelx, nely)
    np.testing.assert_allclose(out[1:-1, 1:-1], phase[1:-1, 1:-1] - 0.5, atol=1e-14)


def _reference_density_filter(nelx: int, nely: int, rmin: float) -> np.ndarray:
    """Independent transliteration of generate_fixtures.m's density-filter loop (0-indexed).

    Deliberately doesn't reuse `_neighbor_offsets`/any of `filters.py`'s own machinery,
    so it can catch a window-size or cutoff bug that the MATLAB fixture comparison above
    can't: that fixture only exercises `rmin == LRMIN == 2` (an integer), so a too-large
    square window is invisible there (the extra offsets all carry weight 0, silently
    dropped) -- see review discussion. `rmin=2.5` here is non-integer and gives a
    genuinely nonzero diagonal-neighbor weight, pinning both the square window's size and
    the circular cutoff independent of any other test's `rmin`/`lrmin` coincidence.
    """
    n = nelx * nely
    H = np.zeros((n, n))
    r = int(np.ceil(rmin)) - 1
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            for i2 in range(max(i1 - r, 0), min(i1 + r, nelx - 1) + 1):
                for j2 in range(max(j1 - r, 0), min(j1 + r, nely - 1) + 1):
                    e2 = i2 * nely + j2
                    H[e1, e2] += max(0.0, rmin - np.hypot(i1 - i2, j1 - j2))
    return H


def test_density_filter_window_and_cutoff_against_independent_reference():
    nelx, nely, rmin = 4, 3, 2.5  # non-integer rmin, deliberately != LRMIN
    H, Hs = density_filter(nelx, nely, rmin)
    expected = _reference_density_filter(nelx, nely, rmin)
    assert_close(H.toarray(), expected, tier="algebraic")
    assert_close(Hs, expected.sum(axis=1), tier="algebraic")


@pytest.mark.parametrize("beta,eta", [(1.0, 0.5), (4.0, 0.3), (0.5, 0.7)])
def test_heaviside_projection_derivative_matches_fd(beta, eta):
    rng = np.random.default_rng(0)
    # asymmetric grid, away from 0/1 edges
    xTilde = rng.uniform(0.02, 0.98, size=(5, 7))
    h = 1e-6
    fd = (
        heaviside_projection(xTilde + h, beta, eta)
        - heaviside_projection(xTilde - h, beta, eta)
    ) / (2 * h)
    analytic = heaviside_projection_derivative(xTilde, beta, eta)
    np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-6)


def test_heaviside_projection_derivative_at_eta_exactly():
    # The trivial case every fixture's xPhys0 happens to hit (xTilde == eta) -- included
    # here so a genuinely broken projection can't hide behind the fixture's blind spot.
    beta, eta = 2.0, 0.5
    xTilde = np.array(eta)
    h = 1e-6
    fd = (
        heaviside_projection(xTilde + h, beta, eta)
        - heaviside_projection(xTilde - h, beta, eta)
    ) / (2 * h)
    analytic = heaviside_projection_derivative(xTilde, beta, eta)
    np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-6)


def test_heaviside_projection_derivative_near_domain_edges():
    beta, eta = 3.0, 0.5
    for xTilde in (1e-3, 1 - 1e-3):
        h = 1e-6
        fd = (
            heaviside_projection(xTilde + h, beta, eta)
            - heaviside_projection(xTilde - h, beta, eta)
        ) / (2 * h)
        analytic = heaviside_projection_derivative(xTilde, beta, eta)
        np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "x,beta,eta,expected",
    [
        (0.5, 2.0, 0.5, 0.5),
        (0.6, 4.0, 0.3, 0.9129507086430348),
        (0.15, 0.5, 0.7, 0.14034596134119473),
        (0.9, 6.0, 0.4, 0.9982578858793695),
    ],
)
def test_heaviside_projection_interior_values(x, beta, eta, expected):
    # Golden values from an independent exp-ratio tanh identity (not np.tanh), pinning
    # the projection's own output at interior points -- the FD tests above only check
    # consistency between the projection and its derivative, so a bug shared by both
    # (e.g. a sign error) wouldn't be caught there.
    assert heaviside_projection(np.array(x), beta, eta) == pytest.approx(
        expected, rel=1e-12
    )


def test_heaviside_projection_endpoints():
    # f(0) = 0, f(1) = 1 by construction, regardless of beta/eta.
    for beta, eta in [(1.0, 0.5), (5.0, 0.2), (0.3, 0.8)]:
        assert heaviside_projection(np.array(0.0), beta, eta) == pytest.approx(
            0.0, abs=1e-12
        )
        assert heaviside_projection(np.array(1.0), beta, eta) == pytest.approx(
            1.0, abs=1e-12
        )
