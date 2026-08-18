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
