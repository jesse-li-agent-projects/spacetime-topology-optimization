"""Red-team findings: robustness gaps and test-strength gaps found by adversarial review.

Nothing here is a MATLAB-vs-Python numerical discrepancy -- the sweep in
`test_reference_sweep.py` found none. These cover the other two categories: places
where the port is faithful but fragile, and places where an existing test claims more
than it checks.

Two of these started life as `xfail(strict=True)` markers describing behaviour the port
did not have -- the pyplot figure leak in `viz`, and the `0*inf = NaN` overflow in the
neighbour sigmoid's derivative. Both are fixed now, so they are plain regression tests;
each carries a companion test pinning the property the fix must not break (that a
caller-supplied `Axes` is still honoured, and that the overflow-free sigmoid is a pure
reformulation of the MATLAB expression rather than a change of value).
"""

import io

import numpy as np
import scipy.sparse as sp
from conftest import e2e_rtol, load_fixture

import sttopt.conductivity as conductivity
import sttopt.mma as mma
import sttopt.optimize as optimize
import sttopt.viz as viz

NELX, NELY, NSTAGE, NLOOP = 7, 5, 3, 3


def test_e2e_agreement_is_at_machine_precision():
    """`conftest.e2e_rtol` loosens by a decade per iteration (1e-9, 1e-8, 1e-7 over the
    fixture's three), on the stated rationale that "subsolv's inner Newton line search
    amplifies small per-iteration differences ... iteration 1 matches to 1e-9,
    iteration 5 to 1e-4".

    That amplification does not actually happen at this problem size: measured
    agreement against `e2e.mat`/`constraints.mat` is <= ~3e-14 at *every* iteration,
    including the third. The ladder therefore leaves six to eight decades of slack, so
    `test_e2e_trajectory_matches_fixture` and `test_constraints_stacking_matches_fixture`
    would still pass with a substantial regression present.

    This test pins the agreement that is actually achieved. It is the sensitive version
    of those two tests; if a future change genuinely does introduce iteration-dependent
    drift, this is where it will show up, and *then* is the time to decide whether the
    ladder is justified -- rather than assuming it up front.
    """
    e2e = load_fixture("e2e")
    cons = load_fixture("constraints")
    result = optimize.run(NELX, NELY, NLOOP, NSTAGE, 0.5, 0.1, 0.8, 3, 2.0, 2.0, 3.0)

    strict = 1e-12
    for k in range(1, NLOOP + 1):
        rec = result.records[k - 1]
        for name, got, want in [
            ("xPhys", result.xPhys_traj[k], e2e["xPhys_traj"][:, :, k]),
            ("tPhys", result.tPhys_traj[k], e2e["tPhys_traj"][:, :, k]),
            ("obj", np.array([rec.obj]), np.array([e2e["objf"][k - 1]])),
            ("vol", np.array([rec.vol]), np.array([e2e["vol"][k - 1]])),
            ("tru_max", np.array([rec.tru_max]), np.array([e2e["tru_max_all"][k - 1]])),
            ("fval", rec.fval, cons["fval_all"][:, k - 1]),
            ("dfdx", rec.dfdx, cons["dfdx_all"][:, :, k - 1]),
        ]:
            got, want = np.asarray(got, float), np.asarray(want, float)
            err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)
            assert err < strict, (
                f"iteration {k} {name}: rel err {err:.2e} exceeds {strict:g} "
                f"(the loose ladder would have allowed {e2e_rtol(k):.0e})"
            )


def test_subsolv_m_ge_n_solves_the_subproblem():
    """`mma.subsolv`'s `m >= n` elimination is flagged in the source as ported but
    unvalidated, and the existing `test_subsolv_m_gt_n_smoke` -- despite a docstring
    saying it "checks that the returned point satisfies the KKT stationarity/
    feasibility/complementarity conditions" -- only asserts `np.all(np.isfinite(...))`.
    A branch returning finite garbage would pass it.

    This is the check that docstring describes. The MMA subproblem is convex (`p0`,
    `q0`, `P`, `Q` are all non-negative, so every term is convex in `x` on the open
    box), so vanishing KKT residuals are sufficient for global optimality -- no
    external reference optimum is needed to make this conclusive.

    Both branches are covered so the assertion is calibrated: whatever residual the
    `m < n` branch achieves, the `m >= n` branch must achieve too.
    """

    def build(n, m, seed):
        r = np.random.default_rng(seed)
        return dict(
            low=-np.ones(n),
            upp=np.ones(n),
            alfa=-0.9 * np.ones(n),
            beta=0.9 * np.ones(n),
            p0=r.uniform(0.5, 2, n),
            q0=r.uniform(0.5, 2, n),
            P=r.uniform(0.1, 1, (m, n)),
            Q=r.uniform(0.1, 1, (m, n)),
            a0=1.0,
            # Non-zero `a` keeps the z-coupled border terms (axz, azz, bz, and dlam's
            # -dz*(a/diaglamyi)) live; a=0 zeroes all of them.
            a=r.uniform(0.05, 0.4, m),
            b=r.uniform(0.5, 2, m),
            c=np.full(m, 10.0),
            d=np.zeros(m),
        )

    epsimin = 1e-9
    for n, m in [(2, 3), (3, 5), (4, 4), (5, 8), (6, 2), (8, 3)]:
        k = build(n, m, n * 10 + m)
        x, y, z, lam, xsi, eta, mu, zet, s = mma.subsolv(
            m,
            n,
            epsimin,
            k["low"],
            k["upp"],
            k["alfa"],
            k["beta"],
            k["p0"],
            k["q0"],
            k["P"],
            k["Q"],
            k["a0"],
            k["a"],
            k["b"],
            k["c"],
            k["d"],
        )
        ux1, xl1 = k["upp"] - x, x - k["low"]
        plam = k["p0"] + k["P"].T @ lam
        qlam = k["q0"] + k["Q"].T @ lam
        gvec = k["P"] @ (1 / ux1) + k["Q"] @ (1 / xl1)

        residuals = {
            "stationarity_x": plam / ux1**2 - qlam / xl1**2 - xsi + eta,
            "stationarity_y": k["c"] + k["d"] * y - mu - lam,
            "stationarity_z": np.array([k["a0"] - zet - k["a"] @ lam]),
            "primal_feasibility": gvec - k["a"] * z - y + s - k["b"],
            "complementarity": np.concatenate(
                [
                    xsi * (x - k["alfa"]),
                    eta * (k["beta"] - x),
                    mu * y,
                    [zet * z],
                    lam * s,
                ]
            ),
        }
        for name, res in residuals.items():
            worst = np.abs(res).max()
            assert worst <= 10 * epsimin, (
                f"n={n} m={m} ({'m<n' if m < n else 'm>=n'} branch): "
                f"{name} residual {worst:.2e} > {10 * epsimin:.0e}"
            )
        # Interior-point iterates must stay strictly feasible, not just finite.
        assert np.all(x > k["alfa"]) and np.all(x < k["beta"])
        assert np.all(y > 0) and z > 0 and np.all(lam > 0)


def test_viz_does_not_accumulate_figures():
    """Regression: `combination_plot`/`stage_boundary_plot` used to allocate through the
    pyplot *state machine* (`plt.subplots()`) when no `ax` was passed, so every call
    registered a figure that was never released. One call per process (the CLI's usage)
    was harmless, but any loop over frames -- porting `fabrication.m`'s per-timestep
    animation, say -- grew memory without bound, and matplotlib's warning at the 21st
    open figure became a hard error under this project's `-W error` test invocation, so
    the 21st viz test added to the suite would have failed for unrelated reasons.

    No warning filter here on purpose: with the fix there is nothing to suppress, so a
    regression trips this assertion *and*, under `-W error`, matplotlib's own warning.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    rng = np.random.default_rng(0)
    xPhys = rng.uniform(0, 1, (NELY, NELX))
    tPhys = rng.uniform(0, 1, (NELY, NELX))
    try:
        for _ in range(25):
            ax = viz.combination_plot(xPhys, tPhys, eps=0.1)
            viz.stage_boundary_plot(tPhys, NSTAGE)
        assert (
            len(plt.get_fignums()) == 0
        ), f"{len(plt.get_fignums())} pyplot figures left open after 50 calls"
        # A pyplot-free Figure must still render, or the fix would have broken the CLI.
        buf = io.BytesIO()
        ax.figure.savefig(buf, format="png")
        assert buf.getbuffer().nbytes > 0
    finally:
        plt.close("all")


def test_viz_still_honours_a_caller_supplied_axes():
    """The escape hatch the fix relies on: callers who *want* a pyplot-managed figure
    pass their own `ax`, and both functions must draw into exactly that one.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    tPhys = np.zeros((NELY, NELX))
    tPhys[:, NELX // 2 :] = 1.0
    try:
        fig, ax = plt.subplots()
        out = viz.combination_plot(np.ones((NELY, NELX)), tPhys, eps=0.1, ax=ax)
        out2 = viz.stage_boundary_plot(tPhys, 2, ax=ax, combination_coords=True)
        assert out is ax and out2 is ax
        assert plt.get_fignums() == [fig.number]
        assert len(ax.collections) == 2
    finally:
        plt.close("all")


def test_hotspot_gradients_finite_for_large_rouf():
    """Regression: `_pairwise_sigmoid_terms` used to compute `FT = 1/(1+exp(z))` and
    `DFT = FT**2 * rouf * exp(z)` with `z = rouf*(t_b - t_a)`, exactly as the MATLAB
    source does. For `z > ~709` the `exp(z)` overflows to `inf`, `FT` underflows to `0`,
    and `DFT` became `0*inf = NaN` -- silently poisoning `dt1` and, through it, the MMA
    constraint Jacobian.

    Not reachable at the default `rouf=100` (with `tPhys` in [0, 1], `|z| <= 100`), but
    `rouf` is an exposed `build_problem` keyword with no validation, and sharpening the
    print-causality mask is exactly the knob a user would reach for.

    No `errstate`/warning suppression on purpose: the point of the fix is that nothing
    overflows, so a regression surfaces as a numpy overflow warning too.
    """
    nelx, nely = 5, 4
    rng = np.random.default_rng(1)
    xPhys = rng.uniform(0.3, 0.95, (nely, nelx))
    tPhys = rng.uniform(0.0, 1.0, (nely, nelx))
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, 3.0)
    H = sp.eye(nelx * nely, format="csr")
    Hs = np.ones(nelx * nely)
    dx = np.ones((nely, nelx))

    for rouf in [100.0, 800.0, 2000.0, 1e6]:
        fval, df1, dt1 = conductivity.hotspot_constraint(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, 0.8, 25.0, 3.0, 0.05, rouf
        )
        assert np.isfinite(fval)
        assert np.all(
            np.isfinite(df1)
        ), f"rouf={rouf}: {np.isnan(df1).sum()} NaNs in df1"
        assert np.all(
            np.isfinite(dt1)
        ), f"rouf={rouf}: {np.isnan(dt1).sum()} NaNs in dt1"


def test_stable_sigmoid_matches_the_matlab_expression():
    """The overflow fix must be a pure reformulation over the range that matters, and
    strictly better outside it.

    Two regimes, asserted separately because the honest claim differs between them:

    * `|z| <= 300` -- everything physically reachable (at the default `rouf=100` and
      `tPhys` in [0, 1], `|z| <= 100`). Here the source's literal expressions are
      well behaved and the new forms must agree with them to 1e-14. This is the
      "no value change" guarantee.
    * beyond that -- the source's forms degrade in two different ways: at `z ~ +700`
      the intermediate `FT**2` underflows to 0 and drags the whole product to 0, and
      past `z ~ +709` `exp(z)` overflows and the product becomes NaN. The new forms
      stay finite and non-zero. So the fix is not merely equivalent out here, it is
      more accurate, and pinning it against the source would pin the source's bug.
    """
    rng = np.random.default_rng(0)
    z_safe = np.concatenate(
        [
            rng.uniform(-100, 100, 200_000),
            rng.uniform(-1, 1, 20_000),
            np.array([0.0, -50.0, 50.0, -300.0, 300.0]),
        ]
    )
    z_extreme = np.array([-1e4, -745.0, -710.0, 400.0, 700.0, 710.0, 1e4])
    z = np.concatenate([z_safe, z_extreme])
    safe = np.arange(z.size) < z_safe.size

    rouf = 3.7
    t = np.concatenate([np.zeros(z.size), z / rouf])  # so t[b] - t[a] == z / rouf
    a = np.arange(z.size)
    b = a + z.size
    FT, DFT = conductivity._pairwise_sigmoid_terms(t, a, b, rouf)

    with np.errstate(over="ignore", invalid="ignore"):
        FT_matlab = 1.0 / (1.0 + np.exp(z))
        DFT_matlab = FT_matlab**2 * rouf * np.exp(z)

    # `z == 0` is an exact print-time tie, where the source (and so this port) forces
    # DFT to 0 rather than the true rouf/4 -- a documented deviation, not something the
    # overflow fix introduced, so it is pinned separately from the comparison.
    ties = z == 0.0
    assert np.all(DFT[ties] == 0.0)

    np.testing.assert_allclose(FT[safe], FT_matlab[safe], rtol=1e-14, atol=0)
    compare = safe & ~ties
    np.testing.assert_allclose(DFT[compare], DFT_matlab[compare], rtol=1e-14, atol=0)

    # Extreme tail: finite everywhere, and still non-zero wherever the true value is
    # representable -- past |z| ~ 745 it genuinely underflows, so 0 is the correct
    # answer there and only |z| <= 745 can be asserted non-zero.
    assert np.all(np.isfinite(FT)) and np.all(np.isfinite(DFT))
    representable = ~safe & (np.abs(z) <= 745.0)
    assert representable.any(), "sanity: need extreme-but-representable samples"
    assert np.all(DFT[representable] > 0.0)
    degraded = ~np.isfinite(DFT_matlab) | (DFT_matlab == 0.0)
    assert degraded[
        representable
    ].any(), "sanity: the extreme samples should break the source form"
