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
import torch
from conftest import default_run_config, e2e_rtol, load_fixture_npz, tt, tti

import sttopt.conductivity as conductivity
import sttopt.mma as mma
import sttopt.optimize as optimize
import sttopt.torch_util as torch_util
import sttopt.viz as viz
import tests.reference.conductivity as conductivity_ref

NELX, NELY, NSTAGE, NLOOP = 7, 5, 3, 3


def test_e2e_agreement_is_at_machine_precision():
    """`conftest.e2e_rtol` loosens by a decade per iteration (1e-9, 1e-8, 1e-7 over the
    fixture's three), on the stated rationale that "subsolv's inner Newton line search
    amplifies small per-iteration differences ... iteration 1 matches to 1e-9,
    iteration 5 to 1e-4".

    That amplification does not actually happen at this problem size: measured
    agreement against `e2e.npz`/`constraints.npz` is at machine precision at *every*
    iteration, including the third. The ladder therefore leaves many decades of slack,
    so `test_e2e_trajectory_matches_fixture` and `test_constraints_stacking_matches_fixture`
    would still pass with a substantial regression present.

    This test pins the agreement that is actually achieved. It is the sensitive version
    of those two tests; if a future change genuinely does introduce iteration-dependent
    drift, this is where it will show up, and *then* is the time to decide whether the
    ladder is justified -- rather than assuming it up front.
    """
    e2e = load_fixture_npz("e2e")
    cons = load_fixture_npz("constraints")
    config = default_run_config(
        nelx=NELX,
        nely=NELY,
        nStage=NSTAGE,
        volfrac=0.5,
        Theta=0.1,
        Tcr=0.8,
        print_base="opposite_corner",
        rmin=2.0,
        lrmin=2.0,
        rmin_cond=3.0,
    )
    problem = optimize.build_problem(config)
    result = optimize.run_from_state(problem, optimize.init_state(problem, 1.0), NLOOP)

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
        solution = mma.subsolv(
            m,
            n,
            epsimin,
            tt(k["low"]),
            tt(k["upp"]),
            tt(k["alfa"]),
            tt(k["beta"]),
            tt(k["p0"]),
            tt(k["q0"]),
            tt(k["P"]),
            tt(k["Q"]),
            k["a0"],
            tt(k["a"]),
            tt(k["b"]),
            tt(k["c"]),
            tt(k["d"]),
        )
        # The residual/complementarity checks below are pure NumPy, independent of
        # subsolv's implementation.
        x, y, z, lam, xsi, eta, mu, zet, s = (np.asarray(v) for v in solution)
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
    H = torch_util.csr_to_tensor(
        sp.eye(nelx * nely, format="csr"), "cpu", torch.float64
    )
    Hs = torch.ones(nelx * nely, dtype=torch.float64)
    dx = torch.ones(nely, nelx, dtype=torch.float64)

    for rouf in [100.0, 800.0, 2000.0, 1e6]:
        result = conductivity_ref.hotspot_constraint(
            tt(xPhys),
            tt(tPhys),
            tti(e1),
            tti(e2),
            tt(w),
            dx,
            H,
            Hs,
            1.0,
            0.8,
            25.0,
            3.0,
            0.05,
            rouf,
        )
        assert np.isfinite(result.fval)
        assert torch.all(
            torch.isfinite(result.df1)
        ), f"rouf={rouf}: {torch.isnan(result.df1).sum()} NaNs in df1"
        assert torch.all(
            torch.isfinite(result.dt1)
        ), f"rouf={rouf}: {torch.isnan(result.dt1).sum()} NaNs in dt1"


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
    FT, DFT = conductivity._pairwise_sigmoid_terms(tt(t), tti(a), tti(b), rouf)
    FT, DFT = FT.numpy(), DFT.numpy()

    with np.errstate(over="ignore", invalid="ignore"):
        FT_matlab = 1.0 / (1.0 + np.exp(z))
        DFT_matlab = FT_matlab**2 * rouf * np.exp(z)

    # `a`/`b` are always distinct here, so `z == 0` is a genuine tie between distinct
    # elements: DFT should be the ordinary rouf/4, matching DFT_matlab, not 0 (the
    # source's `if TPhys(N_ele(o))==ti` zeroed it there too, but that was a bug --
    # correct only for a == b self-pairs, not distinct-element ties; see
    # conventions.md). So no special-casing needed: compare everywhere in `safe`.
    np.testing.assert_allclose(FT[safe], FT_matlab[safe], rtol=1e-14, atol=0)
    np.testing.assert_allclose(DFT[safe], DFT_matlab[safe], rtol=1e-14, atol=0)

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


def test_dft_zero_only_at_self_pairs():
    """Direct unit-level pin of the `a == b` branch in `_pairwise_sigmoid_terms`,
    separate from the aggregate FD checks in test_conductivity.py: `DFT` must be
    exactly 0 at every self-pair (`a == b`, where `FT(t_a, t_a)` is the constant 0.5
    regardless of `t_a`, so its true derivative really is 0) and must be the ordinary
    nonzero `rouf/4` at a genuine tie between two *distinct* elements (`a != b`,
    `t[a] == t[b]`) -- the fix this module's other sigmoid test exercises only
    indirectly (via distinct, never-equal index pairs) and test_conductivity.py's FD
    checks exercise only in aggregate through the full `hotspot_constraint` pipeline.
    """
    rng = np.random.default_rng(3)
    rouf = 17.0
    n = 50
    t = rng.uniform(0.0, 1.0, n)

    # Self-pairs: a == b for every index, t[a] == t[b] trivially. FT == 0.5 is the
    # actual reason DFT == 0 is correct here (FT(t_a, t_a) is that constant regardless
    # of t_a, so its true derivative is 0) -- pinned alongside DFT so the self-pair
    # branch is self-justifying, not just asserted.
    idx = np.arange(n)
    FT_self, DFT_self = conductivity._pairwise_sigmoid_terms(
        tt(t), tti(idx), tti(idx), rouf
    )
    assert torch.all(FT_self == 0.5)
    assert torch.all(DFT_self == 0.0)

    # Distinct-element ties: two disjoint index ranges sharing the same t values, so
    # t[a[i]] == t[b[i]] for every i while a[i] != b[i] (b[i] = a[i] + n).
    t_dup = np.concatenate([t, t])
    a = np.arange(n)
    b = a + n
    _, DFT_tied = conductivity._pairwise_sigmoid_terms(tt(t_dup), tti(a), tti(b), rouf)
    np.testing.assert_allclose(DFT_tied, rouf / 4.0, rtol=1e-14, atol=0)
