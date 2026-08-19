"""Red-team findings: robustness gaps and test-strength gaps found by adversarial review.

Nothing here is a MATLAB-vs-Python numerical discrepancy -- the sweep in
`test_reference_sweep.py` found none. These cover the other two categories: places
where the port is faithful but fragile, and places where an existing test claims more
than it checks.

The two `xfail(strict=True)` tests below describe behaviour the port does *not* have
today. They are written as the assertion the fixed code should satisfy, so removing
the marker is the whole fix-verification step; `strict=True` means they also fail if
someone fixes the underlying issue and forgets to unmark them.
"""

import warnings

import numpy as np
import pytest
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


@pytest.mark.xfail(
    strict=True,
    reason="FINDING: viz creates figures via plt.subplots() and never closes them, so "
    "every call leaks one. Fix: build the Figure off the pyplot state machine "
    "(matplotlib.figure.Figure) or document that callers own closing.",
)
def test_viz_does_not_accumulate_figures():
    """`combination_plot`/`stage_boundary_plot` allocate through the pyplot *state
    machine* (`plt.subplots()`) when no `ax` is passed, so every call registers a figure
    that is never released. One call per process (the CLI's usage) is harmless, but any
    loop over frames -- e.g. porting `fabrication.m`'s per-timestep animation, or simply
    adding more viz tests -- grows memory without bound.

    It also interacts badly with this project's own `-W error` test invocation:
    matplotlib warns at the 21st open figure, which becomes a hard error, so the 21st
    viz test added to the suite would fail for reasons unrelated to what it tests.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    rng = np.random.default_rng(0)
    xPhys = rng.uniform(0, 1, (NELY, NELX))
    tPhys = rng.uniform(0, 1, (NELY, NELX))
    try:
        with warnings.catch_warnings():
            # Ignore matplotlib's own >20-figures warning so the assertion below is what
            # reports the leak, rather than -W error tripping on the symptom first.
            warnings.simplefilter("ignore")
            for _ in range(25):
                viz.combination_plot(xPhys, tPhys, eps=0.1)
        assert (
            len(plt.get_fignums()) <= 1
        ), f"{len(plt.get_fignums())} figures left open after 25 calls"
    finally:
        plt.close("all")


@pytest.mark.xfail(
    strict=True,
    reason="FINDING: the neighbour sigmoid uses 1/(1+exp(z)) directly, so DFT evaluates "
    "0*inf = NaN once rouf*dt > ~709. Faithful to MATLAB, but conventions.md claims "
    "the port replaced this with a stable form -- it did not. Fix: compute DFT as "
    "rouf*FT*(1-FT), which is algebraically identical and cannot overflow.",
)
def test_hotspot_gradients_finite_for_large_rouf():
    """`_pairwise_sigmoid_terms` computes `FT = 1/(1+exp(z))` and then
    `DFT = FT**2 * rouf * exp(z)` with `z = rouf*(t_b - t_a)`. For `z > ~709` the
    `exp(z)` overflows to `inf`, `FT` underflows to `0`, and `DFT` becomes `0*inf =
    NaN` -- silently poisoning `dt1`, and through it the MMA constraint Jacobian.

    Not reachable at the default `rouf=100` (with `tPhys` in [0, 1], `|z| <= 100`), but
    `rouf` is an exposed `build_problem` keyword with no validation, and sharpening the
    print-causality mask is exactly the knob a user would reach for. MATLAB has the
    identical failure, so this is a latent robustness gap rather than a port
    discrepancy -- but `sttopt/conventions.md` explicitly lists "`(1+exp(z))^-1` as a
    stable sigmoid instead of the mathematically-equivalent but overflow-prone form"
    as a deliberate improvement the port made, and that claim is not true of the code
    as written: it is the same expression as the MATLAB source.

    `DFT = rouf * FT * (1 - FT)` is the same quantity with no overflow.
    """
    nelx, nely = 5, 4
    rng = np.random.default_rng(1)
    xPhys = rng.uniform(0.3, 0.95, (nely, nelx))
    tPhys = rng.uniform(0.0, 1.0, (nely, nelx))
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, 3.0)
    H = sp.eye(nelx * nely, format="csr")
    Hs = np.ones(nelx * nely)
    dx = np.ones((nely, nelx))

    with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, df1, dt1 = conductivity.hotspot_constraint(
            xPhys, tPhys, e1, e2, w, dx, H, Hs, 1.0, 0.8, 25.0, 3.0, 0.05, 2000.0
        )

    assert np.all(np.isfinite(df1)), f"{np.isnan(df1).sum()} NaNs in df1"
    assert np.all(np.isfinite(dt1)), f"{np.isnan(dt1).sum()} NaNs in dt1"
