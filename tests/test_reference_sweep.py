"""Port-vs-oracle sweep over parameter points the committed `.mat` fixtures never reach.

Every fixture in `tests/fixtures/` comes from a single MATLAB run at one parameter
point: `nelx=7, nely=5, rmin=2, lrmin=2, rmin_cond=3, tfield=3, nStage=3, nloop=3`.
A bug that only bites at a different filter radius, a square/tall grid, another
`tfield` variant, or an iteration past the third (where the periodic `rou`/`beta`/
`factor` schedules first fire) would pass the entire fixture suite untouched.

These tests close that gap using `matlab_reference.py` / `matlab_reference_loop.py`
-- literal transliterations of the MATLAB source, independent of `sttopt/` -- as the
oracle, so no MATLAB installation is needed to run them.

Tolerances here are deliberately tight (see `TIGHT`/`SOLVED`): the two
implementations agree to a few ULP everywhere except downstream of a linear solve,
so anything looser would stop catching regressions. The one documented exception is
`ti` values that drive the stage mask near zero, where the assembled stiffness matrix
reaches cond ~1e10 and the dense-vs-sparse solvers legitimately part company around
1e-8 (see `test_gravity_compliance_matches_reference`).
"""

import matlab_reference as ref
import numpy as np
import pytest
from matlab_reference_loop import run_reference_loop

import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.fem as fem
import sttopt.filters as filters
import sttopt.gravity as gravity
import sttopt.optimize as optimize
import sttopt.timefield as timefield

TIGHT = 1e-11  # purely algebraic quantities
SOLVED = 1e-8  # downstream of a linear solve

GRIDS = [(7, 5), (5, 7), (9, 3), (4, 4), (3, 1), (1, 4)]


def rel(got, want):
    got, want = np.asarray(got, float), np.asarray(want, float)
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    return np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)


@pytest.mark.parametrize("nelx,nely", GRIDS)
@pytest.mark.parametrize("rmin", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
def test_density_filter_matches_reference(nelx, nely, rmin):
    """`density_filter` skips zero-weight pairs the MATLAB loop stores explicitly; that
    is only sparsity-pattern-deep, so H and Hs must still agree exactly. Non-integer
    radii probe the `ceil(rmin)-1` window fencepost.
    """
    H, Hs = filters.density_filter(nelx, nely, rmin)
    H_ref, Hs_ref = ref.ref_density_filter(nelx, nely, rmin)
    assert rel(H.toarray(), H_ref) < TIGHT
    assert rel(Hs, Hs_ref) < TIGHT


@pytest.mark.parametrize("nelx,nely", GRIDS)
@pytest.mark.parametrize("lrmin", [2.0, 2.5, 3.0])
def test_continuity_filter_matches_reference(nelx, nely, lrmin):
    """The port builds `I - D^-1 A` from sparse ops; MATLAB materializes a dense
    `eye(n) - L./M`. Same matrix, and it must stay that way at every radius/grid.
    """
    L = filters.continuity_filter(nelx, nely, lrmin)
    assert rel(L.toarray(), ref.ref_continuity_filter(nelx, nely, lrmin)) < TIGHT


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (9, 3), (4, 4)])
@pytest.mark.parametrize("rmin_cond", [1.0, 2.0, 3.0, 3.5, 4.0, 12.0])
def test_neighbor_weights_match_reference(nelx, nely, rmin_cond):
    """Only `rmin_cond=3` on one grid is fixture-covered; the production script uses 12."""
    nel = nelx * nely
    N_el, w_el = ref.ref_neighbors(nelx, nely, rmin_cond)
    want = np.zeros((nel, nel))
    for i in range(nel):
        for j, e2 in enumerate(N_el[i]):
            want[i, e2 - 1] += w_el[i][j]

    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    got = np.zeros((nel, nel))
    np.add.at(got, (e1, e2), w)
    assert rel(got, want) < TIGHT


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (9, 3), (4, 4)])
@pytest.mark.parametrize("rmin_cond", [2.0, 3.0, 4.0, 12.0])
def test_WE_equals_w_el_generally(nelx, nely, rmin_cond):
    """`conductivity.py` drops MATLAB's separate `WE` cell array and reuses the forward
    weight for both pair directions. `test_conductivity.py` checks that against the one
    committed fixture; this checks the underlying symmetry holds at every grid/radius,
    including grids whose boundary truncation is asymmetric.
    """
    nel = nelx * nely
    N_el, w_el = ref.ref_neighbors(nelx, nely, rmin_cond)
    WE = ref.ref_WE(N_el, w_el, nel)
    for i in range(nel):
        assert np.array_equal(WE[i], w_el[i]), f"WE != w_el at element {i}"


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (9, 3), (2, 2), (4, 1), (1, 4)])
@pytest.mark.parametrize("variant", [1, 2, 3])
def test_timefield_matches_reference(nelx, nely, variant):
    """`timefield.mat` covers all three variants but only one grid; the `linspace(0, n, n)`
    spacing means variants 1 and 3 change shape (not just scale) with the aspect ratio.

    `(4, 1)`/`(1, 4)` are lone-1 meshes: well-defined and finite for every variant here
    (only `nelx == nely == 1` raises, and only for CORNER -- see `timefield.py`).
    """
    got = timefield.init_timefield(nelx, nely, variant)
    assert rel(got, ref.ref_timefield(nelx, nely, variant)) < TIGHT


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (9, 3), (1, 1), (3, 1)])
def test_gravity_matrix_matches_reference(nelx, nely):
    got = gravity.gravity_load_matrix(nelx, nely).toarray()
    assert rel(got, ref.ref_gravity_C(nelx, nely)) < TIGHT


@pytest.mark.parametrize("nu", [0.0, 0.3, 0.45])
def test_KE_matches_reference(nu):
    assert rel(fem.plane_stress_KE(nu), ref.ref_KE(nu)) < TIGHT


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (9, 3), (2, 3)])
def test_edof_map_matches_reference(nelx, nely):
    assert rel(fem.element_dof_map(nelx, nely), ref.ref_edofMat(nelx, nely) - 1) < TIGHT


def _fe_setup(nelx, nely):
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    F[ndof - 1] = -1.0
    freedofs1 = np.setdiff1d(np.arange(1, ndof + 1), np.arange(1, 2 * (nely + 1) + 1))
    return ndof, F, freedofs1


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (9, 3)])
def test_whole_compliance_matches_reference(nelx, nely):
    ndof, F, freedofs1 = _fe_setup(nelx, nely)
    KE = ref.ref_KE(0.3)
    xP = np.random.default_rng(7).uniform(0.2, 1.0, (nely, nelx))
    c_ref, dcx_ref = ref.ref_whole_compliance(
        nelx, nely, KE, xP, 1e-9, 1.0, 3, freedofs1, F
    )
    c, dcx = compliance.whole_compliance(
        xP, KE, fem.element_dof_map(nelx, nely), 1e-9, 1.0, 3, freedofs1 - 1, F, ndof
    )
    assert abs(c - c_ref) / abs(c_ref) < SOLVED
    assert rel(dcx, dcx_ref) < SOLVED


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (9, 3)])
@pytest.mark.parametrize("ti", [0.25, 0.5, 1.0])
def test_gravity_compliance_matches_reference(nelx, nely, ti):
    """Tolerance here is *derived* from the conditioning of the system actually solved,
    not fixed.

    At small `ti` the stage mask drives most of `xtJoint` down to ~1e-8, the SIMP floor
    `Emin` takes over, and the free partition of the assembled stiffness reaches cond
    ~1e12 -- at which point this test's dense `numpy.linalg.solve` and the port's
    sparse `spsolve` legitimately disagree in the low digits. A single blanket
    tolerance would then have to be loose enough for the worst case (~1e-4) and would
    stop testing anything at all for the well-conditioned `ti` values, where the two
    agree to ~1e-14.

    So instead: `tol = 10 * eps * cond(K_free)`. Across this parametrization the
    observed disagreement never exceeds `0.2 * eps * cond(K)`, so the factor of 10
    leaves ~50x headroom while keeping the tolerance proportional to what the solve
    can actually deliver -- 1e-11 for the well-conditioned cases, 1e-3 only for the
    genuinely singular-ish one.
    """
    ndof, F, freedofs1 = _fe_setup(nelx, nely)
    KE = ref.ref_KE(0.3)
    rng = np.random.default_rng(11)
    xP = rng.uniform(0.2, 1.0, (nely, nelx))
    tP = rng.uniform(0.0, 1.0, (nely, nelx))
    C_ref = ref.ref_gravity_C(nelx, nely)
    c_ref, dcx_ref, dct_ref = ref.ref_gravity_compliance(
        nelx, nely, KE, xP, tP, 1e-9, 1.0, 3, ti, C_ref, 10.0, freedofs1
    )
    c, dcx, dct = compliance.gravity_compliance(
        xP,
        tP,
        KE,
        fem.element_dof_map(nelx, nely),
        1e-9,
        1.0,
        3,
        ti,
        gravity.gravity_load_matrix(nelx, nely),
        10.0,
        freedofs1 - 1,
        ndof,
    )
    ft = compliance.time_mask(tP, ti, 10.0)
    dens = 1e-9 + (xP * ft).flatten(order="F") ** 3 * (1.0 - 1e-9)
    K_free = ref._assemble(KE, dens, ref.ref_edofMat(nelx, nely), ndof)[
        np.ix_(freedofs1 - 1, freedofs1 - 1)
    ]
    tol = 10 * np.finfo(float).eps * np.linalg.cond(K_free)

    assert abs(c - c_ref) / abs(c_ref) < tol
    assert rel(dcx, dcx_ref) < tol
    assert rel(dct, dct_ref) < tol


@pytest.mark.parametrize("nelx,nely", [(7, 5), (5, 7), (9, 3)])
@pytest.mark.parametrize("rmin_cond", [2.0, 3.0, 4.0])
@pytest.mark.parametrize("factor", [1.0, 0.7])
def test_hotspot_matches_reference(nelx, nely, rmin_cond, factor):
    """The hardest block in the port: MATLAB's per-element neighbour loop with its
    diagonal/off-diagonal `N_sub1`/`N_sub2` split and the `WE`/`FT_ba` role reversal,
    rewritten as flat COO arithmetic. Swept over grid shape, neighbourhood radius and
    the stateful `factor`, on a tie-free time field.
    """
    nel = nelx * nely
    rng = np.random.default_rng(3)
    xP = rng.uniform(0.2, 1.0, (nely, nelx))
    tP = rng.uniform(0.0, 1.0, (nely, nelx))
    dx = rng.uniform(0.5, 1.5, (nely, nelx))
    assert len(np.unique(tP)) == nel, "fixture field must be tie-free"

    N_el, w_el = ref.ref_neighbors(nelx, nely, rmin_cond)
    WE = ref.ref_WE(N_el, w_el, nel)
    H_ref, Hs_ref = ref.ref_density_filter(nelx, nely, 2.0)
    fv_ref, df_ref, dt_ref, K_ref, _ = ref.ref_hotspot(
        nelx, nely, xP, tP, N_el, w_el, WE, H_ref, Hs_ref, dx, factor, 0.8
    )

    H, Hs = filters.density_filter(nelx, nely, 2.0)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
    result = conductivity.hotspot_constraint(
        xP, tP, e1, e2, w, dx, H, Hs, factor, 0.8, 25.0, 3.0, 0.05, 100.0
    )
    K = conductivity.estimated_conductivity(xP, tP, e1, e2, w, 3.0, 100.0)

    assert rel(K, K_ref) < TIGHT
    assert abs(result.fval - fv_ref) / abs(fv_ref) < TIGHT
    assert rel(result.df1, df_ref) < 1e-10
    assert rel(result.dt1, dt_ref) < 1e-10


@pytest.mark.parametrize(
    "p,q,r,rouf", [(8, 2, 0.2, 40.0), (12, 4, 0.5, 250.0), (25, 3, 1.0, 100.0)]
)
def test_hotspot_non_default_constants(p, q, r, rouf):
    """`p=25, q=3, r=0.05, rouf=100` are the only values any fixture pins, and they are
    hardcoded in the MATLAB block -- so a constant accidentally baked into the port
    instead of read from its argument would go unnoticed. These are non-default.
    """
    nelx, nely = 7, 5
    rng = np.random.default_rng(2)
    xP = rng.uniform(0.3, 0.95, (nely, nelx))
    tP = rng.uniform(0.05, 0.95, (nely, nelx))
    dx = np.ones((nely, nelx))
    N_el, w_el = ref.ref_neighbors(nelx, nely, 3.0)
    WE = ref.ref_WE(N_el, w_el, nelx * nely)
    H_ref, Hs_ref = ref.ref_density_filter(nelx, nely, 2.0)
    fv_ref, df_ref, dt_ref, _, _ = ref.ref_hotspot(
        nelx,
        nely,
        xP,
        tP,
        N_el,
        w_el,
        WE,
        H_ref,
        Hs_ref,
        dx,
        1.0,
        0.8,
        p=p,
        q=q,
        r=r,
        rouf=rouf,
    )
    H, Hs = filters.density_filter(nelx, nely, 2.0)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, 3.0)
    result = conductivity.hotspot_constraint(
        xP, tP, e1, e2, w, dx, H, Hs, 1.0, 0.8, float(p), float(q), r, rouf
    )
    assert abs(result.fval - fv_ref) / abs(fv_ref) < TIGHT
    assert rel(result.df1, df_ref) < TIGHT
    assert rel(result.dt1, dt_ref) < TIGHT


# ----------------------------------------------------------------- full loop

LOOP_CASES = [
    # (nelx, nely, nloop, nStage, volfrac, Theta, Tcr, tfield, rmin, lrmin, rmin_cond)
    pytest.param(
        7, 5, 3, 3, 0.5, 0.1, 0.8, 1, 2.0, 2.0, 3.0, id="tfield1-single-startpoint"
    ),
    pytest.param(6, 4, 3, 2, 0.4, 0.2, 0.7, 3, 3.0, 2.0, 4.0, id="wider-radii"),
    pytest.param(4, 4, 3, 2, 0.5, 0.1, 0.8, 3, 2.0, 2.0, 3.0, id="square-grid"),
    pytest.param(3, 6, 3, 2, 0.5, 0.1, 0.8, 3, 2.0, 2.0, 3.0, id="tall-grid"),
    pytest.param(6, 3, 3, 4, 0.3, 0.5, 0.6, 1, 3.0, 3.0, 2.0, id="tfield1-4-stages"),
]


@pytest.mark.parametrize(
    "nelx,nely,nloop,nStage,volfrac,Theta,Tcr,tfield,rmin,lrmin,rmin_cond", LOOP_CASES
)
def test_full_loop_matches_reference(
    nelx, nely, nloop, nStage, volfrac, Theta, Tcr, tfield, rmin, lrmin, rmin_cond
):
    """`optimize.step`'s constraint row order and state threading, against the literal
    main-loop transliteration. `tfield=1` matters most here: it is the only variant
    where `Nei` (and so the constraint-row count `m`) collapses from `nely` rows to
    one, and no fixture exercises it.
    """
    trace = run_reference_loop(
        nelx, nely, nloop, nStage, volfrac, Theta, Tcr, tfield, rmin, lrmin, rmin_cond
    )
    result = optimize.run(
        nelx, nely, nloop, nStage, volfrac, Theta, Tcr, tfield, rmin, lrmin, rmin_cond
    )
    for k, (rec, want) in enumerate(zip(result.records, trace), start=1):
        assert len(rec.fval) == want["m"], f"iteration {k}: constraint count"
        assert rel([rec.f0val], [want["f0val"]]) < SOLVED, f"iteration {k}: f0val"
        assert rel(rec.fval, want["fval"]) < SOLVED, f"iteration {k}: fval"
        assert rel(rec.dfdx, want["dfdx"]) < SOLVED, f"iteration {k}: dfdx"
        assert rel(rec.xmma, want["xmma"]) < SOLVED, f"iteration {k}: xmma"


def test_periodic_schedules_match_reference():
    """The `loop % 25` hotspot-`factor` refresh, `loop % 30` `rou` bump and `loop % 50`
    `beta` doubling all sit past the fixture's `nloop=3`. This runs far enough to fire
    all three and checks the trajectory still tracks the reference loop, so a
    misplaced schedule (off-by-one iteration, or applying the new `beta` to the wrong
    field) shows up as a trajectory divergence rather than passing silently.
    """
    nelx, nely, nloop, nStage = 5, 3, 51, 2
    args = (nelx, nely, nloop, nStage, 0.5, 0.1, 0.8, 3, 2.0, 2.0, 3.0)
    trace = run_reference_loop(*args)
    result = optimize.run(*args)

    fired = {(w["loop"], w["rou"], w["beta"], round(w["factor"], 12)) for w in trace}
    assert (25, 10.0, 1.0, round(trace[24]["factor"], 12)) in fired
    assert trace[24]["factor"] != 1.0, "factor refresh at loop 25 did not fire"
    assert trace[29]["rou"] == 15.0, "rou bump at loop 30 did not fire"
    assert trace[49]["beta"] == 2.0, "beta doubling at loop 50 did not fire"

    for k, (rec, want) in enumerate(zip(result.records, trace), start=1):
        assert rel([rec.f0val], [want["f0val"]]) < SOLVED, f"iteration {k}: f0val"
        assert rel([rec.tru_max], [want["tru_max"]]) < SOLVED, f"iteration {k}: tru_max"
        assert rel(rec.fval, want["fval"]) < SOLVED, f"iteration {k}: fval"
        assert rel(rec.dfdx, want["dfdx"]) < SOLVED, f"iteration {k}: dfdx"
