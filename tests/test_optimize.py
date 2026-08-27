"""First-principles tests for `sttopt.optimize`: the wiring layer.

Every module `optimize.step` calls owns its own FD/fixture tests, and every one of those
passing tells you nothing about whether `step` *assembles* them correctly. The assembly
is where the `Theta` weighting, the `H @ (... / Hs)` chain rules and the constraint row
order live, and until now it was covered only by the `.mat` trajectory fixtures and the
MATLAB-transliteration oracle in `matlab_reference_loop.py` -- i.e. only ever by
"matches MATLAB", never by "is the derivative of the thing it claims to differentiate".

The FD test here closes that: it takes `step`'s own stacked `df0dx`/`dfdx` and checks
them against central differences of `step`'s own `f0val`/`fval`, with respect to the raw
design vector `[x; t]` that MMA actually optimizes. It is also the acceptance criterion
for swapping the hand-derived sensitivities for autodiff -- the replacement has to pass
this unchanged.

`init_state`'s tests state the initialization invariant directly, rather than pinning
whatever the current code happens to produce.
"""

import dataclasses

import numpy as np
import pytest
import torch

import sttopt.compliance as compliance
import sttopt.fem as fem
import sttopt.filters as filters
import sttopt.optimize as optimize
import sttopt.timefield as timefield
import sttopt.torch_util as torch_util

VOLFRAC = 0.4
TCR = 0.8
RMIN = LRMIN = 2
RMIN_COND = 3
BETA_D = 1.0

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)


def _filter_field(problem, raw):
    """`H @ raw / Hs`, on the (nely, nelx) grid -- the density filter's action.

    `raw` may be a plain NumPy array (an FD test's perturbed design point) or a
    `State` tensor field; `problem.H`/`.Hs` are always tensors. Returns a tensor,
    matching what `optimize.step`/`init_state` themselves compute.
    """
    device, dtype = problem.device, problem.dtype
    flat = (problem.H @ torch_util.to_tensor(raw, device, dtype).flatten()) / problem.Hs
    return flat.reshape((problem.nely, problem.nelx))


# --- init_state: the initialization invariant ----------------------------------------
#
# `State` carries two fields per design variable, and their relationship is the whole
# point of the split (see optimize.py's module docstring): `x`/`t` are the *raw* design
# variables MMA reads and writes, and `xTilde`/`xPhys`/`tPhys` are *derived* from them by
# the density filter (and, for density, the Heaviside projection). Every iteration of
# `step` maintains exactly that relationship on the state it emits:
#
#     xTilde = H @ x / Hs      xPhys = heaviside(xTilde)      tPhys = H @ t / Hs
#
# `init_state` seeds the raw fields and must therefore derive the physics fields from
# them the same way, so that iteration 1 sees a state indistinguishable in kind from the
# one iteration 2 sees. Anything else is a forward/backward inconsistency: `step`
# differentiates through the filter it assumes produced `tPhys`, but at iteration 1 no
# filter did.


def _assert_state_fields_are_consistent(problem, state, beta_d):
    """The invariant above, asserted on a `State` from any source."""
    np.testing.assert_allclose(
        state.xTilde, _filter_field(problem, state.x), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        state.xPhys,
        filters.heaviside_projection(state.xTilde, beta_d, problem.eta),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        state.tPhys, _filter_field(problem, state.t), rtol=1e-12, atol=1e-14
    )


def _problem(nelx=7, nely=5, nStage=3, tfield=3, Theta=1.0):
    return optimize.build_problem(
        nelx, nely, nStage, VOLFRAC, Theta, TCR, tfield, RMIN, LRMIN, RMIN_COND
    )


# --- Phase 3.1 (plans/torch_port_part2.md): the tensor boundary -----------------------


def _tensor_fields(obj) -> list[tuple[str, torch.Tensor]]:
    """Every tensor-valued field of a `Problem`/`State`, as `(name, value)` pairs."""
    return [
        (f.name, getattr(obj, f.name))
        for f in dataclasses.fields(obj)
        if isinstance(getattr(obj, f.name), torch.Tensor)
    ]


def test_build_problem_default_device_and_dtype():
    """`build_problem`'s default (`device="cpu"`, `dtype=torch.float64`) must cost every
    pre-Phase-3.1 caller nothing: every real-valued tensor field lands on CPU/float64,
    matching what those fields were (as NumPy arrays) before this phase."""
    problem = _problem()
    assert problem.device == torch.device("cpu")
    assert problem.dtype == torch.float64
    for name, t in _tensor_fields(problem):
        assert t.device == torch.device("cpu"), name
        if t.dtype.is_floating_point:
            assert t.dtype == torch.float64, name


def test_build_problem_honors_requested_dtype():
    """A non-default floating `dtype` reaches every real-valued tensor field; index
    (`edofMat`/`freedofs`/`e1`/`e2`/`Nei`) and mask (`free_mask`) fields keep their own
    int64/bool dtype regardless -- `dtype` governs the problem's real-valued fields, not
    every tensor it happens to hold."""
    problem = optimize.build_problem(
        7, 5, 3, VOLFRAC, 1.0, TCR, 3, RMIN, LRMIN, RMIN_COND, dtype=torch.float32
    )
    assert problem.dtype == torch.float32
    for name, t in _tensor_fields(problem):
        if t.dtype.is_floating_point:
            assert t.dtype == torch.float32, name
        else:
            assert t.dtype in (torch.int64, torch.bool), name


def test_init_state_and_step_output_are_tensors_on_problem_device_and_dtype():
    """`State`'s fields are tensors (Phase 3.1), on `problem`'s own device/dtype, both
    fresh out of `init_state` and after a `step` call -- the tensor boundary inside
    `step` must land back on `problem.device`/`.dtype`, not wherever the (still-NumPy)
    leaf math happened to leave its output."""
    problem = _problem()
    state = optimize.init_state(problem, BETA_D)
    for name, t in _tensor_fields(state):
        assert t.device == problem.device, name
        assert t.dtype == problem.dtype, name

    state, _ = optimize.step(problem, state)
    for name, t in _tensor_fields(state):
        assert t.device == problem.device, name
        assert t.dtype == problem.dtype, name


@requires_cuda
def test_build_problem_on_cuda_has_no_lingering_cpu_tensor():
    """The plan's Phase 3.1 test: every tensor field of a CUDA `Problem` is actually on
    CUDA. A field left on a stray default device would pass every CPU-only test and
    only surface as silently wrong, or a device-mismatch crash, once later phases run
    the loop on the GPU."""
    problem = optimize.build_problem(
        7, 5, 3, VOLFRAC, 1.0, TCR, 3, RMIN, LRMIN, RMIN_COND, device="cuda"
    )
    assert problem.device.type == "cuda"
    for name, t in _tensor_fields(problem):
        assert t.device.type == "cuda", name


@pytest.mark.parametrize("tfield", [1, 2, 3])
def test_build_problem_rejects_the_1x1_mesh(tfield):
    """A 1x1 mesh degenerates two of `build_problem`'s pieces -- the distance time fields
    normalize by a zero max distance, and the continuity filter divides by a zero
    neighbour count -- so it must be rejected up front, before either produces a `nan` or
    a divide-by-zero warning. Lone-1 meshes stay legal (see `test_timefield.py`)."""
    with pytest.raises(ValueError):
        _problem(nelx=1, nely=1, tfield=tfield)

    _problem(nelx=1, nely=4, tfield=tfield)
    _problem(nelx=4, nely=1, tfield=tfield)


def test_density_filter_fixes_constant_fields():
    """`H @ 1 / Hs == 1` up to rounding, because `Hs` is by construction `H`'s row sum.
    This is what makes filtering the uniform density seed a no-op in practice, so that
    the correction to `init_state` (PR #26) changed only the time half: measured, the
    filtered seed differs from `volfrac` by ~1.7e-16, one ulp, not bit-for-bit -- the
    row-sum division is exact in exact arithmetic, not in floating point."""
    for nelx, nely in [(7, 5), (10, 8), (4, 4)]:
        H, Hs = filters.density_filter(nelx, nely, RMIN)
        ones = np.ones(nelx * nely)
        np.testing.assert_allclose(H @ ones / Hs, ones, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("tfield", [1, 2, 3])
def test_init_state_seeds_the_raw_fields(tfield):
    """The raw fields are the seed: uniform `volfrac` for density, `init_timefield` for
    print time. These are the variables MMA's move limits are measured against, so they
    are what "initial design" means."""
    problem = _problem(tfield=tfield)
    state = optimize.init_state(problem, BETA_D)

    np.testing.assert_allclose(state.x, VOLFRAC, rtol=1e-14)
    np.testing.assert_allclose(
        state.t,
        timefield.init_timefield(problem.nelx, problem.nely, tfield),
        rtol=1e-14,
        atol=1e-15,
    )


@pytest.mark.parametrize("tfield", [1, 2, 3])
def test_init_state_density_half_is_derived_from_its_seed(tfield):
    """The density half of the invariant. It holds today only because the seed is
    uniform and `H @ 1 / Hs == 1` makes filtering a no-op -- but it holds *exactly*, so
    it pins the intended relationship rather than tolerating it: `xTilde` must equal the
    filtered raw field, and `xPhys` the projection of that."""
    problem = _problem(tfield=tfield)
    state = optimize.init_state(problem, BETA_D)

    np.testing.assert_allclose(
        state.xTilde, _filter_field(problem, state.x), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(state.xTilde, VOLFRAC, rtol=1e-14)
    np.testing.assert_allclose(
        state.xPhys,
        filters.heaviside_projection(state.xTilde, BETA_D, problem.eta),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("tfield", [1, 2, 3])
def test_init_state_time_half_is_derived_from_its_seed(tfield):
    """The time half of the same invariant -- the half the MATLAB source got wrong.

    It assigns `t = tPhys` unfiltered, so at iteration 1 the physics field was the raw
    seed while `step` went on to differentiate it as though `tPhys = H @ t / Hs`.

    Unlike the density half, this cannot be waved away by the filter's constant-field
    fixed point: no `init_timefield` variant is constant (a constant print-time field
    would mean the whole part is deposited at once), so `H @ t / Hs != t` here for real.
    The premise assertion below pins exactly that, so the test cannot pass vacuously.
    """
    problem = _problem(tfield=tfield)
    state = optimize.init_state(problem, BETA_D)

    seed = timefield.init_timefield(problem.nelx, problem.nely, tfield)
    assert not np.allclose(_filter_field(problem, seed), seed), (
        "premise: the seed is not a fixed point of the density filter, so filtering it "
        "is observable"
    )
    np.testing.assert_allclose(
        state.tPhys, _filter_field(problem, state.t), rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("tfield", [1, 3])
def test_step_output_state_is_self_consistent(tfield):
    """The same invariant on the state `step` emits -- which does hold, at every
    iteration. This is the behaviour `init_state` is being specified against above, so
    pinning it here keeps the target from drifting."""
    problem = _problem(tfield=tfield)
    state = optimize.init_state(problem, BETA_D)
    for _ in range(3):
        state, _ = optimize.step(problem, state)
        _assert_state_fields_are_consistent(problem, state, state.beta_d)


# --- step: finite-difference check of the assembled sensitivities ---------------------


def _state_from_raw(problem, x_raw, t_raw, *, beta_d=BETA_D, factor=1.0, beta_t=10.0):
    """A `State` at raw design point `[x_raw; t_raw]`, with the physics fields derived
    per the invariant above -- i.e. the state `step` itself would have produced.

    `loop` is left at 0 so the ensuing `step` runs as iteration 1, which is none of
    30/50/25: `beta_t`, `beta_d` and `factor` all stay fixed across the call, so `f0val`/`fval`
    are smooth functions of the raw variables alone. (At a refresh iteration `factor`
    jumps as a function of the design, and the reported gradient deliberately does not
    account for that -- a different question from the one this test asks.)
    """
    nely, nelx = problem.nely, problem.nelx
    device, dtype = problem.device, problem.dtype
    xTilde = _filter_field(problem, x_raw)
    return optimize.State(
        x=torch_util.to_tensor(x_raw, device, dtype),
        xTilde=xTilde,
        xPhys=filters.heaviside_projection(xTilde, beta_d, problem.eta),
        t=torch_util.to_tensor(t_raw, device, dtype),
        tPhys=_filter_field(problem, t_raw),
        xold1=torch.zeros(problem.n, device=device, dtype=dtype),
        xold2=torch.zeros(problem.n, device=device, dtype=dtype),
        low=torch.zeros(problem.n, device=device, dtype=dtype),
        upp=torch.zeros(problem.n, device=device, dtype=dtype),
        loop=0,
        beta_t=beta_t,
        beta_d=beta_d,
        factor=factor,
    )


# `step` runs one whole-structure FEM solve plus one per stage, each on a differently
# time-masked density field, and any of them can land near a mechanism -- at which point
# FD noise blows up long before the analytic gradient does. Following `test_compliance.py`,
# such draws are rejected and redrawn rather than papered over by loosening tolerances.
#
# The threshold is much looser than test_compliance.py's 1e5, and deliberately so. The
# earliest stage (ti = 1/nStage) masks nearly the whole structure down to the Emin floor,
# so its K_free is ill-conditioned *structurally*, for every draw and every mesh -- a 1e5
# cutoff rejects 100% of draws and the test simply never runs. Measured across draws the
# whole-structure and late-stage solves sit at ~2e3 while the first stage spans 5e4-2e8,
# and an h-sweep (1e-5 / 1e-6 / 1e-7) shows every constraint row still agreeing with FD to
# ~1e-11 absolute at the top of that range: the conditioning does not reach the gradients.
# So the guard is set to catch genuinely degenerate draws, not the normal early-stage
# range, and the tolerances below stay tight.
MAX_COND = 1e10


def _well_conditioned(problem, state, beta_t):
    # `fem.assemble_stiffness` is unported NumPy (Phase 3.1 leaves it that way -- see
    # sttopt/optimize.py's module docstring); `problem`'s tensor fields need the same
    # bridge `optimize.step` uses internally.
    p = problem
    KE = torch_util.to_numpy(p.KE)
    edofMat = torch_util.to_numpy(p.edofMat)
    freedofs = torch_util.to_numpy(p.freedofs)

    fields = [state.xPhys]
    tP = np.linspace(0, 1, p.nStage + 1)
    for i in range(1, p.nStage + 1):
        fields.append(state.xPhys * compliance.time_mask(state.tPhys, tP[i], beta_t))
    for field in fields:
        K = fem.assemble_stiffness(
            KE, torch_util.to_numpy(field), p.Emin, p.Emax, p.penal, edofMat, p.ndof
        )
        Kfree = K[np.ix_(freedofs, freedofs)].toarray()
        if np.linalg.cond(Kfree) >= MAX_COND:
            return False
    return True


def _draw_well_conditioned_state(problem, rng, *, beta_t=10.0, max_tries=50):
    for _ in range(max_tries):
        x_raw = rng.uniform(0.3, 0.7, size=(problem.nely, problem.nelx))
        t_raw = rng.uniform(0.1, 0.9, size=(problem.nely, problem.nelx))
        state = _state_from_raw(problem, x_raw, t_raw, beta_t=beta_t)
        if _well_conditioned(problem, state, beta_t):
            return x_raw, t_raw, state
    raise AssertionError(f"no well-conditioned draw in {max_tries} tries")


@pytest.mark.parametrize(
    "nStage,tfield,Theta",
    [
        (3, 3, 1.0),
        (2, 1, 0.35),  # tfield==1 shrinks Nei to a singleton, changing m
        (4, 2, 2.5),
    ],
)
def test_step_assembled_sensitivities_match_finite_differences(nStage, tfield, Theta):
    """`step`'s stacked `df0dx` and `dfdx` against central differences of its own
    `f0val`/`fval`, w.r.t. the raw design vector `[x; t]`.

    This is the only check in the suite on the *assembly* rather than the parts. In
    particular it is the only thing that pins, without reference to MATLAB:

      * the `Theta` weighting of the per-stage gravity compliances into `f0val`, and the
        matching `Theta` on their contributions to `dc`/`dt`;
      * the `H @ (dct_g / Hs)` chain rule on the time half -- note the density half
        carries an extra `dx` (Heaviside) factor and the time half does not, an asymmetry
        nothing else tests;
      * the constraint *row order* and the row count `m`. A permuted or miscounted stack
        makes row `k` of `dfdx` the derivative of a different row of `fval`, and the
        comparison fails per-row.

    `Theta != 1` and `nStage != 3` are swept because the fixture only ever exercises
    `Theta == 1`, `nStage == 3`: at `Theta == 1` a missing weight is invisible, and at a
    single `nStage` a stage-indexing error can hide.
    """
    nelx, nely = 10, 8
    # f0val is O(1e3) here (a compliance), so its central difference is roundoff-dominated
    # and *improves* with a larger step: an h-sweep gives absolute errors of 1.2e-2 /
    # 1.0e-1 / 1.0e0 at h = 1e-5 / 1e-6 / 1e-7 -- a clean 1/h roundoff slope, no truncation
    # regime in reach. 1e-5 is the best of them. The constraint rows are O(0.1) and match
    # to ~1e-11 at every h in that sweep.
    h = 1e-5
    problem = _problem(nelx=nelx, nely=nely, nStage=nStage, tfield=tfield, Theta=Theta)
    nel = nelx * nely
    assert problem.n == 2 * nel

    rng = np.random.default_rng(0)
    x_raw, t_raw, state = _draw_well_conditioned_state(problem, rng)
    _, record = optimize.step(problem, state)

    # Row count follows from the stack `step` builds: volume, continuity, one row per
    # print-start element, an upper and a lower bound per stage, and the hotspot row.
    assert problem.m == 1 + 1 + len(problem.Nei) + 2 * nStage + 1
    assert record.df0dx.shape == (problem.n,)
    assert record.fval.shape == (problem.m,)
    assert record.dfdx.shape == (problem.m, problem.n)

    # Non-vacuity: an all-but-zero gradient would pass the comparison below regardless.
    assert np.abs(record.df0dx).max() > 1e-3
    assert np.abs(record.dfdx).max(axis=1).min() > 1e-3

    def values_at(x_raw, t_raw):
        _, rec = optimize.step(problem, _state_from_raw(problem, x_raw, t_raw))
        return rec.f0val, rec.fval

    fd_f0 = np.zeros(problem.n)
    fd_f = np.zeros((problem.m, problem.n))
    for e in range(nel):
        j, i = e // nelx, e % nelx

        xp, xm = x_raw.copy(), x_raw.copy()
        xp[j, i] += h
        xm[j, i] -= h
        f0_p, f_p = values_at(xp, t_raw)
        f0_m, f_m = values_at(xm, t_raw)
        fd_f0[e] = (f0_p - f0_m) / (2 * h)
        fd_f[:, e] = (f_p - f_m) / (2 * h)

        tp, tm = t_raw.copy(), t_raw.copy()
        tp[j, i] += h
        tm[j, i] -= h
        f0_p, f_p = values_at(x_raw, tp)
        f0_m, f_m = values_at(x_raw, tm)
        fd_f0[nel + e] = (f0_p - f0_m) / (2 * h)
        fd_f[:, nel + e] = (f_p - f_m) / (2 * h)

    # df0dx spans several orders of magnitude across elements, so an absolute floor has
    # to be set against the gradient's own scale rather than against 1: at h = 1e-5 the
    # observed error is ~2e-6 of max|df0dx|, so a floor at 1e-4 of it leaves ~50x margin
    # while still being far tighter than the entries it is guarding.
    np.testing.assert_allclose(
        record.df0dx, fd_f0, rtol=1e-4, atol=1e-4 * np.abs(record.df0dx).max()
    )
    # The constraint rows are purely algebraic in xPhys/tPhys (no linear solve), and
    # match ~1000x tighter than this.
    for row in range(problem.m):
        np.testing.assert_allclose(
            record.dfdx[row],
            fd_f[row],
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"constraint row {row} of {problem.m}",
        )


def test_step_objective_is_theta_weighted_sum_of_stage_compliances():
    """`f0val` is the whole-structure compliance plus `Theta` times the sum of the
    per-stage gravity compliances -- so it must be exactly affine in `Theta`, with
    intercept the whole-structure compliance alone and slope the stage sum. Recovering
    both from three `Theta` values pins the weighting independently of the FD check
    (which sees the gradient, not the value), and `Theta == 0` recovers the pure
    compliance problem the stage terms are layered onto.
    """
    nelx, nely, nStage = 10, 8, 3
    rng = np.random.default_rng(1)

    def f0val_at(Theta):
        problem = _problem(nelx=nelx, nely=nely, nStage=nStage, Theta=Theta)
        state = _state_from_raw(problem, x_raw, t_raw)
        _, rec = optimize.step(problem, state)
        return rec.f0val, rec.obj

    base = _problem(nelx=nelx, nely=nely, nStage=nStage)
    x_raw, t_raw, _ = _draw_well_conditioned_state(base, rng)

    f0_0, obj_0 = f0val_at(0.0)
    f0_1, _ = f0val_at(1.0)
    f0_3, _ = f0val_at(3.0)

    # At Theta = 0 the stage terms drop out and f0val is the whole-structure compliance,
    # which `IterationRecord.obj` reports separately at every Theta.
    np.testing.assert_allclose(f0_0, obj_0, rtol=1e-12)
    stage_sum = f0_1 - f0_0
    assert stage_sum > 1e-3  # non-vacuous: the stage terms actually contribute
    np.testing.assert_allclose(f0_3, f0_0 + 3.0 * stage_sum, rtol=1e-9)
