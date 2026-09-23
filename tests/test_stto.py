"""First-principles tests for `sttopt.stto`: the wiring layer.

Every module `stto.step` calls owns its own FD/fixture tests, and every one of those
passing tells you nothing about whether `step` wires them together correctly. The FD
test here checks `step`'s own `IterationRecord.df`/`.dg` against central differences of
`step`'s own `.f`/`.g`, with respect to the raw design vector `[x; t]` that MMA actually
optimizes -- "is the derivative of the thing it claims to differentiate", independent
of the MATLAB fixtures.

`init_state`'s tests state the initialization invariant directly, rather than pinning
whatever the current code happens to produce.
"""

import dataclasses

import numpy as np
import pytest
import torch

import sttopt.compliance as compliance
import sttopt.filters as filters
import sttopt.run_config as run_config
import sttopt.stto as stto
import sttopt.timefield as timefield
import sttopt.torch_solve as torch_solve
import sttopt.torch_util as torch_util
import tests.reference.fem as fem_ref
from conftest import FIXTURES_DIR, default_run_config

VOLFRAC = 0.4
TCR = 0.8
RMIN = LRMIN = 2
RMIN_COND = 3
BETA_D = 1.0

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)


def _problem(
    nelx=7,
    nely=5,
    nStage=3,
    tfield=3,
    Theta=1.0,
    uniformity_weight=0.0,
    enable_stage_volume=True,
    config_overrides=None,
    **kwargs,
):
    """A small `Problem` at production defaults bar the arguments named here.

    :param config_overrides: further `RunConfig` fields to override, for a test that
        needs one the parameters above do not name.
    """
    config = default_run_config(
        nelx=nelx,
        nely=nely,
        nStage=nStage,
        enable_stage_volume=enable_stage_volume,
        volfrac=VOLFRAC,
        Theta=Theta,
        uniformity_weight=uniformity_weight,
        Tcr=TCR,
        print_base=timefield.TimeField(tfield).name.lower(),
        rmin=RMIN,
        time_filter_rmin=RMIN,
        lrmin=LRMIN,
        rmin_cond=RMIN_COND,
        **(config_overrides or {}),
    )
    return stto.build_problem(config, **kwargs)


# --- Phase 3.1 (plans/torch_port_part2.md): the tensor boundary -----------------------


def _tensor_fields(obj) -> list[tuple[str, torch.Tensor]]:
    """Every tensor-valued field of a `Problem`/`State`, as `(name, value)` pairs."""
    return [
        (f.name, getattr(obj, f.name))
        for f in dataclasses.fields(obj)
        if isinstance(getattr(obj, f.name), torch.Tensor)
    ]


def test_build_problem_default_device_and_dtype():
    """`build_problem`'s default device (CUDA when available, else CPU) and default
    `dtype=torch.float64` reach every real-valued tensor field of the returned
    `Problem`."""
    # `torch.device("cuda") != torch.device("cuda", 0)`, but every tensor actually
    # built on CUDA reports the latter -- so this test (and the codebase generally)
    # compares device *type*, not exact `torch.device` equality.
    expected_type = "cuda" if torch.cuda.is_available() else "cpu"
    problem = _problem()
    assert problem.device.type == expected_type
    assert problem.dtype == torch.float64
    for name, t in _tensor_fields(problem):
        assert t.device.type == expected_type, name
        if t.dtype.is_floating_point:
            assert t.dtype == torch.float64, name


def test_build_problem_honors_requested_dtype():
    """A non-default floating `dtype` reaches every real-valued tensor field; index
    (`edofMat`/`freedofs`/`e1`/`e2`/`Nei`) and mask (`free_mask`) fields keep their own
    int64/bool dtype regardless -- `dtype` governs the problem's real-valued fields, not
    every tensor it happens to hold."""
    problem = _problem(dtype=torch.float32)
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
    state = stto.init_state(problem)
    for name, t in _tensor_fields(state):
        assert t.device.type == problem.device.type, name
        assert t.dtype == problem.dtype, name

    state, _ = stto.step(problem, state)
    for name, t in _tensor_fields(state):
        assert t.device.type == problem.device.type, name
        assert t.dtype == problem.dtype, name


@requires_cuda
def test_build_problem_on_cuda_has_no_lingering_cpu_tensor():
    """The plan's Phase 3.1 test: every tensor field of a CUDA `Problem` is actually on
    CUDA. A field left on a stray default device would pass every CPU-only test and
    only surface as silently wrong, or a device-mismatch crash, once later phases run
    the loop on the GPU."""
    problem = _problem(device="cuda")
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
    state = stto.init_state(problem)

    np.testing.assert_allclose(torch_util.to_numpy(state.x), VOLFRAC, rtol=1e-14)
    np.testing.assert_allclose(
        torch_util.to_numpy(state.t),
        timefield.init_timefield(problem.config.nelx, problem.config.nely, tfield),
        rtol=1e-14,
        atol=1e-15,
    )


# --- step: finite-difference check of the assembled sensitivities ---------------------

# An iteration that refreshes nothing: not 0, and not a multiple of any refresh period
# these tests use.
QUIET_LOOP = 1


def _state_from_raw(problem, x_raw, t_raw, *, beta_d=BETA_D, beta_t=10.0):
    """A `State` at raw design point `[x_raw; t_raw]`.

    `loop` is `QUIET_LOOP`, not 0, so the ensuing `step` runs an iteration that refreshes
    nothing: iteration 0 always recalibrates the hotspot, and a refresh multiple would too.
    `beta_t`, `beta_d` and the calibration then stay fixed across the call, so `.f`/`.g`
    are smooth functions of the raw variables alone. (At a refresh iteration the
    calibration jumps as a function of the design, and the reported gradient deliberately
    does not account for that -- a different question from the one this test asks.)
    """
    device, dtype = problem.device, problem.dtype
    return stto.State(
        x=torch_util.to_tensor(x_raw, device, dtype),
        t=torch_util.to_tensor(t_raw, device, dtype),
        xold1=torch.zeros(problem.n, device=device, dtype=dtype),
        xold2=torch.zeros(problem.n, device=device, dtype=dtype),
        low=torch.zeros(problem.n, device=device, dtype=dtype),
        upp=torch.zeros(problem.n, device=device, dtype=dtype),
        loop=QUIET_LOOP,
        beta_t=beta_t,
        beta_d=beta_d,
        U=None,
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
    # tests/reference/fem.py's assemble_stiffness -- the NumPy oracle, unrelated to
    # stto.step's own torch/MGCG solve -- is the cheapest way to get a dense
    # `K_free` to condition-number-check, so `problem`'s tensor fields get bridged to
    # NumPy just for this.
    p = problem
    KE = torch_util.to_numpy(p.KE)
    edofMat = torch_util.to_numpy(p.edofMat)
    freedofs = torch_util.to_numpy(p.freedofs)

    xPhys, tPhys = stto.physical_fields(p, state.x, state.t, state.beta_d)
    fields = [xPhys]
    tP = np.linspace(0, 1, p.config.nStage + 1)
    for i in range(1, p.config.nStage + 1):
        fields.append(xPhys * compliance.time_mask(tPhys, tP[i], beta_t))
    for field in fields:
        K = fem_ref.assemble_stiffness(
            KE,
            torch_util.to_numpy(field),
            p.config.Emin,
            p.config.Emax,
            p.config.penal,
            edofMat,
            p.ndof,
        )
        Kfree = K[np.ix_(freedofs, freedofs)].toarray()
        if np.linalg.cond(Kfree) >= MAX_COND:
            return False
    return True


def _draw_well_conditioned_state(problem, rng, *, beta_t=10.0, max_tries=50):
    for _ in range(max_tries):
        x_raw = rng.uniform(0.3, 0.7, size=(problem.config.nely, problem.config.nelx))
        t_raw = rng.uniform(0.1, 0.9, size=(problem.config.nely, problem.config.nelx))
        state = _state_from_raw(problem, x_raw, t_raw, beta_t=beta_t)
        if _well_conditioned(problem, state, beta_t):
            return x_raw, t_raw, state
    raise AssertionError(f"no well-conditioned draw in {max_tries} tries")


def test_physical_time_field_is_scaled_to_a_maximum_of_one():
    """A raw time field whose filtered maximum falls short of 1 is scaled, not shifted,
    so that the build ends at 1 and a start at 0 stays at 0."""
    problem = _problem()
    state = stto.init_state(problem)
    t_raw = 0.6 * state.t
    filtered = filters.apply_density_filter(t_raw, problem.time_H, problem.time_Hs)
    assert float(filtered.max()) < 0.7  # premise: the unscaled field ends early

    _, tPhys = stto.physical_fields(problem, state.x, t_raw, BETA_D)
    assert float(tPhys.max()) == pytest.approx(1.0, rel=1e-14)
    ratio = torch_util.to_numpy(tPhys / filtered)
    np.testing.assert_allclose(ratio, ratio.flat[0], rtol=1e-12)


def test_sensitivity_rows_multi_row_matches_single_row_calls():
    """A `k`-row `_sensitivity_rows` call must agree with `k` independent single-row
    calls: the rows are independent, so assembling them together must not couple them.
    Pin that by feeding a 2-row output through and comparing against two separate calls.
    """
    nelx, nely = 6, 4
    problem = _problem(nelx=nelx, nely=nely)
    device, dtype = problem.device, problem.dtype

    rng = np.random.default_rng(0)
    x = torch.tensor(
        rng.uniform(0.2, 0.8, size=(nely, nelx)), device=device, dtype=dtype
    ).requires_grad_(True)
    t = torch.tensor(
        rng.uniform(0.05, 0.95, size=(nely, nelx)), device=device, dtype=dtype
    ).requires_grad_(True)
    xTilde = filters.apply_density_filter(x, problem.H, problem.Hs)
    tPhys = filters.apply_density_filter(t, problem.H, problem.Hs)

    # Two arbitrary, distinct scalar outputs of (xTilde, tPhys), so the two rows of the
    # multi-row call have genuinely different sensitivities.
    out_a = torch.sum(xTilde**2) + torch.sum(tPhys)
    out_b = torch.sum(xTilde) - torch.sum(tPhys**2)
    outputs = torch.stack([out_a, out_b])

    rows = stto._sensitivity_rows(outputs, x, t)
    row_a = stto._sensitivity_rows(out_a[None], x, t)[0]
    row_b = stto._sensitivity_rows(out_b[None], x, t)[0]

    torch.testing.assert_close(rows[0], row_a, rtol=1e-10, atol=0.0)
    torch.testing.assert_close(rows[1], row_b, rtol=1e-10, atol=0.0)


def test_step_assembled_sensitivities_match_finite_differences(monkeypatch):
    """`step`'s `IterationRecord.df` and `.dg` against central differences of its own
    `.f`/`.g`, along random directions in the raw design vector `[x; t]`.

    Autograd builds every row, so this does not check a chain rule. It checks what
    autograd cannot see: a graph cut by a stray `.detach()` or host round-trip (which
    `allow_unused` turns into a silent zero), or a custom backward misused in context.
    Such an error moves the gradient along a generic direction, so a few random
    directions per field find it for a small fraction of a per-element sweep's `step`
    calls.
    """
    nStage = 3
    nelx, nely = 10, 8
    # .f (a compliance) is noise-dominated at small h; the constraint rows reach
    # truncation error at large h. Worst relative error over the directions below at
    # h = 1e-4 / 1e-5 / 1e-6 / 1e-7: .f 1e-6 / 2e-6 / 3e-5 / 1e-4, .g 9e-6 / 9e-8 / 2e-8
    # / 5e-7 -- measured before the angular stencil weight was on by default.
    #
    # The lobe and its directional divisor both read `grad t`, which makes the hotspot
    # row markedly more curved in `t`: its worst direction goes from 4.6e-8 at
    # `hotspot_kappa = 0` to 1.4e-6 at 2.3666, both at h = 1e-5, decaying as h**2 in
    # each case. So h sits at 1e-6, where that direction reaches 7.5e-9 and `.f` is
    # still an order of magnitude inside its own tolerance.
    h = 1e-6
    # Stage volume bounds on, so every kind of constraint row is present. The hotspot
    # calibration is a detached offset re-measured from the design it is refreshed on,
    # so on a refresh iteration `.g`'s hotspot row moves by an amount `.dg` deliberately
    # does not carry -- `.dg` is the smooth surrogate's gradient, which is what MMA
    # linearizes. Freezing the calibration leaves a row the finite difference can see.
    problem = _problem(
        nelx=nelx,
        nely=nely,
        nStage=nStage,
        enable_stage_volume=True,
        config_overrides={"hotspot_refresh_period": 2**62},
    )
    nel = nelx * nely
    assert problem.n == 2 * nel

    rng = np.random.default_rng(0)
    x_raw, t_raw, state = _draw_well_conditioned_state(problem, rng)
    _, record = stto.step(problem, state)

    # Row count follows from the stack `step` builds: volume, continuity (when enabled),
    # one row per print-start element, an upper and a lower bound per stage, and the
    # hotspot row.
    n_continuity_rows = 1 if problem.config.enable_continuity else 0
    m = 1 + n_continuity_rows + len(problem.Nei) + 2 * nStage + 1
    assert record.df.shape == (problem.n,)
    assert record.g.shape == (m,)
    assert record.dg.shape == (m, problem.n)

    # Non-vacuity: an all-but-zero gradient would pass the comparison below regardless.
    assert np.abs(record.df).max() > 1e-3
    assert np.abs(record.dg).max(axis=1).min() > 1e-3

    # `step`'s gradient treats the time field's scale as a constant, so the finite
    # difference has to hold it at the unperturbed value too.
    def filtered_max(t):
        t = torch_util.to_tensor(t, problem.device, problem.dtype)
        return filters.apply_density_filter(t, problem.time_H, problem.time_Hs).max()

    base_scale = filtered_max(t_raw)
    unscaled_physical_fields = stto.physical_fields

    def physical_fields_at_base_scale(problem, x, t, beta_d):
        xPhys, tPhys = unscaled_physical_fields(problem, x, t, beta_d)
        return xPhys, tPhys * filtered_max(t) / base_scale

    monkeypatch.setattr(stto, "physical_fields", physical_fields_at_base_scale)

    def values_at(x_raw, t_raw):
        _, rec = stto.step(problem, _state_from_raw(problem, x_raw, t_raw))
        return rec.f, rec.g

    # Each field gets its own directions: its gradient block can be orders of magnitude
    # smaller than the other's, and an error in it would hide in a shared direction.
    for field, block in (("x", slice(0, nel)), ("t", slice(nel, 2 * nel))):
        for _ in range(2):
            v = rng.standard_normal((nely, nelx))
            if field == "x":
                f_p, g_p = values_at(x_raw + h * v, t_raw)
                f_m, g_m = values_at(x_raw - h * v, t_raw)
            else:
                f_p, g_p = values_at(x_raw, t_raw + h * v)
                f_m, g_m = values_at(x_raw, t_raw - h * v)
            df_v = record.df[block] @ v.ravel()
            dg_v = record.dg[:, block] @ v.ravel()
            fd_f = (f_p - f_m) / (2 * h)
            fd_g = (g_p - g_m) / (2 * h)
            np.testing.assert_allclose(
                df_v, fd_f, rtol=1e-4, err_msg=f"df, {field} direction"
            )
            # A row that does not depend on this field is exactly zero on both sides.
            np.testing.assert_allclose(
                dg_v,
                fd_g,
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"dg, {field} direction",
            )


def test_step_objective_is_theta_weighted_sum_of_stage_compliances():
    """`IterationRecord.f` is the whole-structure compliance plus `Theta` times the sum
    of the per-stage gravity compliances -- so it must be exactly affine in `Theta`,
    with intercept the whole-structure compliance alone and slope the stage sum.
    Recovering both from three `Theta` values pins the weighting independently of the FD
    check (which sees the gradient, not the value), and `Theta == 0` recovers the pure
    compliance problem the stage terms are layered onto.
    """
    nelx, nely, nStage = 10, 8, 3
    rng = np.random.default_rng(1)

    def f_at(Theta):
        problem = _problem(nelx=nelx, nely=nely, nStage=nStage, Theta=Theta)
        state = _state_from_raw(problem, x_raw, t_raw)
        _, rec = stto.step(problem, state)
        return rec.f, rec

    base = _problem(nelx=nelx, nely=nely, nStage=nStage)
    x_raw, t_raw, _ = _draw_well_conditioned_state(base, rng)

    f0_0, rec_0 = f_at(0.0)
    f0_1, _ = f_at(1.0)
    f0_3, _ = f_at(3.0)

    # At Theta = 0 the stage terms drop out and .f is the whole-structure compliance,
    # which `IterationRecord.obj` reports separately at every Theta, plus the weighted
    # time-field terms, which do not depend on Theta.
    time_terms = (
        base.config.uniformity_weight * rec_0.uniformity
        + rec_0.roughness_weight * rec_0.roughness
    )
    np.testing.assert_allclose(f0_0, rec_0.obj + time_terms, rtol=1e-12)
    stage_sum = f0_1 - f0_0
    assert stage_sum > 1e-3  # non-vacuous: the stage terms actually contribute
    np.testing.assert_allclose(f0_3, f0_0 + 3.0 * stage_sum, rtol=1e-9)


def test_step_objective_adds_the_weighted_uniformity_penalty():
    """`IterationRecord.f` is affine in `uniformity_weight` with slope
    `IterationRecord.uniformity` -- which pins both that the penalty is wired into the
    objective with the right weight and that `uniformity` reports the same quantity the
    weight multiplies.
    """
    nelx, nely, nStage = 10, 8, 3
    rng = np.random.default_rng(1)

    def f_at(weight):
        problem = _problem(
            nelx=nelx, nely=nely, nStage=nStage, uniformity_weight=weight
        )
        state = _state_from_raw(problem, x_raw, t_raw)
        _, rec = stto.step(problem, state)
        return rec.f, rec.uniformity

    base = _problem(nelx=nelx, nely=nely, nStage=nStage)
    x_raw, t_raw, _ = _draw_well_conditioned_state(base, rng)

    f_0, uniformity = f_at(0.0)
    f_100, uniformity_100 = f_at(100.0)

    assert uniformity > 1e-4  # non-vacuous: the field is not already uniform
    np.testing.assert_allclose(uniformity_100, uniformity, rtol=1e-12)
    np.testing.assert_allclose(f_100, f_0 + 100.0 * uniformity, rtol=1e-9)


def test_the_first_step_calibrates_the_hotspot_row_against_the_seed():
    """Iteration 0 already reports the seed's true maximum severity: it is a refresh
    iteration, so the aggregate's bias is calibrated out before the first hotspot row is
    built rather than some periods into the run."""
    problem = _problem()
    uncalibrated = problem.hotspot.calibration
    state = stto.init_state(problem)
    xPhys, tPhys = stto.physical_fields(problem, state.x, state.t, BETA_D)
    K_est = stto.estimated_conductivity(problem, xPhys, tPhys)
    finite = torch.isfinite(K_est)
    true_max = float(
        ((1 - K_est[finite]) * xPhys.flatten()[finite] ** problem.config.r).max()
    )

    _, record = stto.step(problem, state)
    assert problem.hotspot.calibration != uncalibrated  # non-vacuous
    assert record.tru_max == pytest.approx(true_max, rel=1e-12)


def test_enable_continuity_false_drops_the_continuity_constraint_row():
    """Disabling continuity removes its row rather than relaxing it, so the start-point
    rows follow the volume row directly."""
    # Both ends of the comparison are set here rather than left to the default, which
    # the row count would otherwise silently follow -- it has been `false`, which makes
    # this a config compared against itself.
    base_config = _problem().config
    problem = stto.build_problem(
        dataclasses.replace(base_config, enable_continuity=False)
    )
    base = stto.build_problem(dataclasses.replace(base_config, enable_continuity=True))
    _, record = stto.step(problem, stto.init_state(problem))
    _, with_continuity = stto.step(base, stto.init_state(base))

    assert record.g.shape == (with_continuity.g.shape[0] - 1,)
    assert record.dg.shape == (with_continuity.g.shape[0] - 1, problem.n)
    np.testing.assert_allclose(record.g[0], with_continuity.g[0], rtol=1e-12)
    np.testing.assert_allclose(record.g[1:], with_continuity.g[2:], rtol=1e-12)


def test_step_objective_adds_the_scheduled_roughness_term():
    """`IterationRecord.f` moves by this iteration's roughness weight times
    `IterationRecord.roughness`, and a schedule reaches the objective as its value at the
    current iteration rather than as a constant."""
    nelx, nely, nStage = 10, 8, 3
    rng = np.random.default_rng(2)
    base = _problem(nelx=nelx, nely=nely, nStage=nStage)
    x_raw, t_raw, _ = _draw_well_conditioned_state(base, rng)

    def record_at(roughness_weight):
        config = dataclasses.replace(base.config, roughness_weight=roughness_weight)
        problem = stto.build_problem(config)
        _, rec = stto.step(problem, _state_from_raw(problem, x_raw, t_raw))
        return rec

    schedule = {"initial": 1000.0, "decay_iterations": 10, "final": 100.0}
    unweighted, scheduled = record_at(0.0), record_at(schedule)

    # The two records come from separate solves, which agree on `.f` only to the CG
    # tolerance (PR #99).
    noise = 10 * torch_solve.DEFAULT_CG_RTOL * abs(unweighted.f)
    # The weight is the schedule's value at the iteration `_state_from_raw` steps, not
    # its `initial` -- that is the part worth pinning.
    applied = run_config.weight_at(run_config.schedule_from_dict(schedule), QUIET_LOOP)
    assert scheduled.roughness_weight == applied
    assert applied * scheduled.roughness > 100 * noise  # non-vacuous
    np.testing.assert_allclose(
        scheduled.f, unweighted.f + applied * scheduled.roughness, rtol=0, atol=noise
    )


def test_step_gradient_of_the_penalty_matches_its_own_sensitivity():
    """Raising `uniformity_weight` changes `.df` by the weight times the penalty's own
    sensitivity, in both halves: the penalty reads `tPhys` and is weighted by `xPhys`, so
    it moves material as well as print time.

    Both halves are isolated by differencing two separate `step` calls, so the
    compliance sensitivity has to cancel between them. It only cancels to the CG
    tolerance: the solve is iterative, and on CUDA its reduction order is not
    reproducible, so two runs at the same weight land on different iterates within
    `rtol`. Tolerances here are keyed to that noise floor, not to a constant -- see
    PR #99.
    """
    nelx, nely = 10, 8
    rng = np.random.default_rng(3)
    nel = nelx * nely
    base = _problem(nelx=nelx, nely=nely)
    x_raw, t_raw, _ = _draw_well_conditioned_state(base, rng)

    def df_at(weight):
        problem = _problem(nelx=nelx, nely=nely, uniformity_weight=weight)
        _, rec = stto.step(problem, _state_from_raw(problem, x_raw, t_raw))
        return rec.df, problem

    df_0, _ = df_at(0.0)
    weight = 100.0
    df_g, problem = df_at(weight)

    def solve_noise(df_half):
        """How far two solves of the same problem may disagree on `df_half`."""
        return 10 * torch_solve.DEFAULT_CG_RTOL * np.abs(df_half).max()

    # The expected difference, assembled apart from step()'s objective: the penalty's
    # autograd sensitivity w.r.t. both raw fields.
    x_leaf, t_leaf = (
        torch_util.to_tensor(raw, problem.device, problem.dtype).requires_grad_(True)
        for raw in (x_raw, t_raw)
    )
    xPhys, tPhys = stto.physical_fields(problem, x_leaf, t_leaf, BETA_D)
    penalty = timefield.uniformity_penalty(
        tPhys, timefield.UniformityMetric.GRADIENT_CV, weights=xPhys
    )
    expected = weight * np.concatenate(
        [
            torch_util.to_numpy(d).ravel()
            for d in torch.autograd.grad(penalty, (x_leaf, t_leaf))
        ]
    )
    for half in (slice(None, nel), slice(nel, None)):
        noise = solve_noise(df_0[half])
        # Non-vacuous: the penalty's own contribution has to clear the noise it is
        # recovered through, or this asserts nothing.
        assert np.abs(expected[half]).max() > 100 * noise
        np.testing.assert_allclose(
            df_g[half] - df_0[half], expected[half], rtol=1e-4, atol=noise
        )


# --- Batched FEM solves inside step() -------------------------------------------------
# `batched_whole_and_gravity_compliance` matching `whole_compliance` +
# `gravity_compliance` is checked in test_compliance.py, at the level it belongs to.


def test_step_state_U_does_not_carry_grad_across_iterations():
    """`State.U` is a warm-start seed for the *next* iteration's solve, not a value that
    should carry gradient across iterations -- storing it undetached lets `FemSolve`'s
    `x0` argument wire one iteration's whole multigrid hierarchy into the next
    iteration's autograd graph, and the next iteration's `U` does the same to the one
    after that. Confirmed by direct measurement (see `stto.step`'s comment on `U=`):
    left undetached, this chains every iteration's hierarchy into one never-freed graph,
    growing GPU memory ~180 MB/step at 180x60 and OOMing an 8 GB card by iteration ~40;
    detached, memory is flat. Runs a few iterations rather than reproducing the OOM
    directly -- `requires_grad`/`grad_fn` are the property that actually matters, and
    checking it doesn't need a GPU or hundreds of iterations to be a real regression
    guard.
    """
    problem = _problem(nelx=6, nely=4, nStage=3)
    state = stto.init_state(problem)

    for _ in range(3):
        state, _ = stto.step(problem, state)
        assert state.U is not None
        assert not state.U.requires_grad
        assert state.U.grad_fn is None


def test_step_batched_warm_starts_from_previous_iteration():
    """The batched path's second call uses fewer CG iterations than the first, warm-
    started from the `U` the first call left on `State` -- part 1's ~25% saving,
    exercised through `step` rather than through `FemSolve` directly.
    """
    import sttopt.torch_mg as torch_mg

    problem = _problem(nelx=10, nely=8, nStage=3)
    state = stto.init_state(problem)

    counts = []
    orig_pcg = torch_mg.torch_fem.pcg

    def counting_pcg(*args, **kwargs):
        U, n_iter = orig_pcg(*args, **kwargs)
        counts.append(n_iter)
        return U, n_iter

    torch_mg.torch_fem.pcg = counting_pcg
    try:
        state1, _ = stto.step(problem, state)
        cold_iters = counts[-1]
        state2, _ = stto.step(problem, state1)
        warm_iters = counts[-1]
    finally:
        torch_mg.torch_fem.pcg = orig_pcg

    assert state1.U.shape == (1 + problem.config.nStage, problem.ndof)
    assert warm_iters <= cold_iters


# --- Phase 3.4 (plans/torch_port_part2.md): the near-binary NaN regression ------------
#
# The test Phase 0a's original bug (and its autograd resurrection) would have caught:
# a late, near-binary snapshot with exact zeros in xPhys, run through the real
# stto.step wiring end to end.

# The 90x30 snapshot at every third element, for runtime: it keeps exact zeros in xPhys.
# Radii are in element units, so the conduction radius shrinks by the same factor. The
# filter radii stay at 2.0: a third of that would leave an empty stencil.
NEAR_BINARY_STRIDE = 3
NEAR_BINARY_RMIN_COND = 6.0 / NEAR_BINARY_STRIDE


def _near_binary_snapshot() -> tuple[np.ndarray, np.ndarray]:
    s = NEAR_BINARY_STRIDE
    with np.load(FIXTURES_DIR / "torch_port_designs.npz") as data:
        return data["x_90x30_it0800"][1::s, 1::s], data["t_90x30_it0800"][1::s, 1::s]


def test_step_produces_no_nan_gradients_on_a_near_binary_snapshot():
    """`stto.step`'s assembled `IterationRecord.df`/`.dg` must stay finite on a real late-run
    snapshot with exact zeros in `xPhys` (`x_90x30_it0800`, subsampled, from
    `tests/fixtures/torch_port_designs.npz` -- generated by
    `generate_torch_port_designs.py` at the production filter radii/schedules, not
    manufactured). This is the test that would have caught Phase 0a's original NaN bug
    (`hotspot_constraint`'s un-cancelled `x**(r-1)` diagonal term) and would catch its
    resurrection under autograd (`conductivity.PMean`'s NaN-safe rewrite, Phase 3.4) -- see
    `test_step_would_have_produced_nan_without_the_nan_safe_rewrite` for direct proof
    the rewrite, not mere luck on this snapshot, is what keeps it clean.
    """
    x, t = _near_binary_snapshot()
    nely, nelx = x.shape
    assert np.any(x == 0.0)  # premise: exact zeros are actually present

    config = default_run_config(
        nelx=nelx,
        nely=nely,
        nStage=8,
        volfrac=0.5,
        Theta=0.1,
        Tcr=TCR,
        print_base="opposite_corner",
        rmin=2.0,
        lrmin=2.0,
        rmin_cond=NEAR_BINARY_RMIN_COND,
    )
    problem = stto.build_problem(config)
    x = torch_util.to_tensor(x, problem.device, problem.dtype)
    t = torch_util.to_tensor(t, problem.device, problem.dtype)
    xval = torch.cat([x.flatten(), t.flatten()])
    state = stto.State(
        x=x,
        t=t,
        xold1=xval,
        xold2=xval.clone(),
        low=xval - 0.1,
        upp=xval + 0.1,
        loop=800,
        beta_t=50.0,
        beta_d=128.0,
        U=None,
    )

    _, record = stto.step(problem, state)
    assert np.all(np.isfinite(record.df))
    assert np.all(np.isfinite(record.dg))
    assert np.all(np.isfinite(record.g))
    assert np.isfinite(record.f)


def test_step_would_have_produced_nan_without_the_nan_safe_rewrite():
    """Proof the test above is meaningful, not merely lucky: `conductivity.PMean`
    evaluated on the *same* snapshot but through the naive, algebraically equivalent
    `(T_val * x**r) ** p` form (what a mechanical port would have written) produces a
    `nan` gradient, at exactly the exact-zero elements the safe rewrite fixes.
    """
    import sttopt.conductivity as conductivity

    x, t = _near_binary_snapshot()
    nely, nelx = x.shape

    e1, e2, w = conductivity.neighbor_weights(nelx, nely, NEAR_BINARY_RMIN_COND)
    e1_t = torch_util.to_tensor(e1, "cpu", torch.int64)
    e2_t = torch_util.to_tensor(e2, "cpu", torch.int64)
    w_t = torch_util.to_tensor(w, "cpu", torch.float64)
    x_t = torch_util.to_tensor(x, "cpu", torch.float64).flatten().requires_grad_(True)
    t_t = torch_util.to_tensor(t, "cpu", torch.float64).flatten()

    core = conductivity._conductivity_core(x_t, t_t, e1_t, e2_t, w_t, 3.0, 100.0)
    T_val = 1 - core.K_est
    cond_p = (T_val * x_t**0.05) ** 25.0  # naive, pre-rewrite form
    numer = (torch.sum(cond_p) / x_t.numel()) ** (1 / 25.0)
    (d_x,) = torch.autograd.grad(numer, (x_t,))
    assert torch.any(torch.isnan(d_x))


def _curvature_records(tool_radius):
    """One `step` record at the default `tool_radius` and one at `tool_radius`, from the
    same design, at iteration `QUIET_LOOP`."""
    base = _problem(nelx=10, nely=8)
    with_tool = stto.build_problem(
        dataclasses.replace(base.config, tool_radius=tool_radius)
    )
    _, _, state = _draw_well_conditioned_state(base, np.random.default_rng(3))
    _, record = stto.step(base, state)
    _, with_record = stto.step(with_tool, state)
    return record, with_record


def test_tool_radius_appends_one_row_bounding_the_concave_curvature():
    """The row is the tool radius over the largest one the design admits, minus 1,
    since every iteration refreshes the calibration onto the true maximum."""
    assert _problem().config.tool_radius == 0  # premise: the default adds no row
    record, with_record = _curvature_records(2.0)

    assert with_record.g.shape == (record.g.shape[0] + 1,)
    np.testing.assert_allclose(with_record.g[:-1], record.g, rtol=1e-12)
    admissible = with_record.diagnostics["admissible_tool_radius"]
    assert np.isfinite(admissible)  # non-vacuous: a random design has concave fronts
    assert with_record.g[-1] == pytest.approx(2.0 / admissible - 1, rel=1e-10)
    assert np.abs(with_record.dg[-1]).max() > 0


def test_tool_radius_at_zero_mid_schedule_is_an_inactive_row():
    """A ramp that has not yet left 0 already has its row, which a zero-radius tool
    satisfies with no gradient."""
    ramp = run_config.PiecewiseSchedule(
        points=[[0, 0.0], [QUIET_LOOP + 10, 0.0], [QUIET_LOOP + 20, 3.0]]
    )
    record, with_record = _curvature_records(ramp)

    assert with_record.g.shape == (record.g.shape[0] + 1,)
    assert with_record.g[-1] == pytest.approx(-1.0, abs=1e-12)
    assert np.abs(with_record.dg[-1]).max() == 0


def test_tool_radius_needs_an_interior_element():
    config = dataclasses.replace(_problem().config, nelx=2, nely=6, tool_radius=1.0)
    with pytest.raises(ValueError, match="interior element"):
        stto.build_problem(config)


def test_scheduled_tmove_bounds_each_iterations_time_step():
    """A `tmove` schedule sets the trust region of the iteration it resolves at."""
    schedule = run_config.PiecewiseSchedule(points=[[0, 0.02], [1, 0.005]], mode="step")
    problem = _problem(config_overrides=dict(tmove=schedule))
    state, first = stto.step(problem, stto.init_state(problem))
    _, second = stto.step(problem, state)

    assert first.diagnostics["dt_max"] > 0.005  # non-vacuous: the wide limit is used
    assert second.diagnostics["dt_max"] <= 0.005 + 1e-12
