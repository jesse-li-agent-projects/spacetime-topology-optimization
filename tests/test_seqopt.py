"""First-principles tests for `sttopt.seqopt`: the wiring layer, mirroring
`tests/test_stto.py`'s approach for `stto`."""

import dataclasses

import numpy as np
import pytest
import torch

import sttopt.filters as filters
import sttopt.seqopt as seqopt
import sttopt.timefield as timefield
import sttopt.torch_util as torch_util
from conftest import SEQ_ELEMENT_M, default_seq_run_config

NELX, NELY = 7, 5


def _geometry() -> np.ndarray:
    """A small non-square binary geometry: the bottom two rows and the left column
    solid, everything else void -- touches the build plate everywhere along the
    bottom row, but leaves plenty of void for the free-void-time-variable behaviour
    the plan documents."""
    xPhys = np.zeros((NELY, NELX))
    xPhys[-2:, :] = 1.0
    xPhys[:, 0] = 1.0
    return xPhys


def _problem(**overrides) -> seqopt.Problem:
    overrides.setdefault("tmove", 0.05)
    config = default_seq_run_config(
        lrmin_m=1.5 * SEQ_ELEMENT_M, rmin_cond_m=2.5 * SEQ_ELEMENT_M, **overrides
    )
    return seqopt.build_problem(config, _geometry(), SEQ_ELEMENT_M, device="cpu")


# The iteration `_state_from_raw`'s states sit at.
STATE_LOOP = 1


def _state_from_raw(
    problem: seqopt.Problem, t_raw: np.ndarray, base: seqopt.State
) -> seqopt.State:
    t = torch_util.to_tensor(t_raw, problem.device, problem.dtype)
    return dataclasses.replace(base, t=t, loop=STATE_LOOP)


# --- init_state ----------------------------------------------------------------


def test_init_state_is_geodesic_and_finite():
    problem = _problem(void_extension="geodesic")
    state = seqopt.init_state(problem)
    assert state.t.shape == (NELY, NELX)
    assert torch.all(torch.isfinite(state.t))
    assert not state.t.requires_grad
    expected = timefield.init_geodesic_timefield(
        _geometry(), torch_util.to_numpy(problem.Nei)
    )
    np.testing.assert_allclose(torch_util.to_numpy(state.t), expected)


def test_init_state_calibrates_the_hotspot_term_against_the_initial_field():
    """The hotspot term reports the true maximum severity from iteration 1, not the
    aggregate's bias until the first refresh."""
    problem = _problem()
    state = seqopt.init_state(problem)
    K_est = seqopt.estimated_conductivity(
        problem, seqopt.physical_timefield(problem, state.t)
    )
    finite = torch.isfinite(K_est)
    xPhys = problem.xPhys.flatten()[finite]
    true_max = float(((1 - K_est[finite]) * xPhys**problem.config.r).max())

    _, record = seqopt.step(problem, state)
    assert record.hotspot == pytest.approx(true_max, rel=1e-12)


def test_build_problem_drops_solid_that_cannot_reach_the_plate():
    """`Problem.xPhys` is the geometry that can actually be built, so an island is not
    left to contribute to the objective or to the volume the stage budgets use."""
    xPhys = _geometry()
    xPhys[0, -1] = 1.0  # an island in the top-right corner, clear of everything solid
    config = default_seq_run_config(
        lrmin_m=1.5 * SEQ_ELEMENT_M, rmin_cond_m=2.5 * SEQ_ELEMENT_M, tmove=0.05
    )

    with pytest.warns(UserWarning, match="no path of material"):
        problem = seqopt.build_problem(config, xPhys, SEQ_ELEMENT_M, device="cpu")

    np.testing.assert_allclose(torch_util.to_numpy(problem.xPhys), _geometry())


# --- step: shapes, finiteness, xPhys never a design variable --------------------


@pytest.mark.parametrize("nStage", [0, 2])
def test_step_finite_and_design_vector_is_t_only(nStage):
    problem = _problem(nStage=nStage)
    nel = NELX * NELY
    assert problem.n == nel  # not 2*nel: xPhys is fixed, t is the only design variable

    state = seqopt.init_state(problem)
    new_state, record = seqopt.step(problem, state)

    assert not problem.xPhys.requires_grad
    assert torch.all(torch.isfinite(new_state.t))
    assert torch.all(new_state.t >= 0.0) and torch.all(new_state.t <= 1.0)
    assert np.isfinite(record.f)
    assert record.xmma.shape == (nel,)
    assert record.df.shape == (nel,)

    m = (1 if problem.config.enable_continuity else 0) + 1 + 2 * nStage
    assert record.g.shape == (m,)
    assert record.dg.shape == (m, nel)


def test_enable_continuity_false_drops_the_continuity_constraint_row():
    problem = _problem(nStage=0, enable_continuity=False)
    nel = NELX * NELY

    state = seqopt.init_state(problem)
    _, record = seqopt.step(problem, state)
    assert record.g.shape == (1,)  # the start-point row alone
    assert record.dg.shape == (1, nel)


def test_objective_is_the_weighted_sum_of_its_three_terms():
    problem = _problem(nStage=0)
    state = seqopt.init_state(problem)
    _, record = seqopt.step(problem, state)
    assert record.f == pytest.approx(
        problem.config.hotspot_weight * record.hotspot
        + problem.config.uniformity_weight * record.uniformity
        + record.roughness_weight * record.roughness,
        rel=1e-9,
    )


def test_scheduled_roughness_weight_decays_during_the_run():
    """A schedule has to reach the objective, not just the config: the point of the
    continuation is that the term's weight really is released mid-run. A one-iteration
    decay puts both endpoints in the first two steps."""
    problem = _problem(
        nStage=0,
        roughness_weight={"initial": 1.0, "decay_iterations": 1, "final": 0.06},
    )
    state = seqopt.init_state(problem)

    state, first = seqopt.step(problem, state)
    _, second = seqopt.step(problem, state)

    assert first.roughness_weight == 1.0
    assert second.roughness_weight == 0.06
    assert second.f == pytest.approx(
        problem.config.hotspot_weight * second.hotspot
        + problem.config.uniformity_weight * second.uniformity
        + 0.06 * second.roughness,
        rel=1e-9,
    )


# --- finite-difference check of df/dt and dg/dt ---------------------------------


def test_constraint_rows_read_the_time_field():
    """Autograd gets every chain rule right but cannot see a graph cut: a stray
    `.detach()` or host round-trip leaves a row silently zero. `t` is the only leaf, so
    every row must read it. Stage rows and continuity are both on, so every kind of row
    is present."""
    # A tolerance the random draw roughly meets: at the default, its continuity row is
    # violated ~1e3-fold and MMA's subproblem stalls.
    problem = _problem(nStage=2, enable_continuity=True, continuity_tol=1e-2)
    rng = np.random.default_rng(0)
    state = _state_from_raw(
        problem, rng.uniform(0.1, 0.9, size=(NELY, NELX)), seqopt.init_state(problem)
    )
    _, record = seqopt.step(problem, state)
    for i, row in enumerate(record.dg):
        assert np.any(row != 0), f"row {i}"


@pytest.mark.parametrize(
    "term", ["hotspot_weight", "uniformity_weight", "roughness_weight"]
)
def test_each_objective_term_reads_the_time_field(term):
    """With only one term weighted, the objective's sensitivity is that term's alone, so
    a cut in its path to `t` cannot hide behind the others."""
    weights = {"hotspot_weight": 0.0, "uniformity_weight": 0.0, "roughness_weight": 0.0}
    problem = _problem(**{**weights, term: 1.0})
    rng = np.random.default_rng(0)
    state = _state_from_raw(
        problem, rng.uniform(0.1, 0.9, size=(NELY, NELX)), seqopt.init_state(problem)
    )
    _, record = seqopt.step(problem, state)
    assert np.any(record.df != 0)


def test_physical_time_field_holds_its_scale_out_of_the_gradient():
    """The time field is scaled to end the build at 1 by a detached maximum, so its
    gradient is the filtered field's divided by a constant scale."""
    problem = _problem(time_filter_rmin_m=2.0 * SEQ_ELEMENT_M)
    assert problem.H is not None
    rng = np.random.default_rng(5)
    t = torch_util.to_tensor(
        rng.uniform(0.1, 0.9, size=(NELY, NELX)), problem.device, problem.dtype
    ).requires_grad_(True)
    cotangent = torch_util.to_tensor(
        rng.standard_normal((NELY, NELX)), problem.device, problem.dtype
    )

    tPhys = seqopt.physical_timefield(problem, t)
    (got,) = torch.autograd.grad((cotangent * tPhys).sum(), t)

    filtered = filters.apply_density_filter(t, problem.H, problem.Hs)
    scale = float(filtered.detach().max())
    (want,) = torch.autograd.grad((cotangent * filtered).sum() / scale, t)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=0.0)


# --- the optional filter on t ---------------------------------------------------


def test_unfiltered_time_field_is_only_scaled():
    """At radius 0 the physical field is `t` scaled, not shifted, to end the build at 1:
    a field that stops short of 1 is stretched, and a start at 0 stays at 0."""
    problem = _problem(nStage=0, time_filter_rmin_m=0.0)
    assert problem.H is None
    t = 0.6 * seqopt.init_state(problem).t

    tPhys = seqopt.physical_timefield(problem, t)
    assert float(tPhys.max()) == pytest.approx(1.0, rel=1e-14)
    torch.testing.assert_close(tPhys, t / t.max(), rtol=1e-14, atol=0.0)


def test_time_filter_attenuates_a_one_element_sawtooth():
    """The reason a radius is on offer: the mode the uniformity penalty rewards is a
    one-element one, and the filter's whole job is to make those expensive in the
    design variable. Pinning that it attenuates rather than merely alters the field.
    """
    problem = _problem(nStage=0, time_filter_rmin_m=4.0 * SEQ_ELEMENT_M)
    _, j = np.indices((NELY, NELX))
    ramp = np.tile(np.linspace(0.0, 1.0, NELY)[:, None], (1, NELX))

    amplitude = 0.05
    t = torch_util.to_tensor(
        ramp + amplitude * (-1.0) ** j, problem.device, problem.dtype
    )
    xPhys = problem.xPhys
    before = float(timefield.sawtooth_amplitude(t, xPhys))
    after = float(
        timefield.sawtooth_amplitude(seqopt.physical_timefield(problem, t), xPhys)
    )
    assert before == pytest.approx(amplitude, rel=0.05)
    assert after < before / 20


# --- optimization makes progress ------------------------------------------------


def test_hotspot_improves_from_a_bad_initial_field():
    """A handful of iterations from a deliberately bad (reversed) field should reduce
    the hotspot term. Small run, per the repo's rule against production-scale tests."""
    problem = _problem(nStage=0, tmove=0.2)
    state = seqopt.init_state(problem)
    # A "print backwards" field: 1 - geodesic init, a bad ordering the optimizer should
    # improve on.
    bad_t = 1.0 - torch_util.to_numpy(state.t)
    state = _state_from_raw(problem, bad_t, state)

    _, record0 = seqopt.step(problem, state)
    for _ in range(9):
        state, record = seqopt.step(problem, state)

    assert record.hotspot < record0.hotspot


# --- resolution invariance of the uniformity term -------------------------------


def test_uniformity_term_is_resolution_invariant_for_the_same_shape():
    """The property `uniformity_weight`'s portability across resolutions rests on:
    GRADIENT_CV of the same *shape* of time field should be close at two different
    mesh resolutions of the same component."""

    def _cv_at(scale: int) -> float:
        nely, nelx = 8 * scale, 12 * scale
        ys, xs = torch.meshgrid(
            torch.arange(nely, dtype=torch.float64),
            torch.arange(nelx, dtype=torch.float64),
            indexing="ij",
        )
        # Same shape of field at every resolution -- a smooth bilinear ramp, so
        # relative gradient variation is a resolution-independent property of the
        # continuum field, not of the discretization.
        u, v = xs / (nelx - 1), ys / (nely - 1)
        t = u * (1.0 + 0.5 * v)
        weights = torch.ones((nely, nelx), dtype=torch.float64)
        return float(
            timefield.uniformity_penalty(
                t, timefield.UniformityMetric.GRADIENT_CV, weights=weights
            )
        )

    cv_low = _cv_at(1)
    cv_high = _cv_at(4)
    assert cv_low == pytest.approx(cv_high, rel=0.15)
