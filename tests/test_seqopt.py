"""First-principles tests for `sttopt.seqopt`: the wiring layer, mirroring
`tests/test_optimize.py`'s approach for `stto`."""

import dataclasses

import numpy as np
import pytest
import torch

import sttopt.seqopt as seqopt
import sttopt.timefield as timefield
import sttopt.torch_util as torch_util
from conftest import default_seq_run_config

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
    config = default_seq_run_config(lrmin=1.5, rmin_cond=2.5, **overrides)
    return seqopt.build_problem(config, _geometry(), device="cpu")


def _state_from_raw(
    problem: seqopt.Problem, t_raw: np.ndarray, base: seqopt.State
) -> seqopt.State:
    t = torch_util.to_tensor(t_raw, problem.device, problem.dtype)
    return dataclasses.replace(base, t=t)


# --- init_state ----------------------------------------------------------------


def test_init_state_is_geodesic_and_finite():
    problem = _problem()
    state = seqopt.init_state(problem)
    assert state.t.shape == (NELY, NELX)
    assert torch.all(torch.isfinite(state.t))
    assert not state.t.requires_grad
    expected = timefield.init_geodesic_timefield(
        _geometry(), torch_util.to_numpy(problem.Nei)
    )
    np.testing.assert_allclose(torch_util.to_numpy(state.t), expected)


def test_build_problem_drops_solid_that_cannot_reach_the_plate():
    """`Problem.xPhys` is the geometry that can actually be built, so an island is not
    left to contribute to the objective or to the volume the stage budgets use."""
    xPhys = _geometry()
    xPhys[0, -1] = 1.0  # an island in the top-right corner, clear of everything solid
    config = default_seq_run_config(lrmin=1.5, rmin_cond=2.5, tmove=0.05)

    with pytest.warns(UserWarning, match="no path of material"):
        problem = seqopt.build_problem(config, xPhys, device="cpu")

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

    assert problem.m == (
        (1 if problem.config.enable_continuity else 0) + len(problem.Nei) + 2 * nStage
    )
    assert record.g.shape == (problem.m,)
    assert record.dg.shape == (problem.m, nel)


def test_enable_continuity_false_drops_the_continuity_constraint_row():
    problem = _problem(nStage=0, enable_continuity=False)
    nel = NELX * NELY
    assert problem.m == len(problem.Nei)

    state = seqopt.init_state(problem)
    _, record = seqopt.step(problem, state)
    assert record.g.shape == (problem.m,)
    assert record.dg.shape == (problem.m, nel)


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


@pytest.mark.parametrize("nStage", [0, 2])
@pytest.mark.parametrize("time_filter_rmin", [0.0, 2.0])
def test_sensitivities_match_finite_differences(nStage, time_filter_rmin):
    """Covers the filtered case too, where the gradient has to carry the filter's chain
    rule back to the design variable -- a term autograd supplies but nothing else
    checks."""
    h = 1e-5
    problem = _problem(nStage=nStage, time_filter_rmin=time_filter_rmin)
    nel = NELX * NELY
    rng = np.random.default_rng(0)
    t_raw = rng.uniform(0.1, 0.9, size=(NELY, NELX))
    base_state = seqopt.init_state(problem)
    state = _state_from_raw(problem, t_raw, base_state)

    _, record = seqopt.step(problem, state)
    assert np.abs(record.df).max() > 1e-6
    assert np.abs(record.dg).max(axis=1).min() > 1e-8

    def values_at(t):
        _, rec = seqopt.step(problem, _state_from_raw(problem, t, base_state))
        return rec.f, rec.g

    fd_f0 = np.zeros(nel)
    fd_f = np.zeros((problem.m, nel))
    for e in range(nel):
        j, i = e // NELX, e % NELX
        tp, tm = t_raw.copy(), t_raw.copy()
        tp[j, i] += h
        tm[j, i] -= h
        f0_p, f_p = values_at(tp)
        f0_m, f_m = values_at(tm)
        fd_f0[e] = (f0_p - f0_m) / (2 * h)
        fd_f[:, e] = (f_p - f_m) / (2 * h)

    np.testing.assert_allclose(record.df, fd_f0, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(record.dg, fd_f, rtol=1e-4, atol=1e-6)


# --- the optional filter on t ---------------------------------------------------


def test_unfiltered_is_the_default_and_leaves_t_untouched():
    """`tPhys is t` at radius 0, not merely equal to it: `seqopt`'s starting design is
    that the design variable and the physical field are the same object, and a filter
    that quietly copied would put a no-op in every gradient.
    """
    problem = _problem(nStage=0)
    assert problem.config.time_filter_rmin == 0.0
    assert problem.H is None
    t = seqopt.init_state(problem).t
    assert seqopt.physical_timefield(problem, t) is t


def test_time_filter_attenuates_a_one_element_sawtooth():
    """The reason a radius is on offer: the mode the uniformity penalty rewards is a
    one-element one, and the filter's whole job is to make those expensive in the
    design variable. Pinning that it attenuates rather than merely alters the field.
    """
    problem = _problem(nStage=0, time_filter_rmin=4.0)
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
