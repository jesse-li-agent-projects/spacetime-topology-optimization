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

    assert problem.m == 1 + len(problem.Nei) + 2 * nStage
    assert record.g.shape == (problem.m,)
    assert record.dg.shape == (problem.m, nel)


def test_hotspot_weight_and_uniformity_weight_compose_the_objective():
    problem = _problem(nStage=0)
    state = seqopt.init_state(problem)
    _, record = seqopt.step(problem, state)
    assert record.f == pytest.approx(
        problem.config.hotspot_weight * record.hotspot
        + problem.config.uniformity_weight * record.uniformity,
        rel=1e-9,
    )


# --- finite-difference check of df/dt and dg/dt ---------------------------------


@pytest.mark.parametrize("nStage", [0, 2])
def test_sensitivities_match_finite_differences(nStage):
    h = 1e-5
    problem = _problem(nStage=nStage)
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
