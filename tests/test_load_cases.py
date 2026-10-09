"""`load_cases`: force totals and directions, and where each case's supports sit."""

import dataclasses

import numpy as np
import pytest

import sttopt.fem as fem
import sttopt.load_cases as load_cases
from conftest import default_run_config, virtual_heat_config
from sttopt.load_cases import LoadCase
from sttopt.run_config import HeatRunConfig, LaplaceRunConfig, RunConfig

L_BRACKETS = [LoadCase.L_BRACKET_CORNER, LoadCase.L_BRACKET_MIDDLE]


def _resultant(F):
    return F[0::2].sum(), F[1::2].sum()


def _clamped_nodes(freedofs, ndof):
    fixed = np.setdiff1d(np.arange(ndof), freedofs)
    # Every support clamps both dofs of a node
    np.testing.assert_array_equal(fixed[0::2] + 1, fixed[1::2])
    return fixed[0::2] // 2


@pytest.mark.parametrize(
    "case,resultant",
    [
        (LoadCase.CANTILEVER, (0.0, -1.0)),
        (LoadCase.CORNER_LOADS, (0.0, -np.sqrt(2))),
        (LoadCase.EDGE_TRACTION, (1.0, 0.0)),
        (LoadCase.L_BRACKET_CORNER, (1.0, 0.0)),
        (LoadCase.L_BRACKET_MIDDLE, (1.0, 0.0)),
    ],
)
def test_each_case_carries_its_unit_loads(case, resultant):
    F, _ = load_cases.load_case(case, 40, 30, 0.002, 0.004, 0.001)
    np.testing.assert_allclose(_resultant(F), resultant, atol=1e-14)


def test_corner_loads_point_down_right_at_the_top_and_down_left_at_the_bottom():
    nelx, nely = 6, 6
    F, _ = load_cases.load_case(LoadCase.CORNER_LOADS, nelx, nely, 0.0, 0.0, 1.0)
    nodes = fem.node_grid(nelx, nely)
    c = 1 / np.sqrt(2)
    top, bottom = nodes[0, -1], nodes[-1, -1]
    np.testing.assert_allclose(F[[2 * top, 2 * top + 1]], [c, -c])
    np.testing.assert_allclose(F[[2 * bottom, 2 * bottom + 1]], [-c, -c])
    assert np.count_nonzero(F) == 4


@pytest.mark.parametrize("case", [LoadCase.CANTILEVER, LoadCase.CORNER_LOADS])
def test_left_edge_cases_clamp_the_whole_left_edge(case):
    nelx, nely = 5, 4
    F, freedofs = load_cases.load_case(case, nelx, nely, 0.0, 0.0, 1.0)
    np.testing.assert_array_equal(
        _clamped_nodes(freedofs, len(F)), np.sort(fem.node_grid(nelx, nely)[:, 0])
    )


@pytest.mark.parametrize("scale", [1, 2])
def test_edge_traction_patches_sit_at_their_fractions_at_any_resolution(scale):
    """Patch positions, in metres from the left and from the bottom, do not move when
    the mesh is refined."""
    nelx, nely, h = 120 * scale, 90 * scale, 0.001 / scale
    F, freedofs = load_cases.load_case(
        LoadCase.EDGE_TRACTION, nelx, nely, 0.0, 0.004, h
    )
    rows, cols = np.mgrid[: nely + 1, : nelx + 1]
    x_m, y_m = cols * h, (nely - rows) * h

    def within(u, lo, hi):
        return (u > lo - 1e-9) & (u < hi + 1e-9)

    # 25% and 80% of the 90 mm height, 30% of the 120 mm width, each +-2 mm
    expected = (cols == 0) & (within(y_m, 0.0205, 0.0245) | within(y_m, 0.070, 0.074))
    expected |= (rows == nely) & within(x_m, 0.034, 0.038)
    np.testing.assert_array_equal(
        _clamped_nodes(freedofs, len(F)), fem.node_grid(nelx, nely)[expected]
    )


def test_edge_traction_rejects_a_patch_off_the_edge():
    with pytest.raises(ValueError, match="does not fit"):
        load_cases.load_case(LoadCase.EDGE_TRACTION, 12, 9, 0.0, 0.004, 0.001)


def test_run_config_defaults_to_cantilever_for_records_without_a_load_case():
    d = default_run_config().to_dict()
    del d["load_case"], d["support_length_m"]
    config = RunConfig.from_dict(d)
    assert config.load_case == "cantilever"


def test_run_config_rejects_an_unknown_load_case():
    with pytest.raises(ValueError, match="mbb"):
        dataclasses.replace(default_run_config(), load_case="mbb")


@pytest.mark.parametrize(
    "case,col", [(LoadCase.L_BRACKET_CORNER, 0), (LoadCase.L_BRACKET_MIDDLE, 4)]
)
def test_l_brackets_load_down_their_vertical_edge_from_the_top(case, col):
    nelx, nely = 8, 6
    F, _ = load_cases.load_case(case, nelx, nely, 2.0, 0.0, 1.0)
    nodes = fem.node_grid(nelx, nely)
    loaded = np.flatnonzero(F) // 2
    np.testing.assert_array_equal(np.sort(loaded), np.sort(nodes[:3, col]))
    assert not F[1::2].any()  # horizontal


@pytest.mark.parametrize("case", L_BRACKETS)
def test_l_brackets_clamp_the_right_edge_of_their_bottom_half(case):
    nelx, nely = 8, 6
    F, freedofs = load_cases.load_case(case, nelx, nely, 0.0, 0.0, 1.0)
    np.testing.assert_array_equal(
        _clamped_nodes(freedofs, len(F)), np.sort(fem.node_grid(nelx, nely)[3:, -1])
    )


@pytest.mark.parametrize("case", L_BRACKETS)
def test_l_brackets_leave_out_the_top_right_quarter(case):
    active = load_cases.active_mask(case, 8, 6)
    assert not active[:3, 4:].any()
    assert active[3:].all() and active[:, :4].all()


@pytest.mark.parametrize("case", [LoadCase.CANTILEVER, LoadCase.EDGE_TRACTION])
def test_rectangular_cases_use_the_whole_mesh(case):
    assert load_cases.active_mask(case, 5, 3).all()


@pytest.mark.parametrize("nelx,nely", [(7, 6), (8, 5)])
def test_l_brackets_need_an_even_mesh(nelx, nely):
    with pytest.raises(ValueError, match="even"):
        load_cases.active_mask(LoadCase.L_BRACKET_CORNER, nelx, nely)


@pytest.mark.parametrize("cls", [HeatRunConfig, LaplaceRunConfig])
def test_virtual_heat_configs_refuse_a_domain_short_of_the_mesh(cls):
    with pytest.raises(ValueError, match="whole mesh"):
        virtual_heat_config(cls, nelx=40, nely=20, load_case="l_bracket_corner")
