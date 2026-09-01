"""Smoke tests for sttopt.viz -- per the plan's Phase 9 guidance, this is the lowest
correctness-risk phase (visual output, not numerical), so testing here is limited to
"runs without error on a small case" plus a couple of cheap structural sanity checks,
not pixel-level MATLAB fixture comparison (no fixture exists for these two functions).
"""

import numpy as np

import sttopt.viz as viz

NELX, NELY = 7, 5


def test_combination_plot_draws_only_solid_elements():
    rng = np.random.default_rng(0)
    xPhys = rng.uniform(0, 1, size=(NELY, NELX))
    tPhys = rng.uniform(0, 1, size=(NELY, NELX))
    eps = 0.3

    ax = viz.combination_plot(xPhys, tPhys, eps)

    n_expected = int(np.sum(xPhys >= eps))
    coll = ax.collections[0]
    assert len(coll.get_paths()) == n_expected
    assert coll.get_array().size == n_expected


def test_stage_boundary_plot_zero_segments_for_uniform_tfield():
    """No stage transitions exist when every element shares one print time."""
    tPhys = np.full((NELY, NELX), 0.5)

    ax = viz.stage_boundary_plot(tPhys, nStage=4)

    coll = ax.collections[0]
    assert len(coll.get_segments()) == 0


def test_stage_boundary_plot_nonzero_segments_for_two_region_split():
    """A hand-constructed two-region split (left half early, right half late) must
    produce boundary edges along the region seam.
    """
    tPhys = np.zeros((NELY, NELX))
    tPhys[:, NELX // 2 :] = 1.0

    ax = viz.stage_boundary_plot(tPhys, nStage=2)

    coll = ax.collections[0]
    assert len(coll.get_segments()) > 0


def test_gradient_magnitude_plot_draws_only_interior_solid_elements():
    """The gradient magnitude is a central difference, so the border carries no value
    and must stay blank even where it holds material.
    """
    xPhys = np.full((NELY, NELX), 1.0)
    grad = np.ones((NELY - 2, NELX - 2))

    ax = viz.timefield_gradient_magnitude_plot(xPhys, grad)

    coll = ax.collections[0]
    assert len(coll.get_paths()) == (NELY - 2) * (NELX - 2)


def test_combination_and_boundary_compose_on_one_axes():
    """The CLI overlays both plots on one Axes; check that composition actually works
    (the `ax` passed to `stage_boundary_plot` is reused, not replaced).
    """
    xPhys = np.full((NELY, NELX), 1.0)
    tPhys = np.zeros((NELY, NELX))
    tPhys[:, NELX // 2 :] = 1.0

    ax = viz.combination_plot(xPhys, tPhys, eps=0.1)
    ax2 = viz.stage_boundary_plot(tPhys, nStage=2, ax=ax, combination_coords=True)

    assert ax2 is ax
    assert len(ax.collections) == 2


def test_combination_coords_align_boundary_segments_with_combination_frame():
    """The two functions' native coordinate frames don't coincide (see viz.py's module
    docstring) -- this pins the `combination_coords=True` remap actually lands boundary
    segments inside the combination plot's own data extent, not in a disjoint region.
    """
    xPhys = np.full((NELY, NELX), 1.0)
    tPhys = np.zeros((NELY, NELX))
    tPhys[:, NELX // 2 :] = 1.0

    combo_ax = viz.combination_plot(xPhys, tPhys, eps=0.1)
    x0, x1 = combo_ax.dataLim.intervalx
    y0, y1 = combo_ax.dataLim.intervaly

    boundary_ax = viz.stage_boundary_plot(tPhys, nStage=2, combination_coords=True)
    bx0, bx1 = boundary_ax.dataLim.intervalx
    by0, by1 = boundary_ax.dataLim.intervaly

    assert x0 <= bx0 and bx1 <= x1
    assert y0 <= by0 and by1 <= y1
