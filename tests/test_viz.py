"""Smoke tests for sttopt.viz -- per the plan's Phase 9 guidance, this is the lowest
correctness-risk phase (visual output, not numerical), so testing here is limited to
"runs without error on a small case" plus a couple of cheap structural sanity checks,
not pixel-level MATLAB fixture comparison (no fixture exists for these two functions).
"""

import json

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


def test_hotspot_severity_plot_titles_with_the_measured_maximum():
    """The maximum is what the hotspot constraint is about, so it belongs in the title --
    and the unmeasured elements the measure leaves as `nan` must not swallow it.
    """
    xPhys = np.ones((NELY, NELX))
    severity = np.full((NELY, NELX), 0.25)
    severity[0, 0] = 0.75
    severity[-1, -1] = np.nan

    ax = viz.hotspot_severity_plot(xPhys, severity, np.zeros((NELY, NELX)), nStage=1)

    assert "0.75" in ax.get_title()


def test_hotspot_severity_plot_titles_without_a_maximum_when_nothing_is_measured():
    """An all-`nan` severity has no maximum to report, and `nan` is not a number to put
    in front of a reader."""
    xPhys = np.ones((NELY, NELX))
    severity = np.full((NELY, NELX), np.nan)

    ax = viz.hotspot_severity_plot(xPhys, severity, np.zeros((NELY, NELX)), nStage=1)

    assert ax.get_title() == "Hotspot severity"


def test_gradient_magnitude_plot_draws_every_solid_element():
    """The Gauss-point magnitudes scatter back onto every element, border included, so
    the plot covers the whole part rather than leaving a blank frame."""
    xPhys = np.full((NELY, NELX), 1.0)
    grad = np.ones((NELY, NELX))

    ax = viz.timefield_gradient_magnitude_plot(xPhys, grad)

    coll = ax.collections[0]
    assert len(coll.get_paths()) == NELY * NELX


def test_timefield_plots_hold_the_full_scale_on_a_partial_field():
    """Every plot of the time field is on `[0, 1]`, the range the normalization gives
    it, not on the range of the elements that happen to be drawn. A field covering only
    part of the build must not stretch to fill the colorbar -- that would read as a
    complete build, and would move the scale from run to run.
    """
    xPhys = np.ones((NELY, NELX))
    # A build that ends barely past halfway, and never reaches either end of the scale.
    tPhys = np.linspace(0.2, 0.55, NELY * NELX).reshape(NELY, NELX)

    ax = viz.timefield_plot(xPhys, tPhys)
    assert ax.collections[0].get_clim() == viz.TIMEFIELD_RANGE

    ax = viz.timefield_filled_contour_plot(xPhys, tPhys, nContours=4)
    filled = ax.collections[0]
    assert (min(filled.levels), max(filled.levels)) == viz.TIMEFIELD_RANGE


def test_timefield_contour_lines_sit_on_the_filled_contour_boundaries():
    """The line and filled contour plots bin the same scale the same way, so the lines
    one draws land on the edges the other fills between."""
    xPhys = np.ones((NELY, NELX))
    tPhys = np.linspace(0.0, 1.0, NELY * NELX).reshape(NELY, NELX)

    lines = viz.timefield_contour_plot(xPhys, tPhys, nContours=4).collections[-1]
    filled = viz.timefield_filled_contour_plot(xPhys, tPhys, nContours=4).collections[0]
    np.testing.assert_allclose(lines.levels, filled.levels)


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


def test_main_regenerates_plots_from_a_seqopt_run_directory(tmp_path, monkeypatch):
    """`_main` must detect a seqopt run directory (by seq_config.json) and regenerate
    its plots without a Problem/FEM, per the plan's Phase 2 viz.py section."""
    import sttopt.seqopt_cli as seqopt_cli
    from conftest import default_seq_run_config

    monkeypatch.chdir(tmp_path)
    config = default_seq_run_config(
        lrmin=1.5, rmin_cond=2.5, nStage=2, nloop=2, tmove=0.05
    )
    xPhys = np.zeros((NELY, NELX))
    xPhys[-2:, :] = 1.0
    xPhys[:, 0] = 1.0
    geometry_path = tmp_path / "geometry.npz"
    np.savez(geometry_path, xPhys=xPhys)
    config_path = tmp_path / "seq_config.json"
    config_path.write_text(json.dumps(config.to_dict()))

    seqopt_cli.main(
        seqopt_cli.parse_args(
            [
                "--geometry",
                str(geometry_path),
                "--config",
                str(config_path),
                "--tag",
                "seqviz",
            ]
        )
    )

    viz._main(viz._parse_args(["seqviz"]))

    plot_dir = tmp_path / "plot" / "seqviz"
    for name in [
        "hotspot_severity.png",
        "timefield.png",
        "timefield_gradient_magnitude.png",
        "timefield_contour.png",
        "timefield_filled_contour.png",
    ]:
        assert (plot_dir / name).exists()
