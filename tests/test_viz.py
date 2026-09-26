"""Smoke tests for sttopt.viz -- per the plan's Phase 9 guidance, this is the lowest
correctness-risk phase (visual output, not numerical), so testing here is limited to
"runs without error on a small case" plus a couple of cheap structural sanity checks,
not pixel-level MATLAB fixture comparison (no fixture exists for these plots).
"""

import json

import numpy as np
from matplotlib.quiver import Quiver

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


def test_hotspot_severity_plot_quivers_the_print_direction_when_given():
    xPhys = np.ones((NELY, NELX))
    tPhys = np.tile(np.linspace(0.0, 1.0, NELX)[None, :], (NELY, 1))
    direction = (np.ones_like(tPhys), np.zeros_like(tPhys))

    ax = viz.hotspot_severity_plot(
        xPhys, np.zeros_like(tPhys), tPhys, nStage=2, direction=direction
    )

    assert isinstance(ax.collections[-1], Quiver)


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


def test_main_regenerates_plots_from_a_seqopt_run_directory(tmp_path, monkeypatch):
    """`_main` must detect a seqopt run directory (by seq_config.json) and regenerate
    its plots without a Problem/FEM, per the plan's Phase 2 viz.py section."""
    import sttopt.seqopt_cli as seqopt_cli
    from conftest import SEQ_ELEMENT_M, default_seq_run_config

    monkeypatch.chdir(tmp_path)
    config = default_seq_run_config(
        lrmin_m=1.5 * SEQ_ELEMENT_M,
        rmin_cond_m=2.5 * SEQ_ELEMENT_M,
        nStage=2,
        nloop=2,
        tmove=0.05,
    )
    xPhys = np.zeros((NELY, NELX))
    xPhys[-2:, :] = 1.0
    xPhys[:, 0] = 1.0
    geometry_path = tmp_path / "geometry.npz"
    np.savez(
        geometry_path,
        xPhys=xPhys,
        width_m=NELX * SEQ_ELEMENT_M,
        height_m=NELY * SEQ_ELEMENT_M,
    )
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
        "print_direction.png",
        "iso_curvature.png",
        "timefield_contour.png",
        "timefield_filled_contour.png",
    ]:
        assert (plot_dir / name).exists()


def test_print_direction_plot_draws_one_arrow_per_sampled_solid_element():
    """No arrow where there is no direction: void, and anywhere the time field is flat
    enough to leave `grad t` at zero."""
    xPhys = np.ones((NELY, NELX))
    xPhys[0, :] = 0.0
    tPhys = np.tile(np.linspace(0.0, 1.0, NELY)[:, None], (1, NELX))
    ux = np.zeros_like(tPhys)
    uy = np.ones_like(tPhys)
    uy[:, -1] = 0.0  # a column with no direction at all

    ax = viz.print_direction_plot(xPhys, tPhys, (ux, uy), stride=1)

    drawn = (xPhys > 0.5) & ((ux**2 + uy**2) > 0)
    quiver = ax.collections[-1]
    assert len(quiver.get_offsets()) == int(drawn.sum())


def test_print_direction_plot_flips_y_into_the_plot_frame():
    """`combination_plot`'s frame negates y, so a direction toward increasing row must
    draw downward -- otherwise every arrow points the wrong way and the plot silently
    says the opposite of the truth."""
    xPhys = np.ones((2, 2))
    tPhys = np.array([[0.0, 0.0], [1.0, 1.0]])
    ux = np.zeros_like(tPhys)
    uy = np.ones_like(tPhys)  # toward increasing row

    ax = viz.print_direction_plot(xPhys, tPhys, (ux, uy), stride=1)

    quiver = ax.collections[-1]
    assert np.all(quiver.V < 0)


def test_iso_curvature_plot_greys_the_unmeasured_border_and_centres_on_zero():
    """Border elements have no curvature but are still part of the design, and the
    scale is symmetric so concave and convex read at the same strength."""
    xPhys = np.ones((NELY, NELX))
    curvature = np.pad(
        np.linspace(-0.2, 0.1, (NELY - 2) * (NELX - 2)).reshape(NELY - 2, NELX - 2),
        1,
        constant_values=np.nan,
    )

    ax = viz.iso_curvature_plot(xPhys, curvature)

    coll = ax.collections[0]
    assert len(coll.get_paths()) == NELY * NELX
    low, high = coll.get_clim()
    assert high == -low > 0
