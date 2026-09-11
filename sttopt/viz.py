"""Plots for the printed structure: elements colored by print time, and the boundaries
between print stages.

When no `ax` is passed, both functions build their own `Figure` directly rather than
going through `pyplot`, so nothing is registered globally and there's nothing for the
caller to close; `savefig` still works. Pass your own `ax` (e.g. from `plt.subplots()`)
to draw into a pyplot-managed, interactive figure instead.

Coordinate conventions differ between the two functions, on purpose: `combination_plot`
places element `(row, col)` at `x in [col, col+1]`, `y in [-(row+1), -row]` (y flipped);
`stage_boundary_plot` places it at `x in [col+0.5, col+1.5]`, `y in [row+0.5, row+1.5]`
(no flip, half-cell offset). The two frames are related by `x' = x - 0.5, y' = 0.5 - y`;
`stage_boundary_plot(..., combination_coords=True)` applies it to compose both plots on
one `Axes` (as `stto_cli.py` does). See `conventions.md`.

Run as a script (`python -m sttopt.viz <tag>`) to regenerate plots for a saved run from
its `output/<tag>/` artefacts, without rerunning the optimization. This reads
`final_design.npz`'s `xPhys`/`tPhys` -- the state *after* the last MMA update -- whereas
`stto_cli.py`'s own end-of-run plot uses `prev_state` (the state entering that last update).
Pass `--animate` to instead render the time field filled contour as a GIF across a
seqopt run's logged checkpoints.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.dpi"] = "300"

_BOUNDARY_LINEWIDTH = 1.5
_VOID_GRAY = "0.5"
_VOID_ALPHA = 0.65
_NO_MEASURE_GRAY = "0.75"  # solid element, quantity undefined -- not void


def _new_axes() -> Axes:
    """A standalone `Axes` on a `Figure` that pyplot does not own.

    Not registered with pyplot, so it needs no `plt.close()` -- it's reclaimed by
    ordinary garbage collection once the returned `Axes` goes out of scope. Creating
    `Figure` directly is matplotlib's documented route for library/application code.
    """
    return Figure().add_subplot()


def _cell_verts(
    rows: Float[np.ndarray, "n"], cols: Float[np.ndarray, "n"]
) -> Float[np.ndarray, "n 4 2"]:
    """Unit-square corners for elements `(rows[i], cols[i])`, in `combination_plot`'s
    coordinate frame: `x in [col, col+1]`, `y in [-(row+1), -row]`.
    """
    return np.stack(
        [
            np.stack([cols, -rows], axis=1),
            np.stack([cols + 1, -rows], axis=1),
            np.stack([cols + 1, -(rows + 1)], axis=1),
            np.stack([cols, -(rows + 1)], axis=1),
        ],
        axis=1,
    ).astype(float)


def combination_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    values: Float[np.ndarray, "nely nelx"],
    eps: float,
    *,
    cmap: str | Colormap = "viridis",
    colorbar_label: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draws only the elements with density `>= eps`, each colored by `values` (flat
    per-face color, not interpolated). `values` is any per-element scalar (e.g. `tPhys`,
    or a hotspot-severity score), not necessarily a time field.

    An element still draws where its value is `nan`, in the colormap's "bad" color --
    the cell is part of the design whether or not the quantity has a value there, so
    pass a colormap whose bad color says so (the default is fully transparent).

    :param cmap: colormap name or object; callers below pick one that suits `values`.
    :param colorbar_label: if given, adds a horizontal colorbar below `ax` labelled with
        this string; omitted (the default) draws no colorbar.
    """
    if ax is None:
        ax = _new_axes()

    rows, cols = np.nonzero(xPhys >= eps)
    verts = _cell_verts(rows, cols)
    values = values[rows, cols]

    coll = PolyCollection(verts, array=values, cmap=cmap, edgecolors="none")
    ax.add_collection(coll)
    ax.set_aspect("equal")
    ax.autoscale_view()
    if colorbar_label is not None:
        ax.figure.colorbar(coll, ax=ax, orientation="horizontal", label=colorbar_label)
    return ax


def stage_boundary_plot(
    tPhys: Float[np.ndarray, "nely nelx"],
    nStage: int,
    *,
    ax: Axes | None = None,
    combination_coords: bool = False,
) -> Axes:
    """Assigns each element to one of `nStage` print stages by its `tPhys` value
    (half-open `(tt[j], tt[j+1]]` bins except the first, which is closed on both ends),
    then draws a black line along every internal mesh edge whose two adjacent elements
    fall in different stages.

    :param combination_coords: remap edges into `combination_plot`'s coordinate frame
        (`x' = x - 0.5, y' = 0.5 - y`) before drawing, e.g. to overlay onto an `Axes` a
        prior `combination_plot` call already populated; leave `False` for a standalone
        plot. See the module docstring.
    """
    if ax is None:
        ax = _new_axes()

    tt = np.linspace(0.0, 1.0, nStage + 1)
    stage = np.zeros(tPhys.shape, dtype=int)
    for j in range(nStage):
        lo, hi = tt[j], tt[j + 1]
        mask = (tPhys >= lo) & (tPhys <= hi) if j == 0 else (tPhys > lo) & (tPhys <= hi)
        stage[mask] = j + 1

    def _pt(x: float, y: float) -> tuple[float, float]:
        return (x - 0.5, 0.5 - y) if combination_coords else (x, y)

    segments = []
    # Vertical edges: element (row, col) | (row, col+1), shared edge at x = col+1.5.
    rows, cols = np.nonzero(stage[:, :-1] != stage[:, 1:])
    for row, col in zip(rows, cols):
        x = col + 1.5
        segments.append([_pt(x, row + 0.5), _pt(x, row + 1.5)])
    # Horizontal edges: element (row, col) | (row+1, col), shared edge at y = row+1.5.
    rows, cols = np.nonzero(stage[:-1, :] != stage[1:, :])
    for row, col in zip(rows, cols):
        y = row + 1.5
        segments.append([_pt(col + 0.5, y), _pt(col + 1.5, y)])

    ax.add_collection(
        LineCollection(segments, colors="black", linewidths=_BOUNDARY_LINEWIDTH)
    )
    ax.set_aspect("equal")
    ax.autoscale_view()
    return ax


def hotspot_severity_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    hotspot_severity: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    nStage: int,
    *,
    ax: Axes | None = None,
) -> Axes:
    """`combination_plot` (binarized density, colored by `hotspot_severity`, `plasma`
    colormap, labelled horizontal colorbar) with `stage_boundary_plot` overlaid in its
    `combination_coords` frame -- the plot recipe `stto_cli.py` saves as
    `hotspot_severity.png`.

    :param xPhys: physical density field (not yet binarized).
    :param hotspot_severity: per-element overheating severity, e.g. `(1 - K_est) * (xPhys > 0.5)`.
    :param tPhys: physical print-time field.
    :param nStage: number of print stages, for `stage_boundary_plot`'s binning.
    :return: the `Axes` drawn into.
    """
    XPhys = (xPhys > 0.5).astype(xPhys.dtype)
    ax = combination_plot(
        XPhys,
        hotspot_severity,
        eps=1.0e-1,
        # Gray where the measure reports nothing (`nan`), so the design's full extent
        # still reads while the color scale spans only the elements it does report.
        cmap=matplotlib.colormaps["plasma"].with_extremes(bad=_NO_MEASURE_GRAY),
        colorbar_label="Hotspot severity",
        ax=ax,
    )
    stage_boundary_plot(tPhys, nStage, ax=ax, combination_coords=True)
    ax.set_title("Hotspot severity")
    return ax


def _timefield_title(compliance: float | None) -> str:
    if compliance is None:
        return "Time field"
    return f"Time field (compliance: {compliance:.4g})"


def timefield_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    *,
    compliance: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """`combination_plot` (binarized density, colored by `tPhys`, `viridis` colormap,
    labelled horizontal colorbar) -- masked to the same `xPhys > 0.5` elements as
    `hotspot_severity_plot`, so the two plots cover the same printed region.

    :param xPhys: physical density field (not yet binarized).
    :param tPhys: physical print-time field.
    :param compliance: whole-structure compliance to report in the title, if known.
    :return: the `Axes` drawn into.
    """
    XPhys = (xPhys > 0.5).astype(xPhys.dtype)
    ax = combination_plot(
        XPhys, tPhys, eps=1.0e-1, cmap="viridis", colorbar_label="Time field", ax=ax
    )
    ax.set_title(_timefield_title(compliance))
    return ax


def timefield_gradient_magnitude_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    gradient_magnitude: Float[np.ndarray, "nely nelx"],
    *,
    ax: Axes | None = None,
) -> Axes:
    """`combination_plot` (binarized density, `magma` colormap) of the time field's
    gradient magnitude, i.e. the reciprocal of the local deposited-layer thickness: an
    even color means even layers.

    :param xPhys: physical density field (not yet binarized), over the full mesh.
    :param gradient_magnitude: per-element `|grad tPhys|`, e.g. from
        `timefield.gradient_magnitude_elements`.
    :return: the `Axes` drawn into.
    """
    ax = combination_plot(
        (xPhys > 0.5).astype(xPhys.dtype),
        gradient_magnitude,
        eps=1.0e-1,
        cmap="magma",
        colorbar_label="Time field gradient magnitude",
        ax=ax,
    )
    ax.set_title("Time field gradient magnitude")
    return ax


def timefield_contour_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    nContours: int,
    *,
    compliance: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """`timefield_plot` with `nContours` thin black contour lines of `tPhys` overlaid,
    masked to the same `xPhys > 0.5` region.

    :param xPhys: physical density field (not yet binarized).
    :param tPhys: physical print-time field.
    :param nContours: number of contour lines.
    :param compliance: whole-structure compliance to report in the title, if known.
    :return: the `Axes` drawn into.
    """
    ax = timefield_plot(xPhys, tPhys, compliance=compliance, ax=ax)

    nely, nelx = tPhys.shape
    # Cell centers, in combination_plot's coordinate frame: x in [col, col+1],
    # y in [-(row+1), -row].
    X, Y = np.meshgrid(np.arange(nelx) + 0.5, -(np.arange(nely) + 0.5))
    masked_tPhys = np.where(xPhys > 0.5, tPhys, np.nan)
    ax.contour(X, Y, masked_tPhys, levels=nContours, colors="black", linewidths=0.5)
    return ax


def timefield_filled_contour_plot(
    xPhys: Float[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    nContours: int,
    *,
    compliance: float | None = None,
    show_void: bool = False,
    levels: Float[np.ndarray, "nContours+1"] | None = None,
    colorbar: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Filled contour plot of `tPhys` with a colorbar and thin black borders between
    the filled segments.

    Unlike `timefield_contour_plot`, this doesn't mask `tPhys` to `xPhys > 0.5` before
    contouring (which would clip segments to a jagged element boundary); instead it
    contours the full field, then paints over the elements without material in a layer
    above the contour.

    :param xPhys: physical density field (not yet binarized).
    :param tPhys: physical print-time field.
    :param nContours: number of filled contour levels.
    :param compliance: whole-structure compliance to report in the title, if known.
    :param show_void: cover the void with translucent gray instead of opaque white, so
        the time field there stays readable while still reading as "not printed".
    :param levels: contour level boundaries; defaults to `nContours` even bins spanning
        `tPhys`'s own range, e.g. pass a range spanning several `tPhys` fields to keep
        the color scale consistent across an animation's frames.
    :param colorbar: draw a colorbar. `fig.colorbar(ax=ax)` shrinks `ax` to make room
        for it, so redrawing into the same `ax` across an animation's frames (with
        shared `levels`, whose mapping never changes) should pass `False` after the
        first frame -- otherwise the shrink compounds frame over frame.
    :return: the `Axes` drawn into.
    """
    if ax is None:
        ax = _new_axes()

    nely, nelx = tPhys.shape
    # Cell centers, in combination_plot's coordinate frame: x in [col, col+1],
    # y in [-(row+1), -row].
    X, Y = np.meshgrid(np.arange(nelx) + 0.5, -(np.arange(nely) + 0.5))
    if levels is None:
        levels = np.linspace(tPhys.min(), tPhys.max(), nContours + 1)

    filled = ax.contourf(X, Y, tPhys, levels=levels, cmap="viridis")
    ax.contour(X, Y, tPhys, levels=levels, colors="black", linewidths=0.5)
    if colorbar:
        ax.figure.colorbar(filled, ax=ax, orientation="horizontal", label="Time field")

    rows, cols = np.nonzero(xPhys <= 0.5)
    empty = PolyCollection(
        _cell_verts(rows, cols),
        facecolors=_VOID_GRAY if show_void else "white",
        alpha=_VOID_ALPHA if show_void else 1.0,
        edgecolors="none",
        zorder=10,
    )
    ax.add_collection(empty)

    ax.set_aspect("equal")
    ax.autoscale_view()
    title = "Time field (filled contour"
    title += ", void shown)" if show_void else ")"
    if compliance is not None:
        title += f" (compliance: {compliance:.4g})"
    ax.set_title(title)
    return ax


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate plots for a saved run from its output/<tag>/ "
        "artefacts, saved under plot/<tag>/."
    )
    parser.add_argument("tag", help="run identifier, matching output/<tag>/")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="parent directory of run artefacts (default: output)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("plot"),
        help="parent directory to save plots under (default: plot)",
    )
    parser.add_argument(
        "--n-contours",
        type=int,
        default=30,
        help="number of time field contour lines (default: 10)",
    )
    parser.add_argument(
        "--design-file",
        default="final_design.npz",
        help="which output/<tag>/ artefact to plot (default: final_design.npz); pass "
        "e.g. design_it0200.npz to inspect an intermediate checkpoint",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="instead of the usual plots, animate the time field filled contour "
        "across every output/<tag>/design_it*.npz checkpoint plus final_design.npz "
        "(seqopt runs only, saved as timefield_filled_contour_animation.gif)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="animation frame rate, only used with --animate (default: 2)",
    )
    return parser.parse_args(argv)


def _hotspot_severity(
    xPhys: Float[np.ndarray, "nely nelx"], K_est: Float[np.ndarray, "nely nelx"]
) -> Float[np.ndarray, "nely nelx"]:
    """`1 - K_est` over the printed elements, `nan` where the hotspot measure has no
    value to report -- an element an infinitely conductive print base shields
    (`conductivity.infinite_base`) has `K_est == inf`. `nan` keeps it out of the color
    scale, which a severity the run never optimized against would otherwise dominate;
    the element itself still draws (`combination_plot`'s "bad" color).
    """
    severity = np.full(K_est.shape, np.nan)
    measured = np.isfinite(K_est)  # `inf * 0` where it is not, hence no bare `where`
    severity[measured] = (1 - K_est[measured]) * (xPhys[measured] > 0.5)
    return severity


def _load_stto_run(run_dir: Path, design_file: str = "final_design.npz") -> tuple:
    """Loads an `stto` run directory's plotting inputs: `xPhys`, `tPhys`, whole-
    structure compliance, hotspot severity, `|grad tPhys|`, and `nStage`.

    :param design_file: an artefact under `run_dir` holding `xPhys`/`tPhys`, e.g.
        `final_design.npz`. The periodic `design_it*.npz` checkpoints save only the raw
        `x`/`t` design variables (no filtering/projection), so they can't be plotted
        this way -- rerunning with denser checkpointing of `xPhys`/`tPhys` is the fix
        if an intermediate `stto` frame is needed.
    """
    import json

    import sttopt.compliance as compliance
    import sttopt.stto as stto
    import sttopt.timefield as timefield
    import sttopt.torch_util as torch_util
    from sttopt.run_config import RunConfig

    config = RunConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    design = np.load(run_dir / design_file)
    if "xPhys" not in design or "tPhys" not in design:
        raise SystemExit(
            f"{run_dir / design_file}: no xPhys/tPhys (checkpoints only save raw "
            "x/t) -- pass an artefact with filtered/projected fields, e.g. "
            "final_design.npz"
        )
    xPhys, tPhys = design["xPhys"], design["tPhys"]

    problem = stto.build_problem(config)
    xPhys_t = torch_util.to_tensor(xPhys, device=problem.device, dtype=problem.dtype)
    tPhys_t = torch_util.to_tensor(tPhys, device=problem.device, dtype=problem.dtype)
    obj, _ = compliance.whole_compliance(
        xPhys_t,
        problem.KE,
        problem.edofMat,
        config.Emin,
        config.Emax,
        config.penal,
        problem.freedofs,
        problem.F,
        problem.ndof,
    )
    obj = float(obj)
    _, K_est_t = stto.hotspot_value(problem, xPhys_t, tPhys_t)
    K_est = torch_util.to_numpy(K_est_t).reshape(config.nely, config.nelx)
    hotspot_severity = _hotspot_severity(xPhys, K_est)
    grad_magnitude = torch_util.to_numpy(timefield.gradient_magnitude_elements(tPhys_t))
    return xPhys, tPhys, obj, hotspot_severity, grad_magnitude, config.nStage


def _load_seqopt_run(run_dir: Path, design_file: str = "final_design.npz") -> tuple:
    """`_load_stto_run`'s counterpart for a `seqopt` run directory: no FEM, so
    compliance is `None`, and the geometry comes from the run's own artefacts rather
    than from a config.

    :param design_file: an artefact under `run_dir` holding a time field, either
        `final_design.npz` (`xPhys`/`tPhys`) or a periodic `design_it*.npz` checkpoint
        (`t` alone -- `seqopt` never filters the time field, so `t` doubles as `tPhys`;
        the fixed geometry comes from `run_dir/geometry.npz` instead).
    """
    import json

    import torch

    import sttopt.seqopt as seqopt
    import sttopt.timefield as timefield
    import sttopt.torch_util as torch_util
    from sttopt.run_config import SeqRunConfig

    config = SeqRunConfig.from_dict(
        json.loads((run_dir / "seq_config.json").read_text())
    )
    design = np.load(run_dir / design_file)
    xPhys = (
        design["xPhys"]
        if "xPhys" in design
        else np.load(run_dir / "geometry.npz")["xPhys"]
    )
    tPhys = design["tPhys"] if "tPhys" in design else design["t"]

    problem = seqopt.build_problem(config, xPhys, device="cpu", dtype=torch.float64)
    tPhys_t = torch_util.to_tensor(tPhys, device="cpu", dtype=torch.float64)
    _, K_est_t = seqopt.hotspot_value(problem, tPhys_t)
    xPhys = torch_util.to_numpy(problem.xPhys)  # what the run optimized, post-cleanup
    K_est = torch_util.to_numpy(K_est_t).reshape(xPhys.shape)
    hotspot_severity = _hotspot_severity(xPhys, K_est)
    grad_magnitude = torch_util.to_numpy(timefield.gradient_magnitude_elements(tPhys_t))
    return xPhys, tPhys, None, hotspot_severity, grad_magnitude, config.nStage


def _design_checkpoints(run_dir: Path) -> list[str]:
    """Every `design_it*.npz` checkpoint under `run_dir`, in iteration order, followed by
    `final_design.npz`.
    """
    import re

    checkpoints = sorted(
        run_dir.glob("design_it*.npz"),
        key=lambda p: int(re.search(r"\d+", p.stem).group()),
    )
    return [p.name for p in checkpoints] + ["final_design.npz"]


def _animate_timefield_filled_contour(
    run_dir: Path, plot_dir: Path, n_contours: int, fps: float, is_seqopt: bool
) -> None:
    """Renders `timefield_filled_contour_plot` for every design checkpoint in `run_dir`
    and stitches the frames into a GIF, with one shared color scale spanning every
    frame's `tPhys` range.

    Only seqopt runs are supported: `stto`'s `design_it*.npz` checkpoints save raw,
    unfiltered `x`/`t` rather than `xPhys`/`tPhys` (see `_load_stto_run`), so they can't
    be plotted this way.
    """
    from matplotlib.animation import PillowWriter

    if not is_seqopt:
        raise SystemExit(
            "--animate only supports seqopt runs -- stto's design_it*.npz checkpoints "
            "don't carry filtered xPhys/tPhys (see _load_stto_run)"
        )

    design_files = _design_checkpoints(run_dir)
    frames = [_load_seqopt_run(run_dir, design_file) for design_file in design_files]
    levels = np.linspace(
        min(tPhys.min() for _, tPhys, *_ in frames),
        max(tPhys.max() for _, tPhys, *_ in frames),
        n_contours + 1,
    )

    fig = Figure()
    ax = fig.add_subplot()
    out_path = plot_dir / "timefield_filled_contour_animation.gif"
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out_path, dpi=150):
        for i, (design_file, (xPhys, tPhys, obj, *_rest)) in enumerate(
            zip(design_files, frames)
        ):
            ax.clear()
            # Only the first frame draws a colorbar: `levels` (and so the color
            # mapping) is shared across every frame, so it never needs redrawing --
            # and redrawing it would reshrink ax, which ax.clear() does not undo.
            timefield_filled_contour_plot(
                xPhys,
                tPhys,
                n_contours,
                compliance=obj,
                levels=levels,
                colorbar=(i == 0),
                ax=ax,
            )
            ax.set_title(f"{ax.get_title()} -- {Path(design_file).stem}")
            writer.grab_frame()
    print(f"Saved time field filled contour animation to {out_path}")


def _main(args: argparse.Namespace) -> None:
    run_dir = args.output_dir / args.tag
    is_seqopt = (run_dir / "seq_config.json").exists()

    if args.animate:
        plot_dir = args.plot_dir / args.tag
        plot_dir.mkdir(parents=True, exist_ok=True)
        _animate_timefield_filled_contour(
            run_dir, plot_dir, args.n_contours, args.fps, is_seqopt
        )
        return

    if is_seqopt:
        xPhys, tPhys, obj, hotspot_severity, grad_magnitude, nStage = _load_seqopt_run(
            run_dir, args.design_file
        )
    else:
        xPhys, tPhys, obj, hotspot_severity, grad_magnitude, nStage = _load_stto_run(
            run_dir, args.design_file
        )

    # Nest non-default design files under their own subdirectory so an intermediate
    # checkpoint's plots don't overwrite the final design's.
    design_stem = Path(args.design_file).stem
    plot_dir = args.plot_dir / args.tag
    if design_stem != "final_design":
        plot_dir /= design_stem
    plot_dir.mkdir(parents=True, exist_ok=True)
    ax = hotspot_severity_plot(xPhys, hotspot_severity, tPhys, nStage)
    out_path = plot_dir / "hotspot_severity.png"
    ax.figure.savefig(out_path)
    print(f"Saved hotspot severity plot to {out_path}")

    ax = timefield_plot(xPhys, tPhys, compliance=obj)
    out_path = plot_dir / "timefield.png"
    ax.figure.savefig(out_path)
    print(f"Saved time field plot to {out_path}")

    ax = timefield_gradient_magnitude_plot(xPhys, grad_magnitude)
    out_path = plot_dir / "timefield_gradient_magnitude.png"
    ax.figure.savefig(out_path)
    print(f"Saved time field gradient magnitude plot to {out_path}")

    ax = timefield_contour_plot(xPhys, tPhys, args.n_contours, compliance=obj)
    out_path = plot_dir / "timefield_contour.png"
    ax.figure.savefig(out_path)
    print(f"Saved time field contour plot to {out_path}")

    ax = timefield_filled_contour_plot(xPhys, tPhys, args.n_contours, compliance=obj)
    out_path = plot_dir / "timefield_filled_contour.png"
    ax.figure.savefig(out_path)
    print(f"Saved time field filled contour plot to {out_path}")

    ax = timefield_filled_contour_plot(
        xPhys, tPhys, args.n_contours, compliance=obj, show_void=True
    )
    out_path = plot_dir / "timefield_filled_contour_with_void.png"
    ax.figure.savefig(out_path)
    print(f"Saved time field filled contour (void shown) plot to {out_path}")


if __name__ == "__main__":
    _args = _parse_args()  # early-exits on --help before the heavy imports below
    _main(_args)
