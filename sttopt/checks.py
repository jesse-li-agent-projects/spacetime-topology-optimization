"""Checks on a finished `stto` design, read off the binarized design a print would
produce rather than the continuous one MMA optimized (`plans/post_run_checks.md`).

Two conditions are hard, since a design that breaks them cannot be printed as
sequenced: the print starts on the base, and every element off the part's boundary is
printed onto material already there. Everything else -- each constraint's value, compliance, the count of
saddles in the time field -- is reported as a margin, not judged. A failed hard check warns rather than failing the run: the
design is saved either way.
"""

import math
import warnings
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from torch import Tensor

import sttopt.compliance as compliance
import sttopt.geometry as geometry
import sttopt.run_config as run_config
import sttopt.stto as stto
import sttopt.torch_util as torch_util

# How far above `t = 0` a solid base element may start. The mean start-point constraint
# lets runs sit at a few 1e-6.
START_TOLERANCE = 1e-5

# How many offending elements a report lists by position.
_MAX_LISTED = 20


def report_path(design: Path) -> Path:
    """
    Where a saved design's check report is written: beside it.

    :param design: path of the saved design
    :return: path of its report
    """
    return design.with_name(f"{design.stem}_checks.json")


def unsupported(
    solid: Bool[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    base: Int[np.ndarray, " k"],
) -> Bool[np.ndarray, "nely nelx"]:
    """
    Solid elements off `base` with no 8-connected solid neighbor printed strictly
    before them.

    Empty exactly when every solid element connects to a solid base element through
    material printed in order: following earlier neighbors from any solid element
    must end somewhere, and only the base may end it.

    :param solid: which elements are solid
    :param tPhys: each element's print time
    :param base: flat indices of the base elements
    :return: which solid elements are unsupported
    """
    nely, nelx = solid.shape
    s, t = solid.flatten(), tPhys.flatten()
    src, dst, _ = geometry.neighbor_pairs(nelx, nely)
    onto_earlier = s[src] & s[dst] & (t[dst] < t[src])
    supported = np.zeros_like(s)
    supported[src[onto_earlier]] = True
    supported[base] = True
    return (s & ~supported).reshape(nely, nelx)


def interior(solid: Bool[np.ndarray, "nely nelx"]) -> Bool[np.ndarray, "nely nelx"]:
    """
    Solid elements whose 4 edge neighbors are all solid; outside the domain is void.

    :param solid: which elements are solid
    :return: which solid elements are off the part's boundary
    """
    p = np.pad(solid, 1)
    return solid & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def support_report(
    solid: Bool[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    base: Int[np.ndarray, " k"],
) -> dict:
    """
    The support check: no `unsupported` element in the part's `interior`.

    An unsupported element on the boundary reads as a steep overhang, which is accepted
    as printable (a heuristic, PR #189); those are counted but not judged.

    :param solid: which elements are solid
    :param tPhys: each element's print time
    :param base: flat indices of the base elements
    :return: `passed`, both counts, and the interior offenders' positions
    """
    orphans = unsupported(solid, tPhys, base)
    inside = orphans & interior(solid)
    return dict(
        passed=not inside.any(),
        unsupported=int(orphans.sum()),
        interior_unsupported=int(inside.sum()),
        interior_unsupported_at=np.argwhere(inside)[:_MAX_LISTED].tolist(),
    )


def warn_unsupported(support: dict) -> None:
    """Warn that a `support_report` failed."""
    warnings.warn(
        f"print support: {support['interior_unsupported']} interior solid element(s) have no solid neighbor printed before them, e.g. at (row, col) {support['interior_unsupported_at'][:5]}"
    )


#: A ring difference below this fraction of the field's range is a tie: no sign, so an
#: iso-line through the ring (e.g. of a linear field) is not read as a sign change.
SADDLE_TIE = 1e-9

# The 8-ring as (row, col) offsets, in order around the centre
_RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def saddles(
    values: Float[np.ndarray, "rows cols"],
    where: Bool[np.ndarray, "rows cols"],
    among: Bool[np.ndarray, "rows cols"] | None = None,
) -> Bool[np.ndarray, "rows cols"]:
    """
    Cells of `where` where `values - values[cell]` changes sign 4 or more times around
    the 8-ring: a discrete saddle (a strict extremum has no change at all).

    Works on any grid, elements or nodes. Ring cells outside `among` or off the grid are
    dropped from the sign sequence before the changes are counted, and so are ties:
    differences under `SADDLE_TIE` of the range of `values` over `among`.

    :param values: the field, e.g. `tPhys`
    :param where: the cells to test, e.g. the solid
    :param among: the cells a ring may read, e.g. the solid; `None` is every cell
    :return: which cells are saddles
    """
    rows, cols = values.shape
    mask = np.ones_like(where) if among is None else among

    def ring(a):
        padded = np.pad(a, 1)
        return np.stack(
            [padded[1 + i : 1 + i + rows, 1 + j : 1 + j + cols] for i, j in _RING],
            axis=-1,
        )

    ring_inside = ring(mask)
    diff = ring(values) - values[..., None]
    tie = SADDLE_TIE * float(np.ptp(values[mask])) if mask.any() else 0.0
    signs = np.where(ring_inside & (np.abs(diff) > tie), np.sign(diff), 0)
    out = np.zeros_like(where)
    for r, c in np.argwhere(where):
        s = signs[r, c][signs[r, c] != 0]
        out[r, c] = np.count_nonzero(s != np.roll(s, 1)) >= 4
    return out


def check_design(
    problem: stto.Problem,
    x: Float[Tensor, "nely nelx"],
    t: Float[Tensor, "nely nelx"],
    loop: int,
) -> dict:
    """
    Check a design, as raw design variables, at iteration `loop`'s settings.

    Warns once per failed hard check. Refreshes `problem`'s smooth-max calibrations.

    :param problem: the problem the design was optimized for
    :param x: design densities
    :param t: design time field
    :param loop: iteration whose schedules (`beta`, `penal`, ...) apply
    :return: a JSON-serializable report; `passed` is whether both hard checks passed
    """
    beta_d = run_config.weight_at(problem.config.beta_d_schedule, loop)
    xPhys, _ = stto.physical_fields(problem, x, t, beta_d)
    xBin, tPhys = stto.physical_fields(problem, x, t, math.inf)

    xBin_np = torch_util.to_numpy(xBin)
    tPhys_np = torch_util.to_numpy(tPhys)
    solid = geometry.solid_mask(xBin_np).reshape(xBin_np.shape)
    base = torch_util.to_numpy(problem.Nei)

    base_t = tPhys_np.flatten()[base][solid.flatten()[base]]
    max_base_t = float(base_t.max()) if base_t.size else None
    start = dict(
        passed=max_base_t is not None and max_base_t <= START_TOLERANCE,
        solid_base_elements=int(base_t.size),
        max_base_t=max_base_t,
        tolerance=START_TOLERANCE,
    )
    support = support_report(solid, tPhys_np, base)

    if not start["passed"]:
        warnings.warn(
            f"print start: {start['solid_base_elements']} solid base element(s), the latest starting at t = {max_base_t}, against a tolerance of {START_TOLERANCE}"
        )
    if not support["passed"]:
        warn_unsupported(support)

    xPhys_np = torch_util.to_numpy(xPhys)
    report = dict(
        loop=loop,
        passed=start["passed"] and support["passed"],
        start=start,
        support=support,
        saddles=int(saddles(tPhys_np, solid, solid).sum()),
        volume_fraction=float(solid.mean()),
        grey_fraction=float(((xPhys_np > 0.05) & (xPhys_np < 0.95)).mean()),
    )
    # The physics has nothing to measure on a design without material.
    if solid.any():
        report.update(_physics_report(problem, xBin, tPhys, loop))
    return report


def _physics_report(
    problem: stto.Problem,
    xBin: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    loop: int,
) -> dict:
    """
    Every constraint's values and the compliance, at iteration `loop`'s settings.

    :param xBin: binarized densities
    :param tPhys: physical time field
    :param loop: iteration whose schedules apply
    :return: `constraints` (name to values) and `compliance`
    """
    config = problem.config
    K_est = problem.terms.estimated_conductivity(xBin, tPhys, loop)
    g = stto.constraint_values(
        problem,
        xBin,
        tPhys,
        K_est,
        loop,
        run_config.weight_at(config.beta_t_schedule, loop),
    )
    c, _ = compliance.whole_compliance(
        xBin,
        problem.KE,
        problem.edofMat,
        config.Emin,
        config.Emax,
        run_config.weight_at(config.penal, loop),
        problem.freedofs,
        problem.F,
        problem.ndof,
    )
    return dict(
        constraints={name: torch_util.to_numpy(v).tolist() for name, v in g.items()},
        compliance=float(c),
    )


def summary(report: dict) -> str:
    """
    `check_design`'s report as a few lines for the console.

    :param report: a report from `check_design`, or from a virtual-heat script's
    :return: the summary text
    """
    start, support = report["start"], report["support"]
    # A virtual-heat report judges the start against the earliest element off the base
    bound = (
        f"tolerance {start['tolerance']}"
        if "tolerance" in start
        else f"earliest off the base at t = {start['earliest_off_base_t']}"
    )
    lines = [
        f"start:   {'pass' if start['passed'] else 'FAIL'} ({start['solid_base_elements']} solid base element(s), latest at t = {start['max_base_t']}, {bound})",
        f"support: {'pass' if support['passed'] else 'FAIL'} ({support['interior_unsupported']} interior unsupported element(s), {support['unsupported']} with the boundary){' (reported, not judged)' if 'domain' in report else ''}",
        f"saddles in the time field over the part: {report['saddles']} (reported, not judged)",
    ]
    if "domain" in report:
        domain = report["domain"]
        lines.append(
            f"support over the domain: {'pass' if domain['passed'] else 'FAIL'} ({domain['unsupported']} unsupported element(s), {domain['saddles']} saddle(s))"
        )
    if "constraints" in report:
        lines.append("binarized constraints, worst row (<= 0 satisfied):")
        width = max(map(len, report["constraints"]))
        lines += [
            f"  {name:<{width}} {max(values):+.4g}"
            for name, values in report["constraints"].items()
        ]
        lines.append(f"binarized compliance: {report['compliance']:.6g}")
    lines.append(
        f"volume fraction: {report['volume_fraction']:.4f}, grey fraction before binarizing: {report['grey_fraction']:.4f}"
    )
    return "\n".join(lines)
