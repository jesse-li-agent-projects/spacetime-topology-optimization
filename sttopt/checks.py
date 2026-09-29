"""Checks on a finished `stto` design, read off the binarized design a print would
produce rather than the continuous one MMA optimized (`plans/post_run_checks.md`).

Two conditions are hard, since a design that breaks them cannot be printed as
sequenced: the print starts on the base, and every element is printed onto material
already there. Everything else -- each constraint's value, compliance -- is reported
as a margin, not judged. A failed hard check warns rather than failing the run: the
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
    """Where a saved design's check report is written: beside it."""
    return design.with_name(f"{design.stem}_checks.json")


def unsupported(
    solid: Bool[np.ndarray, "nely nelx"],
    tPhys: Float[np.ndarray, "nely nelx"],
    base: Int[np.ndarray, " k"],
) -> Bool[np.ndarray, "nely nelx"]:
    """Solid elements off `base` with no 8-connected solid neighbor printed strictly
    before them.

    Empty exactly when every solid element connects to a solid base element through
    material printed in order: following earlier neighbors from any solid element
    must end somewhere, and only the base may end it.
    """
    nely, nelx = solid.shape
    s, t = solid.flatten(), tPhys.flatten()
    src, dst, _ = geometry.neighbor_pairs(nelx, nely)
    onto_earlier = s[src] & s[dst] & (t[dst] < t[src])
    supported = np.zeros_like(s)
    supported[src[onto_earlier]] = True
    supported[base] = True
    return (s & ~supported).reshape(nely, nelx)


def check_design(
    problem: stto.Problem,
    x: Float[Tensor, "nely nelx"],
    t: Float[Tensor, "nely nelx"],
    loop: int,
) -> dict:
    """Check a design, as raw design variables, at iteration `loop`'s settings.

    Warns once per failed hard check. Refreshes `problem`'s smooth-max calibrations.

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
    orphans = unsupported(solid, tPhys_np, base)
    support = dict(
        passed=not orphans.any(),
        unsupported=int(orphans.sum()),
        unsupported_at=np.argwhere(orphans)[:_MAX_LISTED].tolist(),
    )

    if not start["passed"]:
        warnings.warn(
            f"print start: {start['solid_base_elements']} solid base element(s), the latest starting at t = {max_base_t}, against a tolerance of {START_TOLERANCE}"
        )
    if not support["passed"]:
        warnings.warn(
            f"print support: {support['unsupported']} solid element(s) have no solid neighbor printed before them, e.g. at (row, col) {support['unsupported_at'][:5]}"
        )

    xPhys_np = torch_util.to_numpy(xPhys)
    report = dict(
        loop=loop,
        passed=start["passed"] and support["passed"],
        start=start,
        support=support,
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
    """Every constraint's values and the compliance, at iteration `loop`'s settings."""
    config = problem.config
    K_est = stto.estimated_conductivity(problem, xBin, tPhys, loop)
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
    """`check_design`'s report as a few lines for the console."""
    start, support = report["start"], report["support"]
    verdict = lambda passed: "pass" if passed else "FAIL"
    lines = [
        f"start:   {verdict(start['passed'])} ({start['solid_base_elements']} solid base element(s), latest at t = {start['max_base_t']}, tolerance {start['tolerance']})",
        f"support: {verdict(support['passed'])} ({support['unsupported']} unsupported element(s))",
    ]
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
