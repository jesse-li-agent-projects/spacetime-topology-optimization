"""Fixed-geometry fabrication-sequence optimization (`seqopt`): optimize only the
print-time field `t` for a fixed, prescribed geometry, to minimize overheating subject
to layer-uniformity and print-continuity constraints.

Mirrors `stto.py`'s shape (`Problem`/`State`/`IterationRecord`/`build_problem`/
`init_state`/`step`/`run`/`run_from_state`) so that a reader who knows one knows the
other, but the two `step`s are not shared: there is no FEM here (no `K U = F` solve,
no `Emin`/`Emax`/`nu`/`penal`/`eta`/`beta_d`), and `xPhys` is a fixed input rather than
a design variable, so `n = nel` (not `2*nel`) and `t` is the sole autograd leaf.

**The filter on `t` is off by default: `tPhys = t` unless `config.time_filter_rmin`
is set.** STTO reuses the density filter to smooth `tPhys`, which also smooths *across
void* -- two branches separated by a gap get their print times mixed with no material
between them to carry heat, and void `t`, which is pinned by nothing physical, bleeds
into the `tPhys` of solid within `rmin` of the boundary. `seqopt` therefore starts from
no filtering at all (see `plans/archive/fixed_geometry_sequence_optimization.md`), and
`t` is then both the autograd leaf and the physical field.

The radius exists because the sawtooth the uniformity penalty rewards
(`timefield._gradient_cv`) is a one-element mode, and the filter attenuates the
one-element modes by construction -- ~200x at `rmin=4`, so reproducing a given
corrugation in `tPhys` would take a sawtooth in `t` large enough to dominate the design
variable. Whether that eliminates the mode or merely hides it behind the filter is what
the `sawtooth`/`sawtooth_raw` pair in `IterationRecord` is there to answer: a smooth
`tPhys` over a jagged `t` is a failure, not a fix.

Void elements keep their time variables as free design variables, exactly as in STTO
-- `seqopt` is a proving ground for the full space-time approach, where density is
itself a design variable and there is no fixed void set to special-case. See the
plan's "time field over void stays a free design variable" section for a measured
artefact this implies and deliberately does not paper over.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

import sttopt.conductivity as conductivity
import sttopt.constraints as constraints
import sttopt.filters as filters
import sttopt.geometry as geometry
import sttopt.mma as mma
import sttopt.run_config as run_config
import sttopt.sensitivity as sensitivity
import sttopt.timefield as timefield
import sttopt.torch_util as torch_util


@dataclass(frozen=True)
class Problem:
    """Fixed problem setup: everything that doesn't change across iterations.

    Built once by `build_problem` and passed unchanged to every `step` call.
    """

    config: run_config.SeqRunConfig
    device: torch.device
    dtype: torch.dtype

    xPhys: Float[Tensor, "nely nelx"]  # the fixed geometry -- never requires grad
    nelx: int
    nely: int

    L: Tensor  # sparse CSR continuity filter, shape (nel, nel)
    # The density filter on `t`, or None when `config.time_filter_rmin` is 0 and
    # `tPhys` is `t` itself -- see the module docstring.
    H: torch_util.SymmetricCsr | None
    Hs: Float[Tensor, " nel"] | None
    e1: Int[Tensor, " npairs"]
    e2: Int[Tensor, " npairs"]
    w: Float[Tensor, " npairs"]
    # Constant `K_est` divisor for `config.hotspot_normalization`, or None where the
    # divisor is per-element -- `conductivity.constant_denominator`.
    hotspot_denom: float | None
    # Print base elements the hotspot measure treats as infinitely dense, or None where
    # the normalization needs no print base -- `conductivity.infinite_base`.
    hotspot_base: Int[Tensor, " k"] | None
    Nei: Int[Tensor, " k"]  # print-start element(s), already filtered to solid ones

    m: int  # number of MMA constraint rows: continuity + start-point(s) + 2*nStage
    n: int  # number of MMA design variables: nel (t alone -- xPhys is fixed)


@dataclass(frozen=True)
class State:
    """Iteration-dependent state carried from one `step` call to the next."""

    t: Float[Tensor, "nely nelx"]  # the raw design variable; `physical_timefield`
    # turns it into the field `step`'s physics reads
    xold1: Float[Tensor, " n"]
    xold2: Float[Tensor, " n"]
    low: Float[Tensor, " n"]
    upp: Float[Tensor, " n"]
    loop: int
    beta_t: float  # stage-mask sigmoid sharpness
    # Carries the hotspot aggregate onto the true maximum severity; a ratio or an
    # offset, whichever `config.hotspot_aggregation`'s bias calls for. Periodically
    # refreshed -- `conductivity.Aggregation.calibration`.
    hotspot_calibration: float


@dataclass(frozen=True)
class IterationRecord:
    """Per-iteration diagnostics and raw MMA outputs."""

    f: float  # objective: the weighted sum of the three raw terms below
    hotspot: float  # raw hotspot p-mean (`numer`), before any rescaling
    uniformity: float  # raw layer-uniformity penalty
    roughness: float  # smoothness regularizer, as a fraction of a layer thickness
    tru_max: float  # calibrated hotspot severity, comparable across runs

    # Diagnostics only -- nothing below enters the objective or a constraint. They exist
    # because `uniformity` cannot be taken at face value: it is blind to the transverse
    # sawtooth (`timefield._gradient_cv`), so a run can drive it down while the field
    # gets worse. `true_cv` is the same measurement through an operator that annihilates
    # that mode, and `sawtooth` is the mode's own amplitude; the gap between `uniformity`
    # and `true_cv` is the evidence of gaming.
    true_cv: float  # `timefield.central_difference_cv` of tPhys, sawtooth-blind
    sawtooth: float  # tPhys corrugation depth as a fraction of a layer thickness
    # The same depth in the raw design variable. Equal to `sawtooth` when no filter is
    # configured; when one is, the two together separate "the mode is gone" from "the
    # mode is still there and the filter is hiding it".
    sawtooth_raw: float

    roughness_weight: float  # this iteration's weight, which a schedule may vary
    df: Float[np.ndarray, " n"]
    xmma: Float[np.ndarray, " n"]
    low: Float[np.ndarray, " n"]
    upp: Float[np.ndarray, " n"]
    lam: Float[np.ndarray, " m"]
    g: Float[np.ndarray, " m"]
    dg: Float[np.ndarray, "m n"]


@dataclass(frozen=True)
class RunResult:
    state: State  # final state after nloop iterations
    t_traj: list[
        Float[Tensor, "nely nelx"]
    ]  # length nloop+1, index 0 the initial field
    records: list[IterationRecord]  # length nloop


def build_problem(
    config: run_config.SeqRunConfig,
    xPhys: Float[np.ndarray, "nely nelx"],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Problem:
    """Build the fixed geometry/filter setup once, before the loop starts.

    Loading a geometry file is the CLI's job (`geometry.load_geometry`), not this
    function's -- `xPhys` arrives as a plain array so callers (e.g. tests) can pass
    one directly.

    :param config: the run's hyperparameters
    :param xPhys: fixed density field, shape `(nely, nelx)`; solid that does not
        connect to the build plate is dropped (`geometry.drop_disconnected`), so
        `Problem.xPhys` is what the run actually optimizes and is not always this
    :param device: device every tensor field lives on; defaults to CUDA if available
    :param dtype: floating dtype every real-valued tensor field is cast to
    """
    nely, nelx = xPhys.shape
    nStage = config.nStage
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    tfield = timefield.TimeField[config.print_base.upper()]
    candidates = timefield.base_elements(nelx, nely, tfield)
    Nei = geometry.base_elements(xPhys, candidates)
    xPhys = geometry.drop_disconnected(xPhys, Nei)

    L = filters.continuity_filter(nelx, nely, config.lrmin)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, config.rmin_cond)
    hotspot_denom = conductivity.constant_denominator(
        conductivity.Normalization(config.hotspot_normalization), config.rmin_cond
    )
    hotspot_base = conductivity.infinite_base(
        conductivity.Normalization(config.hotspot_normalization), Nei
    )

    H = Hs = None
    if config.time_filter_rmin > 0:
        H_np, Hs_np = filters.density_filter(nelx, nely, config.time_filter_rmin)
        H = torch_util.symmetric_csr_to_tensor(H_np, device, dtype)
        Hs = torch_util.to_tensor(Hs_np, device, dtype)

    n = nelx * nely
    m = (1 if config.enable_continuity else 0) + len(Nei) + 2 * nStage

    xPhys_t = torch_util.to_tensor(xPhys, device, dtype)
    int_fields = torch_util.to_tensors(
        {"e1": e1, "e2": e2, "Nei": Nei}, device, torch.int64
    )
    return Problem(
        config=config,
        device=device,
        dtype=dtype,
        xPhys=xPhys_t,
        nelx=nelx,
        nely=nely,
        L=torch_util.csr_to_tensor(L, device, dtype),
        H=H,
        Hs=Hs,
        w=torch_util.to_tensor(w, device, dtype),
        hotspot_denom=hotspot_denom,
        hotspot_base=None if hotspot_base is None else int_fields["Nei"],
        m=m,
        n=n,
        **int_fields,
    )


def physical_timefield(
    problem: Problem, t: Float[Tensor, "nely nelx"]
) -> Float[Tensor, "nely nelx"]:
    """The physical time field the objective and the constraints read: `t` filtered,
    or `t` itself when no radius is configured. Differentiable, so the chain rule back
    to the design variable is autograd's.

    Recomputed from `t` wherever it is needed rather than cached on `State`, so the two
    cannot drift apart.
    """
    if problem.H is None:
        return t
    return filters.apply_density_filter(t, problem.H, problem.Hs)


def init_state(problem: Problem) -> State:
    """Initial state: `t` is the normalized geodesic distance from the print-start
    element(s) through the material, with the void filled per `config.void_extension`
    (`timefield.init_geometry_timefield`). Nothing else to set up -- both objective
    terms are dimensionless and order 1
    (`plans/archive/fixed_geometry_sequence_optimization.md`), so no scale is latched
    from the initial field.
    """
    xPhys_np = torch_util.to_numpy(problem.xPhys)
    Nei_np = torch_util.to_numpy(problem.Nei)
    t0 = timefield.init_geometry_timefield(
        xPhys_np, Nei_np, timefield.VoidExtension(problem.config.void_extension)
    )
    t = torch_util.to_tensor(t0, problem.device, problem.dtype)

    xold = t.flatten().clone()
    return State(
        t=t,
        xold1=xold,
        xold2=xold.clone(),
        low=torch.zeros(problem.n, device=problem.device, dtype=problem.dtype),
        upp=torch.zeros(problem.n, device=problem.device, dtype=problem.dtype),
        loop=0,
        beta_t=10.0,
        hotspot_calibration=conductivity.Aggregation(
            problem.config.hotspot_aggregation
        ).uncalibrated,
    )


def hotspot_value(
    problem: Problem, tPhys: Float[Tensor, "nely nelx"]
) -> tuple[Float[Tensor, ""], Float[Tensor, " nel"]]:
    """The run's hotspot term and the `K_est` field behind it, with every conductivity
    setting taken from `problem`.

    The one path to that field, so offline tooling (`viz.py`) cannot report a hotspot
    measure the run never optimized -- it once plotted the print base as the worst
    hotspot in the domain by rebuilding this call by hand (PR #98).
    """
    config = problem.config
    return conductivity.hotspot_value(
        problem.xPhys,
        tPhys,
        problem.e1,
        problem.e2,
        problem.w,
        config.p,
        config.q,
        config.r,
        config.rouf,
        problem.hotspot_denom,
        problem.hotspot_base,
        conductivity.Aggregation(config.hotspot_aggregation),
        config.hotspot_beta,
        config.hotspot_density_exponent,
    )


def step(problem: Problem, state: State) -> tuple[State, IterationRecord]:
    """Run one optimization iteration: build the objective and every constraint's
    value, differentiate by autograd (`t` is the sole leaf), call `mma.mmasub`, and
    unpack the result into the next state.

    Constraints are stacked in a fixed order: `constraints.time_field_continuity`
    (when `config.enable_continuity`), then `constraints.start_point`, then (when
    `config.nStage > 0`) the interleaved upper/lower stage-volume rows.
    `constraints.stage_volume_bounds` is called with
    `volfrac = xPhys.mean()`, which makes its scale factor `nelx*nely*volfrac ==
    xPhys.sum()` -- the row becomes "fraction *of the part* deposited by `t_stage`,
    versus `t_stage`", the right statement once density is fixed rather than a design
    variable.

    Both periodic updates (`beta_t += 5` every 30 iterations capped at 50, the hotspot
    hotspot calibration refresh every 25 iterations) are deferred to take effect the *next*
    iteration, matching `stto.step`'s convention. There is no Heaviside sharpening
    here: there is no density projection to sharpen.
    """
    config = problem.config
    nely, nelx, nStage = problem.nely, problem.nelx, config.nStage
    device, dtype = problem.device, problem.dtype

    loop = state.loop + 1
    beta_t = state.beta_t
    xPhys = problem.xPhys

    # -- Gradient region begins: t is the sole autograd leaf. --
    t = state.t.clone().requires_grad_(True)
    tPhys = physical_timefield(problem, t)

    numer_t, K_est_t = hotspot_value(problem, tPhys)
    # The roughness regularizer is not optional garnish: uniformity alone rewards a
    # sawtooth across the print direction (`timefield._gradient_cv`).
    metric = timefield.UniformityMetric(config.uniformity_metric)
    penalty_t = timefield.uniformity_penalty(tPhys, metric, weights=xPhys)
    rough_t = timefield.relative_roughness(tPhys, weights=xPhys)

    roughness_weight = run_config.weight_at(config.roughness_weight, loop)
    f_val_t = (
        config.hotspot_weight * numer_t
        + config.uniformity_weight * penalty_t
        + roughness_weight * rough_t
    )
    f_val = float(f_val_t.detach())
    (df_dt,) = sensitivity.jacobian_rows(f_val_t[None], (t,))

    # -- Bounds and trust region for this iteration's raw MMA variable --
    tflat = state.t.flatten()
    xmin = torch.zeros_like(tflat)
    xmax = torch.ones_like(tflat)
    mma_trust = mma.trust_region_params(
        torch.full_like(tflat, config.tmove),
        xmax - xmin,
        asyclamp_min_ratio=config.asyclamp_min_ratio,
        asyclamp_max_ratio=config.asyclamp_max_ratio,
    )
    xval = tflat

    # -- Constraints, in the fixed documented order above. --
    g_start_t = constraints.start_point(tPhys, problem.Nei)
    g_parts = [g_start_t]
    if config.enable_continuity:
        g_cont_t = constraints.time_field_continuity(
            tPhys, problem.L, config.continuity_tol
        )
        g_parts.insert(0, g_cont_t[None])

    if nStage > 0:
        stage_times = [float(ti) for ti in np.linspace(0, 1, nStage + 1)[1:]]
        volfrac = float(xPhys.mean())  # xPhys.sum() == nelx*nely*volfrac, see docstring
        stage_upper_t = torch.stack(
            [
                constraints.stage_volume_bounds(xPhys, tPhys, t_stage, volfrac, beta_t)
                for t_stage in stage_times
            ]
        )
        g_parts.append(
            torch.stack([stage_upper_t, -stage_upper_t - 1.0e-5], dim=1).flatten()
        )

    g_all = torch.cat(g_parts)
    dg_dt = torch.cat([sensitivity.jacobian_rows(g, (t,))[0] for g in g_parts], dim=0)

    # -- Periodic state updates, deferred to take effect starting *next* iteration. --
    numer = float(numer_t.detach())
    aggregation = conductivity.Aggregation(config.hotspot_aggregation)
    calibration = state.hotspot_calibration
    if loop % 25 == 0:
        K_est = K_est_t.detach()
        severity = (1 - K_est) * xPhys.flatten() ** config.r
        # An infinitely shielded element has no severity to take a maximum over.
        max_g = float(torch.max(severity[torch.isfinite(K_est)]))
        refreshed = aggregation.calibration(numer, max_g)
        if refreshed is not None:
            calibration = refreshed
    tru_max = aggregation.calibrated(numer, calibration)

    if loop % 30 == 0 and beta_t < 50:
        beta_t += 5

    with torch.no_grad():
        physical, raw = tPhys.detach(), t.detach()
        true_cv = float(timefield.central_difference_cv(physical, xPhys))
        sawtooth = float(timefield.relative_sawtooth_amplitude(physical, xPhys))
        sawtooth_raw = float(timefield.relative_sawtooth_amplitude(raw, xPhys))

    # -- Gradient region ends: mmasub is not part of the autograd graph. --
    mma_a = torch.zeros(problem.m, device=device, dtype=dtype)
    mma_c = torch.full((problem.m,), config.mma_c, device=device, dtype=dtype)
    mma_d = torch.zeros(problem.m, device=device, dtype=dtype)
    xmma, ymma, zmma, lam, xsi, mma_eta, mu, zet, s, low, upp = mma.mmasub(
        problem.m,
        problem.n,
        loop,
        xval,
        xmin,
        xmax,
        state.xold1,
        state.xold2,
        f_val,
        df_dt.detach()[0],
        g_all.detach(),
        dg_dt.detach(),
        state.low,
        state.upp,
        config.a0,
        mma_a,
        mma_c,
        mma_d,
        asyincr=config.asyincr,
        asydecr=config.asydecr,
        **mma_trust,
    )

    t_new = xmma.reshape(nely, nelx)

    new_state = State(
        t=t_new,
        xold1=xval,
        xold2=state.xold1,
        low=low,
        upp=upp,
        loop=loop,
        beta_t=beta_t,
        hotspot_calibration=calibration,
    )
    record = IterationRecord(
        f=f_val,
        hotspot=numer,
        uniformity=float(penalty_t.detach()),
        roughness=float(rough_t.detach()),
        tru_max=tru_max,
        true_cv=true_cv,
        sawtooth=sawtooth,
        sawtooth_raw=sawtooth_raw,
        roughness_weight=roughness_weight,
        df=torch_util.to_numpy(df_dt[0]),
        xmma=torch_util.to_numpy(xmma),
        low=torch_util.to_numpy(low),
        upp=torch_util.to_numpy(upp),
        lam=torch_util.to_numpy(lam),
        g=torch_util.to_numpy(g_all),
        dg=torch_util.to_numpy(dg_dt),
    )
    return new_state, record


def run(
    config: run_config.SeqRunConfig,
    xPhys: Float[np.ndarray, "nely nelx"],
    **problem_kwargs,
) -> RunResult:
    """Build the fixed setup and run `config.nloop` optimization iterations from the
    geodesic initialization. `problem_kwargs` forwards to `build_problem`
    (`device`/`dtype`) for callers that need non-default values.
    """
    problem = build_problem(config, xPhys, **problem_kwargs)
    return run_from_state(problem, init_state(problem), config.nloop)


def run_from_state(problem: Problem, state: State, nloop: int) -> RunResult:
    """Run `nloop` iterations from an arbitrary starting state, collecting the
    trajectory."""
    t_traj = [state.t.clone()]
    records: list[IterationRecord] = []

    for _ in range(nloop):
        state, record = step(problem, state)
        t_traj.append(state.t.clone())
        records.append(record)

    return RunResult(state=state, t_traj=t_traj, records=records)
