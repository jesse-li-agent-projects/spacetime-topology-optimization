"""Main-loop orchestration: wires fem/filters/timefield/gravity/compliance/constraints/
conductivity/mma together into the actual space-time topology optimization iteration.

Every module this file calls already owns its own math and its own fixture/FD tests;
this file's only job is *wiring* -- building the stacked objective/constraint arrays
MMA expects, in the exact constraint order the MATLAB main loop uses, and threading the
iteration-dependent state (`beta_d`, `beta_t`, `xold1`/`xold2`, `low`/`upp`, and the
raw x/t fields) from one call to the next. See
`conventions.md` for array-order conventions and `tests/matlab_reference_loop.py` (a
literal transliteration of the MATLAB source this ports) for the authoritative
iteration order.

Two field pairs exist per design variable, and must not be conflated: `x`/`t` are each
iteration's *raw* MMA output (unfiltered, unprojected -- what next iteration's
move-limit bounds read); `xPhys`/`tPhys` are the *filtered* fields the physics uses
(density also Heaviside-projected, time also scaled to a maximum of 1). `State` carries only the raw pair;
`physical_fields` derives the other wherever it is needed.

**The tensor boundary (`plans/torch_port_part2.md`).** `Problem` and `State` hold torch
tensors -- `Problem.device`/`.dtype` say where -- and so does every leaf module `step`
calls: `filters`/`compliance`/`constraints`/`conductivity`/`mma` all take and return
tensors, and `compliance.py`'s FEM solve runs through `torch_solve.FemSolve`'s
multigrid-CG, not a NumPy round trip. `fem.py`'s NumPy `assemble_stiffness`/`solve_fe`
stay only as `tests/reference/`'s independent oracle; nothing in `sttopt/` calls them.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import sttopt.compliance as compliance
import sttopt.constraints as constraints
import sttopt.fem as fem
import sttopt.field_terms as field_terms
import sttopt.filters as filters
import sttopt.gravity as gravity
import sttopt.load_cases as load_cases
import sttopt.mma as mma
import sttopt.run_config as run_config
import sttopt.sensitivity as sensitivity
import sttopt.timefield as timefield
import sttopt.torch_fem as torch_fem
import sttopt.torch_util as torch_util
import sttopt.units as units


@dataclass(frozen=True)
class Problem:
    """Fixed problem setup: everything that doesn't change across iterations, except
    the smooth maxima's calibrations in `terms`.

    Built once by `build_problem` and passed to every `step` call. `step` refreshes that
    calibration in place, so it is not a pure function of `State`.
    """

    # Hyperparameters -- see RunConfig for field docs. Read as `problem.config.nelx`,
    # etc., rather than duplicated as Problem's own fields.
    config: run_config.RunConfig

    device: torch.device
    dtype: torch.dtype

    KE: Float[Tensor, "8 8"]
    edofMat: Int[Tensor, "nelx*nely 8"]
    freedofs: Int[Tensor, " n_free"]
    free_mask: Bool[Tensor, " ndof"]  # True at free dofs; the matrix-free path's mask
    F: Float[Tensor, " ndof"]
    ndof: int
    H: torch_util.SymmetricCsr  # shape (nel, nel)
    Hs: Float[Tensor, " nel"]
    # The density filter on `t`, or None when `config.time_filter_rmin_m` is 0.
    time_H: torch_util.SymmetricCsr | None
    time_Hs: Float[Tensor, " nel"] | None
    L: Tensor  # sparse CSR, shape (nel, nel)
    C: Tensor  # sparse CSR, shape ((nelx+1)*(nely+1), nel)
    # The hotspot measure and the time-field constraints and regularizers
    terms: field_terms.FieldTerms
    Nei: Int[Tensor, " k"]

    n: int  # number of MMA design variables: 2*nelx*nely (density half + time half)


@dataclass(frozen=True)
class State:
    """Iteration-dependent state carried from one `step` call to the next."""

    x: Float[Tensor, "nely nelx"]  # raw density (unfiltered MMA output)
    t: Float[Tensor, "nely nelx"]  # raw time field (unfiltered MMA output)
    mma: mma.History
    loop: int  # iterations already done, i.e. the index of the one `step` runs next
    beta_t: float  # gravity/stage-mask sigmoid sharpness
    beta_d: float  # Heaviside projection sharpness

    # Batched FEM solution from this iteration's whole_compliance + gravity_compliance
    # solves, `(1 + nStage, ndof)`, row 0 the whole-structure solve and row `1 + i`
    # stage `i` -- carried forward to warm-start the next iteration's solves (part 1
    # measured ~25% fewer CG iterations). `None` only at `init_state`, which has no
    # previous solution to carry.
    U: Float[Tensor, "n_stage_plus_1 ndof"] | None


@dataclass(frozen=True)
class IterationRecord:
    """Per-iteration diagnostics and raw MMA outputs, for E2E/MMA/constraint-order tests."""

    obj: float  # whole-structure compliance (doesn't include intermediate structures)
    vol: float  # volume fraction (mean xPhys)
    tru_max: float  # calibrated hotspot severity, comparable across runs
    uniformity: float  # layer-uniformity penalty, before uniformity_weight
    roughness: float  # smoothness regularizer, as a fraction of a layer thickness
    roughness_weight: float  # this iteration's weight, which a schedule may vary
    f: float  # objective (compliance terms plus the weighted time-field terms)
    df: Float[np.ndarray, " n"]
    xmma: Float[np.ndarray, " n"]
    low: Float[np.ndarray, " n"]
    upp: Float[np.ndarray, " n"]
    lam: Float[np.ndarray, " m"]
    g: Float[np.ndarray, " m"]
    dg: Float[np.ndarray, "m n"]
    # Scalars for following the optimization dynamics: this iteration's continuation
    # values, the true hotspot maximum and where it sits, and the MMA step size.
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    state: State  # final state after nloop iterations
    # length nloop+1, index 0 is initial field
    xPhys_traj: list[Float[Tensor, "nely nelx"]]
    tPhys_traj: list[Float[Tensor, "nely nelx"]]  # length nloop+1
    # the raw (unfiltered, unprojected) counterpart of xPhys_traj/tPhys_traj -- what
    # `state.x`/`state.t` held at each loop, i.e. the input `step` would need to
    # reproduce that loop's xPhys/tPhys, not the fields themselves
    x_traj: list[Float[Tensor, "nely nelx"]]
    t_traj: list[Float[Tensor, "nely nelx"]]  # length nloop+1
    records: list[IterationRecord]  # length nloop; `records[i]` is iteration `i`


def build_problem(
    config: run_config.RunConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Problem:
    """Build the fixed FEM/filter/geometry setup once, before the loop starts.

    The config's lengths are in metres; this is where they become the element units the
    filters and neighbourhoods are built in.

    :param device: device every tensor field of the returned `Problem` lives on. Defaults
        to CUDA when available, else CPU.
    :param dtype: floating dtype every real-valued tensor field is cast to; integer
        (index/mask) fields keep their own integer/bool dtype regardless.
    """
    nelx, nely = config.nelx, config.nely
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # A 1x1 mesh has neither extent nor neighbours, so two of the pieces built below
    # degenerate: the CORNER/OPPOSITE_CORNER time fields normalize by a zero max
    # distance, and the continuity filter divides by a zero neighbour count. This is the
    # first point where both exist, so the check belongs here rather than in either one.
    if nelx == 1 and nely == 1:
        raise ValueError(
            f"nelx and nely cannot both be 1, got nelx={nelx}, nely={nely}"
        )
    h = config.element_size_m

    tfield = timefield.TimeField[config.print_base.upper()]
    KE = fem.plane_stress_KE(config.nu)
    edofMat = fem.element_dof_map(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)
    F, freedofs = load_cases.load_case(
        load_cases.LoadCase(config.load_case),
        nelx,
        nely,
        config.load_length_m,
        config.support_length_m,
        h,
    )

    H, Hs = filters.density_filter(nelx, nely, units.in_elements(config.rmin_m, h))
    time_H = time_Hs = None
    if config.time_filter_rmin_m > 0:
        time_H_np, time_Hs_np = filters.density_filter(
            nelx, nely, units.in_elements(config.time_filter_rmin_m, h)
        )
        time_H = torch_util.symmetric_csr_to_tensor(time_H_np, device, dtype)
        time_Hs = torch_util.to_tensor(time_Hs_np, device, dtype)
    L = filters.continuity_filter(nelx, nely, units.in_elements(config.lrmin_m, h))
    C = gravity.gravity_load_matrix(nelx, nely)

    # Print-start element(s), per constraints.start_point's own docstring.
    Nei = timefield.base_elements(nelx, nely, tfield)

    n = 2 * nelx * nely

    # Batch every float-valued and every int-valued raw array into one boundary crossing
    # each, rather than a `to_tensor` call per field (plans/torch_port_review_followup.md
    # Phase 5). Keys match `Problem`'s field names so they splat straight in below.
    float_fields = torch_util.to_tensors({"KE": KE, "F": F, "Hs": Hs}, device, dtype)
    int_fields = torch_util.to_tensors(
        {"edofMat": edofMat, "freedofs": freedofs, "Nei": Nei}, device, torch.int64
    )
    return Problem(
        config=config,
        device=device,
        dtype=dtype,
        free_mask=torch_fem.free_mask(ndof, int_fields["freedofs"], device=device),
        ndof=ndof,
        H=torch_util.symmetric_csr_to_tensor(H, device, dtype),
        time_H=time_H,
        time_Hs=time_Hs,
        L=torch_util.csr_to_tensor(L, device, dtype),
        C=torch_util.csr_to_tensor(C, device, dtype),
        terms=field_terms.FieldTerms.build(config, Nei, device, dtype),
        n=n,
        **float_fields,
        **int_fields,
    )


def physical_fields(
    problem: Problem,
    x: Float[Tensor, "nely nelx"],
    t: Float[Tensor, "nely nelx"],
    beta_d: float,
) -> tuple[Float[Tensor, "nely nelx"], Float[Tensor, "nely nelx"]]:
    """The density and time fields the physics reads, from the raw design variables:
    `x` filtered and Heaviside-projected at sharpness `beta_d`, and `t` filtered and
    scaled so its maximum is 1.

    The one definition of that map, so a trajectory or a saved design cannot disagree
    with what `step` optimized. Differentiable: `step` gets the chain rule back to
    `x`/`t` from autograd.

    Filtering pulls the time field's maximum below 1, but the stage times and the
    hotspot term read absolute print times, so the build must end at 1. The start-point
    constraint already pins the minimum to 0, so a scale is enough. The gradient treats
    that scale as a constant.

    :return: `(xPhys, tPhys)`
    """
    xTilde = filters.apply_density_filter(x, problem.H, problem.Hs)
    xPhys = filters.heaviside_projection(xTilde, beta_d, problem.config.eta)
    tTilde = (
        t
        if problem.time_H is None
        else filters.apply_density_filter(t, problem.time_H, problem.time_Hs)
    )
    tPhys = tTilde / tTilde.max().detach()
    return xPhys, tPhys


def init_state(problem: Problem) -> State:
    """Initial state: `x`/`t` are the raw seed (uniform `problem.config.volfrac`, time
    field per `problem.config.print_base`).

    The MATLAB source leaves `tPhys` unfiltered at initialization, so its first iteration
    reads a different forward map from every later one (PR #26). Here every iteration,
    the first included, derives the physical fields through `physical_fields`.
    """
    config = problem.config
    nely, nelx = config.nely, config.nelx
    nel = nelx * nely
    device, dtype = problem.device, problem.dtype

    x = torch.full((nely, nelx), config.volfrac, device=device, dtype=dtype)

    # init_timefield is a NumPy builder (plans/torch_port_part2.md Phase 3.2 item 2);
    # converted once here, at the tensor boundary.
    t = torch_util.to_tensor(
        timefield.init_timefield(
            nelx, nely, timefield.TimeField[config.print_base.upper()]
        ),
        device,
        dtype,
    )

    # MATLAB's xold1=xold2=[x(:); zeros(nel,1)], reproduced for fidelity
    xold = torch.cat([x.flatten(), torch.zeros(nel, device=device, dtype=dtype)])

    beta_d = run_config.weight_at(config.beta_d_schedule, 0)
    beta_t = run_config.weight_at(config.beta_t_schedule, 0)

    return State(
        x=x,
        t=t,
        mma=mma.History.initial(xold),
        loop=0,
        beta_t=beta_t,
        beta_d=beta_d,
        U=None,
    )


def save_checkpoint(problem: Problem, state: State, path: Path, **extra) -> None:
    """
    Write what resuming a run at `state` needs: the state and the calibrations
    `step` refreshes in place. Written to a temporary file first, so a run stopped
    mid-write leaves the previous checkpoint whole.

    :param problem: the problem whose calibrations are saved
    :param state: the state to resume from
    :param path: checkpoint file to write
    :param extra: further values to keep with it, e.g. the elapsed time
    """
    checkpoint = dict(
        state=dataclasses.asdict(state),
        calibrations=problem.terms.calibrations(),
        extra=extra,
    )
    partial = path.with_name(path.name + ".partial")
    torch.save(checkpoint, partial)
    partial.replace(path)


def load_checkpoint(problem: Problem, path: Path) -> tuple[State, dict]:
    """
    Read a `save_checkpoint` file onto `problem`'s device, and restore its
    calibrations into `problem`.

    :param problem: the problem to restore calibrations into
    :param path: checkpoint file to read
    :return: the state, and the `extra` values saved with it
    """
    checkpoint = torch.load(path, map_location=problem.device, weights_only=True)
    saved = checkpoint["state"]
    state = State(**(saved | {"mma": mma.History(**saved["mma"])}))
    problem.terms.restore_calibrations(checkpoint["calibrations"])
    return state, checkpoint["extra"]


def _flatten_pair(density_part: Tensor, time_part: Tensor) -> Float[Tensor, " n"]:
    """Concatenate a density-half tensor and a time-half tensor into the single `n`-
    length layout MMA's design/gradient vectors use -- the one place the
    `[density; time]` ordering is spelled out, so every caller (design variables,
    bounds, sensitivity rows) shares it rather than repeating the concatenation.

    :param density_part: `(..., nel)` or `(nel,)`, flattened along its last dim.
    :param time_part: same shape as `density_part`.
    :return: `density_part` and `time_part` flattened and concatenated along the last
        dim.
    """
    return torch.cat(
        [density_part.flatten(start_dim=-1), time_part.flatten(start_dim=-1)],
        dim=-1,
    )


def _sensitivity_rows(
    outputs: Float[Tensor, " k"],
    x: Float[Tensor, "nely nelx"],
    t: Float[Tensor, "nely nelx"],
) -> Float[Tensor, "k n"]:
    """Sensitivities of `k` independent scalar outputs (e.g. one per element, or one per
    stage, or a single value passed as `value[None]`) w.r.t. both raw leaves, as
    `(k, n)` in MMA's `[density; time]` layout.

    Every step of the chain is autograd's, the density filter included. That is only
    affordable because `filters.apply_density_filter` multiplies through
    `torch_util.symmetric_matmul`: autograd's backward for a plain `H @ x` transposes to
    CSC, which is catastrophically slow to multiply on CPU, while the symmetric backward
    costs the same as the forward. What remains is that the graph applies the adjoint
    once per row where a hand-applied one would batch all `k` into a single
    sparse-times-dense product -- ~1 ms per row block at 180x60, which is not worth
    keeping a hand-derived step for. See PR #89 for the measurements.

    `allow_unused` covers rows that depend on only one field (e.g. a density-only
    constraint never touches `t`): the unused field's block is exactly zero, which is
    what a hand-derived predecessor returned too.

    :param outputs: `k` independent scalars sharing one autograd graph.
    :param x: raw density leaf.
    :param t: raw time-field leaf.
    :return: `(k, n)` sensitivity rows.
    """
    return _flatten_pair(*sensitivity.jacobian_rows(outputs, (x, t)))


def _stage_times(nStage: int) -> list[float]:
    """
    The gravity stages' boundaries, as fractions of the build.

    :param nStage: number of stages
    :return: the `nStage` upper boundaries, ending at 1
    """
    return [float(ti) for ti in np.linspace(0, 1, nStage + 1)[1:]]


def constraint_values(
    problem: Problem,
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    K_est: Float[Tensor, " nel"],
    loop: int,
    beta_t: float,
) -> dict[str, Float[Tensor, " k"]]:
    """
    Every constraint's values at iteration `loop`'s settings, keyed by the
    `constraints` function (or `"hotspot"`) that gives them, in the reference loop's
    row order (`tests/matlab_reference_loop.py` is the authority for it).

    The one statement of the constraint stack, so a check on a finished design reads
    the constraints the run optimized. Each smooth maximum refreshes its calibration,
    so its value is the true maximum and its gradient the smooth surrogate's.

    :param problem: the problem being optimized
    :param xPhys: physical densities
    :param tPhys: physical time field
    :param K_est: `FieldTerms.estimated_conductivity` of the same fields at `loop`
    :param loop: iteration whose schedules apply
    :param beta_t: time-field projection sharpness
    :return: each constraint's rows, in stack order
    """
    config = problem.config
    g = {
        "global_volume_fraction": constraints.global_volume_fraction(
            xPhys, config.volfrac
        )
    }
    if config.enable_continuity:
        g["time_field_continuity"] = constraints.time_field_continuity(
            tPhys, problem.L, config.continuity_tol
        )
    g["start_point"] = constraints.start_point(tPhys, problem.Nei)
    if config.enable_stage_volume:
        g["stage_volume_bounds"] = constraints.stage_volume_bounds(
            xPhys, tPhys, _stage_times(config.nStage), config.volfrac, beta_t
        )
    return g | problem.terms.constraints(xPhys, tPhys, K_est, loop)


def step(problem: Problem, state: State) -> tuple[State, IterationRecord]:
    """Run one optimization iteration: build the objective + every constraint's value
    in the reference's exact order, differentiate the whole graph by autograd
    (`plans/torch_port_part2.md` Phase 3.4 -- `x`/`t` are the autograd leaves, per
    Decision 4; the filter and Heaviside projection are ordinary forward operations,
    not accompanied by a hand-derived `dx` chain-rule factor), call `mma.mmasub`, and
    unpack the result into the next state.

    Iterations are 0-indexed: `state.loop` counts the iterations already done, so it is
    also the index of the one this call runs.

    The projection sharpnesses `beta_t`/`beta_d` are read at the *next* index and stored
    on the returned state, so a scheduled change takes effect starting the next `step`
    call rather than rescaling this iteration's own `g_all`/`dg_dx`/`xPhys` mid-loop --
    a deliberate simplification, not a fidelity gap. The hotspot calibration refresh
    instead applies to this iteration's own hotspot constraint, as in the MATLAB source.
    """
    config = problem.config
    nely, nelx, nStage = config.nely, config.nelx, config.nStage
    nel = nelx * nely

    loop = state.loop
    beta_t = state.beta_t
    beta_d = state.beta_d
    penal = run_config.weight_at(config.penal, loop)
    Tcr = run_config.weight_at(config.Tcr, loop)
    rouf = run_config.weight_at(config.rouf, loop)

    # -- Gradient region begins: x/t become autograd leaves. --
    x = state.x.clone().requires_grad_(True)
    t = state.t.clone().requires_grad_(True)
    xPhys, tPhys = physical_fields(problem, x, t, beta_d)

    # -- Objective: whole-structure compliance + Theta-weighted per-stage gravity
    # compliance + weighted layer-uniformity penalty and roughness regularizer --
    # whole_compliance's solve and every gravity stage's go into one batched FemSolve
    # call; `torch_fem.pcg` retires each row as it converges, so the batch costs no more
    # row-iterations than solving the rows one at a time would.
    stage_times = _stage_times(nStage)
    c_t, stage_cs, U_new = compliance.batched_whole_and_gravity_compliance(
        xPhys,
        tPhys,
        problem.KE,
        problem.edofMat,
        config.Emin,
        config.Emax,
        penal,
        problem.freedofs,
        problem.F,
        problem.ndof,
        problem.C,
        beta_t,
        stage_times,
        x0=state.U,
    )

    # compliance of final structure only, saved for logging
    obj_final_only = float(c_t.detach())

    f_val_t = c_t
    for cg_t in stage_cs:
        f_val_t = f_val_t + config.Theta * cg_t
    stage_obj = config.Theta * sum(float(cg_t.detach()) for cg_t in stage_cs)

    objective = problem.terms.objective(f_val_t, xPhys, tPhys, loop)
    f_val_t = objective.value

    f_val = float(f_val_t.detach())
    df_dx = _sensitivity_rows(f_val_t[None], x, t)[0]

    # -- Bounds and trust region for this iteration's raw MMA variables. Density and
    # time carry separate move limits, so the trust region is per-variable. --
    xflat = state.x.flatten()
    tflat = state.t.flatten()
    xval = _flatten_pair(xflat, tflat)
    move_limit = _flatten_pair(
        torch.full_like(xflat, run_config.weight_at(config.move, loop)),
        torch.full_like(tflat, run_config.weight_at(config.tmove, loop)),
    )

    vol_diag = float(xPhys.detach().sum() / (nelx * nely))

    K_est_t = problem.terms.estimated_conductivity(xPhys, tPhys, loop)
    g_parts = constraint_values(problem, xPhys, tPhys, K_est_t, loop, beta_t)
    g_hotspot_t = g_parts["hotspot"][0]
    tru_max = (float(g_hotspot_t.detach()) + 1) * Tcr
    tool_radius_m = run_config.weight_at(config.tool_radius_m, loop)

    g_all = torch.cat(list(g_parts.values()))
    # Per part, not on `g_all`: a row of the stack would also backpropagate zeros
    # through every other part's graph, the hotspot's included (~35% slower per step, PR #173).
    dg_dx = torch.cat(
        [_sensitivity_rows(g, x, t) for g in g_parts.values()],
        dim=0,
    )
    diagnostics = problem.terms.hotspot_diagnostics(g_hotspot_t, K_est_t, xPhys)
    with torch.no_grad():
        unit_m = timefield.unit_length_m(tPhys, config.element_size_m)
        grad_p10, grad_p50, grad_p90 = (
            g / unit_m
            for g in timefield.gradient_percentiles(tPhys.detach(), xPhys.detach())
        )
    diagnostics.update(
        stage_obj=stage_obj,
        grad_p10_per_m=grad_p10,
        grad_p50_per_m=grad_p50,
        grad_p90_per_m=grad_p90,
        hotspot_kappa=run_config.weight_at(config.hotspot_kappa, loop),
        grey=float(((xPhys > 0.05) & (xPhys < 0.95)).double().mean()),
        calibration=problem.terms.hotspot.calibration,
        penal=penal,
        uniformity_weight=objective.uniformity_weight,
        Tcr=Tcr,
        rouf=rouf,
        hotspot_beta=run_config.weight_at(config.hotspot_beta, loop),
        beta_d=beta_d,
        beta_t=beta_t,
        tool_radius_m=tool_radius_m,
        admissible_tool_radius_m=problem.terms.admissible_tool_radius_m(
            xPhys.detach(), tPhys.detach()
        ),
        min_gradient_fraction=run_config.weight_at(config.min_gradient_fraction, loop),
        gradient_smoothness_m=run_config.weight_at(config.gradient_smoothness_m, loop),
    )

    # -- Periodic state updates, deferred to take effect starting *next* iteration's
    # step() call, never rescaling this iteration's own g_all/dg_dx mid-loop. --
    beta_t = run_config.weight_at(config.beta_t_schedule, loop + 1)
    beta_d = run_config.weight_at(config.beta_d_schedule, loop + 1)

    # -- Gradient region ends: mmasub is not part of the autograd graph. df_dx/g_all/
    # dg_dx are the last gradient-carrying values, detached here on their way in.
    # xval never required grad -- it's built from state.x/state.t, the detached
    # fields, not the x/t leaves above.
    xmma, lam, mma_history = mma.unit_box_step(
        state.mma,
        loop,
        xval,
        move_limit,
        f_val,
        df_dx.detach(),
        g_all.detach(),
        dg_dx.detach(),
        a0=config.a0,
        c=config.mma_c,
        raa0=config.raa0_total / problem.n,
    )

    new_state = State(
        x=xmma[:nel].reshape(nely, nelx),
        t=xmma[nel:].reshape(nely, nelx),
        mma=mma_history,
        loop=loop + 1,
        beta_t=beta_t,
        beta_d=beta_d,
        # Required: femsolve() asserts its x0 argument arrives already detached (the
        # next iteration passes state.U as x0). Undetached, this leaked ~180 MB/step
        # of multigrid hierarchy across iterations -- see commit 855eb76.
        U=U_new.detach(),
    )
    step_size = (xmma - xval).abs()
    diagnostics.update(
        dx_max=float(step_size[:nel].max()),
        dx_mean=float(step_size[:nel].mean()),
        dt_max=float(step_size[nel:].max()),
        dt_mean=float(step_size[nel:].mean()),
    )
    record = IterationRecord(
        obj=obj_final_only,
        vol=vol_diag,
        tru_max=tru_max,
        uniformity=float(objective.uniformity.detach()),
        roughness=float(objective.roughness.detach()),
        roughness_weight=objective.roughness_weight,
        f=float(f_val),
        df=torch_util.to_numpy(df_dx),
        xmma=torch_util.to_numpy(xmma),
        low=torch_util.to_numpy(mma_history.low),
        upp=torch_util.to_numpy(mma_history.upp),
        lam=torch_util.to_numpy(lam),
        g=torch_util.to_numpy(g_all),
        dg=torch_util.to_numpy(dg_dx),
        diagnostics=diagnostics,
    )
    return new_state, record


def run(config: run_config.RunConfig, **problem_kwargs) -> RunResult:
    """Build the fixed setup and run `config.nloop` optimization iterations from the
    standard initialization (uniform density at `config.volfrac`,
    `timefield.init_timefield` for the print-time field). `problem_kwargs` forwards to
    `build_problem` (`device`/`dtype`) for callers that need non-default values.
    """
    problem = build_problem(config, **problem_kwargs)
    return run_from_state(problem, init_state(problem), config.nloop)


def run_from_state(problem: Problem, state: State, nloop: int) -> RunResult:
    """Run `nloop` iterations from an arbitrary starting state, collecting the
    trajectory. Split out of `run` so that where the iteration starts is a caller's
    choice -- warm-starting from a previous run's final state, or entering at a
    trajectory recorded elsewhere.
    """
    xPhys_traj, tPhys_traj = [], []

    def record_fields(state: State) -> None:
        xPhys, tPhys = physical_fields(problem, state.x, state.t, state.beta_d)
        xPhys_traj.append(xPhys)
        tPhys_traj.append(tPhys)

    record_fields(state)
    x_traj = [state.x.clone()]
    t_traj = [state.t.clone()]
    records: list[IterationRecord] = []

    for _ in range(nloop):
        state, record = step(problem, state)
        record_fields(state)
        x_traj.append(state.x.clone())
        t_traj.append(state.t.clone())
        records.append(record)

    return RunResult(
        state=state,
        xPhys_traj=xPhys_traj,
        tPhys_traj=tPhys_traj,
        x_traj=x_traj,
        t_traj=t_traj,
        records=records,
    )
