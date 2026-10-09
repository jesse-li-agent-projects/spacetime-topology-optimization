"""Space-time topology optimization with the time field from a virtual heat equation
(Variant 1 of `plans/archive/virtual_heat_timefield.md`, after Wu2025 §2.3).

The design variables are the density `x` and a diffusivity field `mu`, both on
`[0, 1]`. `tPhys` is not a design variable: `virtual_heat.heat_time_field` solves for it
through the material, so it has no local minimum over the part by construction. The
start-point, continuity and time-filter machinery of `stto` has no counterpart here;
everything else (compliance, gravity stages, the hotspot and time-field constraints and
regularizers, MMA) is the same, through `field_terms` and `mma`, except that the
objective takes the log of the compliance terms.

`State` carries the last heat and Poisson solutions as the next iteration's warm start,
detached like `stto`'s `U`.
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import sttopt.checks as checks
import sttopt.compliance as compliance
import sttopt.constraints as constraints
import sttopt.fem as fem
import sttopt.field_terms as field_terms
import sttopt.filters as filters
import sttopt.geometry as geometry
import sttopt.gravity as gravity
import sttopt.load_cases as load_cases
import sttopt.mma as mma
import sttopt.run_config as run_config
import sttopt.sensitivity as sensitivity
import sttopt.timefield as timefield
import sttopt.torch_fem as torch_fem
import sttopt.torch_util as torch_util
import sttopt.units as units
import sttopt.virtual_heat as virtual_heat


@dataclass(frozen=True)
class Problem:
    """Fixed problem setup, except the smooth maxima's calibrations in `terms`."""

    config: run_config.HeatRunConfig
    device: torch.device
    dtype: torch.dtype

    KE: Float[Tensor, "8 8"]
    edofMat: Int[Tensor, "nelx*nely 8"]
    freedofs: Int[Tensor, " n_free"]
    free_mask: Bool[Tensor, " ndof"]
    F: Float[Tensor, " ndof"]
    ndof: int
    H: torch_util.SymmetricCsr  # density filter, shape (nel, nel)
    Hs: Float[Tensor, " nel"]
    C: Tensor  # gravity load matrix, sparse CSR, shape ((nelx+1)*(nely+1), nel)
    # The hotspot measure and the time-field constraints and regularizers
    terms: field_terms.FieldTerms
    # The scalar mesh the heat equation is solved on, with the build plate
    mesh: virtual_heat.ScalarMesh
    Nei: Int[Tensor, " k"]  # the elements on the build plate
    n: int  # MMA design variables: 2*nelx*nely (density half + diffusivity half)


@dataclass(frozen=True)
class State:
    """Iteration-dependent state carried from one `step` call to the next."""

    x: Float[Tensor, "nely nelx"]  # raw density (unfiltered MMA output)
    mu: Float[Tensor, "nely nelx"]  # diffusivity design variable (MMA output)
    mma: mma.History
    loop: int  # iterations already done, i.e. the index of the one `step` runs next
    beta_t: float  # gravity/stage-mask sigmoid sharpness
    beta_d: float  # Heaviside projection sharpness
    # The last iteration's FEM solutions, `(1 + nStage, ndof)`, the next warm start;
    # `None` at `init_state`
    U: Float[Tensor, "n_stage_plus_1 ndof"] | None
    # The last iteration's heat (and Poisson) solutions, the next warm start
    heat: virtual_heat.TimeFieldSolution | None


@dataclass(frozen=True)
class IterationRecord:
    """Per-iteration diagnostics and raw MMA outputs, as in `stto.IterationRecord`."""

    obj: float  # whole-structure compliance
    vol: float  # volume fraction (mean xPhys)
    tru_max: float  # calibrated hotspot severity
    uniformity: float  # layer-uniformity penalty, before uniformity_weight
    roughness: float  # smoothness regularizer, as a fraction of a layer thickness
    roughness_weight: float
    f: float  # objective
    df: Float[np.ndarray, " n"]
    xmma: Float[np.ndarray, " n"]
    low: Float[np.ndarray, " n"]
    upp: Float[np.ndarray, " n"]
    lam: Float[np.ndarray, " m"]
    g: Float[np.ndarray, " m"]
    dg: Float[np.ndarray, "m n"]
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """A finished run: the final state, the trajectory, and every iteration's record."""

    state: State  # final state after nloop iterations
    xPhys_traj: list[Float[Tensor, "nely nelx"]]  # length nloop+1, index 0 initial
    tPhys_traj: list[Float[Tensor, "nely nelx"]]
    x_traj: list[Float[Tensor, "nely nelx"]]  # the raw design variables
    mu_traj: list[Float[Tensor, "nely nelx"]]
    records: list[IterationRecord]  # length nloop; `records[i]` is iteration `i`


def build_problem(
    config: run_config.HeatRunConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Problem:
    """Build the fixed setup once, before the loop starts.

    :param config: the run's settings
    :param device: device of every tensor field; CUDA when available, else CPU
    :param dtype: floating dtype of every real-valued tensor field
    :return: the problem
    """
    nelx, nely = config.nelx, config.nely
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    h = config.element_size_m

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
    C = gravity.gravity_load_matrix(nelx, nely)
    Nei = timefield.base_elements(
        nelx, nely, timefield.TimeField[config.print_base.upper()]
    )

    floats = torch_util.to_tensors({"KE": KE, "F": F, "Hs": Hs}, device, dtype)
    ints = torch_util.to_tensors(
        {"edofMat": edofMat, "freedofs": freedofs, "Nei": Nei}, device, torch.int64
    )
    return Problem(
        config=config,
        device=device,
        dtype=dtype,
        free_mask=torch_fem.free_mask(ndof, ints["freedofs"], device=device),
        ndof=ndof,
        H=torch_util.symmetric_csr_to_tensor(H, device, dtype),
        C=torch_util.csr_to_tensor(C, device, dtype),
        terms=field_terms.FieldTerms.build(config, Nei, device, dtype),
        mesh=virtual_heat.ScalarMesh.build(
            nelx, nely, config.print_base, device, dtype
        ),
        n=2 * nelx * nely,
        **floats,
        **ints,
    )


def density(
    problem: Problem, x: Float[Tensor, "nely nelx"], beta_d: float
) -> Float[Tensor, "nely nelx"]:
    """
    The density the physics reads: `x` filtered and Heaviside-projected.

    :param problem: the problem
    :param x: raw density design variable
    :param beta_d: projection sharpness
    :return: the projected density
    """
    xTilde = filters.apply_density_filter(x, problem.H, problem.Hs)
    return filters.heaviside_projection(xTilde, beta_d, problem.config.eta)


def physical_fields(
    problem: Problem,
    x: Float[Tensor, "nely nelx"],
    mu: Float[Tensor, "nely nelx"],
    beta_d: float,
    loop: int,
    previous: virtual_heat.TimeFieldSolution | None = None,
) -> tuple[Float[Tensor, "nely nelx"], virtual_heat.TimeFieldSolution]:
    """The density and time fields the physics reads, from the design variables.

    The one definition of that map, so a trajectory or a saved design cannot disagree
    with what `step` optimized.

    :param x: raw density, filtered and Heaviside-projected at sharpness `beta_d`
    :param mu: diffusivity design field
    :param beta_d: projection sharpness
    :param loop: iteration whose `penal` a penalized conduction takes
    :param previous: the last solution, the warm start
    :return: `(xPhys, solution)`, `solution.tPhys` the time field
    """
    config = problem.config
    xPhys = density(problem, x, beta_d)
    penal = run_config.weight_at(config.penal, loop)
    solution = virtual_heat.heat_time_field(
        config,
        problem.mesh,
        xPhys,
        mu,
        previous,
        conduction_penal=penal if config.penalize_conduction else 1.0,
    )
    return xPhys, solution


def init_state(problem: Problem) -> State:
    """The initial state: uniform density at `volfrac`, uniform diffusivity `chi = 1`
    (`mu = 1/2`)."""
    config = problem.config
    shape = (config.nely, config.nelx)
    kw = dict(device=problem.device, dtype=problem.dtype)
    x = torch.full(shape, config.volfrac, **kw)
    mu = torch.full(shape, 0.5, **kw)
    return State(
        x=x,
        mu=mu,
        mma=mma.History.initial(torch.cat([x.flatten(), mu.flatten()])),
        loop=0,
        beta_t=run_config.weight_at(config.beta_t_schedule, 0),
        beta_d=run_config.weight_at(config.beta_d_schedule, 0),
        U=None,
        heat=None,
    )


def constraint_values(
    problem: Problem,
    xPhys: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    K_est: Float[Tensor, " nel"],
    loop: int,
) -> dict[str, Float[Tensor, " k"]]:
    """
    Every constraint's values at iteration `loop`'s settings, in stack order: the
    global volume, then `FieldTerms.constraints`.

    :param xPhys: physical densities
    :param tPhys: physical time field
    :param K_est: `FieldTerms.estimated_conductivity` of the same fields at `loop`
    :param loop: iteration whose schedules apply
    :return: each constraint's rows
    """
    g = {
        "global_volume_fraction": constraints.global_volume_fraction(
            xPhys, problem.config.volfrac
        )
    }
    return g | problem.terms.constraints(xPhys, tPhys, K_est, loop)


def _sensitivity_rows(
    outputs: Float[Tensor, " k"],
    x: Float[Tensor, "nely nelx"],
    mu: Float[Tensor, "nely nelx"],
) -> Float[Tensor, "k n"]:
    """Sensitivities of `k` scalar outputs w.r.t. both design fields, as `(k, n)` in
    MMA's `[density; diffusivity]` layout. Each row passes through the heat adjoint."""
    return torch.cat(sensitivity.jacobian_rows(outputs, (x, mu)), dim=-1)


def step(problem: Problem, state: State) -> tuple[State, IterationRecord]:
    """One optimization iteration: the objective and every constraint, their gradients
    by autograd (through the heat equation's adjoint), and an MMA step on `[x; mu]`
    with one scheduled move limit for both.

    :param state: the state after `state.loop` iterations
    :return: the next state, and this iteration's record
    """
    config = problem.config
    nely, nelx = config.nely, config.nelx
    nel = nelx * nely
    loop, beta_t, beta_d = state.loop, state.beta_t, state.beta_d
    penal = run_config.weight_at(config.penal, loop)
    Tcr = run_config.weight_at(config.Tcr, loop)

    x = state.x.clone().requires_grad_(True)
    mu = state.mu.clone().requires_grad_(True)
    xPhys, heat = physical_fields(problem, x, mu, beta_d, loop, state.heat)
    tPhys = heat.tPhys

    c_t, stage_cs, U_new = compliance.batched_whole_and_gravity_compliance(
        xPhys,
        # The stage masks assume t <= 1, but void can be later than the whole part;
        # beyond 1 a mask turns negative, and a negative density NaNs under SIMP
        torch.clamp(tPhys, max=1),
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
        # The gravity stages' upper boundaries, as fractions of the build
        [float(ti) for ti in np.linspace(0, 1, config.nStage + 1)[1:]],
        x0=state.U,
    )
    f_val_t = c_t
    for cg_t in stage_cs:
        f_val_t = f_val_t + config.Theta * cg_t
    # Log, so the time-field weights act relative to the compliance (PR #189)
    objective = problem.terms.objective(torch.log(f_val_t), xPhys, tPhys, loop)
    f_val = float(objective.value.detach())
    df_dx = _sensitivity_rows(objective.value[None], x, mu)[0]

    K_est_t = problem.terms.estimated_conductivity(xPhys, tPhys, loop)
    g_parts = constraint_values(problem, xPhys, tPhys, K_est_t, loop)
    g_hotspot_t = g_parts["hotspot"][0]
    g_all = torch.cat(list(g_parts.values()))
    # Per part, not on `g_all`, as in `stto.step` (PR #173)
    g_rows = {name: _sensitivity_rows(g, x, mu) for name, g in g_parts.items()}
    dg_dx = torch.cat(list(g_rows.values()), dim=0)
    hotspot_weight = problem.terms.add_time_only_hotspot(
        df_dx, g_rows["hotspot"][0], nel, loop
    )

    diagnostics = problem.terms.hotspot_diagnostics(g_hotspot_t, K_est_t, xPhys, loop)
    with torch.no_grad():
        unit_m = timefield.unit_length_m(tPhys, config.element_size_m)
        grad_p10, grad_p50, grad_p90 = (
            g / unit_m
            for g in timefield.gradient_percentiles(tPhys.detach(), xPhys.detach())
        )
        log_chi = torch.log(virtual_heat.diffusivity(state.mu, config.chi_contrast))
    diagnostics.update(
        stage_obj=config.Theta * sum(float(cg.detach()) for cg in stage_cs),
        grad_p10_per_m=grad_p10,
        grad_p50_per_m=grad_p50,
        grad_p90_per_m=grad_p90,
        grey=_grey_fraction(xPhys),
        calibration=problem.terms.hotspot.calibration,
        penal=penal,
        uniformity_weight=objective.uniformity_weight,
        hotspot_weight=hotspot_weight,
        Tcr=Tcr,
        rouf=run_config.weight_at(config.rouf, loop),
        hotspot_beta=run_config.weight_at(config.hotspot_beta, loop),
        hotspot_kappa=run_config.weight_at(config.hotspot_kappa, loop),
        beta_d=beta_d,
        beta_t=beta_t,
        move=run_config.weight_at(config.move, loop),
        tool_radius_m=run_config.weight_at(config.tool_radius_m, loop),
        min_gradient_fraction=run_config.weight_at(config.min_gradient_fraction, loop),
        gradient_smoothness_m=run_config.weight_at(config.gradient_smoothness_m, loop),
        admissible_tool_radius_m=problem.terms.admissible_tool_radius_m(
            xPhys.detach(), tPhys.detach(), loop
        ),
        log_chi_roughness=float(timefield.roughness(log_chi)),
    )

    xval = torch.cat([state.x.flatten(), state.mu.flatten()])
    move_limit = torch.full_like(xval, run_config.weight_at(config.move, loop))
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
        mu=xmma[nel:].reshape(nely, nelx),
        mma=mma_history,
        loop=loop + 1,
        # Read at the next index, so a scheduled change starts with the next step
        beta_t=run_config.weight_at(config.beta_t_schedule, loop + 1),
        beta_d=run_config.weight_at(config.beta_d_schedule, loop + 1),
        # Detached: femsolve asserts a warm start carries no graph (commit 855eb76)
        U=U_new.detach(),
        heat=virtual_heat.TimeFieldSolution(
            *(None if v is None else v.detach() for v in heat)
        ),
    )
    step_size = (xmma - xval).abs()
    diagnostics.update(
        dx_max=float(step_size[:nel].max()),
        dx_mean=float(step_size[:nel].mean()),
        dmu_max=float(step_size[nel:].max()),
        dmu_mean=float(step_size[nel:].mean()),
    )
    record = IterationRecord(
        obj=float(c_t.detach()),
        vol=float(xPhys.detach().mean()),
        tru_max=(float(g_hotspot_t.detach()) + 1) * Tcr,
        uniformity=float(objective.uniformity.detach()),
        roughness=float(objective.roughness.detach()),
        roughness_weight=objective.roughness_weight,
        f=f_val,
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


def run(config: run_config.HeatRunConfig, **problem_kwargs) -> RunResult:
    """Build the setup and run `config.nloop` iterations from `init_state`.

    :param problem_kwargs: forwarded to `build_problem` (`device`, `dtype`)
    :return: the run
    """
    problem = build_problem(config, **problem_kwargs)
    return run_from_state(problem, init_state(problem), config.nloop)


def run_from_state(problem: Problem, state: State, nloop: int) -> RunResult:
    """Run `nloop` iterations from any starting state, collecting the trajectory.

    :return: the run
    """
    xPhys_traj, tPhys_traj, x_traj, mu_traj, records = [], [], [], [], []

    def record_fields(state: State) -> None:
        with torch.no_grad():
            xPhys, heat = physical_fields(
                problem, state.x, state.mu, state.beta_d, state.loop, state.heat
            )
        xPhys_traj.append(xPhys)
        tPhys_traj.append(heat.tPhys)
        x_traj.append(state.x.clone())
        mu_traj.append(state.mu.clone())

    record_fields(state)
    for _ in range(nloop):
        state, record = step(problem, state)
        record_fields(state)
        records.append(record)
    return RunResult(state, xPhys_traj, tPhys_traj, x_traj, mu_traj, records)


def _grey_fraction(xPhys: Float[Tensor, "nely nelx"]) -> float:
    """The fraction of elements neither near void nor near solid."""
    return float(((xPhys > 0.05) & (xPhys < 0.95)).double().mean())


def check_design(
    problem: Problem,
    x: Float[Tensor, "nely nelx"],
    mu: Float[Tensor, "nely nelx"],
    loop: int,
) -> dict:
    """
    Check a design, as raw design variables, on the binarized density with the run's own
    `tPhys`, as `checks.check_design` does for `stto`. Solving `tPhys` again through the
    binarized density gives a misleading hotspot row (PR #189). The report has the layout
    of `checks.check_design`'s, so `checks.summary` reads it.

    Start as in `virtual_heat.start_report`. Support and the saddle count are as in
    `checks.check_design`.
    A failed hard check warns.

    :param problem: the problem the design was optimized for
    :param x: design densities
    :param mu: design diffusivity field
    :param loop: iteration whose schedules apply
    :return: a JSON-serializable report
    """
    config = problem.config
    beta_d = run_config.weight_at(config.beta_d_schedule, loop)
    with torch.no_grad():
        xPhys, heat = physical_fields(problem, x, mu, beta_d, loop)
        xBin = density(problem, x, math.inf)
    tPhys = heat.tPhys
    xBin_np, tPhys_np = torch_util.to_numpy(xBin), torch_util.to_numpy(tPhys)
    solid = geometry.solid_mask(xBin_np).reshape(xBin_np.shape)
    base = torch_util.to_numpy(problem.Nei)

    start = virtual_heat.start_report(problem.mesh, heat, solid, base)
    support = checks.support_report(solid, tPhys_np, base)
    if not start["passed"]:
        virtual_heat.warn_start(start)
    if not support["passed"]:
        checks.warn_unsupported(support)
    report = dict(
        loop=loop,
        passed=start["passed"] and support["passed"],
        start=start,
        support=support,
        saddles=int(checks.saddles(tPhys_np, solid, solid).sum()),
        volume_fraction=float(solid.mean()),
        grey_fraction=_grey_fraction(xPhys),
    )
    if solid.any():

        def rows(density):
            K_est = problem.terms.estimated_conductivity(density, tPhys, loop)
            g = constraint_values(problem, density, tPhys, K_est, loop)
            return {k: torch_util.to_numpy(v).tolist() for k, v in g.items()}

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
        report.update(
            constraints=rows(xBin),
            constraints_continuous=rows(xPhys),
            compliance=float(c),
        )
        checks.warn_on_binarized_hotspot(report)
    return report
