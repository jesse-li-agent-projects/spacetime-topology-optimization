"""Main-loop orchestration: wires fem/filters/timefield/gravity/compliance/constraints/
conductivity/mma together into the actual space-time topology optimization iteration.

Every module this file calls already owns its own math and its own fixture/FD tests;
this file's only job is *wiring* -- building the stacked objective/constraint arrays
MMA expects, in the exact row order the MATLAB main loop uses, and threading the
iteration-dependent state (`beta_d`, `beta_t`, `factor`, `xold1`/`xold2`, `low`/`upp`, and
the raw-vs-filtered x/t fields) from one call to the next. See `conventions.md` for
array-order conventions and `tests/matlab_reference_loop.py` (a literal transliteration
of the MATLAB source this ports) for the authoritative iteration order.

Two field pairs are threaded per design variable, and must not be conflated: `x`/`t`
are each iteration's *raw* MMA output (unfiltered, unprojected -- what next
iteration's move-limit bounds read); `xTilde`/`xPhys`/`tPhys` are the *filtered* (and,
for density, Heaviside-projected) fields the physics uses.
`xTilde` alone is carried an extra step (needed for the Heaviside-derivative chain rule
at the *start* of the next iteration, before that iteration's own update).

**The tensor boundary (`plans/torch_port_part2.md`).** `Problem` and `State` hold torch
tensors -- `Problem.device`/`.dtype` say where -- and, as of Phase 3.2, so does every
leaf module `step` calls: `filters`/`compliance`/`constraints`/`conductivity` all take
and return tensors. Two things are deliberately not yet ported (Phases 3.3 and 3.5) and
`step` bridges to NumPy narrowly around just their calls: `compliance.py`'s own
`fem.assemble_stiffness`/`fem.solve_fe` calls (inside `whole_compliance`/
`gravity_compliance`, not visible here), and `mma.mmasub` below.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.constraints as constraints
import sttopt.fem as fem
import sttopt.filters as filters
import sttopt.gravity as gravity
import sttopt.mma as mma
import sttopt.timefield as timefield
import sttopt.torch_fem as torch_fem
import sttopt.torch_util as torch_util


@dataclass(frozen=True)
class Problem:
    """Fixed problem setup: everything that doesn't change across iterations.

    Built once by `build_problem` and passed unchanged to every `step` call.
    """

    nelx: int
    nely: int
    nStage: int
    volfrac: float
    Theta: float
    Tcr: float
    tfield: timefield.TimeField

    device: torch.device
    dtype: torch.dtype

    KE: Float[Tensor, "8 8"]
    edofMat: Int[Tensor, "nelx*nely 8"]
    freedofs: Int[Tensor, " n_free"]
    free_mask: Bool[Tensor, " ndof"]  # True at free dofs; the matrix-free path's mask
    F: Float[Tensor, " ndof"]
    ndof: int
    H: Tensor  # sparse CSR, shape (nel, nel)
    Hs: Float[Tensor, " nel"]
    L: Tensor  # sparse CSR, shape (nel, nel)
    C: Tensor  # sparse CSR, shape ((nelx+1)*(nely+1), nel)
    e1: Int[Tensor, " npairs"]
    e2: Int[Tensor, " npairs"]
    w: Float[Tensor, " npairs"]
    Nei: Int[Tensor, " k"]

    Emin: float
    Emax: float
    penal: float
    eta: float
    beta_d_max: float
    p: float
    q: float
    r: float
    rouf: float  # hotspot/conductivity-selection sigmoid sharpness (Das 2023 zeta) -- unrelated to beta_t/beta_d
    a0: float
    mma_c: float
    move: float
    tmove: float

    m: int  # number of MMA constraint rows: vol + continuity + start-point(s) + 2*nStage + hotspot
    n: int  # number of MMA design variables: 2*nelx*nely (density half + time half)


@dataclass(frozen=True)
class State:
    """Iteration-dependent state carried from one `step` call to the next."""

    x: Float[Tensor, "nely nelx"]  # raw density (unfiltered MMA output)
    xTilde: Float[Tensor, "nely nelx"]  # density-filtered x
    xPhys: Float[Tensor, "nely nelx"]  # density for physics purposes
    t: Float[Tensor, "nely nelx"]  # raw time field (unfiltered MMA output)
    tPhys: Float[Tensor, "nely nelx"]  # time for physics purposes
    xold1: Float[Tensor, " n"]
    xold2: Float[Tensor, " n"]
    low: Float[Tensor, " n"]
    upp: Float[Tensor, " n"]
    loop: int
    beta_t: float  # gravity/stage-mask sigmoid sharpness
    beta_d: float  # Heaviside projection sharpness
    factor: float  # hotspot-constraint rescaling constant, periodically refreshed


@dataclass(frozen=True)
class IterationRecord:
    """Per-iteration diagnostics and raw MMA outputs, for E2E/MMA/constraint-order tests."""

    obj: float  # whole-structure compliance (doesn't include intermediate structures)
    vol: float  # volume fraction (mean xPhys)
    tru_max: float  # Estimated max hotspot severity (debiased from P-mean)
    f0val: float  # objective (weighted sum of whole & per-stage compliances)
    df0dx: Float[np.ndarray, " n"]
    xmma: Float[np.ndarray, " n"]
    low: Float[np.ndarray, " n"]
    upp: Float[np.ndarray, " n"]
    lam: Float[np.ndarray, " m"]
    fval: Float[np.ndarray, " m"]
    dfdx: Float[np.ndarray, "m n"]


@dataclass(frozen=True)
class RunResult:
    state: State  # final state after nloop iterations
    # length nloop+1, index 0 is initial field
    xPhys_traj: list[Float[Tensor, "nely nelx"]]
    tPhys_traj: list[Float[Tensor, "nely nelx"]]  # length nloop+1
    records: list[IterationRecord]  # length nloop


def build_problem(
    nelx: int,
    nely: int,
    nStage: int,
    volfrac: float,
    Theta: float,
    Tcr: float,
    tfield: timefield.TimeField,
    rmin: float,
    lrmin: float,
    rmin_cond: float,
    *,
    Emin: float = 1e-9,
    Emax: float = 1.0,
    nu: float = 0.3,
    penal: float = 3.0,
    eta: float = 0.5,
    beta_d_max: float = 128.0,
    p: float = 25.0,
    q: float = 3.0,
    r: float = 0.05,
    rouf: float = 100.0,
    a0: float = 1.0,
    mma_c: float = 2500.0,
    move: float = 0.01,
    tmove: float = 0.01,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Problem:
    """Build the fixed FEM/filter/geometry setup once, before the loop starts.

    `rmin`/`lrmin`/`rmin_cond` are filter radii (density filter, continuity filter,
    conductivity neighborhood) -- passed explicitly rather than hardcoded so callers
    (e.g. the E2E test) can match whatever grid they're running on, since the fixture's
    radii differ from the original full-scale script's.

    :param device: device every tensor field of the returned `Problem` lives on.
    :param dtype: floating dtype every real-valued tensor field is cast to; integer
        (index/mask) fields keep their own integer/bool dtype regardless.
    """
    device = torch.device(device)
    # A 1x1 mesh has neither extent nor neighbours, so two of the pieces built below
    # degenerate: the CORNER/OPPOSITE_CORNER time fields normalize by a zero max
    # distance, and the continuity filter divides by a zero neighbour count. This is the
    # first point where both exist, so the check belongs here rather than in either one.
    if nelx == 1 and nely == 1:
        raise ValueError(
            f"nelx and nely cannot both be 1, got nelx={nelx}, nely={nely}"
        )

    tfield = timefield.TimeField(tfield)
    KE = fem.plane_stress_KE(nu)
    edofMat = fem.element_dof_map(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Fixed cantilever load case, stated geometrically rather than as a linear-index
    # formula so it survives a change of node numbering: unit downward point load on the
    # bottom-right node, left edge clamped in both directions.
    nodes = fem.node_grid(nelx, nely)
    F = np.zeros(ndof)
    F[2 * nodes[-1, -1] + 1] = -1.0
    left_edge = nodes[:, 0]
    fixeddofs = np.stack([2 * left_edge, 2 * left_edge + 1], axis=-1).ravel()
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    H, Hs = filters.density_filter(nelx, nely, rmin)
    L = filters.continuity_filter(nelx, nely, lrmin)
    C = gravity.gravity_load_matrix(nelx, nely)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)

    # Print-start element(s): the whole first mesh column for tfield != CORNER, the
    # single origin element for tfield == CORNER (constraints.start_point's own
    # docstring). Element `row*nelx` is grid position `(row, 0)` per conventions.md's
    # C-order element enumeration -- i.e. column 0, every row.
    Nei = (
        np.array([0])
        if tfield == timefield.TimeField.CORNER
        else np.arange(nely) * nelx
    )

    n = 2 * nelx * nely
    # MATLAB hardcodes `m = 1 + 1 + nely + 2*nStage + 1` -- only self-consistent when
    # tfield != CORNER (Nei has nely rows); computing from len(Nei) generalizes correctly.
    m = 1 + 1 + len(Nei) + 2 * nStage + 1

    freedofs_t = torch_util.to_tensor(freedofs, device, torch.int64)
    return Problem(
        nelx=nelx,
        nely=nely,
        nStage=nStage,
        volfrac=volfrac,
        Theta=Theta,
        Tcr=Tcr,
        tfield=tfield,
        device=device,
        dtype=dtype,
        KE=torch_util.to_tensor(KE, device, dtype),
        edofMat=torch_util.to_tensor(edofMat, device, torch.int64),
        freedofs=freedofs_t,
        free_mask=torch_fem.free_mask(ndof, freedofs_t, device=device),
        F=torch_util.to_tensor(F, device, dtype),
        ndof=ndof,
        H=torch_util.csr_to_tensor(H, device, dtype),
        Hs=torch_util.to_tensor(Hs, device, dtype),
        L=torch_util.csr_to_tensor(L, device, dtype),
        C=torch_util.csr_to_tensor(C, device, dtype),
        e1=torch_util.to_tensor(e1, device, torch.int64),
        e2=torch_util.to_tensor(e2, device, torch.int64),
        w=torch_util.to_tensor(w, device, dtype),
        Nei=torch_util.to_tensor(Nei, device, torch.int64),
        Emin=Emin,
        Emax=Emax,
        penal=penal,
        eta=eta,
        beta_d_max=beta_d_max,
        p=p,
        q=q,
        r=r,
        rouf=rouf,
        a0=a0,
        mma_c=mma_c,
        move=move,
        tmove=tmove,
        m=m,
        n=n,
    )


def init_state(problem: Problem, beta_d: float) -> State:
    """Initial state: `x`/`t` are the raw seed (uniform `problem.volfrac`, time field per
    `problem.tfield`); the physics fields are derived from them exactly as in `step`.

    The MATLAB source instead assigns `xTilde = x` and `t = tPhys` unfiltered here. For
    the density half that is a numeric no-op -- `x` is uniform and the filter fixes
    constants -- but no `init_timefield` variant is constant, so leaving `tPhys`
    unfiltered made iteration 1 the one iteration whose forward map disagreed with the
    `tPhys = H @ t / Hs` that `step`'s chain rule differentiates. See PR #26.
    """
    nely, nelx = problem.nely, problem.nelx
    nel = nelx * nely
    device, dtype = problem.device, problem.dtype

    def filtered(field: Tensor) -> Tensor:
        return ((problem.H @ field.flatten()) / problem.Hs).reshape(nely, nelx)

    x = torch.full((nely, nelx), problem.volfrac, device=device, dtype=dtype)
    xTilde = filtered(x)
    xPhys = filters.heaviside_projection(xTilde, beta_d, problem.eta)

    # init_timefield is a NumPy builder (plans/torch_port_part2.md Phase 3.2 item 2);
    # converted once here, at the tensor boundary.
    t = torch_util.to_tensor(
        timefield.init_timefield(nelx, nely, problem.tfield), device, dtype
    )
    tPhys = filtered(t)

    # MATLAB's xold1=xold2=[x(:); zeros(nel,1)] -- provably unread (iterations 1 and 2
    # both take mmasub's `iteration < 2.5` reinit branch), reproduced anyway for fidelity.
    xold = torch.cat([x.flatten(), torch.zeros(nel, device=device, dtype=dtype)])

    return State(
        x=x,
        xTilde=xTilde,
        xPhys=xPhys,
        t=t,
        tPhys=tPhys,
        xold1=xold,
        xold2=xold.clone(),
        low=torch.zeros(problem.n, device=device, dtype=dtype),
        upp=torch.zeros(problem.n, device=device, dtype=dtype),
        loop=0,
        beta_t=10.0,
        beta_d=beta_d,
        factor=1.0,
    )


def step(problem: Problem, state: State) -> tuple[State, IterationRecord]:
    """Run one optimization iteration: build the objective + every constraint's
    value/sensitivity in the reference's exact row order, call `mma.mmasub`, and
    unpack the result into the next state.

    The three periodic state updates below (`beta_t += 5` at loop%30==0, `beta_d *= 2` at
    loop%50==0, the hotspot `factor` refresh at loop%25==0) never trigger against the
    small E2E fixture (`nloop=3`) -- unexercised by that fixture, not unimplemented or
    worked around. The `factor` refresh deviates from the MATLAB reference: it takes
    effect starting the *next* iteration rather than rescaling this iteration's own
    `fval`/`df1`/`dt1` mid-loop -- a deliberate simplification, not a fidelity gap.
    """
    nely, nelx, nStage = problem.nely, problem.nelx, problem.nStage
    nel = nelx * nely
    device, dtype = problem.device, problem.dtype

    loop = state.loop + 1
    beta_t = state.beta_t
    if loop % 30 == 0 and beta_t < 50:
        beta_t += 5
    beta_d = state.beta_d
    if loop % 50 == 0 and beta_d <= problem.beta_d_max:
        beta_d *= 2
    if beta_d > problem.beta_d_max:
        beta_d = problem.beta_d_max

    xPhys, tPhys = state.xPhys, state.tPhys
    # Heaviside-derivative chain rule for this iteration's density sensitivities, from
    # the xTilde carried over from the *previous* iteration's update (or init).
    dx = filters.heaviside_projection_derivative(state.xTilde, beta_d, problem.eta)

    # -- Objective: whole-structure compliance + Theta-weighted per-stage gravity compliance --
    c, dcx = compliance.whole_compliance(
        xPhys,
        problem.KE,
        problem.edofMat,
        problem.Emin,
        problem.Emax,
        problem.penal,
        problem.freedofs,
        problem.F,
        problem.ndof,
    )
    obj_final_only = c  # compliance of final structure only, saved for logging
    obj = c
    dc = problem.H @ (dcx.flatten() * dx.flatten() / problem.Hs)
    dt = torch.zeros(nel, device=device, dtype=dtype)

    for ti in np.linspace(0, 1, nStage + 1)[1:]:
        cg, dcx_g, dct_g = compliance.gravity_compliance(
            xPhys,
            tPhys,
            problem.KE,
            problem.edofMat,
            problem.Emin,
            problem.Emax,
            problem.penal,
            float(ti),
            problem.C,
            beta_t,
            problem.freedofs,
            problem.ndof,
        )
        obj += problem.Theta * cg
        dc = dc + problem.Theta * (problem.H @ (dcx_g * dx.flatten() / problem.Hs))
        dt = dt + problem.Theta * (problem.H @ (dct_g / problem.Hs))

    df0dx = torch.cat([dc, dt])
    f0val = obj

    # -- Move-limit bounds on this iteration's raw MMA variables --
    xflat = state.x.flatten()
    tflat = state.t.flatten()
    xminx = torch.clamp(xflat - problem.move, min=0.0)
    xmaxx = torch.clamp(xflat + problem.move, max=1.0)
    xmint = torch.clamp(tflat - problem.tmove, min=0.0)
    xmaxt = torch.clamp(tflat + problem.tmove, max=1.0)
    xmin = torch.cat([xminx, xmint])
    xmax = torch.cat([xmaxx, xmaxt])
    xval = torch.cat([xflat, tflat])

    # -- Constraints, stacked in the reference loop's exact order --
    fval_parts: list[Tensor] = []
    dfdx_parts: list[Tensor] = []

    fv, dfx, dft = constraints.global_volume_fraction(
        xPhys, dx, problem.H, problem.Hs, problem.volfrac
    )
    vol_diag = float(torch.sum(xPhys) / (nelx * nely))
    fval_parts.append(torch.tensor([fv], device=device, dtype=dtype))
    dfdx_parts.append(torch.cat([dfx, dft])[None, :])

    fv, dfx, dft = constraints.time_field_continuity(
        tPhys, problem.L, problem.H, problem.Hs
    )
    fval_parts.append(torch.tensor([fv], device=device, dtype=dtype))
    dfdx_parts.append(torch.cat([dfx, dft])[None, :])

    fv, dfx, dft = constraints.start_point(tPhys, problem.Nei, problem.H, problem.Hs)
    fval_parts.append(fv)
    dfdx_parts.append(torch.cat([dfx, dft], dim=1))

    for t_stage in np.linspace(0, 1, nStage + 1)[1:]:
        fu, fl, dfx, dft = constraints.stage_volume_bounds(
            xPhys,
            tPhys,
            dx,
            problem.H,
            problem.Hs,
            float(t_stage),
            problem.volfrac,
            beta_t,
        )
        fval_parts.append(torch.tensor([fu, fl], device=device, dtype=dtype))
        dfdx_parts.append(torch.stack([torch.cat([dfx, dft]), torch.cat([-dfx, -dft])]))

    # Hotspot constraint, evaluated at this iteration's (possibly stale) `factor`.
    # `factor` is refreshed every 25 iterations from this same call's `numer`/`K_est`
    # (both independent of `factor`), but the refresh only takes effect starting next
    # iteration's `step` call -- `fv`/`df1`/`dt1` below are never rescaled mid-iteration.
    hotspot = conductivity.hotspot_constraint(
        xPhys,
        tPhys,
        problem.e1,
        problem.e2,
        problem.w,
        dx,
        problem.H,
        problem.Hs,
        state.factor,
        problem.Tcr,
        problem.p,
        problem.q,
        problem.r,
        problem.rouf,
    )
    fv, df1, dt1 = hotspot.fval, hotspot.df1, hotspot.dt1
    factor = state.factor
    if loop % 25 == 0:
        max_g = float(torch.max((1 - hotspot.K_est) * xPhys.flatten() ** problem.r))
        factor = max_g / hotspot.numer
    tru_max = factor * hotspot.numer
    fval_parts.append(torch.tensor([fv], device=device, dtype=dtype))
    dfdx_parts.append(torch.cat([df1, dt1])[None, :])

    fval = torch.cat(fval_parts)
    dfdx = torch.cat(dfdx_parts, dim=0)

    # -- MMA subproblem solve and update -- (mma.py isn't ported yet, Phase 3.5: bridge
    # to NumPy narrowly around just this call.)
    mma_a = np.zeros(problem.m)
    mma_c = np.full(problem.m, problem.mma_c)
    mma_d = np.zeros(problem.m)
    xmma, ymma, zmma, lam, xsi, mma_eta, mu, zet, s, low, upp = mma.mmasub(
        problem.m,
        problem.n,
        loop,
        torch_util.to_numpy(xval),
        torch_util.to_numpy(xmin),
        torch_util.to_numpy(xmax),
        torch_util.to_numpy(state.xold1),
        torch_util.to_numpy(state.xold2),
        float(f0val),
        torch_util.to_numpy(df0dx),
        torch_util.to_numpy(fval),
        torch_util.to_numpy(dfdx),
        torch_util.to_numpy(state.low),
        torch_util.to_numpy(state.upp),
        problem.a0,
        mma_a,
        mma_c,
        mma_d,
    )

    x_new = torch_util.to_tensor(xmma[:nel].reshape(nely, nelx), device, dtype)
    t_new = torch_util.to_tensor(xmma[nel:].reshape(nely, nelx), device, dtype)

    xTilde_new = ((problem.H @ x_new.flatten()) / problem.Hs).reshape(nely, nelx)
    xPhys_new = filters.heaviside_projection(xTilde_new, beta_d, problem.eta)
    tPhys_new = ((problem.H @ t_new.flatten()) / problem.Hs).reshape(nely, nelx)

    new_state = State(
        x=x_new,
        xTilde=xTilde_new,
        xPhys=xPhys_new,
        t=t_new,
        tPhys=tPhys_new,
        xold1=xval,
        xold2=state.xold1,
        low=torch_util.to_tensor(low, device, dtype),
        upp=torch_util.to_tensor(upp, device, dtype),
        loop=loop,
        beta_t=beta_t,
        beta_d=beta_d,
        factor=factor,
    )
    record = IterationRecord(
        obj=obj_final_only,
        vol=vol_diag,
        tru_max=tru_max,
        f0val=float(f0val),
        df0dx=torch_util.to_numpy(df0dx),
        xmma=xmma,
        low=low,
        upp=upp,
        lam=lam,
        fval=torch_util.to_numpy(fval),
        dfdx=torch_util.to_numpy(dfdx),
    )
    return new_state, record


def run(
    nelx: int,
    nely: int,
    nloop: int,
    nStage: int,
    volfrac: float,
    Theta: float,
    Tcr: float,
    tfield: timefield.TimeField,
    rmin: float,
    lrmin: float,
    rmin_cond: float,
    *,
    beta_d: float = 1.0,
    **problem_kwargs,
) -> RunResult:
    """Build the fixed setup and run `nloop` optimization iterations from the standard
    initialization (uniform density at `volfrac`, `timefield.init_timefield` for the
    print-time field). `problem_kwargs` forwards to `build_problem` (SIMP/Heaviside/
    hotspot/MMA constants) for callers that need non-default values.
    """
    problem = build_problem(
        nelx,
        nely,
        nStage,
        volfrac,
        Theta,
        Tcr,
        tfield,
        rmin,
        lrmin,
        rmin_cond,
        **problem_kwargs,
    )
    return run_from_state(problem, init_state(problem, beta_d), nloop)


def run_from_state(problem: Problem, state: State, nloop: int) -> RunResult:
    """Run `nloop` iterations from an arbitrary starting state, collecting the
    trajectory. Split out of `run` so that where the iteration starts is a caller's
    choice -- warm-starting from a previous run's final state, or entering at a
    trajectory recorded elsewhere.
    """
    xPhys_traj = [state.xPhys.clone()]
    tPhys_traj = [state.tPhys.clone()]
    records: list[IterationRecord] = []

    for _ in range(nloop):
        state, record = step(problem, state)
        xPhys_traj.append(state.xPhys.clone())
        tPhys_traj.append(state.tPhys.clone())
        records.append(record)

    return RunResult(
        state=state, xPhys_traj=xPhys_traj, tPhys_traj=tPhys_traj, records=records
    )
