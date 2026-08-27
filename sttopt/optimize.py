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

**The tensor boundary (`plans/torch_port_part2.md` Phase 3.1).** `Problem` and `State`
hold torch tensors -- `Problem.device`/`.dtype` say where -- but `compliance.py`/
`constraints.py`/`conductivity.py`/`filters.py`/`mma.py` are still the original
NumPy/SciPy implementations (later phases in that plan port them). `step` and
`init_state` bridge the two: convert the tensor fields they need to NumPy/SciPy once at
the top, run the existing array-library computation unchanged, then convert the results
back to tensors for the returned `State`. This bridging -- not any change to the
arithmetic -- is the only thing this phase adds to `step`/`init_state`; it goes away
once Phase 3.2 ports the leaf math itself.
"""

from dataclasses import dataclass
from types import SimpleNamespace

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

    # Tensor boundary (see module docstring): H/Hs are still applied via SciPy here,
    # since `filters.heaviside_projection` is unported NumPy.
    H = torch_util.csr_to_scipy(problem.H)
    Hs = torch_util.to_numpy(problem.Hs)

    def filtered(field):
        return (H @ field.flatten() / Hs).reshape(nely, nelx)

    x = np.full((nely, nelx), problem.volfrac)
    xTilde = filtered(x)
    xPhys = filters.heaviside_projection(xTilde, beta_d, problem.eta)

    t = timefield.init_timefield(nelx, nely, problem.tfield)
    tPhys = filtered(t)

    # MATLAB's xold1=xold2=[x(:); zeros(nel,1)] -- provably unread (iterations 1 and 2
    # both take mmasub's `iteration < 2.5` reinit branch), reproduced anyway for fidelity.
    xold = np.concatenate([x.flatten(), np.zeros(nel)])

    device, dtype = problem.device, problem.dtype
    return State(
        x=torch_util.to_tensor(x, device, dtype),
        xTilde=torch_util.to_tensor(xTilde, device, dtype),
        xPhys=torch_util.to_tensor(xPhys, device, dtype),
        t=torch_util.to_tensor(t, device, dtype),
        tPhys=torch_util.to_tensor(tPhys, device, dtype),
        xold1=torch_util.to_tensor(xold, device, dtype),
        xold2=torch_util.to_tensor(xold.copy(), device, dtype),
        low=torch_util.to_tensor(np.zeros(problem.n), device, dtype),
        upp=torch_util.to_tensor(np.zeros(problem.n), device, dtype),
        loop=0,
        beta_t=10.0,
        beta_d=beta_d,
        factor=1.0,
    )


def _problem_numpy_view(problem: Problem) -> SimpleNamespace:
    """
    NumPy/SciPy view of `problem`'s tensor fields, for the still-unported leaf math
    (`compliance`/`constraints`/`conductivity`/`filters`/`mma`); every non-tensor field
    passes through unchanged. Transitional scaffolding, per this module's docstring --
    removed once Phase 3.2 of `plans/torch_port_part2.md` ports that leaf math to torch
    directly.

    :param problem: the tensor-valued `Problem`.
    :return: a namespace with the same field names as `Problem`, tensor fields converted.
    """
    return SimpleNamespace(
        nelx=problem.nelx,
        nely=problem.nely,
        nStage=problem.nStage,
        volfrac=problem.volfrac,
        Theta=problem.Theta,
        Tcr=problem.Tcr,
        tfield=problem.tfield,
        KE=torch_util.to_numpy(problem.KE),
        edofMat=torch_util.to_numpy(problem.edofMat),
        freedofs=torch_util.to_numpy(problem.freedofs),
        F=torch_util.to_numpy(problem.F),
        ndof=problem.ndof,
        H=torch_util.csr_to_scipy(problem.H),
        Hs=torch_util.to_numpy(problem.Hs),
        L=torch_util.csr_to_scipy(problem.L),
        C=torch_util.csr_to_scipy(problem.C),
        e1=torch_util.to_numpy(problem.e1),
        e2=torch_util.to_numpy(problem.e2),
        w=torch_util.to_numpy(problem.w),
        Nei=torch_util.to_numpy(problem.Nei),
        Emin=problem.Emin,
        Emax=problem.Emax,
        penal=problem.penal,
        eta=problem.eta,
        beta_d_max=problem.beta_d_max,
        p=problem.p,
        q=problem.q,
        r=problem.r,
        rouf=problem.rouf,
        a0=problem.a0,
        mma_c=problem.mma_c,
        move=problem.move,
        tmove=problem.tmove,
        m=problem.m,
        n=problem.n,
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
    # Tensor boundary (see module docstring): `prob` exposes `problem`'s tensor fields
    # as NumPy/SciPy for the unported leaf math below; converted once here, then this
    # function's arithmetic is exactly what it was pre-Phase-3.1.
    prob = _problem_numpy_view(problem)
    nely, nelx, nStage = prob.nely, prob.nelx, prob.nStage
    nel = nelx * nely

    loop = state.loop + 1
    beta_t = state.beta_t
    if loop % 30 == 0 and beta_t < 50:
        beta_t += 5
    beta_d = state.beta_d
    if loop % 50 == 0 and beta_d <= prob.beta_d_max:
        beta_d *= 2
    if beta_d > prob.beta_d_max:
        beta_d = prob.beta_d_max

    xPhys, tPhys = torch_util.to_numpy(state.xPhys), torch_util.to_numpy(state.tPhys)
    # Heaviside-derivative chain rule for this iteration's density sensitivities, from
    # the xTilde carried over from the *previous* iteration's update (or init).
    dx = filters.heaviside_projection_derivative(
        torch_util.to_numpy(state.xTilde), beta_d, prob.eta
    )

    # -- Objective: whole-structure compliance + Theta-weighted per-stage gravity compliance --
    c, dcx = compliance.whole_compliance(
        xPhys,
        prob.KE,
        prob.edofMat,
        prob.Emin,
        prob.Emax,
        prob.penal,
        prob.freedofs,
        prob.F,
        prob.ndof,
    )
    obj_final_only = c  # compliance of final structure only, saved for logging
    obj = c
    dc = prob.H @ (dcx.flatten() * dx.flatten() / prob.Hs)
    dt = np.zeros(nel)

    for ti in np.linspace(0, 1, nStage + 1)[1:]:
        cg, dcx_g, dct_g = compliance.gravity_compliance(
            xPhys,
            tPhys,
            prob.KE,
            prob.edofMat,
            prob.Emin,
            prob.Emax,
            prob.penal,
            ti,
            prob.C,
            beta_t,
            prob.freedofs,
            prob.ndof,
        )
        obj += prob.Theta * cg
        dc = dc + prob.Theta * (prob.H @ (dcx_g * dx.flatten() / prob.Hs))
        dt = dt + prob.Theta * (prob.H @ (dct_g / prob.Hs))

    df0dx = np.concatenate([dc, dt])
    f0val = obj

    # -- Move-limit bounds on this iteration's raw MMA variables --
    xflat = torch_util.to_numpy(state.x).flatten()
    tflat = torch_util.to_numpy(state.t).flatten()
    xminx = np.maximum(0.0, xflat - prob.move)
    xmaxx = np.minimum(1.0, xflat + prob.move)
    xmint = np.maximum(0.0, tflat - prob.tmove)
    xmaxt = np.minimum(1.0, tflat + prob.tmove)
    xmin = np.concatenate([xminx, xmint])
    xmax = np.concatenate([xmaxx, xmaxt])
    xval = np.concatenate([xflat, tflat])

    # -- Constraints, stacked in the reference loop's exact order --
    fval_parts: list[np.ndarray] = []
    dfdx_parts: list[np.ndarray] = []

    fv, dfx, dft = constraints.global_volume_fraction(
        xPhys, dx, prob.H, prob.Hs, prob.volfrac
    )
    vol_diag = float(np.sum(xPhys) / (nelx * nely))
    fval_parts.append(np.array([fv]))
    dfdx_parts.append(np.concatenate([dfx, dft])[None, :])

    fv, dfx, dft = constraints.time_field_continuity(tPhys, prob.L, prob.H, prob.Hs)
    fval_parts.append(np.array([fv]))
    dfdx_parts.append(np.concatenate([dfx, dft])[None, :])

    fv, dfx, dft = constraints.start_point(tPhys, prob.Nei, prob.H, prob.Hs)
    fval_parts.append(fv)
    dfdx_parts.append(np.concatenate([dfx, dft], axis=1))

    for t_stage in np.linspace(0, 1, nStage + 1)[1:]:
        fu, fl, dfx, dft = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, prob.H, prob.Hs, t_stage, prob.volfrac, beta_t
        )
        fval_parts.append(np.array([fu, fl]))
        dfdx_parts.append(
            np.stack([np.concatenate([dfx, dft]), np.concatenate([-dfx, -dft])])
        )

    # Hotspot constraint, evaluated at this iteration's (possibly stale) `factor`.
    # `factor` is refreshed every 25 iterations from this same call's `numer`/`K_est`
    # (both independent of `factor`), but the refresh only takes effect starting next
    # iteration's `step` call -- `fv`/`df1`/`dt1` below are never rescaled mid-iteration.
    hotspot = conductivity.hotspot_constraint(
        xPhys,
        tPhys,
        prob.e1,
        prob.e2,
        prob.w,
        dx,
        prob.H,
        prob.Hs,
        state.factor,
        prob.Tcr,
        prob.p,
        prob.q,
        prob.r,
        prob.rouf,
    )
    fv, df1, dt1 = hotspot.fval, hotspot.df1, hotspot.dt1
    factor = state.factor
    if loop % 25 == 0:
        max_g = float(np.max((1 - hotspot.K_est) * xPhys.flatten() ** prob.r))
        factor = max_g / hotspot.numer
    tru_max = factor * hotspot.numer
    fval_parts.append(np.array([fv]))
    dfdx_parts.append(np.concatenate([df1, dt1])[None, :])

    fval = np.concatenate(fval_parts)
    dfdx = np.concatenate(dfdx_parts, axis=0)

    # -- MMA subproblem solve and update --
    mma_a = np.zeros(prob.m)
    mma_c = np.full(prob.m, prob.mma_c)
    mma_d = np.zeros(prob.m)
    xmma, ymma, zmma, lam, xsi, mma_eta, mu, zet, s, low, upp = mma.mmasub(
        prob.m,
        prob.n,
        loop,
        xval,
        xmin,
        xmax,
        torch_util.to_numpy(state.xold1),
        torch_util.to_numpy(state.xold2),
        f0val,
        df0dx,
        fval,
        dfdx,
        torch_util.to_numpy(state.low),
        torch_util.to_numpy(state.upp),
        prob.a0,
        mma_a,
        mma_c,
        mma_d,
    )

    x_new = xmma[:nel].reshape(nely, nelx)
    t_new = xmma[nel:].reshape(nely, nelx)

    xTilde_new = (prob.H @ x_new.flatten() / prob.Hs).reshape(nely, nelx)
    xPhys_new = filters.heaviside_projection(xTilde_new, beta_d, prob.eta)
    tPhys_new = (prob.H @ t_new.flatten() / prob.Hs).reshape(nely, nelx)

    # Tensor boundary again: back to `problem`'s own device/dtype for the returned State.
    device, dtype = problem.device, problem.dtype
    new_state = State(
        x=torch_util.to_tensor(x_new, device, dtype),
        xTilde=torch_util.to_tensor(xTilde_new, device, dtype),
        xPhys=torch_util.to_tensor(xPhys_new, device, dtype),
        t=torch_util.to_tensor(t_new, device, dtype),
        tPhys=torch_util.to_tensor(tPhys_new, device, dtype),
        xold1=torch_util.to_tensor(xval, device, dtype),
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
        f0val=f0val,
        df0dx=df0dx,
        xmma=xmma,
        low=low,
        upp=upp,
        lam=lam,
        fval=fval,
        dfdx=dfdx,
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
