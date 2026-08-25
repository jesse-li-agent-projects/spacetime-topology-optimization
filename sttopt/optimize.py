"""Main-loop orchestration: wires fem/filters/timefield/gravity/compliance/constraints/
conductivity/mma together into the actual space-time topology optimization iteration.

Every module this file calls already owns its own math and its own fixture/FD tests;
this file's only job is *wiring* -- building the stacked objective/constraint arrays
MMA expects, in the exact row order the MATLAB main loop uses, and threading the
iteration-dependent state (`beta`, `rou`, `factor`, `xold1`/`xold2`, `low`/`upp`, and
the raw-vs-filtered x/t fields) from one call to the next. See `conventions.md` for
array-order conventions and `tests/fixtures/generate_fixtures.m` (the literal MATLAB
reference this ports) for the authoritative iteration order.

Two field pairs are threaded per design variable, and must not be conflated: `x`/`t`
are each iteration's *raw* MMA output (unfiltered, unprojected -- what next
iteration's move-limit bounds read); `xTilde`/`xPhys`/`tPhys` are the *filtered* (and,
for density, Heaviside-projected) fields the physics uses.
`xTilde` alone is carried an extra step (needed for the Heaviside-derivative chain rule
at the *start* of the next iteration, before that iteration's own update).
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from jaxtyping import Float, Int

import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.constraints as constraints
import sttopt.fem as fem
import sttopt.filters as filters
import sttopt.gravity as gravity
import sttopt.mma as mma
import sttopt.timefield as timefield


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
    tfield: int

    KE: Float[np.ndarray, "8 8"]
    edofMat: Int[np.ndarray, "nelx*nely 8"]
    freedofs: Int[np.ndarray, " n_free"]
    F: Float[np.ndarray, " ndof"]
    ndof: int
    H: sp.spmatrix | sp.sparray
    Hs: Float[np.ndarray, " nel"]
    L: sp.spmatrix | sp.sparray
    C: sp.spmatrix | sp.sparray
    e1: Int[np.ndarray, " npairs"]
    e2: Int[np.ndarray, " npairs"]
    w: Float[np.ndarray, " npairs"]
    Nei: Int[np.ndarray, " k"]

    Emin: float
    Emax: float
    penal: float
    eta: float
    beta_max: float
    p: float
    q: float
    r: float
    rouf: float
    a0: float
    mma_c: float
    move: float
    tmove: float

    m: int  # number of MMA constraint rows: vol + continuity + start-point(s) + 2*nStage + hotspot
    n: int  # number of MMA design variables: 2*nelx*nely (density half + time half)


@dataclass(frozen=True)
class State:
    """Iteration-dependent state carried from one `step` call to the next."""

    x: Float[np.ndarray, "nely nelx"]  # raw density (unfiltered MMA output)
    xTilde: Float[np.ndarray, "nely nelx"]  # density-filtered x
    xPhys: Float[np.ndarray, "nely nelx"]  # density for physics purposes
    t: Float[np.ndarray, "nely nelx"]  # raw time field (unfiltered MMA output)
    tPhys: Float[np.ndarray, "nely nelx"]  # time for physics purposes
    xold1: Float[np.ndarray, " n"]
    xold2: Float[np.ndarray, " n"]
    low: Float[np.ndarray, " n"]
    upp: Float[np.ndarray, " n"]
    loop: int
    rou: float  # gravity/stage-mask sigmoid sharpness
    beta: float  # Heaviside projection sharpness
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
    xPhys_traj: list[Float[np.ndarray, "nely nelx"]]
    tPhys_traj: list[Float[np.ndarray, "nely nelx"]]  # length nloop+1
    records: list[IterationRecord]  # length nloop


def build_problem(
    nelx: int,
    nely: int,
    nStage: int,
    volfrac: float,
    Theta: float,
    Tcr: float,
    tfield: int,
    rmin: float,
    lrmin: float,
    rmin_cond: float,
    *,
    Emin: float = 1e-9,
    Emax: float = 1.0,
    nu: float = 0.3,
    penal: float = 3.0,
    eta: float = 0.5,
    beta_max: float = 128.0,
    p: float = 25.0,
    q: float = 3.0,
    r: float = 0.05,
    rouf: float = 100.0,
    a0: float = 1.0,
    mma_c: float = 2500.0,
    move: float = 0.01,
    tmove: float = 0.01,
) -> Problem:
    """Build the fixed FEM/filter/geometry setup once, before the loop starts.

    `rmin`/`lrmin`/`rmin_cond` are filter radii (density filter, continuity filter,
    conductivity neighborhood) -- passed explicitly rather than hardcoded so callers
    (e.g. the E2E test) can match whatever grid they're running on, since the fixture's
    radii differ from the original full-scale script's.
    """
    KE = fem.plane_stress_KE(nu)
    edofMat = fem.element_dof_map(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Fixed point load + boundary condition, matching generate_fixtures.m / test_fem.py:
    # MATLAB's 1-indexed `F(2*(nelx+1)*(nely+1))=-1`, `fixeddofs=1:2*(nely+1)`.
    F = np.zeros(ndof)
    F[2 * (nelx + 1) * (nely + 1) - 1] = -1.0
    fixeddofs = np.arange(2 * (nely + 1))
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    H, Hs = filters.density_filter(nelx, nely, rmin)
    L = filters.continuity_filter(nelx, nely, lrmin)
    C = gravity.gravity_load_matrix(nelx, nely)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)

    # Print-start element(s): the whole first mesh column for tfield != 1, the single
    # origin element for tfield == 1 (constraints.start_point's own docstring).
    Nei = np.array([0]) if tfield == 1 else np.arange(nely)

    n = 2 * nelx * nely
    # MATLAB hardcodes `m = 1 + 1 + nely + 2*nStage + 1` -- only self-consistent when
    # tfield != 1 (Nei has nely rows); computing from len(Nei) generalizes correctly.
    m = 1 + 1 + len(Nei) + 2 * nStage + 1

    return Problem(
        nelx=nelx,
        nely=nely,
        nStage=nStage,
        volfrac=volfrac,
        Theta=Theta,
        Tcr=Tcr,
        tfield=tfield,
        KE=KE,
        edofMat=edofMat,
        freedofs=freedofs,
        F=F,
        ndof=ndof,
        H=H,
        Hs=Hs,
        L=L,
        C=C,
        e1=e1,
        e2=e2,
        w=w,
        Nei=Nei,
        Emin=Emin,
        Emax=Emax,
        penal=penal,
        eta=eta,
        beta_max=beta_max,
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


def init_state(problem: Problem, beta: float) -> State:
    """Initial state: uniform density at `problem.volfrac`, time field per `problem.tfield`,
    matching `generate_fixtures.m`'s pre-loop setup (lines ~199-208).
    """
    nely, nelx = problem.nely, problem.nelx
    nel = nelx * nely

    x = np.full((nely, nelx), problem.volfrac)
    # unfiltered at init, unlike every later iteration's H-filtered xTilde
    xTilde = x.copy()
    xPhys = filters.heaviside_projection(xTilde, beta, problem.eta)

    tPhys = timefield.init_timefield(nelx, nely, problem.tfield)
    t = tPhys.copy()  # also unfiltered at init, unlike every later iteration's tPhys

    # MATLAB's xold1=xold2=[x(:); zeros(nel,1)] -- provably unread (iterations 1 and 2
    # both take mmasub's `iteration < 2.5` reinit branch), reproduced anyway for fidelity.
    xold = np.concatenate([x.flatten(order="F"), np.zeros(nel)])

    return State(
        x=x,
        xTilde=xTilde,
        xPhys=xPhys,
        t=t,
        tPhys=tPhys,
        xold1=xold,
        xold2=xold.copy(),
        low=np.zeros(problem.n),
        upp=np.zeros(problem.n),
        loop=0,
        rou=10.0,
        beta=beta,
        factor=1.0,
    )


def step(problem: Problem, state: State) -> tuple[State, IterationRecord]:
    """Run one optimization iteration: build the objective + every constraint's
    value/sensitivity in the reference's exact row order, call `mma.mmasub`, and
    unpack the result into the next state.

    The three periodic state updates below (`rou += 5` at loop%30==0, `beta *= 2` at
    loop%50==0, the hotspot `factor` refresh at loop%25==0) are implemented faithfully
    but never trigger against the small E2E fixture (`nloop=3`) -- unexercised by that
    fixture, not unimplemented or worked around.
    """
    p = problem
    nely, nelx, nStage = p.nely, p.nelx, p.nStage
    nel = nelx * nely

    loop = state.loop + 1
    rou = state.rou
    if loop % 30 == 0 and rou < 50:
        rou += 5
    beta = state.beta
    if loop % 50 == 0 and beta <= p.beta_max:
        beta *= 2
    if beta > p.beta_max:
        beta = p.beta_max

    xPhys, tPhys = state.xPhys, state.tPhys
    # Heaviside-derivative chain rule for this iteration's density sensitivities, from
    # the xTilde carried over from the *previous* iteration's update (or init).
    dx = filters.heaviside_projection_derivative(state.xTilde, beta, p.eta)

    # -- Objective: whole-structure compliance + Theta-weighted per-stage gravity compliance --
    c, dcx = compliance.whole_compliance(
        xPhys, p.KE, p.edofMat, p.Emin, p.Emax, p.penal, p.freedofs, p.F, p.ndof
    )
    obj_final_only = c  # compliance of final structure only, saved for logging
    obj = c
    dc = p.H @ (dcx.flatten(order="F") * dx.flatten(order="F") / p.Hs)
    dt = np.zeros(nel)

    tP = np.linspace(0, 1, nStage + 1)
    for i in range(1, nStage + 1):
        ti = tP[i]
        cg, dcx_g, dct_g = compliance.gravity_compliance(
            xPhys,
            tPhys,
            p.KE,
            p.edofMat,
            p.Emin,
            p.Emax,
            p.penal,
            ti,
            p.C,
            rou,
            p.freedofs,
            p.ndof,
        )
        obj += p.Theta * cg
        dc = dc + p.Theta * (p.H @ (dcx_g * dx.flatten(order="F") / p.Hs))
        dt = dt + p.Theta * (p.H @ (dct_g / p.Hs))

    df0dx = np.concatenate([dc, dt])
    f0val = obj

    # -- Move-limit bounds on this iteration's raw MMA variables --
    xflat, tflat = state.x.flatten(order="F"), state.t.flatten(order="F")
    xminx = np.maximum(0.0, xflat - p.move)
    xmaxx = np.minimum(1.0, xflat + p.move)
    xmint = np.maximum(0.0, tflat - p.tmove)
    xmaxt = np.minimum(1.0, tflat + p.tmove)
    xmin = np.concatenate([xminx, xmint])
    xmax = np.concatenate([xmaxx, xmaxt])
    xval = np.concatenate([xflat, tflat])

    # -- Constraints, stacked in the reference loop's exact order --
    fval_parts: list[np.ndarray] = []
    dfdx_parts: list[np.ndarray] = []

    fv, dfx, dft = constraints.global_volume_fraction(xPhys, dx, p.H, p.Hs, p.volfrac)
    vol_diag = float(np.sum(xPhys) / (nelx * nely))
    fval_parts.append(np.array([fv]))
    dfdx_parts.append(np.concatenate([dfx, dft])[None, :])

    fv, dfx, dft = constraints.time_field_continuity(tPhys, p.L, p.H, p.Hs)
    fval_parts.append(np.array([fv]))
    dfdx_parts.append(np.concatenate([dfx, dft])[None, :])

    fv, dfx, dft = constraints.start_point(tPhys, p.Nei, p.H, p.Hs)
    fval_parts.append(fv)
    dfdx_parts.append(np.concatenate([dfx, dft], axis=1))

    for stage in range(1, nStage + 1):
        fu, fl, dfx, dft = constraints.stage_volume_bounds(
            xPhys, tPhys, dx, p.H, p.Hs, stage, nStage, p.volfrac, rou
        )
        fval_parts.append(np.array([fu, fl]))
        dfdx_parts.append(
            np.stack([np.concatenate([dfx, dft]), np.concatenate([-dfx, -dft])])
        )

    # Hotspot constraint: `factor` is refreshed every 25 iterations. `numer`/`K_est` come
    # back from this one call, so the refresh below is a pure rescale, not a recompute.
    fv, df1, dt1, numer, K_est = conductivity.hotspot_constraint(
        xPhys,
        tPhys,
        p.e1,
        p.e2,
        p.w,
        dx,
        p.H,
        p.Hs,
        state.factor,
        p.Tcr,
        p.p,
        p.q,
        p.r,
        p.rouf,
    )
    factor = state.factor
    if loop % 25 == 0:
        max_g = float(np.max((1 - K_est) * xPhys.flatten(order="F") ** p.r))
        factor = max_g / numer
        # Exact rescale: fval and its sensitivities are both linear in `factor`
        # (conductivity.hotspot_constraint's `fval`/`scale`), so no need to recompute.
        rescale = factor / state.factor
        fv = factor * numer / p.Tcr - 1
        df1 = df1 * rescale
        dt1 = dt1 * rescale
    tru_max = factor * numer
    fval_parts.append(np.array([fv]))
    dfdx_parts.append(np.concatenate([df1, dt1])[None, :])

    fval = np.concatenate(fval_parts)
    dfdx = np.concatenate(dfdx_parts, axis=0)

    # -- MMA subproblem solve and update --
    mma_a = np.zeros(p.m)
    mma_c = np.full(p.m, p.mma_c)
    mma_d = np.zeros(p.m)
    xmma, ymma, zmma, lam, xsi, mma_eta, mu, zet, s, low, upp = mma.mmasub(
        p.m,
        p.n,
        loop,
        xval,
        xmin,
        xmax,
        state.xold1,
        state.xold2,
        f0val,
        df0dx,
        fval,
        dfdx,
        state.low,
        state.upp,
        p.a0,
        mma_a,
        mma_c,
        mma_d,
    )

    xnew = xmma.reshape(nely, -1, order="F")
    s_new = xnew[:, :nelx]
    t_new = xnew[:, nelx:]

    xTilde_new = (p.H @ s_new.flatten(order="F") / p.Hs).reshape(nely, nelx, order="F")
    xPhys_new = filters.heaviside_projection(xTilde_new, beta, p.eta)
    tPhys_new = (p.H @ t_new.flatten(order="F") / p.Hs).reshape(nely, nelx, order="F")

    new_state = State(
        x=s_new,
        xTilde=xTilde_new,
        xPhys=xPhys_new,
        t=t_new,
        tPhys=tPhys_new,
        xold1=xval,
        xold2=state.xold1,
        low=low,
        upp=upp,
        loop=loop,
        rou=rou,
        beta=beta,
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
    tfield: int,
    rmin: float,
    lrmin: float,
    rmin_cond: float,
    *,
    beta: float = 1.0,
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
    state = init_state(problem, beta)

    xPhys_traj = [state.xPhys.copy()]
    tPhys_traj = [state.tPhys.copy()]
    records: list[IterationRecord] = []

    for _ in range(nloop):
        state, record = step(problem, state)
        xPhys_traj.append(state.xPhys.copy())
        tPhys_traj.append(state.tPhys.copy())
        records.append(record)

    return RunResult(
        state=state, xPhys_traj=xPhys_traj, tPhys_traj=tPhys_traj, records=records
    )
